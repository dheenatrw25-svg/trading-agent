from __future__ import annotations

import csv
import datetime
import json
import math
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Any

from .paths import LOGS_DIR
from .polymarket_client import PolymarketClient


STATE_FILE = LOGS_DIR / "polymarket_paper_state.json"
TRADES_FILE = LOGS_DIR / "polymarket_paper_trades.csv"
RESULTS_FILE = LOGS_DIR / "polymarket_paper_results.csv"
DEFAULT_BANKROLL = 100.0
DEFAULT_STAKE = 5.0
CRYPTO_TAKER_FEE_RATE = 0.07
ENTRY_GRACE_SECONDS = 60
WINDOW_MATCH_TOLERANCE_SECONDS = 60

TRADE_FIELDS = [
    "trade_id",
    "status",
    "market_id",
    "question",
    "forecast_reference",
    "entry_time",
    "outcome_bought",
    "entry_odds",
    "stake",
    "shares",
    "expiry",
    "spread",
    "fee",
    "resolved_outcome",
    "pnl",
    "virtual_balance",
    "skip_reason",
]
RESULT_FIELDS = [
    "trade_id",
    "market_id",
    "question",
    "forecast_reference",
    "entry_time",
    "outcome_bought",
    "entry_odds",
    "stake",
    "shares",
    "expiry",
    "spread",
    "fee",
    "resolved_outcome",
    "settled_at",
    "payout",
    "pnl",
    "virtual_balance",
]

_STATE_LOCK = threading.RLock()


class PaperTradeError(ValueError):
    pass


def _utc_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _iso(value: datetime.datetime) -> str:
    if value.tzinfo is None:
        raise PaperTradeError("Paper-trading timestamps must be timezone-aware.")
    return value.astimezone(datetime.timezone.utc).isoformat()


def _parse_time(value: object, label: str) -> datetime.datetime:
    if not isinstance(value, str) or not value.strip():
        raise PaperTradeError(f"{label} is missing.")
    try:
        parsed = datetime.datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise PaperTradeError(f"{label} is invalid.") from exc
    if parsed.tzinfo is None:
        raise PaperTradeError(f"{label} must include a timezone.")
    return parsed.astimezone(datetime.timezone.utc)


def _positive_config(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError as exc:
        raise PaperTradeError(f"{name} must be a positive number.") from exc
    if not math.isfinite(value) or value <= 0:
        raise PaperTradeError(f"{name} must be a positive number.")
    return value


def default_state() -> dict[str, Any]:
    bankroll = _positive_config("POLYMARKET_PAPER_BANKROLL", DEFAULT_BANKROLL)
    return {
        "starting_balance": bankroll,
        "current_balance": bankroll,
        "open_positions": [],
        "entered_market_ids": [],
        "processed_forecasts": [],
    }


def _load_state(state_path: Path) -> dict[str, Any]:
    if not state_path.exists():
        return default_state()
    with state_path.open("r", encoding="utf-8") as file:
        state = json.load(file)
    if not isinstance(state, dict):
        raise PaperTradeError(f"Paper state {state_path} must contain a JSON object.")
    merged = default_state()
    merged.update(state)
    for key in ("open_positions", "entered_market_ids", "processed_forecasts"):
        if not isinstance(merged[key], list):
            raise PaperTradeError(f"Paper state field {key} must be a list.")
    balance = merged.get("current_balance")
    if isinstance(balance, bool) or not isinstance(balance, (int, float)):
        raise PaperTradeError("Paper-trading balance is invalid.")
    if not math.isfinite(float(balance)) or float(balance) < 0:
        raise PaperTradeError("Paper-trading balance is invalid.")
    return merged


def _save_state(state_path: Path, state: dict[str, Any]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = state_path.with_suffix(state_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(state, file, indent=2)
    temporary_path.replace(state_path)


def _append_csv(path: Path, fields: list[str], row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open("r", newline="", encoding="utf-8-sig") as file:
            existing_fields = next(csv.reader(file), [])
        if existing_fields != fields:
            raise PaperTradeError(
                f"Paper log {path} has an unexpected CSV header; it was not modified."
            )
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="raise")
        if file.tell() == 0:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fields})


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return isinstance(value, str) and value.strip().casefold() in {
        "true",
        "1",
        "yes",
    }


def _list_value(value: object, label: str) -> list[Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise PaperTradeError(f"Market {label} is invalid JSON.") from exc
    if not isinstance(value, list):
        raise PaperTradeError(f"Market {label} must be a list.")
    return value


def _forecast_details(
    row: dict[str, Any],
    forecast_type: str,
) -> tuple[str, str, datetime.datetime, datetime.datetime, int]:
    if forecast_type == "advisory":
        direction = str(row.get("direction", "")).strip().upper()
        issue = _parse_time(row.get("forecast_issue_timestamp"), "Advisory issue time")
        target = _parse_time(row.get("target_timestamp"), "Advisory target time")
        reference = f"advisory:{row.get('forecast_issue_timestamp', '')}"
        required_horizon = 4
    elif forecast_type == "forecast":
        direction = str(row.get("predicted_direction", "")).strip().upper()
        issue = _parse_time(
            row.get("forecast_issue_timestamp"), "2H forecast issue time"
        )
        target = _parse_time(
            row.get("intended_evaluation_timestamp"), "2H forecast target time"
        )
        reference = str(row.get("forecast_id", "")).strip()
        required_horizon = 2
    else:
        raise PaperTradeError("Forecast type must be 'advisory' or 'forecast'.")

    allowed_directions = (
        {"UP", "DOWN"}
        if forecast_type == "advisory"
        else {"BULLISH", "BEARISH", "SIDEWAYS"}
    )
    if direction not in allowed_directions:
        raise PaperTradeError("Forecast direction is not a tradable UP/DOWN direction.")
    raw_horizon = row.get("forecast_horizon_hours")
    if raw_horizon in (None, ""):
        raise PaperTradeError("Forecast horizon is missing.")
    try:
        horizon = int(float(raw_horizon))
    except (TypeError, ValueError) as exc:
        raise PaperTradeError("Forecast horizon is invalid.") from exc
    if horizon != required_horizon:
        raise PaperTradeError(
            f"{forecast_type} forecast horizon must be {required_horizon} hours."
        )
    if target - issue != datetime.timedelta(hours=required_horizon):
        raise PaperTradeError(
            f"{forecast_type} forecast timestamps do not span exactly "
            f"{required_horizon} hours."
        )
    if not reference:
        raise PaperTradeError("Forecast reference is missing.")
    outcome = (
        "UP"
        if direction in {"UP", "BULLISH"}
        else "DOWN"
        if direction in {"DOWN", "BEARISH"}
        else ""
    )
    return outcome, reference, issue, target, required_horizon


def _is_btc_up_down_market(market: dict[str, Any]) -> bool:
    question = str(market.get("question", "")).casefold()
    if not (re.search(r"\bbitcoin\b|\bbtc\b", question)):
        return False
    if not re.search(r"\bup\b", question) or not re.search(r"\bdown\b", question):
        return False
    try:
        outcomes = [
            str(value).strip().casefold()
            for value in _list_value(market.get("outcomes"), "outcomes")
        ]
    except PaperTradeError:
        return False
    return "up" in outcomes and "down" in outcomes


def _rules_are_verifiable(market: dict[str, Any]) -> bool:
    description = str(market.get("description", "")).casefold()
    source = str(market.get("resolutionSource", "")).strip()
    has_open_reference = bool(
        re.search(r"\b(beginning|start(?:ing)?|opening|price to beat)\b", description)
    )
    has_close_reference = bool(
        re.search(r"\b(end|ending|close|closing|expiry|expiration)\b", description)
    )
    has_price_comparison = "price" in description and any(
        word in description
        for word in ("higher", "lower", "greater", "less", "above", "below")
    )
    return bool(source and has_open_reference and has_close_reference and has_price_comparison)


def _market_outcome_token(
    market: dict[str, Any],
    outcome: str,
) -> tuple[str, str]:
    outcomes = _list_value(market.get("outcomes"), "outcomes")
    token_ids = _list_value(market.get("clobTokenIds"), "CLOB token IDs")
    if len(outcomes) != len(token_ids):
        raise PaperTradeError("Market outcomes do not align with CLOB token IDs.")
    target = outcome.casefold()
    for label, token_id in zip(outcomes, token_ids):
        if str(label).strip().casefold() == target:
            if not isinstance(token_id, (str, int)) or not str(token_id).strip():
                break
            return str(label), str(token_id)
    raise PaperTradeError(f"Market does not provide a {outcome} outcome token.")


def _matching_market(
    markets: list[dict[str, Any]],
    issue: datetime.datetime,
    target: datetime.datetime,
    horizon: int,
    now: datetime.datetime,
) -> tuple[dict[str, Any] | None, str]:
    tolerance = datetime.timedelta(seconds=WINDOW_MATCH_TOLERANCE_SECONDS)
    duration = datetime.timedelta(hours=horizon)
    for market in markets:
        if _truthy(market.get("closed")):
            continue
        if "active" in market and not _truthy(market.get("active")):
            continue
        if not _is_btc_up_down_market(market) or not _rules_are_verifiable(market):
            continue
        try:
            start = _parse_time(market.get("startDate"), "Market opening time")
            end = _parse_time(market.get("endDate"), "Market closing time")
        except PaperTradeError:
            continue
        if abs((end - start) - duration) > datetime.timedelta(seconds=1):
            continue
        if abs(start - issue) > tolerance or abs(end - target) > tolerance:
            continue
        age_seconds = (now - start).total_seconds()
        if age_seconds < 0:
            return None, "Market measurement window has not started; entry is not made early."
        if age_seconds > ENTRY_GRACE_SECONDS:
            return None, "Forecast is stale; the market window entry grace period has passed."
        if not market.get("id"):
            continue
        return market, ""
    return (
        None,
        f"No verified BTC Up/Down market matches the forecast's exact {horizon}H "
        "measurement window and resolution rules.",
    )


def _book_levels(book: dict[str, Any], side: str) -> list[tuple[float, float]]:
    raw_levels = book.get(side)
    if not isinstance(raw_levels, list):
        raise PaperTradeError(f"Order book {side} levels are missing.")
    levels: list[tuple[float, float]] = []
    for level in raw_levels:
        if not isinstance(level, dict):
            raise PaperTradeError(f"Order book contains an invalid {side} level.")
        try:
            price = float(level["price"])
            size = float(level["size"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PaperTradeError(f"Order book contains an invalid {side} level.") from exc
        if (
            not math.isfinite(price)
            or not math.isfinite(size)
            or not 0 < price < 1
            or size <= 0
        ):
            raise PaperTradeError(
                f"Order book contains an invalid {side} price or size."
            )
        levels.append((price, size))
    levels.sort(key=lambda level: level[0], reverse=(side == "bids"))
    return levels


def _fee_rate(market: dict[str, Any]) -> float:
    enabled_value = market.get("feesEnabled")
    enabled = _truthy(enabled_value)
    for key in ("takerFeeRate", "feeRate", "fee_rate"):
        if market.get(key) is not None:
            try:
                rate = float(market[key])
            except (TypeError, ValueError) as exc:
                raise PaperTradeError("Market taker fee rate is invalid.") from exc
            if not math.isfinite(rate) or not 0 <= rate <= 1:
                raise PaperTradeError("Market taker fee rate is outside 0-1.")
            return rate if enabled_value is None or enabled else 0.0
    if enabled:
        return CRYPTO_TAKER_FEE_RATE
    if enabled_value is not None:
        return 0.0
    raise PaperTradeError("Market fee status is unavailable; refusing to guess the fee.")


def _simulate_buy(
    book: dict[str, Any],
    stake: float,
    fee_rate: float,
) -> dict[str, float]:
    asks = _book_levels(book, "asks")
    bids = _book_levels(book, "bids")
    if not asks or not bids:
        raise PaperTradeError("Order book has no two-sided quote; spread cannot be verified.")
    best_ask = asks[0][0]
    best_bid = bids[0][0]
    if best_ask < best_bid:
        raise PaperTradeError("Order book is crossed; quote is invalid.")
    spread = best_ask - best_bid
    remaining = stake
    shares = 0.0
    fees = 0.0
    for price, available_shares in asks:
        fill_cost = min(remaining, price * available_shares)
        fill_shares = fill_cost / price
        shares += fill_shares
        fees += fill_shares * fee_rate * price * (1 - price)
        remaining -= fill_cost
        if remaining <= 1e-9:
            break
    if remaining > 1e-6:
        raise PaperTradeError(
            f"Insufficient ask liquidity for the full ${stake:.2f} paper stake."
        )
    fees = round(fees, 5)
    return {
        "entry_odds": stake / shares,
        "shares": shares,
        "stake": stake,
        "spread": spread,
        "fee": fees,
        "best_ask": best_ask,
    }


def _skip(
    state: dict[str, Any],
    state_path: Path,
    trades_path: Path,
    forecast_reference: str,
    reason: str,
    market_id: str = "",
    question: str = "",
    now: datetime.datetime | None = None,
) -> dict[str, Any]:
    row = {
        "trade_id": "",
        "status": "SKIPPED",
        "market_id": market_id,
        "question": question,
        "forecast_reference": forecast_reference,
        "entry_time": _iso(now or _utc_now()),
        "outcome_bought": "",
        "entry_odds": "",
        "stake": "",
        "shares": "",
        "expiry": "",
        "spread": "",
        "fee": "",
        "resolved_outcome": "",
        "pnl": "",
        "virtual_balance": round(float(state["current_balance"]), 2),
        "skip_reason": reason,
    }
    _append_csv(trades_path, TRADE_FIELDS, row)
    state["processed_forecasts"].append(forecast_reference)
    _save_state(state_path, state)
    return {"status": "skipped", "reason": reason}


def process_forecast(
    row: dict[str, Any],
    forecast_type: str,
    *,
    client: PolymarketClient | None = None,
    state_path: Path = STATE_FILE,
    trades_path: Path = TRADES_FILE,
    now: datetime.datetime | None = None,
) -> dict[str, Any]:
    with _STATE_LOCK:
        outcome, reference, issue, target, horizon = _forecast_details(
            row, forecast_type
        )
        state = _load_state(state_path)
        if reference in state["processed_forecasts"]:
            return {"status": "ignored", "reason": "Forecast was already processed."}
        current_time = now or _utc_now()
        if current_time.tzinfo is None:
            raise PaperTradeError("Paper-trading clock must be timezone-aware.")
        current_time = current_time.astimezone(datetime.timezone.utc)
        if not outcome:
            return _skip(
                state,
                state_path,
                trades_path,
                reference,
                "2H forecast direction is SIDEWAYS; no directional paper entry was attempted.",
                now=current_time,
            )
        api = client or PolymarketClient()
        markets = api.find_markets(issue, target)
        current_time = now or _utc_now()
        current_time = current_time.astimezone(datetime.timezone.utc)
        market, reason = _matching_market(markets, issue, target, horizon, current_time)
        if market is None:
            return _skip(
                state,
                state_path,
                trades_path,
                reference,
                reason,
                now=current_time,
            )

        market_id = str(market["id"])
        question = str(market.get("question", ""))
        if market_id in state["entered_market_ids"]:
            return _skip(
                state,
                state_path,
                trades_path,
                reference,
                "A paper position was already entered for this market.",
                market_id,
                question,
                current_time,
            )

        try:
            bought_label, token_id = _market_outcome_token(market, outcome)
            book = api.get_order_book(token_id)
            indicated_ask = api.get_price(token_id, "BUY")
            stake = _positive_config("POLYMARKET_PAPER_STAKE", DEFAULT_STAKE)
            quote = _simulate_buy(book, stake, _fee_rate(market))
            if abs(indicated_ask - quote["best_ask"]) > 1e-6:
                raise PaperTradeError(
                    "CLOB market price does not agree with the order-book best ask."
                )
        except PaperTradeError as exc:
            return _skip(
                state,
                state_path,
                trades_path,
                reference,
                str(exc),
                market_id,
                question,
                current_time,
            )

        entry_time = now or _utc_now()
        if entry_time.tzinfo is None:
            raise PaperTradeError("Paper-trading clock must be timezone-aware.")
        entry_time = entry_time.astimezone(datetime.timezone.utc)
        still_fresh, stale_reason = _matching_market(
            [market], issue, target, horizon, entry_time
        )
        if still_fresh is None:
            return _skip(
                state,
                state_path,
                trades_path,
                reference,
                stale_reason,
                market_id,
                question,
                entry_time,
            )

        total_cost = stake + quote["fee"]
        balance = float(state["current_balance"])
        if total_cost > balance + 1e-9:
            return _skip(
                state,
                state_path,
                trades_path,
                reference,
                f"Insufficient virtual balance for stake plus fees "
                f"(${total_cost:.5f} required; ${balance:.2f} available).",
                market_id,
                question,
                entry_time,
            )

        trade_id = f"{market_id}:{reference}"
        trade = {
            "trade_id": trade_id,
            "status": "OPEN",
            "market_id": market_id,
            "question": question,
            "forecast_reference": reference,
            "entry_time": _iso(entry_time),
            "outcome_bought": bought_label,
            "entry_odds": round(quote["entry_odds"], 8),
            "stake": round(stake, 2),
            "shares": round(quote["shares"], 8),
            "expiry": _iso(_parse_time(market.get("endDate"), "Market closing time")),
            "spread": round(quote["spread"], 8),
            "fee": quote["fee"],
            "resolved_outcome": "",
            "pnl": "",
            "virtual_balance": round(balance - total_cost, 2),
            "skip_reason": "",
        }
        position = {
            **trade,
            "token_id": token_id,
            "entry_fee": quote["fee"],
        }
        state["current_balance"] = round(balance - total_cost, 2)
        state["open_positions"].append(position)
        state["entered_market_ids"].append(market_id)
        state["processed_forecasts"].append(reference)
        _save_state(state_path, state)
        _append_csv(trades_path, TRADE_FIELDS, trade)
        return {"status": "entered", "trade": trade}


def _resolved_outcome(market: dict[str, Any]) -> str | None:
    if not _truthy(market.get("closed")):
        return None
    resolution_status = str(market.get("umaResolutionStatus", "")).strip().casefold()
    if resolution_status not in {"resolved", "settled"} and not _truthy(
        market.get("resolved")
    ):
        return None
    try:
        outcomes = _list_value(market.get("outcomes"), "outcomes")
        prices = _list_value(market.get("outcomePrices"), "outcome prices")
    except PaperTradeError:
        return None
    if len(outcomes) != len(prices) or len(outcomes) < 2:
        return None
    final_labels = []
    for label, raw_price in zip(outcomes, prices):
        try:
            price = float(raw_price)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(price) or not 0 <= price <= 1:
            return None
        if price >= 0.999:
            final_labels.append(str(label))
        elif price > 0.001:
            return None
    return final_labels[0] if len(final_labels) == 1 else None


def settle_open_positions(
    *,
    client: PolymarketClient | None = None,
    state_path: Path = STATE_FILE,
    results_path: Path = RESULTS_FILE,
    now: datetime.datetime | None = None,
) -> dict[str, int]:
    with _STATE_LOCK:
        state = _load_state(state_path)
        api = client or PolymarketClient()
        current_time = now or _utc_now()
        if current_time.tzinfo is None:
            raise PaperTradeError("Paper-trading clock must be timezone-aware.")
        current_time = current_time.astimezone(datetime.timezone.utc)
        settled = 0
        pending = 0
        still_open = []
        for position in state["open_positions"]:
            market = api.get_market(str(position["market_id"]))
            winner = _resolved_outcome(market)
            if winner is None:
                still_open.append(position)
                pending += 1
                continue
            won = winner.casefold() == str(position["outcome_bought"]).casefold()
            payout = float(position["shares"]) if won else 0.0
            pnl = payout - float(position["stake"]) - float(position["entry_fee"])
            state["current_balance"] = round(
                float(state["current_balance"]) + payout, 2
            )
            result = {
                field: position.get(field, "") for field in RESULT_FIELDS
            }
            result.update(
                {
                    "resolved_outcome": winner,
                    "settled_at": _iso(current_time),
                    "payout": round(payout, 8),
                    "pnl": round(pnl, 8),
                    "virtual_balance": round(float(state["current_balance"]), 2),
                }
            )
            _append_csv(results_path, RESULT_FIELDS, result)
            settled += 1
        state["open_positions"] = still_open
        _save_state(state_path, state)
        return {"settled": settled, "pending": pending}
