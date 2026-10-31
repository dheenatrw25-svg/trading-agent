import json
import math
from typing import Any

import numpy as np
import pandas as pd
import requests
import yfinance as yf


BINANCE_FUTURES_BASE_URL = "https://fapi.binance.com"
REQUEST_TIMEOUT_SECONDS = 10


def _confirmed_candles(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    if df.empty:
        raise RuntimeError("yfinance returned no BTC-USD spot candles.")
    required_columns = {"Open", "High", "Low", "Close", "Volume"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(
            "BTC-USD candle data is missing columns: "
            + ", ".join(sorted(missing_columns))
            + "."
        )

    interval_delta = pd.Timedelta(interval)
    timestamps = pd.to_datetime(df.index, utc=True, errors="coerce")
    now_utc = pd.Timestamp.now(tz="UTC")
    confirmed = timestamps.notna() & (timestamps + interval_delta <= now_utc)
    result = df.loc[confirmed].copy()
    result.index = timestamps[confirmed]
    result = result.loc[~result.index.duplicated(keep="last")].sort_index()
    result = result.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    try:
        result[["Open", "High", "Low", "Close", "Volume"]] = result[
            ["Open", "High", "Low", "Close", "Volume"]
        ].apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("BTC-USD OHLCV candle values must be numeric.") from exc
    ohlcv = result[["Open", "High", "Low", "Close", "Volume"]].to_numpy(dtype=float)
    valid_rows = (
        np.isfinite(ohlcv).all(axis=1)
        & (ohlcv[:, :4] > 0).all(axis=1)
        & (ohlcv[:, 1] >= ohlcv[:, 0])
        & (ohlcv[:, 1] >= ohlcv[:, 2])
        & (ohlcv[:, 1] >= ohlcv[:, 3])
        & (ohlcv[:, 2] <= ohlcv[:, 0])
        & (ohlcv[:, 2] <= ohlcv[:, 3])
        & (ohlcv[:, 4] >= 0)
    )
    result = result.loc[valid_rows]
    if result.empty:
        raise RuntimeError("No complete BTC-USD spot candles are available.")

    if len(result) < 2:
        raise RuntimeError("At least two complete BTC-USD spot candles are required.")
    return result


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def fetch_spot_data() -> pd.DataFrame:
    print("[DATA] Fetching BTC-USD spot candles (4h, 15d)...")
    df = yf.download(
        "BTC-USD",
        period="15d",
        interval="4h",
        progress=False,
    )
    df = _flatten_columns(df)
    df = _confirmed_candles(df, "4h")
    print(f"[DATA] Retrieved {len(df)} spot candles.")
    return df


def _fetch_15m_indicators() -> dict[str, float]:
    print("[DATA] Fetching BTC-USD 15-minute candles for RSI...")
    df = yf.download(
        "BTC-USD",
        period="5d",
        interval="15m",
        progress=False,
    )
    df = _flatten_columns(df)
    confirmed = _confirmed_candles(df, "15m")
    close = confirmed["Close"].astype(float)
    delta = close.diff()
    average_gain = delta.clip(lower=0).ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False,
    ).mean()
    average_loss = (-delta.clip(upper=0)).ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False,
    ).mean()
    latest_gain = float(average_gain.iloc[-1])
    latest_loss = float(average_loss.iloc[-1])
    if not math.isfinite(latest_gain) or not math.isfinite(latest_loss):
        raise RuntimeError("Insufficient completed 15-minute candles to calculate RSI.")
    if latest_gain == 0 and latest_loss == 0:
        rsi = 50.0
    elif latest_loss == 0:
        rsi = 100.0
    else:
        relative_strength = latest_gain / latest_loss
        rsi = 100 - (100 / (1 + relative_strength))
    latest_close = float(close.iloc[-1])
    print(f"[DATA] BTC-USD 15-minute close={latest_close:.2f}, RSI={rsi:.2f}.")
    return {
        "current_price": latest_close,
        "rsi_15m": round(rsi, 2),
    }


def fetch_macro_context() -> dict[str, float]:
    print("[DATA] Fetching US Dollar Index (DX-Y.NYB)...")
    df = yf.download(
        "DX-Y.NYB",
        period="5d",
        interval="1d",
        progress=False,
    )
    df = _flatten_columns(df).dropna(subset=["Close"])
    timestamps = pd.to_datetime(df.index, utc=True, errors="coerce")
    now_utc = pd.Timestamp.now(tz="UTC")
    confirmed = timestamps.notna() & (
        timestamps + pd.Timedelta(days=1) <= now_utc
    )
    df = df.loc[confirmed].copy()
    df.index = timestamps[confirmed]
    df = df.loc[~df.index.duplicated(keep="last")].sort_index()
    if len(df) < 2:
        raise RuntimeError(
            "yfinance must return at least two completed daily DXY closing prices "
            "to calculate the daily percentage change."
        )

    previous_close = float(df["Close"].iloc[-2])
    latest_close = float(df["Close"].iloc[-1])
    if (
        not math.isfinite(previous_close)
        or not math.isfinite(latest_close)
        or previous_close <= 0
        or latest_close <= 0
    ):
        raise RuntimeError(
            "Cannot calculate DXY daily change from invalid completed closing prices."
        )

    daily_change_pct = ((latest_close - previous_close) / previous_close) * 100
    print(
        f"[DATA] DXY close={latest_close:.4f}, "
        f"daily change={daily_change_pct:.4f}%."
    )
    return {
        "dxy_latest_close": latest_close,
        "dxy_daily_change_pct": daily_change_pct,
    }


def _fetch_open_interest_trend(symbol: str) -> dict[str, Any]:
    unavailable = {
        "open_interest_change_pct": None,
        "open_interest_change_interval_hours": None,
        "open_interest_trend": "unavailable",
        "open_interest_trend_reason": "",
    }
    try:
        response = requests.get(
            f"{BINANCE_FUTURES_BASE_URL}/futures/data/openInterestHist",
            params={"symbol": symbol, "period": "4h", "limit": 2},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        history = response.json()
    except (requests.RequestException, ValueError) as exc:
        unavailable["open_interest_trend_reason"] = (
            f"Binance historical open-interest data unavailable: {exc}"
        )
        print(f"[DATA] {unavailable['open_interest_trend_reason']}")
        return unavailable

    if not isinstance(history, list) or len(history) < 2:
        unavailable["open_interest_trend_reason"] = (
            "Binance returned fewer than two historical open-interest observations."
        )
        print(f"[DATA] {unavailable['open_interest_trend_reason']}")
        return unavailable
    try:
        observations = sorted(
            (
                int(record["timestamp"]),
                float(record["sumOpenInterest"]),
            )
            for record in history
        )
    except (KeyError, TypeError, ValueError) as exc:
        unavailable["open_interest_trend_reason"] = (
            f"Binance historical open-interest data had an invalid shape: {exc}"
        )
        print(f"[DATA] {unavailable['open_interest_trend_reason']}")
        return unavailable

    previous_timestamp, previous_value = observations[-2]
    latest_timestamp, latest_value = observations[-1]
    interval_hours = (latest_timestamp - previous_timestamp) / 3_600_000
    now_milliseconds = pd.Timestamp.now(tz="UTC").timestamp() * 1000
    if (
        not math.isfinite(previous_value)
        or not math.isfinite(latest_value)
        or previous_value <= 0
        or latest_value <= 0
        or not 1 <= interval_hours <= 8
        or now_milliseconds - latest_timestamp > 8 * 3_600_000
    ):
        unavailable["open_interest_trend_reason"] = (
            "Historical open-interest observations were invalid, stale, or not "
            "separated by a reliable interval."
        )
        print(f"[DATA] {unavailable['open_interest_trend_reason']}")
        return unavailable

    change_pct = ((latest_value - previous_value) / previous_value) * 100
    if change_pct > 1:
        trend = "rising"
    elif change_pct < -1:
        trend = "falling"
    else:
        trend = "stagnant"
    result = {
        "open_interest_change_pct": change_pct,
        "open_interest_change_interval_hours": interval_hours,
        "open_interest_trend": trend,
        "open_interest_trend_reason": (
            "Compared the two latest valid Binance 4h-period observations; "
            "changes within +/-1% are operationally classified as stagnant."
        ),
    }
    print(
        f"[DATA] Historical open interest changed {change_pct:.4f}% "
        f"over {interval_hours:.2f} hours ({trend})."
    )
    return result


def fetch_order_book_walls(symbol: str = "BTCUSDT") -> dict[str, Any]:
    try:
        response = requests.get(
            f"{BINANCE_FUTURES_BASE_URL}/fapi/v1/depth",
            params={"symbol": symbol, "limit": 1000},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        order_book = response.json()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Binance order book request failed for {symbol}: {exc}"
        ) from exc
    except ValueError as exc:
        raise RuntimeError(
            f"Binance returned invalid JSON for the {symbol} order book."
        ) from exc

    if not isinstance(order_book, dict):
        raise RuntimeError(f"Binance returned an invalid {symbol} order book.")

    def largest_wall(side_name: str, side_label: str) -> dict[str, float]:
        levels = order_book.get(side_name)
        if not isinstance(levels, list) or not levels:
            raise RuntimeError(
                f"Binance returned no {side_label} levels for {symbol}."
            )

        largest: tuple[float, float] | None = None
        for level in levels:
            if not isinstance(level, (list, tuple)) or len(level) < 2:
                raise RuntimeError(
                    f"Binance returned a malformed {side_label} level for {symbol}."
                )
            try:
                price, quantity = float(level[0]), float(level[1])
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"Binance returned a non-numeric {side_label} level for {symbol}."
                ) from exc
            if (
                not math.isfinite(price)
                or not math.isfinite(quantity)
                or price <= 0
                or quantity < 0
            ):
                raise RuntimeError(
                    f"Binance returned an invalid {side_label} level for {symbol}."
                )
            if largest is None or quantity > largest[1]:
                largest = (price, quantity)

        if largest is None or largest[1] <= 0:
            raise RuntimeError(
                f"Binance returned no positive-quantity {side_label} level for {symbol}."
            )
        price, quantity = largest
        return {
            "price_usd": price,
            "quantity_btc": quantity,
            "notional_usd": price * quantity,
        }

    return {
        "symbol": symbol,
        "depth_limit": 1000,
        "snapshot_timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "largest_buy_wall": largest_wall("bids", "bid"),
        "largest_sell_wall": largest_wall("asks", "ask"),
        "interpretation_note": (
            "A single snapshot shows displayed resting liquidity only; it cannot "
            "establish persistence or identify spoofing."
        ),
    }


def _fetch_taker_pressure(symbol: str) -> dict[str, Any]:
    unavailable = {
        "taker_pressure_status": "Unavailable",
        "taker_buy_volume_btc": None,
        "taker_sell_volume_btc": None,
        "net_taker_volume_btc": None,
        "taker_buy_sell_ratio": None,
        "taker_pressure_window_hours": None,
        "taker_pressure_timestamp_utc": None,
        "taker_pressure_reason": "",
    }
    try:
        response = requests.get(
            f"{BINANCE_FUTURES_BASE_URL}/futures/data/takerlongshortRatio",
            params={"symbol": symbol, "period": "15m", "limit": 12},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        history = response.json()
    except (requests.RequestException, ValueError) as exc:
        unavailable["taker_pressure_reason"] = (
            f"Binance taker-volume data unavailable: {exc}"
        )
        print(f"[DATA] {unavailable['taker_pressure_reason']}")
        return unavailable

    if not isinstance(history, list) or len(history) < 8:
        unavailable["taker_pressure_reason"] = (
            "Binance returned fewer than eight 15-minute taker-volume observations."
        )
        print(f"[DATA] {unavailable['taker_pressure_reason']}")
        return unavailable

    try:
        observations = sorted(
            (
                int(record["timestamp"]),
                float(record["buyVol"]),
                float(record["sellVol"]),
            )
            for record in history
        )
    except (KeyError, TypeError, ValueError) as exc:
        unavailable["taker_pressure_reason"] = (
            f"Binance taker-volume data had an invalid shape: {exc}"
        )
        print(f"[DATA] {unavailable['taker_pressure_reason']}")
        return unavailable

    if any(
        not math.isfinite(buy_volume)
        or not math.isfinite(sell_volume)
        or buy_volume < 0
        or sell_volume < 0
        for _, buy_volume, sell_volume in observations
    ):
        unavailable["taker_pressure_reason"] = (
            "Binance taker-volume observations contained invalid quantities."
        )
        print(f"[DATA] {unavailable['taker_pressure_reason']}")
        return unavailable

    observations = observations[-12:]
    latest_timestamp = observations[-1][0]
    now_milliseconds = pd.Timestamp.now(tz="UTC").timestamp() * 1000
    if (
        len({timestamp for timestamp, _, _ in observations}) != len(observations)
        or latest_timestamp > now_milliseconds + 60_000
        or now_milliseconds - latest_timestamp > 30 * 60_000
    ):
        unavailable["taker_pressure_reason"] = (
            "Binance taker-volume observations were duplicated or the latest "
            "observation was stale or future-dated."
        )
        print(f"[DATA] {unavailable['taker_pressure_reason']}")
        return unavailable

    buy_volume = sum(buy for _, buy, _ in observations)
    sell_volume = sum(sell for _, _, sell in observations)
    total_volume = buy_volume + sell_volume
    if total_volume <= 0:
        unavailable["taker_pressure_reason"] = (
            "Binance taker-volume observations contained no traded volume."
        )
        print(f"[DATA] {unavailable['taker_pressure_reason']}")
        return unavailable

    net_volume = buy_volume - sell_volume
    buy_sell_ratio = buy_volume / sell_volume if sell_volume > 0 else None
    imbalance = net_volume / total_volume
    if imbalance >= 0.05:
        status = "Aggressive Bid Lifting (Bullish CVD)"
    elif imbalance <= -0.05:
        status = "Aggressive Ask Hitting (Bearish CVD)"
    else:
        status = "Neutral/Choppy"

    window_hours = len(observations) * 15 / 60
    timestamp = pd.to_datetime(latest_timestamp, unit="ms", utc=True).isoformat()
    print(
        f"[DATA] Taker pressure over {window_hours:.2f} hours: {status} "
        f"(net {net_volume:.4f} BTC)."
    )
    return {
        "taker_pressure_status": status,
        "taker_buy_volume_btc": buy_volume,
        "taker_sell_volume_btc": sell_volume,
        "net_taker_volume_btc": net_volume,
        "taker_buy_sell_ratio": buy_sell_ratio,
        "taker_pressure_window_hours": window_hours,
        "taker_pressure_timestamp_utc": timestamp,
        "taker_pressure_reason": (
            "Aggregated Binance 15-minute taker buy/sell volumes; statuses require "
            "at least a 5% net-volume imbalance. This is a short-term flow proxy, "
            "not a guarantee of continued price direction."
        ),
    }


def fetch_derivatives_data(symbol: str = "BTCUSDT") -> dict[str, Any]:
    print(f"[DATA] Fetching Binance futures data for {symbol}...")
    endpoints = {
        "open_interest": f"{BINANCE_FUTURES_BASE_URL}/fapi/v1/openInterest",
        "premium_index": f"{BINANCE_FUTURES_BASE_URL}/fapi/v1/premiumIndex",
    }
    results: dict[str, dict[str, Any]] = {}

    for name, url in endpoints.items():
        try:
            response = requests.get(
                url,
                params={"symbol": symbol},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            results[name] = response.json()
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Binance {name.replace('_', ' ')} request failed for {symbol}: {exc}"
            ) from exc
        except ValueError as exc:
            raise RuntimeError(
                f"Binance returned invalid JSON for {name.replace('_', ' ')} "
                f"({symbol})."
            ) from exc

    try:
        open_interest = float(results["open_interest"]["openInterest"])
        funding_rate = float(results["premium_index"]["lastFundingRate"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Binance response for {symbol} is missing valid open interest or funding data."
        ) from exc
    if (
        not math.isfinite(open_interest)
        or open_interest <= 0
        or not math.isfinite(funding_rate)
    ):
        raise RuntimeError(
            f"Binance response for {symbol} contains invalid open interest or funding data."
        )

    annualized_funding_rate_pct = funding_rate * 3 * 365 * 100
    print(
        f"[DATA] Open interest={open_interest}, "
        f"annualized funding={annualized_funding_rate_pct:.4f}%."
    )
    return {
        "symbol": symbol,
        "open_interest": open_interest,
        "funding_rate": funding_rate,
        "annualized_funding_rate_pct": annualized_funding_rate_pct,
        **_fetch_open_interest_trend(symbol),
        **_fetch_taker_pressure(symbol),
    }


def _build_price_action_context(spot_data: pd.DataFrame) -> dict[str, Any]:
    confirmed = _confirmed_candles(spot_data, "4h")
    latest = confirmed.iloc[-1]
    latest_timestamp = pd.Timestamp(confirmed.index[-1])
    latest_volume = float(latest["Volume"])
    baseline = confirmed["Volume"].iloc[-21:-1]
    if len(baseline) < 5:
        volume_context: dict[str, Any] = {
            "latest_volume": latest_volume,
            "baseline_median_volume": None,
            "latest_to_baseline_volume_ratio": None,
            "volume_state": "unavailable",
            "volume_state_note": (
                "At least five prior completed candles are required for a volume comparison."
            ),
        }
    else:
        baseline_median = float(baseline.median())
        if not math.isfinite(baseline_median) or baseline_median <= 0:
            ratio = None
            state = "unavailable"
            note = "Prior completed-candle median volume is not positive."
        else:
            ratio = latest_volume / baseline_median
            if ratio >= 1.5:
                state = "expanding"
            elif ratio < 0.8:
                state = "declining"
            elif 0.8 <= ratio <= 1.2:
                state = "flat"
            else:
                state = "moderately_above_baseline"
            note = (
                "Operational thresholds versus the median of up to 20 prior "
                "completed candles: expanding >=1.5x; flat 0.8x-1.2x; "
                "declining <0.8x. These are explicit heuristics, not validated "
                "trading thresholds."
            )
        volume_context = {
            "latest_volume": latest_volume,
            "baseline_median_volume": baseline_median,
            "latest_to_baseline_volume_ratio": ratio,
            "volume_state": state,
            "volume_state_note": note,
        }

    swing_window = confirmed.tail(20)
    swing_high_index = swing_window["High"].idxmax()
    swing_low_index = swing_window["Low"].idxmin()
    latest_candle = {
        "timestamp": latest_timestamp.isoformat(),
        "close_timestamp": (latest_timestamp + pd.Timedelta(hours=4)).isoformat(),
        "open": float(latest["Open"]),
        "high": float(latest["High"]),
        "low": float(latest["Low"]),
        "close": float(latest["Close"]),
        "volume": latest_volume,
    }
    recent_completed_candles = []
    for timestamp, candle in confirmed.tail(20).iterrows():
        candle_timestamp = pd.Timestamp(timestamp)
        recent_completed_candles.append(
            {
                "timestamp": candle_timestamp.isoformat(),
                "close_timestamp": (
                    candle_timestamp + pd.Timedelta(hours=4)
                ).isoformat(),
                "open": float(candle["Open"]),
                "high": float(candle["High"]),
                "low": float(candle["Low"]),
                "close": float(candle["Close"]),
                "volume": float(candle["Volume"]),
            }
        )
    return {
        "candle_interval": "4h",
        "latest_completed_candle": latest_candle,
        "recent_completed_candles": recent_completed_candles,
        "volume_comparison": volume_context,
        "recent_swing_high": {
            "price": float(swing_window.loc[swing_high_index, "High"]),
            "timestamp": pd.Timestamp(swing_high_index).isoformat(),
            "definition": "Highest actual candle high in the latest 20 completed candles.",
        },
        "recent_swing_low": {
            "price": float(swing_window.loc[swing_low_index, "Low"]),
            "timestamp": pd.Timestamp(swing_low_index).isoformat(),
            "definition": "Lowest actual candle low in the latest 20 completed candles.",
        },
    }


def compile_market_context(spot_data: pd.DataFrame | None = None) -> dict[str, Any]:
    print("[DATA] Compiling market context...")
    if spot_data is None:
        spot_data = fetch_spot_data()
    if spot_data.empty:
        raise RuntimeError("Cannot compile market context from empty BTC-USD candles.")
    price_action = _build_price_action_context(spot_data)
    _confirmed_candles(spot_data, "4h")
    macro_context = fetch_macro_context()
    derivatives_data = fetch_derivatives_data()
    order_book_data = fetch_order_book_walls()
    technical_indicators = _fetch_15m_indicators()

    current_spot_close = technical_indicators["current_price"]
    context = {
        "spot": {
            "symbol": "BTC-USD",
            "current_close": current_spot_close,
        },
        "price_action": price_action,
        "macro": macro_context,
        "derivatives": derivatives_data,
        "order_book": order_book_data,
        "technical_indicators": {
            "rsi_15m": technical_indicators["rsi_15m"],
            "dxy_macro_trend": f"{macro_context['dxy_daily_change_pct']:+.1f}%",
        },
        "institutional_order_flow": _build_institutional_order_flow(
            order_book_data,
            derivatives_data,
        ),
    }
    context["current_price"] = technical_indicators["current_price"]
    print("[DATA] Market context compiled.")
    return context


def _build_institutional_order_flow(
    order_book: dict[str, Any],
    derivatives: dict[str, Any],
) -> dict[str, Any]:
    buy_wall = order_book.get("largest_buy_wall", {})
    sell_wall = order_book.get("largest_sell_wall", {})
    buy_price = buy_wall.get("price_usd")
    sell_price = sell_wall.get("price_usd")
    buy_notional = buy_wall.get("notional_usd")
    sell_notional = sell_wall.get("notional_usd")

    taker_status = derivatives.get("taker_pressure_status", "Unavailable")
    cvd_statuses = {
        "Aggressive Bid Lifting (Bullish CVD)": "Aggressive Market Buying (Bullish CVD)",
        "Aggressive Ask Hitting (Bearish CVD)": "Aggressive Market Selling (Bearish CVD)",
        "Neutral/Choppy": "Neutral/Choppy",
        "Unavailable": "Unavailable",
    }
    open_interest_statuses = {
        "rising": "Rising OI (Trend Supported)",
        "falling": "Falling OI (Position Closing/Reversal Risk)",
        "stagnant": "Stagnant OI (Limited Trend Confirmation)",
        "unavailable": "Unavailable",
    }
    return {
        "largest_buy_wall_price": buy_price,
        "largest_buy_wall_usd": (
            f"${buy_notional:,.0f}"
            if isinstance(buy_notional, (int, float))
            and math.isfinite(buy_notional)
            else "Unavailable"
        ),
        "largest_sell_wall_price": sell_price,
        "largest_sell_wall_usd": (
            f"${sell_notional:,.0f}"
            if isinstance(sell_notional, (int, float))
            and math.isfinite(sell_notional)
            else "Unavailable"
        ),
        "cvd_status": cvd_statuses.get(taker_status, "Unavailable"),
        "open_interest_trend": open_interest_statuses.get(
            derivatives.get("open_interest_trend"),
            "Unavailable",
        ),
    }


def build_ai_prompt_payload(market_context: dict[str, Any]) -> dict[str, Any]:
    """Project the rich internal context into the stable AI-facing schema."""
    return {
        "current_price": market_context["current_price"],
        "technical_indicators": market_context["technical_indicators"],
        "institutional_order_flow": market_context["institutional_order_flow"],
    }


if __name__ == "__main__":
    print(json.dumps(compile_market_context(), indent=2))
