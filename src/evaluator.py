import csv
import datetime
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from .paths import TRADE_MEMORIES_DIR
from .prediction_logs import (
    ADVISORY_LOG_FIELDS,
    ADVISORY_LOG_FILE,
    BTC_FORECAST_FIELDS,
    BTC_FORECAST_FILE,
    get_binance_close_at_timestamp,
    forecast_expiration_timestamp,
    initialize_advisory_log,
    initialize_btc_forecast_log,
)


FORECAST_HORIZON_HOURS = 2
ADVISORY_HORIZON_HOURS = 4


def _utc_timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError(f"Forecast timestamp must include a valid timezone: {value!r}.")
    return timestamp.tz_convert("UTC")


def _positive_number(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite positive number.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite positive number.") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{field} must be a finite positive number.")
    return number


def _validate_forecast_timestamps(row: dict[str, Any]) -> tuple[pd.Timestamp, pd.Timestamp]:
    if not str(row.get("forecast_id", "")).strip():
        raise ValueError("Forecast ID is missing.")
    if str(row.get("ticker", "")).strip().upper() != "BTC-USD":
        raise ValueError("Only BTC-USD forecast records can be evaluated.")
    issue_time = _utc_timestamp(row.get("forecast_issue_timestamp"))
    source_time = _utc_timestamp(row.get("source_candle_timestamp"))
    target_time = _utc_timestamp(row.get("intended_evaluation_timestamp"))
    try:
        horizon = float(row.get("forecast_horizon_hours"))
    except (TypeError, ValueError) as exc:
        raise ValueError("Forecast horizon is missing or invalid.") from exc
    if horizon != FORECAST_HORIZON_HOURS:
        raise ValueError("Only fixed two-hour BTC forecasts can be evaluated.")
    if source_time > issue_time:
        raise ValueError("Source candle close is later than the forecast issue time.")
    if target_time != pd.Timestamp(
        forecast_expiration_timestamp(issue_time.to_pydatetime(), FORECAST_HORIZON_HOURS)
    ):
        raise ValueError("Evaluation timestamp does not match the aligned two-hour expiry.")
    _positive_number(row.get("starting_btc_price"), "Starting BTC price")
    try:
        confidence = float(row.get("confidence_score"))
    except (TypeError, ValueError) as exc:
        raise ValueError("Confidence score is missing or invalid.") from exc
    if not math.isfinite(confidence) or not 1 <= confidence <= 10:
        raise ValueError("Confidence score must be between 1 and 10.")
    if not str(row.get("scenario", "")).strip():
        raise ValueError("Forecast scenario is missing.")
    direction = str(row.get("predicted_direction", "")).strip().upper()
    if direction not in {"BULLISH", "BEARISH"}:
        raise ValueError("Binary forecast direction must be BULLISH or BEARISH.")
    return issue_time, target_time


def evaluate_forecast(
    row: dict[str, Any],
    actual_close: float,
) -> dict[str, Any]:
    _, target_time = _validate_forecast_timestamps(row)
    actual_close = _positive_number(actual_close, "Exact-expiry BTC close")
    starting_price = float(row["starting_btc_price"])
    direction = str(row["predicted_direction"]).strip().upper()
    direction_correct = (
        actual_close > starting_price
        if direction == "BULLISH"
        else actual_close < starting_price
    )
    verdict = "WIN" if direction_correct else "LOSS"
    return {
        "actual_btc_close": actual_close,
        "actual_close_timestamp": target_time.isoformat(),
        "actual_high": "",
        "actual_low": "",
        "direction_correct": direction_correct,
        "final_close_in_predicted_range": "",
        "full_period_range_inside_predicted_range": "",
        "projected_price_absolute_error": "",
        "projected_price_percentage_error": "",
        "evaluated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "evaluation_status": verdict,
        "evaluation_note": (
            f"{verdict}: {direction} forecast scored only by comparing the exact "
            "Binance 1-minute expiry close with the starting BTC price."
        ),
    }


def _memory_id(row_data: dict[str, Any]) -> str:
    forecast_id = str(row_data.get("forecast_id", "")).strip()
    if forecast_id:
        return forecast_id
    issue_timestamp = str(
        row_data.get("forecast_issue_timestamp") or row_data.get("timestamp") or ""
    ).strip()
    if not issue_timestamp:
        raise ValueError("Cannot store a forecast memory without an issue timestamp.")
    return f"advisory-{issue_timestamp}"


def _memory_file_path(row_data: dict[str, Any], memory_id: str) -> Path:
    timestamp = _utc_timestamp(
        row_data.get("forecast_issue_timestamp") or row_data.get("timestamp")
    )
    date = timestamp.strftime("%Y-%m-%d")
    ticker = str(row_data.get("ticker") or "BTC-USD").strip().upper()
    ticker = re.sub(r"[^A-Z0-9.-]+", "_", ticker.split("-", 1)[0]) or "UNKNOWN"
    direction = str(
        row_data.get("predicted_direction") or row_data.get("direction") or "UNKNOWN"
    ).strip().upper()
    direction = {"BULLISH": "UP", "BEARISH": "DOWN"}.get(direction, direction)
    direction = re.sub(r"[^A-Z0-9_-]+", "_", direction) or "UNKNOWN"
    base_path = TRADE_MEMORIES_DIR / f"{date}_{ticker}_{direction}.md"
    for existing_path in TRADE_MEMORIES_DIR.glob(f"{base_path.stem}*.md"):
        if f"Memory ID: `{memory_id}`" in existing_path.read_text(encoding="utf-8"):
            return existing_path
    if not base_path.exists():
        return base_path
    safe_id = re.sub(r"[^A-Za-z0-9_-]+", "_", memory_id)
    return TRADE_MEMORIES_DIR / f"{base_path.stem}_{safe_id}.md"


def store_memory(row_data: dict[str, Any], outcome_reason: str | None = None) -> None:
    memory_id = _memory_id(row_data)
    reason = outcome_reason or str(row_data.get("outcome_reason", "")).strip()
    if not reason:
        raise ValueError("Cannot store a forecast memory without its outcome reason.")
    TRADE_MEMORIES_DIR.mkdir(parents=True, exist_ok=True)
    memory_path = _memory_file_path(row_data, memory_id)
    context = json.dumps(row_data, indent=2, ensure_ascii=False, default=str)
    memory = (
        f"# {row_data.get('ticker', 'BTC-USD')} "
        f"{row_data.get('predicted_direction') or row_data.get('direction', '')} "
        "forecast\n\n"
        f"**Memory ID:** `{memory_id}`\n\n"
        "## Outcome\n\n"
        f"{reason}\n\n"
        "## Full forecast and evaluation context\n\n"
        "```json\n"
        f"{context}\n"
        "```\n"
    )
    memory_path.write_text(memory, encoding="utf-8")


def retrieve_past_lessons(
    current_market_regime: str,
    forecast_horizon_hours: int | None = None,
) -> str:
    if not current_market_regime.strip():
        raise ValueError("current_market_regime must not be empty.")
    if forecast_horizon_hours not in (None, 2, 4):
        raise ValueError("Forecast horizon must be two or four hours.")
    memories = sorted(
        TRADE_MEMORIES_DIR.glob("*.md"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not memories:
        return "No past trade lessons are available yet."
    terms = {
        term.casefold()
        for term in re.findall(r"[A-Za-z0-9_-]{3,}", current_market_regime)
    }
    ranked_memories: list[tuple[int, float, Path, str]] = []
    for memory_path in memories:
        document = memory_path.read_text(encoding="utf-8")
        if forecast_horizon_hours is not None:
            match = re.search(
                r"```json\s*(\{.*?\})\s*```", document, flags=re.DOTALL
            )
            if not match:
                continue
            try:
                context = json.loads(match.group(1))
                memory_horizon = int(context.get("forecast_horizon_hours"))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            if memory_horizon != forecast_horizon_hours:
                continue
        score = sum(document.casefold().count(term) for term in terms)
        ranked_memories.append((score, memory_path.stat().st_mtime, memory_path, document))
    if not ranked_memories:
        return f"No {forecast_horizon_hours or 'matching'}-hour trade lessons are available yet."
    ranked_memories.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return "\n\n".join(
        f"### Past trade lesson {index}\n\n{document}"
        for index, (_, _, _, document) in enumerate(ranked_memories[:3], start=1)
    )


def _write_rows_atomic(
    path: Path,
    rows: list[dict[str, str]],
    fields: list[str] = BTC_FORECAST_FIELDS,
) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}-",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
        Path(temporary_name).replace(path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def grade_pending_predictions(
    csv_file: str | Path = BTC_FORECAST_FILE,
    now: pd.Timestamp | None = None,
) -> dict[str, int]:
    path = Path(csv_file)
    initialize_btc_forecast_log(path)
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != BTC_FORECAST_FIELDS:
            raise ValueError(
                f"Forecast log {path} has an unexpected schema; it was not evaluated."
            )
        rows = list(reader)

    now_utc = (
        _utc_timestamp(now)
        if now is not None
        else pd.Timestamp.now(tz="UTC")
    )
    summary = {
        "graded": 0,
        "lost": 0,
        "memorized": 0,
        "errors": 0,
        "pending": 0,
        "needs_review": 0,
    }
    changed = False

    for row in rows:
        if row.get("evaluation_status", "").strip().upper() != "PENDING":
            continue
        try:
            issue_time, target_time = _validate_forecast_timestamps(row)
            if target_time > now_utc:
                summary["pending"] += 1
                continue
        except (TypeError, ValueError) as exc:
            row["evaluation_status"] = "NEEDS_REVIEW"
            row["evaluation_note"] = str(exc)
            summary["needs_review"] += 1
            changed = True
            continue

        try:
            actual_close = get_binance_close_at_timestamp(
                "BTCUSDT", target_time.isoformat()
            )
            result = evaluate_forecast(row, actual_close)
            evaluated_row = {**row, **{key: str(value) for key, value in result.items()}}
            store_memory(
                evaluated_row,
                outcome_reason=_two_hour_outcome_reason(evaluated_row),
            )
            row.update(evaluated_row)
            summary["graded"] += 1
            summary["lost"] += int(not result["direction_correct"])
            summary["memorized"] += 1
            changed = True
        except Exception as exc:
            summary["errors"] += 1
            print(
                f"[EVALUATOR] Could not evaluate forecast "
                f"{row.get('forecast_id')}: {exc}"
            )

    if changed:
        _write_rows_atomic(path, rows)
    return summary


def _two_hour_outcome_reason(row: dict[str, Any]) -> str:
    direction = str(row["predicted_direction"]).strip().upper()
    correct = str(row["direction_correct"]).strip().casefold() == "true"
    verdict = "WIN" if correct else "LOSS"
    return (
        f"**{verdict}.** The {direction} forecast was scored by comparing exact "
        f"expiry close {row['actual_btc_close']} with starting price "
        f"{row['starting_btc_price']}."
    )


def _validate_advisory_timestamps(
    row: dict[str, Any],
) -> tuple[pd.Timestamp, pd.Timestamp]:
    issue_time = _utc_timestamp(
        row.get("forecast_issue_timestamp") or row.get("timestamp")
    )
    source_time = _utc_timestamp(row.get("source_candle_timestamp"))
    target_time = _utc_timestamp(row.get("target_timestamp"))
    try:
        horizon = float(row.get("forecast_horizon_hours"))
    except (TypeError, ValueError) as exc:
        raise ValueError("Advisory forecast horizon is missing or invalid.") from exc
    if horizon != ADVISORY_HORIZON_HOURS:
        raise ValueError("Only fixed four-hour BTC advisories can be evaluated.")
    if source_time > issue_time:
        raise ValueError("Source candle close is later than the advisory issue time.")
    if target_time != pd.Timestamp(
        forecast_expiration_timestamp(issue_time.to_pydatetime(), ADVISORY_HORIZON_HOURS)
    ):
        raise ValueError("Advisory target timestamp does not match the aligned four-hour expiry.")
    _positive_number(row.get("btc_price_at_prediction"), "Starting BTC price")
    direction = str(row.get("direction", "")).strip().upper()
    if direction not in {"UP", "DOWN"}:
        raise ValueError("Advisory direction is invalid.")
    return issue_time, target_time


def evaluate_advisory(
    row: dict[str, Any],
    actual_close: float,
) -> dict[str, Any]:
    _, target_time = _validate_advisory_timestamps(row)
    actual_close = _positive_number(actual_close, "Exact-expiry BTC close")
    starting_price = float(row["btc_price_at_prediction"])
    direction = str(row["direction"]).strip().upper()
    correct = actual_close > starting_price if direction == "UP" else actual_close < starting_price
    verdict = "WIN" if correct else "LOSS"
    return {
        "actual_price_timestamp": target_time.isoformat(),
        "actual_outcome": verdict,
        "graded": verdict,
        "direction_correct": correct,
        "actual_btc_close": actual_close,
        "evaluation_note": (
            f"{verdict}: {direction} advisory scored only by comparing the exact "
            "Binance 1-minute expiry close with the starting BTC price."
        ),
    }


def grade_pending_advisories(
    csv_file: str | Path = ADVISORY_LOG_FILE,
    now: pd.Timestamp | None = None,
) -> dict[str, int]:
    path = Path(csv_file)
    initialize_advisory_log(path)
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != ADVISORY_LOG_FIELDS:
            raise ValueError(
                f"Advisory log {path} has an unexpected schema; it was not evaluated."
            )
        rows = list(reader)

    now_utc = _utc_timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC")
    summary = {
        "graded": 0,
        "lost": 0,
        "memorized": 0,
        "errors": 0,
        "pending": 0,
        "needs_review": 0,
    }
    changed = False
    for row in rows:
        if row.get("graded", "").strip().casefold() not in {"false", "pending"}:
            continue
        try:
            issue_time, target_time = _validate_advisory_timestamps(row)
            if target_time > now_utc:
                summary["pending"] += 1
                continue
        except (TypeError, ValueError) as exc:
            row["graded"] = "NEEDS_REVIEW"
            row["actual_outcome"] = str(exc)
            summary["needs_review"] += 1
            changed = True
            continue

        try:
            actual_close = get_binance_close_at_timestamp(
                "BTCUSDT", target_time.isoformat()
            )
            result = evaluate_advisory(row, actual_close)
            memory_data = {**row, **{key: str(value) for key, value in result.items()}}
            memory_data["ticker"] = "BTC-USD"
            store_memory(memory_data, outcome_reason=str(result["actual_outcome"]))
            row.update(
                {
                    key: str(result[key])
                    for key in ("actual_price_timestamp", "actual_outcome", "graded")
                }
            )
            summary["graded"] += 1
            summary["lost"] += int(not result["direction_correct"])
            summary["memorized"] += 1
            changed = True
        except Exception as exc:
            summary["errors"] += 1
            print(
                f"[EVALUATOR] Could not evaluate advisory "
                f"{row.get('forecast_issue_timestamp')}: {exc}"
            )
    if changed:
        _write_rows_atomic(path, rows, ADVISORY_LOG_FIELDS)
    return summary


if __name__ == "__main__":
    for log_name, grade in (
        ("2-hour forecasts", grade_pending_predictions),
        ("4-hour advisories", grade_pending_advisories),
    ):
        try:
            result = grade()
        except Exception as exc:
            print(f"[EVALUATOR] Could not process {log_name}: {exc}")
            continue
        print(
            f"[EVALUATOR] {log_name}: "
            f"{result['graded']} evaluated, {result['memorized']} memories written, "
            f"{result['pending']} pending, {result['needs_review']} need review, "
            f"{result['errors']} errors."
        )
