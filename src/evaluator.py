import csv
import datetime
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from .paths import TRADE_MEMORIES_DIR
from .prediction_logs import (
    ADVISORY_LOG_FIELDS,
    ADVISORY_LOG_FILE,
    BTC_FORECAST_FIELDS,
    BTC_FORECAST_FILE,
    initialize_advisory_log,
    initialize_btc_forecast_log,
)


FORECAST_HORIZON_HOURS = 2
ADVISORY_HORIZON_HOURS = 4
MAX_CLOSE_DELAY = pd.Timedelta(minutes=1)


class EvaluationDataUnavailable(RuntimeError):
    pass


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
    if target_time != issue_time + pd.Timedelta(hours=FORECAST_HORIZON_HOURS):
        raise ValueError("Evaluation timestamp does not match the two-hour horizon.")
    _positive_number(row.get("starting_btc_price"), "Starting BTC price")
    _positive_number(row.get("expected_low"), "Expected low")
    _positive_number(row.get("expected_high"), "Expected high")
    _positive_number(row.get("projected_price"), "Projected price")
    try:
        confidence = float(row.get("confidence_score"))
    except (TypeError, ValueError) as exc:
        raise ValueError("Confidence score is missing or invalid.") from exc
    if not math.isfinite(confidence) or not 1 <= confidence <= 10:
        raise ValueError("Confidence score must be between 1 and 10.")
    if not str(row.get("scenario", "")).strip():
        raise ValueError("Forecast scenario is missing.")
    if float(row["expected_low"]) > float(row["expected_high"]):
        raise ValueError("Expected low exceeds expected high.")
    direction = str(row.get("predicted_direction", "")).strip().upper()
    if direction not in {"BULLISH", "BEARISH", "SIDEWAYS"}:
        raise ValueError("Forecast direction is invalid.")
    return issue_time, target_time


def _normalize_candles(history: pd.DataFrame, interval: str) -> pd.DataFrame:
    if history.empty:
        return history
    if isinstance(history.columns, pd.MultiIndex):
        history = history.copy()
        history.columns = history.columns.get_level_values(0)
    required = {"Open", "High", "Low", "Close"}
    if not required.issubset(history.columns):
        raise EvaluationDataUnavailable("Historical candle data is missing OHLC columns.")

    result = history.copy()
    timestamps = pd.to_datetime(result.index, utc=True, errors="coerce")
    result = result.loc[timestamps.notna()].copy()
    result.index = timestamps[timestamps.notna()]
    result = result.loc[~result.index.duplicated(keep="last")].sort_index()
    for column in ("Open", "High", "Low", "Close"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.dropna(subset=["Open", "High", "Low", "Close"])
    ohlc = result[["Open", "High", "Low", "Close"]].to_numpy(dtype=float)
    valid = (
        np.isfinite(ohlc).all(axis=1)
        & (ohlc > 0).all(axis=1)
        & (result["High"].to_numpy() >= result["Low"].to_numpy())
    )
    result = result.loc[valid]
    result.attrs["interval"] = interval
    return result


def _download_candles(
    ticker: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    interval: str,
) -> pd.DataFrame:
    raw = yf.download(
        ticker,
        start=start.to_pydatetime(),
        end=end.to_pydatetime(),
        interval=interval,
        progress=False,
        auto_adjust=False,
    )
    return _normalize_candles(raw, interval)


def _fetch_evaluation_candles(
    ticker: str,
    issue_time: pd.Timestamp,
    target_time: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fifteen_minute_start = issue_time.floor("15min")
    fifteen_minute_end = target_time.ceil("15min") + pd.Timedelta(minutes=15)
    candles_15m = _download_candles(
        ticker, fifteen_minute_start, fifteen_minute_end, "15m"
    )

    one_minute_start = issue_time.floor("min") - pd.Timedelta(minutes=1)
    one_minute_end = target_time.ceil("min") + pd.Timedelta(minutes=1)
    candles_1m = _download_candles(
        ticker, one_minute_start, one_minute_end, "1m"
    )
    return candles_15m, candles_1m


def _completed_candles(
    candles: pd.DataFrame,
    interval: pd.Timedelta,
    now_utc: pd.Timestamp,
) -> pd.DataFrame:
    if candles.empty:
        return candles
    close_times = candles.index + interval
    return candles.loc[close_times <= now_utc]


def _target_close(
    target_time: pd.Timestamp,
    candles_15m: pd.DataFrame,
    candles_1m: pd.DataFrame,
) -> tuple[float, pd.Timestamp]:
    for candles, interval in (
        (candles_15m, pd.Timedelta(minutes=15)),
        (candles_1m, pd.Timedelta(minutes=1)),
    ):
        if candles.empty:
            continue
        close_times = candles.index + interval
        exact = candles.loc[close_times == target_time]
        if not exact.empty:
            return float(exact["Close"].iloc[-1]), target_time

    if candles_1m.empty:
        raise EvaluationDataUnavailable("No minute candles are available at expiry.")
    close_times = candles_1m.index + pd.Timedelta(minutes=1)
    eligible = close_times <= target_time
    if not eligible.any():
        raise EvaluationDataUnavailable("No completed candle is available by expiry.")
    eligible_times = close_times[eligible]
    close_time = eligible_times.max()
    if target_time - close_time > MAX_CLOSE_DELAY:
        raise EvaluationDataUnavailable(
            "The nearest minute candle close is more than one minute before expiry."
        )
    position = candles_1m.index.get_loc(close_time - pd.Timedelta(minutes=1))
    return float(candles_1m["Close"].iloc[position]), close_time


def _range_extremes(
    issue_time: pd.Timestamp,
    actual_close_time: pd.Timestamp,
    candles_15m: pd.DataFrame,
    candles_1m: pd.DataFrame,
) -> tuple[float, float, str]:
    aligned_to_15m = (
        issue_time == issue_time.floor("15min")
        and actual_close_time == actual_close_time.floor("15min")
    )
    candidates = []
    if aligned_to_15m:
        candidates.append(
            (candles_15m, pd.Timedelta(minutes=15), "15-minute")
        )
    candidates.append((candles_1m, pd.Timedelta(minutes=1), "1-minute"))

    unavailable_reason = "No candles overlap the forecast period."
    for candles, interval, resolution in candidates:
        if candles.empty:
            unavailable_reason = f"No {resolution} candles cover the forecast period."
            continue
        opens = candles.index
        closes = opens + interval
        overlapping = candles.loc[(opens < actual_close_time) & (closes > issue_time)]
        if overlapping.empty:
            unavailable_reason = "No candles overlap the forecast period."
            continue

        first_expected_open = issue_time.floor(interval)
        final_expected_open = (
            actual_close_time - pd.Timedelta(nanoseconds=1)
        ).floor(interval)
        expected_opens = pd.date_range(
            first_expected_open,
            final_expected_open,
            freq=interval,
            tz="UTC",
        )
        if not expected_opens.isin(overlapping.index).all():
            unavailable_reason = (
                f"Missing or misaligned {resolution} candles in the forecast period."
            )
            continue
        return (
            float(overlapping["High"].max()),
            float(overlapping["Low"].min()),
            resolution,
        )

    raise EvaluationDataUnavailable(unavailable_reason)


def evaluate_forecast(
    row: dict[str, Any],
    candles_15m: pd.DataFrame,
    candles_1m: pd.DataFrame,
) -> dict[str, Any]:
    issue_time, target_time = _validate_forecast_timestamps(row)
    actual_close, close_time = _target_close(target_time, candles_15m, candles_1m)
    actual_high, actual_low, resolution = _range_extremes(
        issue_time, close_time, candles_15m, candles_1m
    )
    expected_low = float(row["expected_low"])
    expected_high = float(row["expected_high"])
    projected_price = float(row["projected_price"])
    direction = str(row["predicted_direction"]).strip().upper()

    if direction == "BULLISH":
        direction_correct: bool | str = actual_high >= expected_low
    elif direction == "BEARISH":
        direction_correct = actual_low <= expected_high
    else:
        # A SIDEWAYS forecast has no up/down direction; range scores measure it.
        direction_correct = ""

    absolute_error = abs(actual_close - projected_price)
    return {
        "actual_btc_close": actual_close,
        "actual_close_timestamp": close_time.isoformat(),
        "actual_high": actual_high,
        "actual_low": actual_low,
        "direction_correct": direction_correct,
        "final_close_in_predicted_range": expected_low <= actual_close <= expected_high,
        "full_period_range_inside_predicted_range": (
            actual_low >= expected_low and actual_high <= expected_high
        ),
        "projected_price_absolute_error": absolute_error,
        "projected_price_percentage_error": (absolute_error / projected_price) * 100,
        "evaluated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "evaluation_status": "EVALUATED",
        "evaluation_note": (
            f"Range measured with complete {resolution} OHLC candles overlapping "
            "the forecast interval; directional forecasts are scored by whether "
            "the window reached the predicted target zone. The close uses the exact "
            "expiry candle when "
            "available, otherwise the latest minute close no more than one minute "
            "before expiry. SIDEWAYS direction is not applicable; use range scores."
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


def retrieve_past_lessons(current_market_regime: str) -> str:
    if not current_market_regime.strip():
        raise ValueError("current_market_regime must not be empty.")
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
        score = sum(document.casefold().count(term) for term in terms)
        ranked_memories.append((score, memory_path.stat().st_mtime, memory_path, document))
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
            candles_15m, candles_1m = _fetch_evaluation_candles(
                str(row["ticker"]), issue_time, target_time
            )
            candles_15m = _completed_candles(
                candles_15m, pd.Timedelta(minutes=15), now_utc
            )
            candles_1m = _completed_candles(
                candles_1m, pd.Timedelta(minutes=1), now_utc
            )
            result = evaluate_forecast(row, candles_15m, candles_1m)
            evaluated_row = {**row, **{key: str(value) for key, value in result.items()}}
            store_memory(
                evaluated_row,
                outcome_reason=_two_hour_outcome_reason(evaluated_row),
            )
            row.update(evaluated_row)
            summary["graded"] += 1
            summary["memorized"] += 1
            changed = True
        except EvaluationDataUnavailable as exc:
            row["evaluation_note"] = str(exc)
            summary["pending"] += 1
            changed = True
            print(f"[EVALUATOR] {row.get('forecast_id')}: {exc}")
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
    final_in_range = (
        str(row["final_close_in_predicted_range"]).strip().casefold() == "true"
    )
    if direction == "SIDEWAYS":
        won = final_in_range
        directional_detail = (
            "The final close landed inside the predicted range."
            if final_in_range
            else "The final close landed outside the predicted range."
        )
    else:
        won = correct
        directional_detail = (
            f"The {direction} forecast was correct: actual close "
            f"{row['actual_btc_close']} versus starting price "
            f"{row['starting_btc_price']}."
            if correct
            else f"The {direction} forecast was incorrect: actual close "
            f"{row['actual_btc_close']} versus starting price "
            f"{row['starting_btc_price']}."
        )
    verdict = "WON" if won else "LOST"
    return (
        f"**{verdict}.** {directional_detail} The final close was "
        f"{'inside' if final_in_range else 'outside'} the predicted range "
        f"{row['expected_low']}–{row['expected_high']}; actual high/low were "
        f"{row['actual_high']}/{row['actual_low']}."
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
    if target_time != issue_time + pd.Timedelta(hours=ADVISORY_HORIZON_HOURS):
        raise ValueError("Advisory target timestamp does not match the four-hour horizon.")
    _positive_number(row.get("btc_price_at_prediction"), "Starting BTC price")
    low = _positive_number(row.get("target_low"), "Target low")
    high = _positive_number(row.get("target_high"), "Target high")
    if low > high:
        raise ValueError("Advisory target low exceeds target high.")
    direction = str(row.get("direction", "")).strip().upper()
    if direction not in {"UP", "DOWN"}:
        raise ValueError("Advisory direction is invalid.")
    return issue_time, target_time


def evaluate_advisory(
    row: dict[str, Any],
    candles_15m: pd.DataFrame,
    candles_1m: pd.DataFrame,
) -> dict[str, Any]:
    issue_time, target_time = _validate_advisory_timestamps(row)
    actual_close, close_time = _target_close(target_time, candles_15m, candles_1m)
    actual_high, actual_low, resolution = _range_extremes(
        issue_time, close_time, candles_15m, candles_1m
    )
    direction = str(row["direction"]).strip().upper()
    target_low = float(row["target_low"])
    target_high = float(row["target_high"])
    correct = (
        actual_high >= target_low
        if direction == "UP"
        else actual_low <= target_high
    )
    in_range = float(row["target_low"]) <= actual_close <= float(row["target_high"])
    verdict = "WON" if correct else "LOST"
    touch_reason = (
        f"actual high {actual_high} "
        f"{'reached or exceeded' if correct else 'did not reach'} target low {target_low}"
        if direction == "UP"
        else f"actual low {actual_low} "
        f"{'reached or fell below' if correct else 'did not reach'} target high {target_high}"
    )
    reason = (
        f"{verdict}: {direction} forecast was "
        f"{'correct' if correct else 'incorrect'} because {touch_reason}. "
        f"Actual BTC close was {actual_close} and was "
        f"{'inside' if in_range else 'outside'} target range "
        f"{row['target_low']}–{row['target_high']}; actual high/low were "
        f"{actual_high}/{actual_low}."
    )
    return {
        "actual_price_timestamp": close_time.isoformat(),
        "actual_outcome": reason,
        "graded": "True",
        "actual_btc_close": actual_close,
        "actual_high": actual_high,
        "actual_low": actual_low,
        "direction_correct": correct,
        "final_close_in_predicted_range": in_range,
        "evaluation_note": (
            f"Range measured with complete {resolution} OHLC candles overlapping "
            "the four-hour advisory interval."
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
    summary = {"graded": 0, "memorized": 0, "errors": 0, "pending": 0, "needs_review": 0}
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
            candles_15m, candles_1m = _fetch_evaluation_candles(
                "BTC-USD", issue_time, target_time
            )
            candles_15m = _completed_candles(
                candles_15m, pd.Timedelta(minutes=15), now_utc
            )
            candles_1m = _completed_candles(
                candles_1m, pd.Timedelta(minutes=1), now_utc
            )
            result = evaluate_advisory(row, candles_15m, candles_1m)
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
            summary["memorized"] += 1
            changed = True
        except EvaluationDataUnavailable as exc:
            row["actual_outcome"] = f"Pending: {exc}"
            summary["pending"] += 1
            changed = True
            print(f"[EVALUATOR] Advisory {row.get('forecast_issue_timestamp')}: {exc}")
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
