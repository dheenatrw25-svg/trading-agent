import csv
import datetime
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
import pandas as pd
import yfinance as yf

from .paths import PROJECT_ROOT
from .prediction_logs import (
    BTC_FORECAST_FIELDS,
    BTC_FORECAST_FILE,
    initialize_btc_forecast_log,
)


CHROMA_CLIENT = chromadb.PersistentClient(path=str(PROJECT_ROOT / "chroma_db"))
TRADE_MEMORIES = CHROMA_CLIENT.get_or_create_collection(name="trade_memories")
FORECAST_HORIZON_HOURS = 2
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
    starting_price = float(row["starting_btc_price"])
    expected_low = float(row["expected_low"])
    expected_high = float(row["expected_high"])
    projected_price = float(row["projected_price"])
    direction = str(row["predicted_direction"]).strip().upper()

    if direction == "BULLISH":
        direction_correct: bool | str = actual_close > starting_price
    elif direction == "BEARISH":
        direction_correct = actual_close < starting_price
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
            "the forecast interval; the close uses the exact expiry candle when "
            "available, otherwise the latest minute close no more than one minute "
            "before expiry. SIDEWAYS direction is not applicable; use range scores."
        ),
    }


def store_memory(row_data: dict[str, Any]) -> None:
    forecast_id = str(row_data.get("forecast_id", "")).strip()
    if not forecast_id:
        raise ValueError("Cannot store a forecast memory without a forecast ID.")
    document = (
        f"Forecast {forecast_id}: {row_data['predicted_direction']} BTC forecast. "
        f"Starting price {row_data['starting_btc_price']}; predicted range "
        f"{row_data['expected_low']} to {row_data['expected_high']}; projected "
        f"price {row_data['projected_price']}. Scenario: {row_data['scenario']} "
        f"Actual close {row_data['actual_btc_close']} at "
        f"{row_data['actual_close_timestamp']}; actual high/low "
        f"{row_data['actual_high']}/{row_data['actual_low']}. Direction correct: "
        f"{row_data['direction_correct']}; final close inside range: "
        f"{row_data['final_close_in_predicted_range']}; full-period range inside: "
        f"{row_data['full_period_range_inside_predicted_range']}."
    )
    TRADE_MEMORIES.upsert(
        ids=[forecast_id],
        documents=[document],
        metadatas=[
            {
                "forecast_id": forecast_id,
                "direction": str(row_data["predicted_direction"]),
                "evaluation_status": str(row_data["evaluation_status"]),
            }
        ],
    )


def retrieve_past_lessons(current_market_regime: str) -> str:
    if not current_market_regime.strip():
        raise ValueError("current_market_regime must not be empty.")
    memory_count = TRADE_MEMORIES.count()
    if memory_count == 0:
        return "No past trade lessons are available yet."
    results = TRADE_MEMORIES.query(
        query_texts=[current_market_regime],
        n_results=min(3, memory_count),
    )
    documents = results.get("documents", [[]])[0]
    if not documents:
        return "No relevant past trade lessons were found."
    return "\n".join(
        f"{index}. {document}" for index, document in enumerate(documents, start=1)
    )


def _write_rows_atomic(path: Path, rows: list[dict[str, str]]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}-",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=BTC_FORECAST_FIELDS, extrasaction="raise")
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
            store_memory(evaluated_row)
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


if __name__ == "__main__":
    result = grade_pending_predictions()
    print(
        "[EVALUATOR] "
        f"{result['graded']} evaluated, {result['memorized']} memories upserted, "
        f"{result['pending']} pending, {result['needs_review']} need review, "
        f"{result['errors']} errors."
    )
