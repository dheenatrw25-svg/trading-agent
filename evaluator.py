import datetime
import math
from pathlib import Path
from typing import Any

import chromadb
import pandas as pd
import yfinance as yf

from prediction_logs import (
    APP_LOG_FILE,
    MARKET_LOG_FILE,
    initialize_app_log,
    initialize_market_log,
)

CHROMA_CLIENT = chromadb.PersistentClient(path="chroma_db")
TRADE_MEMORIES = CHROMA_CLIENT.get_or_create_collection(name="trade_memories")
MARKET_COLUMNS = {
    "forecast_issue_timestamp",
    "source_candle_timestamp",
    "target_timestamp",
    "forecast_horizon_hours",
    "starting_btc_price",
    "predicted_direction",
    "expected_low_2h",
    "expected_high_2h",
    "graded",
    "actual_outcome",
}
APP_COLUMNS = {
    "forecast_issue_timestamp",
    "source_candle_timestamp",
    "target_timestamp",
    "forecast_horizon_hours",
    "direction",
    "target_low",
    "target_high",
    "invalidation_price",
    "graded",
    "actual_outcome",
}


def _utc_timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        local_timezone = datetime.datetime.now().astimezone().tzinfo
        timestamp = timestamp.tz_localize(local_timezone)
    return timestamp.tz_convert("UTC")


def _fetch_price_at(target_time: pd.Timestamp) -> tuple[float, pd.Timestamp]:
    start = target_time - pd.Timedelta(minutes=2)
    end = target_time + pd.Timedelta(minutes=1)
    history = yf.download(
        "BTC-USD",
        start=start.to_pydatetime(),
        end=end.to_pydatetime(),
        interval="1m",
        progress=False,
        auto_adjust=False,
    )
    if isinstance(history.columns, pd.MultiIndex):
        history.columns = history.columns.get_level_values(0)
    history = history.dropna(subset=["Close"])
    if history.empty:
        raise RuntimeError(f"No BTC-USD historical price returned near {target_time}.")

    index_utc = pd.to_datetime(history.index, utc=True, errors="coerce")
    candle_close_times = index_utc + pd.Timedelta(minutes=1)
    eligible = candle_close_times.notna() & (candle_close_times <= target_time)
    eligible_positions = [position for position, is_valid in enumerate(eligible) if is_valid]
    if not eligible_positions:
        raise RuntimeError(
            f"No completed BTC-USD one-minute candle is available at {target_time}."
        )
    nearest_position = max(eligible_positions, key=lambda position: candle_close_times[position])
    price_timestamp = candle_close_times[nearest_position]
    if target_time - price_timestamp > pd.Timedelta(minutes=1):
        raise RuntimeError(
            f"No BTC-USD candle close within one minute before {target_time}; "
            "the forecast remains pending."
        )

    price = float(history["Close"].iloc[nearest_position])
    if not math.isfinite(price) or price <= 0:
        raise RuntimeError(f"Historical BTC-USD price at {target_time} is not finite.")
    return price, price_timestamp


def _grade_outcome(row: pd.Series, actual_price: float) -> str:
    direction = str(row["direction"]).strip().upper()
    target_low = float(row["target_low"])
    target_high = float(row["target_high"])
    invalidation_price = float(row["invalidation_price"])

    if direction == "UP":
        if actual_price >= target_low:
            return "Win"
        if actual_price <= invalidation_price:
            return "Loss"
    elif direction == "DOWN":
        if actual_price <= target_high:
            return "Win"
        if actual_price >= invalidation_price:
            return "Loss"
    return "Neutral"


def _grade_market_outcome(row: pd.Series, actual_price: float) -> str:
    direction = str(row["predicted_direction"]).strip().upper()
    starting_price = float(row["starting_btc_price"])

    if direction == "BULLISH":
        if actual_price > starting_price:
            return "Win"
        return "Loss" if actual_price < starting_price else "Neutral"
    if direction == "BEARISH":
        if actual_price < starting_price:
            return "Win"
        return "Loss" if actual_price > starting_price else "Neutral"
    if direction == "SIDEWAYS":
        target_low = float(row["expected_low_2h"])
        target_high = float(row["expected_high_2h"])
        return "Win" if target_low <= actual_price <= target_high else "Loss"
    raise ValueError(f"Unsupported 2-hour forecast direction: {direction}.")


def _validate_forecast_timestamps(row: pd.Series, expected_horizon: int) -> pd.Timestamp:
    issue_time = _utc_timestamp(row["forecast_issue_timestamp"])
    source_time = _utc_timestamp(row["source_candle_timestamp"])
    target_time = _utc_timestamp(row["target_timestamp"])
    horizon = float(row["forecast_horizon_hours"])
    if horizon != expected_horizon:
        raise ValueError(
            f"Expected a {expected_horizon}-hour forecast, found {horizon:g} hours."
        )
    if source_time > issue_time:
        raise ValueError("Source candle timestamp is later than the forecast issue time.")
    if target_time != issue_time + pd.Timedelta(hours=expected_horizon):
        raise ValueError("Target timestamp does not match the recorded forecast horizon.")
    return target_time


def store_memory(row_data: dict[str, Any]) -> None:
    timestamp = str(row_data.get("timestamp", ""))
    if not timestamp:
        raise ValueError("Cannot store a trade memory without a timestamp.")

    direction = row_data.get("direction") or row_data.get("predicted_direction", "Unknown")
    target_low = row_data.get("target_low") or row_data.get("expected_low_2h", "Unknown")
    target_high = row_data.get("target_high") or row_data.get("expected_high_2h", "Unknown")
    reasoning = row_data.get("reasoning") or row_data.get("scenario") or "Not recorded."
    market_regime = row_data.get("market_regime") or "Not recorded."
    outcome = row_data.get("actual_outcome", "Unknown")
    actual_price = row_data.get("actual_price", "Not available")
    document = (
        f"Timestamp: {timestamp}. Direction predicted: {direction}. "
        f"Target: {target_low} to {target_high}. Market regime: {market_regime}. "
        f"AI Reasoning: {reasoning}. The actual price was: {actual_price}. "
        f"The actual outcome was: {outcome}."
    )
    TRADE_MEMORIES.upsert(
        ids=[timestamp],
        documents=[document],
        metadatas=[{"timestamp": timestamp, "outcome": str(outcome)}],
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


def grade_pending_predictions(
    csv_file: str | Path = MARKET_LOG_FILE,
) -> dict[str, int]:
    csv_file = Path(csv_file)
    if csv_file == MARKET_LOG_FILE:
        initialize_market_log()
    elif csv_file == APP_LOG_FILE:
        initialize_app_log()
    if not csv_file.is_file():
        return {"graded": 0, "memorized": 0, "errors": 0, "pending": 0}

    predictions = pd.read_csv(csv_file)
    market_log = "predicted_direction" in predictions.columns
    required_columns = MARKET_COLUMNS if market_log else APP_COLUMNS
    expected_horizon = 2 if market_log else 4
    missing_columns = required_columns.difference(predictions.columns)
    if missing_columns:
        raise ValueError(
            "Prediction CSV is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    now_utc = pd.Timestamp.now(tz="UTC")
    pending_mask = predictions["graded"].astype(str).str.strip().str.casefold().eq("false")
    summary = {"graded": 0, "memorized": 0, "errors": 0, "pending": 0}
    for column in ("graded", "actual_price_timestamp", "actual_outcome"):
        if column in predictions.columns:
            predictions[column] = predictions[column].astype(object)

    for index in predictions.index[pending_mask]:
        row = predictions.loc[index]
        try:
            target_time = _validate_forecast_timestamps(row, expected_horizon)
            if not str(row["source_candle_timestamp"]).strip():
                raise ValueError("Source candle timestamp is missing.")
            if now_utc < target_time:
                continue
            actual_price, price_timestamp = _fetch_price_at(target_time)
            outcome = (
                _grade_market_outcome(row, actual_price)
                if market_log
                else _grade_outcome(row, actual_price)
            )
            row_data = row.to_dict()
            row_data["actual_price"] = actual_price
            row_data["actual_price_timestamp"] = price_timestamp.isoformat()
            row_data["actual_outcome"] = outcome
            store_memory(row_data)
            predictions.at[index, "graded"] = "True"
            if "actual_price" in predictions.columns:
                predictions.at[index, "actual_price"] = actual_price
            if "actual_price_timestamp" in predictions.columns:
                predictions.at[index, "actual_price_timestamp"] = price_timestamp.isoformat()
            predictions.at[index, "actual_outcome"] = outcome
            summary["graded"] += 1
            summary["memorized"] += 1
            print(
                f"[EVALUATOR] {row['timestamp']} graded {outcome} "
                f"(actual BTC close: {actual_price:.2f} at {price_timestamp})."
            )
        except Exception as exc:
            summary["errors"] += 1
            print(f"[EVALUATOR] Could not grade row {index}: {exc}")

    summary["pending"] = int(
        (predictions["graded"].astype(str).str.strip().str.casefold() == "false").sum()
    )
    if summary["graded"]:
        predictions.to_csv(csv_file, index=False)
    return summary


def grade_app_pending_predictions(
    csv_file: str | Path = APP_LOG_FILE,
) -> dict[str, int]:
    return grade_pending_predictions(csv_file)


if __name__ == "__main__":
    for label, result in (
        ("2-hour market forecasts", grade_pending_predictions()),
        ("4-hour app forecasts", grade_app_pending_predictions()),
    ):
        print(
            f"[EVALUATOR] {label}: "
            f"{result['graded']} graded, {result['memorized']} memorized, "
            f"{result['pending']} pending, {result['errors']} errors."
        )
