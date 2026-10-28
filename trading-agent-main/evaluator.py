import datetime
import math
from typing import Any

import chromadb
import pandas as pd
import yfinance as yf


CHROMA_CLIENT = chromadb.PersistentClient(path="chroma_db")
TRADE_MEMORIES = CHROMA_CLIENT.get_or_create_collection(name="trade_memories")
CSV_COLUMNS = {
    "timestamp",
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


def _fetch_price_at(target_time: pd.Timestamp) -> float:
    start = target_time - pd.Timedelta(hours=2)
    end = target_time + pd.Timedelta(hours=2)
    history = yf.download(
        "BTC-USD",
        start=start.to_pydatetime(),
        end=end.to_pydatetime(),
        interval="1h",
        progress=False,
        auto_adjust=False,
    )
    if isinstance(history.columns, pd.MultiIndex):
        history.columns = history.columns.get_level_values(0)
    history = history.dropna(subset=["Close"])
    if history.empty:
        raise RuntimeError(f"No BTC-USD historical price returned near {target_time}.")

    index_utc = pd.to_datetime(history.index, utc=True)
    candle_close_times = index_utc + pd.Timedelta(hours=1)
    distances = candle_close_times - target_time
    nearest_position = int(distances.to_series().abs().argmin())
    if abs(distances[nearest_position]) > pd.Timedelta(hours=1):
        raise RuntimeError(
            f"No BTC-USD hourly candle close enough to {target_time} "
            "to grade this prediction."
        )

    price = float(history["Close"].iloc[nearest_position])
    if not math.isfinite(price):
        raise RuntimeError(f"Historical BTC-USD price at {target_time} is not finite.")
    return price


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


def store_memory(row_data: dict[str, Any]) -> None:
    timestamp = str(row_data.get("timestamp", ""))
    if not timestamp:
        raise ValueError("Cannot store a trade memory without a timestamp.")

    direction = row_data.get("direction", "Unknown")
    target_low = row_data.get("target_low", "Unknown")
    target_high = row_data.get("target_high", "Unknown")
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
    csv_file: str = "predictions_log.csv",
) -> dict[str, int]:
    predictions = pd.read_csv(csv_file)
    missing_columns = CSV_COLUMNS.difference(predictions.columns)
    if missing_columns:
        legacy_columns = {
            "timestamp",
            "predicted_direction",
            "expected_low_2h",
            "expected_high_2h",
            "scenario",
        }
        if legacy_columns.issubset(predictions.columns):
            print(
                "[EVALUATOR] Skipping legacy 2-hour forecast log; "
                "it has no 4-hour grading status."
            )
            return {"graded": 0, "memorized": 0, "errors": 0, "pending": 0}
        raise ValueError(
            "Prediction CSV is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    now_utc = pd.Timestamp.now(tz="UTC")
    pending_mask = predictions["graded"].astype(str).str.strip().str.casefold().eq("false")
    summary = {"graded": 0, "memorized": 0, "errors": 0, "pending": 0}
    predictions["graded"] = predictions["graded"].astype(object)

    for index in predictions.index[pending_mask]:
        row = predictions.loc[index]
        try:
            prediction_time = _utc_timestamp(row["timestamp"])
            target_time = prediction_time + pd.Timedelta(hours=4)
            if now_utc - prediction_time < pd.Timedelta(hours=4):
                continue
            actual_price = _fetch_price_at(target_time)
            outcome = _grade_outcome(row, actual_price)
            row_data = row.to_dict()
            row_data["actual_price"] = actual_price
            row_data["actual_outcome"] = outcome
            store_memory(row_data)
            predictions.at[index, "graded"] = "True"
            predictions.at[index, "actual_outcome"] = outcome
            summary["graded"] += 1
            summary["memorized"] += 1
            print(
                f"[EVALUATOR] {row['timestamp']} graded {outcome} "
                f"(actual BTC close: {actual_price:.2f})."
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


if __name__ == "__main__":
    result = grade_pending_predictions()
    print(
        "[EVALUATOR] Summary: "
        f"{result['graded']} graded, {result['memorized']} memorized, "
        f"{result['pending']} pending, {result['errors']} errors."
    )
