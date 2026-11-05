import csv
import datetime
import math
import os
import tempfile
import time

import schedule

import chart_engine
import data_engine
import market_predictor
import pro_strategist


CSV_FILENAME = "predictions_log.csv"
CSV_HEADERS = [
    "timestamp",
    "btc_price_at_prediction",
    "direction",
    "target_low",
    "target_high",
    "invalidation_price",
    "confidence",
    "graded",
    "actual_outcome",
    "market_regime",
    "reasoning",
]
ADVISORY_FIELDS = {
    "direction",
    "target_range_low",
    "target_range_high",
    "invalidation_price",
    "confidence_score",
    "market_regime",
    "reasoning",
}


def initialize_csv(filename: str = CSV_FILENAME) -> None:
    if not os.path.exists(filename):
        with open(filename, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(CSV_HEADERS)
        return

    with open(filename, "r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames == CSV_HEADERS:
            return
        old_rows = list(reader)

    migrated_rows = []
    for old_row in old_rows:
        direction = old_row.get("direction") or old_row.get("predicted_direction", "")
        direction = {
            "BULLISH": "UP",
            "BEARISH": "DOWN",
            "SIDEWAYS": "NEUTRAL",
        }.get(direction, direction)
        legacy = not old_row.get("graded")
        migrated_rows.append(
            {
                "timestamp": old_row.get("timestamp", ""),
                "btc_price_at_prediction": old_row.get(
                    "btc_price_at_prediction", old_row.get("current_price", "")
                ),
                "direction": direction,
                "target_low": old_row.get(
                    "target_low", old_row.get("expected_low_2h", "")
                ),
                "target_high": old_row.get(
                    "target_high", old_row.get("expected_high_2h", "")
                ),
                "invalidation_price": old_row.get("invalidation_price", ""),
                "confidence": old_row.get(
                    "confidence", old_row.get("confidence_score", "")
                ),
                "graded": "Legacy" if legacy else old_row.get("graded", "False"),
                "actual_outcome": (
                    "Not graded" if legacy else old_row.get("actual_outcome", "Pending")
                ),
                "market_regime": old_row.get("market_regime", ""),
                "reasoning": old_row.get(
                    "reasoning", old_row.get("scenario", "")
                ),
            }
        )

    directory = os.path.dirname(os.path.abspath(filename))
    file_descriptor, temporary_path = tempfile.mkstemp(
        dir=directory,
        prefix=f".{os.path.basename(filename)}.",
        suffix=".tmp",
        text=True,
    )
    try:
        with os.fdopen(
            file_descriptor,
            "w",
            newline="",
            encoding="utf-8",
        ) as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_HEADERS)
            writer.writeheader()
            writer.writerows(migrated_rows)
        os.replace(temporary_path, filename)
    except Exception:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
        raise


def _validate_advisory(advisory: object) -> bool:
    if not isinstance(advisory, dict) or not ADVISORY_FIELDS.issubset(advisory):
        return False
    if advisory["direction"] not in {"UP", "DOWN", "NEUTRAL", "WAIT"}:
        return False
    if isinstance(advisory["confidence_score"], bool) or not isinstance(
        advisory["confidence_score"], int
    ):
        return False
    if not 1 <= advisory["confidence_score"] <= 10:
        return False
    for field in ("target_range_low", "target_range_high", "invalidation_price"):
        value = advisory[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        if not math.isfinite(value):
            return False
    return isinstance(advisory["market_regime"], str) and isinstance(
        advisory["reasoning"], str
    )


def run_agent_pipeline() -> None:
    timestamp = datetime.datetime.now().isoformat(timespec="seconds")
    print(f"\n[{timestamp}] Gatekeeper triggered.")

    try:
        market_context = data_engine.compile_market_context()
        spot_data = data_engine.fetch_spot_data()
        btc_price = market_context["spot"]["current_close"]

        volume_profile = chart_engine.generate_pro_chart(spot_data)
        full_context = {
            **market_context,
            "volume_profile": volume_profile,
        }
        advisory = pro_strategist.generate_trading_advisory(full_context)

        if not _validate_advisory(advisory):
            print("[GATEKEEPER] Invalid advisory payload; prediction was not logged.")
            return

        initialize_csv()
        row = {
            "timestamp": timestamp,
            "btc_price_at_prediction": btc_price,
            "direction": advisory["direction"],
            "target_low": advisory["target_range_low"],
            "target_high": advisory["target_range_high"],
            "invalidation_price": advisory["invalidation_price"],
            "confidence": advisory["confidence_score"],
            "graded": "False",
            "actual_outcome": "Pending",
            "market_regime": advisory["market_regime"],
            "reasoning": advisory["reasoning"],
        }
        with open(CSV_FILENAME, "a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_HEADERS)
            writer.writerow(row)
        print(f"[GATEKEEPER] Advisory logged to {CSV_FILENAME}.")
    except Exception as exc:
        print(f"[GATEKEEPER] Pipeline failed: {exc}")


def run_market_prediction() -> None:
    print(
        f"\n--- Waking up AI Predictor at "
        f"{datetime.datetime.now().strftime('%H:%M:%S')} ---"
    )
    try:
        market_predictor.run_predictor("BTC-USD")
    except Exception as exc:
        print(f"Error during prediction: {exc}")


if __name__ == "__main__":
    initialize_csv()
    run_agent_pipeline()
    run_market_prediction()
    schedule.every(4).hours.do(run_agent_pipeline)
    schedule.every(60).minutes.do(run_market_prediction)

    print("Starting trading agent and hourly predictor scheduler.")
    print("Leave this terminal open. Press Ctrl+C to stop.")

    while True:
        schedule.run_pending()
        time.sleep(60)
