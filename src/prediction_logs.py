import csv
import datetime
import math
from pathlib import Path
from typing import Any

from .paths import LOGS_DIR

LEGACY_LOG_FILE = LOGS_DIR / "predictions_log.csv"
APP_LOG_FILE = LOGS_DIR / "app_predictions_log.csv"
MARKET_LOG_FILE = LOGS_DIR / "market_predictions_log.csv"

APP_LOG_FIELDS = [
    "timestamp",
    "forecast_issue_timestamp",
    "source_candle_timestamp",
    "target_timestamp",
    "forecast_horizon_hours",
    "btc_price_at_prediction",
    "direction",
    "target_low",
    "target_high",
    "invalidation_price",
    "confidence",
    "graded",
    "actual_price_timestamp",
    "actual_outcome",
    "market_regime",
    "reasoning",
]
MARKET_LOG_FIELDS = [
    "timestamp",
    "forecast_issue_timestamp",
    "source_candle_timestamp",
    "target_timestamp",
    "forecast_horizon_hours",
    "ticker",
    "starting_btc_price",
    "predicted_direction",
    "confidence_score",
    "expected_low_2h",
    "expected_high_2h",
    "projected_price_2h",
    "scenario",
    "graded",
    "actual_price",
    "actual_price_timestamp",
    "actual_outcome",
]

LEGACY_APP_FIELDS = [
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
MARKET_DIRECTIONS = {"BULLISH", "BEARISH", "SIDEWAYS"}


def _number(value: str) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _legacy_rows(
    legacy_file: Path = LEGACY_LOG_FILE,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    if not legacy_file.is_file():
        return [], [], 0

    app_rows: list[dict[str, Any]] = []
    market_rows: list[dict[str, Any]] = []
    skipped = 0
    with legacy_file.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.reader(file)
        header = next(reader, [])
        if header != LEGACY_APP_FIELDS:
            print(
                f"[LOG] Unrecognized legacy CSV header in {legacy_file}; "
                "leaving its records untouched."
            )
            return [], [], 0

        for line_number, values in enumerate(reader, start=2):
            if len(values) == len(LEGACY_APP_FIELDS):
                old = dict(zip(LEGACY_APP_FIELDS, values))
                direction = old["direction"].strip().upper()
                if direction in {"UP", "DOWN", "NEUTRAL", "WAIT"}:
                    old_grade = old["graded"].strip()
                    if not old_grade or old_grade.casefold() == "false":
                        old_grade = "Legacy"
                    app_rows.append(
                        {
                            "timestamp": old["timestamp"],
                            "forecast_issue_timestamp": old["timestamp"],
                            "source_candle_timestamp": "",
                            "target_timestamp": "",
                            "forecast_horizon_hours": 4,
                            "btc_price_at_prediction": old["btc_price_at_prediction"],
                            "direction": direction,
                            "target_low": old["target_low"],
                            "target_high": old["target_high"],
                            "invalidation_price": old["invalidation_price"],
                            "confidence": old["confidence"],
                            "graded": old_grade,
                            "actual_price_timestamp": "",
                            "actual_outcome": old["actual_outcome"] or "Not graded",
                            "market_regime": old["market_regime"],
                            "reasoning": old["reasoning"],
                        }
                    )
                    continue

            if len(values) == 9:
                issue_timestamp, ticker, starting_price, direction = values[:4]
                confidence, target_low, target_high, projected_price, scenario = values[4:]
                normalized_direction = direction.strip().upper()
                price_values = [
                    _number(starting_price),
                    _number(confidence),
                    _number(target_low),
                    _number(target_high),
                    _number(projected_price),
                ]
                if (
                    normalized_direction in MARKET_DIRECTIONS
                    and all(value is not None and value > 0 for value in price_values)
                ):
                    market_rows.append(
                        {
                            "timestamp": issue_timestamp,
                            "forecast_issue_timestamp": issue_timestamp,
                            "source_candle_timestamp": "",
                            "target_timestamp": "",
                            "forecast_horizon_hours": 2,
                            "ticker": ticker,
                            "starting_btc_price": starting_price,
                            "predicted_direction": normalized_direction,
                            "confidence_score": confidence,
                            "expected_low_2h": target_low,
                            "expected_high_2h": target_high,
                            "projected_price_2h": projected_price,
                            "scenario": scenario,
                            "graded": "Legacy",
                            "actual_price": "",
                            "actual_price_timestamp": "",
                            "actual_outcome": "Not graded",
                        }
                    )
                    continue

            skipped += 1
            print(
                f"[LOG] Could not safely classify legacy CSV row {line_number}; "
                "it remains preserved in the original file."
            )

    return app_rows, market_rows, skipped


def _initialize_log(
    path: Path,
    fields: list[str],
    migrated_rows: list[dict[str, Any]],
) -> None:
    if path.exists():
        with path.open("r", newline="", encoding="utf-8-sig") as file:
            existing_fields = next(csv.reader(file), [])
        if existing_fields != fields:
            raise ValueError(
                f"Prediction log {path} has an unexpected CSV header; "
                "it was not modified."
            )
        return

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(migrated_rows)


def initialize_app_log(
    path: Path = APP_LOG_FILE,
    legacy_file: Path | None = None,
) -> None:
    if legacy_file is None and path == APP_LOG_FILE:
        legacy_file = LEGACY_LOG_FILE
    app_rows, _, _ = _legacy_rows(legacy_file) if legacy_file else ([], [], 0)
    _initialize_log(path, APP_LOG_FIELDS, app_rows)


def initialize_market_log(
    path: Path = MARKET_LOG_FILE,
    legacy_file: Path | None = None,
) -> None:
    if legacy_file is None and path == MARKET_LOG_FILE:
        legacy_file = LEGACY_LOG_FILE
    _, market_rows, _ = _legacy_rows(legacy_file) if legacy_file else ([], [], 0)
    _initialize_log(path, MARKET_LOG_FIELDS, market_rows)


def append_record(path: Path, fields: list[str], row: dict[str, Any]) -> None:
    missing_fields = set(fields).difference(row)
    if missing_fields:
        raise ValueError(
            "Prediction record is missing required fields: "
            + ", ".join(sorted(missing_fields))
        )
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="raise")
        writer.writerow(row)


def utc_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def to_utc_iso(value: datetime.datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("Forecast timestamps must be timezone-aware.")
    return value.astimezone(datetime.timezone.utc).isoformat(timespec="seconds")
