import csv
import datetime
import hashlib
import shutil
import uuid
from pathlib import Path
from typing import Any

from .paths import LOGS_DIR


BTC_FORECAST_FILE = LOGS_DIR / "btc_forecasts_2h.csv"
ADVISORY_LOG_FILE = LOGS_DIR / "advisory_log.csv"

# Previous log paths are read-only migration sources. They are never appended to.
LEGACY_APP_LOG_FILE = LOGS_DIR / "app_predictions_log.csv"
LEGACY_MARKET_LOG_FILE = LOGS_DIR / "market_predictions_log.csv"
LEGACY_SHARED_LOG_FILE = LOGS_DIR / "predictions_log.csv"

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
ADVISORY_LOG_FIELDS = APP_LOG_FIELDS

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
LEGACY_MARKET_FIELDS = [
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
BTC_FORECAST_FIELDS = [
    "forecast_id",
    "forecast_issue_timestamp",
    "ticker",
    "starting_btc_price",
    "source_candle_timestamp",
    "forecast_horizon_hours",
    "intended_evaluation_timestamp",
    "predicted_direction",
    "expected_high",
    "expected_low",
    "projected_price",
    "confidence_score",
    "scenario",
    "evaluation_status",
    "actual_btc_close",
    "actual_close_timestamp",
    "actual_high",
    "actual_low",
    "direction_correct",
    "final_close_in_predicted_range",
    "full_period_range_inside_predicted_range",
    "projected_price_absolute_error",
    "projected_price_percentage_error",
    "evaluated_at_utc",
    "evaluation_note",
    "legacy_source_file",
    "legacy_source_row",
]


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        return list(reader.fieldnames or []), list(reader)


def _backup_migration_sources() -> None:
    sources = (
        LEGACY_APP_LOG_FILE,
        LEGACY_MARKET_LOG_FILE,
        LEGACY_SHARED_LOG_FILE,
    )
    backup_dir = LOGS_DIR / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for source in sources:
        if source.is_file():
            backup = backup_dir / f"{source.name}.pre-phase1.bak"
            if not backup.exists():
                shutil.copy2(source, backup)


def _legacy_id(source: Path, line_number: int, row: dict[str, str]) -> str:
    row_digest = hashlib.sha256(
        repr(sorted(row.items())).encode("utf-8")
    ).hexdigest()
    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"{source.name}:{line_number}:{row_digest}",
        )
    )


def _read_legacy_market_rows() -> list[dict[str, Any]]:
    source = LEGACY_MARKET_LOG_FILE
    if not source.is_file():
        return []
    fields, rows = _read_rows(source)
    if fields != LEGACY_MARKET_FIELDS:
        raise ValueError(
            f"Legacy market log {source} has an unrecognized header; "
            "it was preserved and not migrated."
        )

    migrated: list[dict[str, Any]] = []
    for line_number, row in enumerate(rows, start=2):
        if None in row or None in row.values():
            print(f"[LOG] Skipping malformed legacy forecast row {line_number}; original retained.")
            continue
        migrated.append(
            {
                "forecast_id": _legacy_id(source, line_number, row),
                "forecast_issue_timestamp": row.get("forecast_issue_timestamp", ""),
                "ticker": row.get("ticker", ""),
                "starting_btc_price": row.get("starting_btc_price", ""),
                "source_candle_timestamp": row.get("source_candle_timestamp", ""),
                "forecast_horizon_hours": row.get("forecast_horizon_hours", ""),
                "intended_evaluation_timestamp": row.get("target_timestamp", ""),
                "predicted_direction": row.get("predicted_direction", ""),
                "expected_high": row.get("expected_high_2h", ""),
                "expected_low": row.get("expected_low_2h", ""),
                "projected_price": row.get("projected_price_2h", ""),
                "confidence_score": row.get("confidence_score", ""),
                "scenario": row.get("scenario", ""),
                "evaluation_status": "NEEDS_REVIEW",
                "actual_btc_close": "",
                "actual_close_timestamp": "",
                "actual_high": "",
                "actual_low": "",
                "direction_correct": "",
                "final_close_in_predicted_range": "",
                "full_period_range_inside_predicted_range": "",
                "projected_price_absolute_error": "",
                "projected_price_percentage_error": "",
                "evaluated_at_utc": "",
                "evaluation_note": (
                    "Legacy record retained; not automatically regraded because its "
                    "timestamps and candle-boundary semantics are not verified."
                ),
                "legacy_source_file": source.name,
                "legacy_source_row": line_number,
            }
        )
    return migrated


def _normalize_advisory_row(
    fields: list[str],
    row: dict[str, str],
) -> dict[str, str]:
    if fields == APP_LOG_FIELDS:
        return {field: row.get(field, "") or "" for field in ADVISORY_LOG_FIELDS}
    if fields == LEGACY_APP_FIELDS:
        legacy = {field: row.get(field, "") or "" for field in LEGACY_APP_FIELDS}
        return {
            "timestamp": legacy["timestamp"],
            "forecast_issue_timestamp": "",
            "source_candle_timestamp": "",
            "target_timestamp": "",
            "forecast_horizon_hours": "",
            "btc_price_at_prediction": legacy["btc_price_at_prediction"],
            "direction": legacy["direction"],
            "target_low": legacy["target_low"],
            "target_high": legacy["target_high"],
            "invalidation_price": legacy["invalidation_price"],
            "confidence": legacy["confidence"],
            "graded": legacy["graded"] or "Legacy",
            "actual_price_timestamp": "",
            "actual_outcome": legacy["actual_outcome"] or "Not graded",
            "market_regime": legacy["market_regime"],
            "reasoning": legacy["reasoning"],
        }
    raise ValueError(
        f"Legacy advisory log {source} has an unrecognized header; "
        "it was preserved and not migrated."
    )


def _read_legacy_advisory_rows() -> list[dict[str, str]]:
    migrated: list[dict[str, str]] = []
    seen: set[tuple[str, ...]] = set()
    for source in (LEGACY_APP_LOG_FILE, LEGACY_SHARED_LOG_FILE):
        if not source.is_file():
            continue
        fields, rows = _read_rows(source)
        if fields not in (APP_LOG_FIELDS, LEGACY_APP_FIELDS):
            # The shared legacy file may contain another schema. Its source and
            # backup remain untouched rather than being guessed at.
            if source == LEGACY_SHARED_LOG_FILE:
                print(f"[LOG] Preserving unrecognized legacy file without migration: {source}")
                continue
            raise ValueError(
                f"Legacy advisory log {source} has an unrecognized header; "
                "it was preserved and not migrated."
            )
        for line_number, row in enumerate(rows, start=2):
            if None in row or None in row.values():
                print(f"[LOG] Skipping malformed legacy advisory row {line_number}; original retained.")
                continue
            normalized = _normalize_advisory_row(fields, row)
            identity = tuple(
                normalized[field]
                for field in (
                    "timestamp",
                    "btc_price_at_prediction",
                    "direction",
                    "target_low",
                    "target_high",
                    "reasoning",
                )
            )
            if identity not in seen:
                migrated.append(normalized)
                seen.add(identity)
    return migrated


def _initialize_log(
    path: Path,
    fields: list[str],
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open("r", newline="", encoding="utf-8-sig") as file:
            existing_fields = next(csv.reader(file), [])
        if existing_fields != fields:
            raise ValueError(
                f"Prediction log {path} has an unexpected CSV header; it was not modified."
            )
        return

    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def initialize_btc_forecast_log(path: Path | None = None) -> None:
    path = path or BTC_FORECAST_FILE
    if path == BTC_FORECAST_FILE and not path.exists():
        _backup_migration_sources()
        rows = _read_legacy_market_rows()
    else:
        rows = []
    _initialize_log(path, BTC_FORECAST_FIELDS, rows)


def initialize_advisory_log(path: Path | None = None) -> None:
    path = path or ADVISORY_LOG_FILE
    if path == ADVISORY_LOG_FILE and not path.exists():
        _backup_migration_sources()
        rows = _read_legacy_advisory_rows()
    else:
        rows = []
    _initialize_log(path, ADVISORY_LOG_FIELDS, rows)


def append_record(path: Path, fields: list[str], row: dict[str, Any]) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Prediction log must be initialized before appending: {path}")
    with path.open("r+", newline="", encoding="utf-8") as file:
        existing_fields = next(csv.reader(file), [])
        if existing_fields != fields:
            raise ValueError(
                f"Prediction log {path} has an unexpected CSV header; row was not appended."
            )
        missing_fields = set(fields).difference(row)
        if missing_fields:
            raise ValueError(
                "Prediction record is missing required fields: "
                + ", ".join(sorted(missing_fields))
            )
        file.seek(0, 2)
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="raise")
        writer.writerow(row)


def utc_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def to_utc_iso(value: datetime.datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("Forecast timestamps must be timezone-aware.")
    return value.astimezone(datetime.timezone.utc).isoformat(timespec="seconds")
