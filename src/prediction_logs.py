import csv
import datetime
import hashlib
import math
import shutil
import threading
import uuid
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from .paths import LOGS_DIR


BTC_FORECAST_FILE = LOGS_DIR / "btc_forecasts_2h.csv"
ADVISORY_LOG_FILE = LOGS_DIR / "advisory_log.csv"
FORECAST_RESULTS_FILE = LOGS_DIR / "forecast_results.csv"
FORECAST_RESULTS_FIELDS = [
    "Analysed Time (IST)",
    "Current Price",
    "Forecast Hours",
    "Target Price Range",
    "Target Price Time (IST)",
    "Exact Target Price",
    "UP / DOWN / WAIT",
    "Reason",
    "Confidence",
]
FORECAST_RESULTS_LOCK = threading.Lock()

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


def _format_ist_timestamp(value: object, hours: int = 0) -> str:
    if not isinstance(value, str) or not value.strip():
        return ""
    try:
        timestamp = datetime.datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return ""
    if timestamp.tzinfo is None:
        return ""
    timestamp = timestamp.astimezone(datetime.timezone.utc) + datetime.timedelta(hours=hours)
    return timestamp.astimezone(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S IST")


def _format_result_price(value: object) -> str:
    if value in (None, ""):
        return ""
    try:
        price = float(value)
    except (TypeError, ValueError):
        return ""
    if price <= 0 or not math.isfinite(price):
        return ""
    return f"${price:,.2f}"


def _format_result_range(low: object, high: object) -> str:
    formatted_low = _format_result_price(low)
    formatted_high = _format_result_price(high)
    if not formatted_low and not formatted_high:
        return ""
    return f"{formatted_low} - {formatted_high}"


def _format_result_confidence(value: object) -> str:
    if value in (None, ""):
        return ""
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return ""
    if not 1 <= confidence <= 10:
        return ""
    return f"{confidence:g}/10"


def _result_direction(direction: object) -> str:
    normalized = str(direction or "").strip().upper()
    return {
        "BULLISH": "UP",
        "BEARISH": "DOWN",
        "SIDEWAYS": "WAIT",
        "NEUTRAL": "WAIT",
    }.get(normalized, normalized if normalized in {"UP", "DOWN", "WAIT"} else "")


def _result_setup_actionable(row: dict[str, Any], direction: str) -> bool:
    value = str(row.get("setup_actionable", "")).strip().lower()
    if value:
        return value in {"true", "1", "yes"}
    try:
        current = float(row.get("starting_btc_price", row.get("btc_price_at_prediction")))
        low = float(row.get("expected_low", row.get("expected_low_2h", row.get("target_low"))))
        high = float(row.get("expected_high", row.get("expected_high_2h", row.get("target_high"))))
        invalidation = float(row.get("invalidation_price"))
    except (TypeError, ValueError):
        return False
    risk = current - invalidation if direction == "UP" else invalidation - current
    midpoint = (low + high) / 2
    reward = midpoint - current if direction == "UP" else current - midpoint
    return risk > 0 and reward > 0 and reward / risk >= 1.5


def build_forecast_result(
    row: dict[str, Any],
    forecast_hours: int | None = None,
) -> dict[str, str]:
    horizon_value = forecast_hours or row.get("forecast_horizon_hours")
    try:
        horizon = int(horizon_value)
    except (TypeError, ValueError):
        horizon = 0
    if horizon not in (2, 4):
        horizon = 0

    issue_time = row.get("forecast_issue_timestamp") or row.get("timestamp")
    low = row.get("expected_low", row.get("expected_low_2h", row.get("target_low")))
    high = row.get("expected_high", row.get("expected_high_2h", row.get("target_high")))
    exact_target = row.get(
        "projected_price",
        row.get("projected_price_2h", row.get("target_price", "")),
    )
    direction = _result_direction(
        row.get("predicted_direction") or row.get("direction")
    )
    confidence = row.get("confidence_score", row.get("confidence", ""))
    reason = row.get("scenario") or row.get("reasoning") or ""

    if horizon == 4:
        try:
            low_number = float(low)
            high_number = float(high)
            target_number = float(exact_target)
        except (TypeError, ValueError):
            pass
        else:
            if not low_number <= target_number <= high_number:
                raise ValueError("4-hour exact target price must be within its target range.")
        reason_text = str(reason).lower()
        setup_actionable = _result_setup_actionable(row, direction)
        if (
            not setup_actionable
            or "not an actionable setup" in reason_text
            or "non-actionable" in reason_text
        ):
            direction = "WAIT"
        try:
            if float(confidence) <= 4:
                direction = "WAIT"
        except (TypeError, ValueError):
            pass

    return {
        "Analysed Time (IST)": _format_ist_timestamp(issue_time),
        "Current Price": _format_result_price(
            row.get("starting_btc_price", row.get("btc_price_at_prediction"))
        ),
        "Forecast Hours": str(horizon) if horizon else "",
        "Target Price Range": _format_result_range(low, high),
        "Target Price Time (IST)": _format_ist_timestamp(issue_time, horizon),
        "Exact Target Price": _format_result_price(exact_target),
        "UP / DOWN / WAIT": direction,
        "Reason": str(reason),
        "Confidence": _format_result_confidence(confidence),
    }


def _migration_result_sources() -> list[Path]:
    return [
        BTC_FORECAST_FILE,
        ADVISORY_LOG_FILE,
        LEGACY_MARKET_LOG_FILE,
        LEGACY_APP_LOG_FILE,
        LEGACY_SHARED_LOG_FILE,
    ]


def _recover_shared_legacy_market_row(row: dict[str, Any]) -> dict[str, str] | None:
    values = list(row.values())
    if len(values) != len(LEGACY_APP_FIELDS) or any(value is not None for value in values[9:]):
        return None
    timestamp, ticker, price, direction, confidence, low, high, target, reason = values[:9]
    if (
        ticker != "BTC-USD"
        or str(direction).upper() not in {"BULLISH", "BEARISH", "SIDEWAYS"}
    ):
        return None
    try:
        numeric_values = [float(value) for value in (price, confidence, low, high, target)]
    except (TypeError, ValueError):
        return None
    if (
        any(not math.isfinite(value) for value in numeric_values)
        or numeric_values[0] <= 0
        or not 1 <= numeric_values[1] <= 10
        or numeric_values[2] <= 0
        or numeric_values[3] < numeric_values[2]
        or not numeric_values[2] <= numeric_values[4] <= numeric_values[3]
        or not isinstance(reason, str)
        or not reason.strip()
    ):
        return None
    return {
        "timestamp": str(timestamp or ""),
        "forecast_issue_timestamp": str(timestamp or ""),
        "forecast_horizon_hours": "2",
        "starting_btc_price": str(price),
        "predicted_direction": str(direction),
        "confidence_score": str(confidence),
        "expected_low_2h": str(low),
        "expected_high_2h": str(high),
        "projected_price_2h": str(target),
        "scenario": reason,
    }


def _source_result_rows(source: Path) -> list[dict[str, str]]:
    if not source.is_file():
        return []
    fields, rows = _read_rows(source)
    market_fields = set(fields)
    if {"starting_btc_price", "predicted_direction", "expected_low", "expected_high"}.issubset(market_fields):
        horizon = 2
    elif {"starting_btc_price", "predicted_direction", "expected_low_2h", "expected_high_2h"}.issubset(market_fields):
        horizon = 2
    elif {"direction", "target_low", "target_high", "btc_price_at_prediction"}.issubset(market_fields):
        horizon = 4
    else:
        print(f"[LOG] Preserving unrecognized forecast log without unified migration: {source}")
        return []

    migrated: list[dict[str, str]] = []
    for line_number, row in enumerate(rows, start=2):
        if None in row or None in row.values():
            if source == LEGACY_SHARED_LOG_FILE:
                recovered_row = _recover_shared_legacy_market_row(row)
                if recovered_row is not None:
                    migrated.append(build_forecast_result(recovered_row, 2))
                    continue
            print(
                f"[LOG] Skipping malformed unified-migration row {line_number} "
                f"in {source}; the source file was retained."
            )
            continue
        try:
            migrated.append(build_forecast_result(row, horizon))
        except ValueError as exc:
            print(f"[LOG] Skipping invalid forecast row in {source}: {exc}")
    return migrated


def initialize_forecast_results_log(path: Path | None = None) -> None:
    path = path or FORECAST_RESULTS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with FORECAST_RESULTS_LOCK:
        if path.exists():
            existing_fields, existing_rows = _read_rows(path)
            if existing_fields != FORECAST_RESULTS_FIELDS:
                raise ValueError(
                    f"Forecast results log {path} has an unexpected CSV header; it was not modified."
                )
            seen = {
                tuple(row.get(field, "") for field in FORECAST_RESULTS_FIELDS)
                for row in existing_rows
            }
            rows: list[dict[str, str]] = []
        else:
            seen = set()
            rows = []
        for source in _migration_result_sources():
            for record in _source_result_rows(source):
                identity = tuple(record[field] for field in FORECAST_RESULTS_FIELDS)
                if identity not in seen:
                    rows.append(record)
                    seen.add(identity)

        if path.exists():
            if rows:
                with path.open("a", newline="", encoding="utf-8") as file:
                    writer = csv.DictWriter(
                        file,
                        fieldnames=FORECAST_RESULTS_FIELDS,
                        extrasaction="raise",
                    )
                    writer.writerows(rows)
        else:
            temporary_path = path.with_suffix(path.suffix + ".tmp")
            with temporary_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=FORECAST_RESULTS_FIELDS,
                    extrasaction="raise",
                )
                writer.writeheader()
                writer.writerows(rows)
            temporary_path.replace(path)


def append_forecast_result(
    row: dict[str, Any],
    forecast_hours: int | None = None,
    path: Path | None = None,
) -> None:
    path = path or FORECAST_RESULTS_FILE
    record = build_forecast_result(row, forecast_hours)
    initialize_forecast_results_log(path)
    with FORECAST_RESULTS_LOCK:
        existing_fields, existing_rows = _read_rows(path)
        if existing_fields != FORECAST_RESULTS_FIELDS:
            raise ValueError(
                f"Forecast results log {path} has an unexpected CSV header; row was not appended."
            )
        identity = tuple(record[field] for field in FORECAST_RESULTS_FIELDS)
        if any(
            tuple(existing.get(field, "") for field in FORECAST_RESULTS_FIELDS) == identity
            for existing in existing_rows
        ):
            return
        with path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=FORECAST_RESULTS_FIELDS, extrasaction="raise")
            writer.writerow(record)


def utc_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def to_utc_iso(value: datetime.datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("Forecast timestamps must be timezone-aware.")
    return value.astimezone(datetime.timezone.utc).isoformat(timespec="seconds")
