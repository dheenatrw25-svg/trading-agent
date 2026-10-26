import csv
import datetime
from pathlib import Path
from threading import Lock
from typing import Any

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

import chart_engine
import data_engine
import pro_strategist


BASE_DIR = Path(__file__).resolve().parent
INDEX_FILE = BASE_DIR / "templates" / "index.html"
CHART_FILE = BASE_DIR / "prediction_chart.png"
HISTORY_FILE = BASE_DIR / "predictions_log.csv"
PREDICTION_LOG_FIELDS = [
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
ANALYSIS_LOCK = Lock()

app = FastAPI(title="PRO-TRADER AI", version="1.0.0")


def log_prediction(decision_json: dict[str, Any]) -> None:
    log_file = Path(__file__).resolve().parent / "predictions_log.csv"
    row = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "btc_price_at_prediction": "",
        "direction": decision_json["direction"],
        "target_low": decision_json["target_range_low"],
        "target_high": decision_json["target_range_high"],
        "invalidation_price": decision_json["invalidation_price"],
        "confidence": decision_json.get("confidence_score", ""),
        "graded": "False",
        "actual_outcome": "Pending",
        "market_regime": decision_json.get("market_regime", ""),
        "reasoning": decision_json.get("reasoning", ""),
    }
    write_headers = not log_file.exists() or log_file.stat().st_size == 0
    with log_file.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=PREDICTION_LOG_FIELDS)
        if write_headers:
            writer.writeheader()
        writer.writerow(row)


@app.get("/", response_class=HTMLResponse)
def dashboard() -> HTMLResponse:
    try:
        page = INDEX_FILE.read_text(encoding="utf-8")
    except OSError as exc:
        return HTMLResponse(
            content=f"Dashboard could not be loaded: {exc}",
            status_code=500,
        )
    return HTMLResponse(content=page)


@app.get("/api/analyze", response_model=None)
def analyze() -> dict[str, Any] | JSONResponse:
    try:
        with ANALYSIS_LOCK:
            market_context = data_engine.compile_market_context()
            spot_data = data_engine.fetch_spot_data()
            volume_profile = chart_engine.generate_pro_chart(
                spot_data,
                filename=str(CHART_FILE),
            )
            full_context = {
                **market_context,
                "volume_profile": volume_profile,
            }
            decision = pro_strategist.generate_trading_advisory(
                full_context,
                chart_image_path=str(CHART_FILE),
            )
            log_prediction(decision)
            return decision
    except Exception as exc:
        print(f"[API] Analysis failed: {exc}")
        return JSONResponse(
            status_code=502,
            content={
                "error": "Market analysis failed.",
                "detail": str(exc),
            },
        )


@app.get("/api/chart", response_model=None)
def get_chart() -> FileResponse | JSONResponse:
    if not CHART_FILE.is_file():
        return JSONResponse(
            status_code=404,
            content={"error": "No analysis chart is available yet. Run analysis first."},
        )
    return FileResponse(
        path=CHART_FILE,
        media_type="image/png",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


def _history_record(row: dict[str, Any]) -> dict[str, Any]:
    direction = row.get("direction") or row.get("predicted_direction") or "UNKNOWN"
    direction = {
        "BULLISH": "UP",
        "BEARISH": "DOWN",
        "SIDEWAYS": "NEUTRAL",
    }.get(str(direction).upper(), str(direction).upper())

    outcome = row.get("actual_outcome")
    if not outcome or pd.isna(outcome):
        outcome = "Pending" if str(row.get("graded", "")).lower() == "false" else "—"

    price = row.get("btc_price_at_prediction")
    if price is None or pd.isna(price):
        price = row.get("current_price")

    confidence = row.get("confidence")
    if confidence is None or pd.isna(confidence):
        confidence = row.get("confidence_score")

    target_low = row.get("target_low")
    if target_low is None or pd.isna(target_low):
        target_low = row.get("expected_low_2h")

    target_high = row.get("target_high")
    if target_high is None or pd.isna(target_high):
        target_high = row.get("expected_high_2h")

    return {
        "timestamp": row.get("timestamp", ""),
        "direction": direction,
        "price": None if price is None or pd.isna(price) else float(price),
        "outcome": str(outcome),
        "confidence": (
            None if confidence is None or pd.isna(confidence) else confidence
        ),
        "target_low": (
            None if target_low is None or pd.isna(target_low) else float(target_low)
        ),
        "target_high": (
            None if target_high is None or pd.isna(target_high) else float(target_high)
        ),
        "market_regime": row.get("market_regime", ""),
        "reasoning": row.get("reasoning") or row.get("scenario") or "",
    }


@app.get("/api/history", response_model=None)
def get_history() -> list[dict[str, Any]] | JSONResponse:
    if not HISTORY_FILE.is_file():
        return []
    try:
        history = pd.read_csv(HISTORY_FILE)
        if "timestamp" not in history.columns:
            return JSONResponse(
                status_code=500,
                content={"error": "Prediction log is missing the timestamp column."},
            )
        history = history.tail(5)
        return [
            _history_record(row)
            for row in history.to_dict(orient="records")
        ][::-1]
    except Exception as exc:
        print(f"[API] History read failed: {exc}")
        return JSONResponse(
            status_code=500,
            content={"error": "Prediction history could not be read.", "detail": str(exc)},
        )
