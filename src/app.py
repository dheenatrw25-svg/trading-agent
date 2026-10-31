import datetime
import csv
from threading import Lock
from typing import Any

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from . import chart_engine, data_engine, pro_strategist
from .paths import ASSETS_DIR, PROJECT_ROOT
from .prediction_logs import (
    ADVISORY_LOG_FIELDS,
    ADVISORY_LOG_FILE,
    append_record,
    append_forecast_result,
    FORECAST_RESULTS_FILE,
    FORECAST_RESULTS_FIELDS,
    initialize_advisory_log,
    initialize_forecast_results_log,
    to_utc_iso,
    utc_now,
)


INDEX_FILE = PROJECT_ROOT / "templates" / "index.html"
CHART_FILE = ASSETS_DIR / "prediction_chart.png"
HISTORY_FILE = ADVISORY_LOG_FILE
RESULTS_FILE = FORECAST_RESULTS_FILE
FORECAST_HORIZON_HOURS = 4
ANALYSIS_LOCK = Lock()

app = FastAPI(title="PRO-TRADER AI", version="1.0.0")


def log_prediction(
    decision_json: dict[str, Any],
    market_context: dict[str, Any],
    spot_data: pd.DataFrame,
) -> None:
    decision_json = pro_strategist.validate_advisory(decision_json)
    if spot_data.empty:
        raise ValueError("Cannot log a forecast without confirmed BTC candles.")
    issue_time = utc_now()
    issue_timestamp = to_utc_iso(issue_time)
    source_timestamp = pd.Timestamp(spot_data.index[-1])
    if source_timestamp.tzinfo is None:
        raise ValueError("Source candle timestamp must include a timezone.")
    source_timestamp = source_timestamp.tz_convert("UTC")
    if source_timestamp > pd.Timestamp(issue_time):
        raise ValueError("Source candle timestamp is later than the forecast issue time.")
    row = {
        "timestamp": issue_timestamp,
        "forecast_issue_timestamp": issue_timestamp,
        "source_candle_timestamp": source_timestamp.isoformat(),
        "target_timestamp": to_utc_iso(
            issue_time + datetime.timedelta(hours=FORECAST_HORIZON_HOURS)
        ),
        "forecast_horizon_hours": FORECAST_HORIZON_HOURS,
        "btc_price_at_prediction": market_context["spot"]["current_close"],
        "direction": decision_json["direction"],
        "target_low": decision_json["target_range_low"],
        "target_high": decision_json["target_range_high"],
        "invalidation_price": decision_json["invalidation_price"],
        "confidence": decision_json.get("confidence_score", ""),
        "graded": "False",
        "actual_price_timestamp": "",
        "actual_outcome": "Pending",
        "market_regime": decision_json.get("market_regime", ""),
        "reasoning": decision_json.get("reasoning", ""),
    }
    initialize_advisory_log(HISTORY_FILE)
    append_record(HISTORY_FILE, ADVISORY_LOG_FIELDS, row)
    result_row = {
        **row,
        "target_price": decision_json["target_price"],
        "setup_actionable": pro_strategist.calculate_risk_reward(
            market_context["spot"]["current_close"],
            decision_json["direction"],
            decision_json["target_range_low"],
            decision_json["target_range_high"],
            decision_json["invalidation_price"],
        )["actionable"],
    }
    append_forecast_result(
        result_row,
        FORECAST_HORIZON_HOURS,
        path=RESULTS_FILE,
    )


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
            spot_data = data_engine.fetch_spot_data()
            market_context = data_engine.compile_market_context(spot_data=spot_data)
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
            log_prediction(decision, market_context, spot_data)
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


@app.get("/api/history", response_model=None)
def get_history() -> list[dict[str, Any]] | JSONResponse:
    try:
        if not RESULTS_FILE.is_file():
            initialize_forecast_results_log(RESULTS_FILE)
        with RESULTS_FILE.open("r", newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames != FORECAST_RESULTS_FIELDS:
                return JSONResponse(
                    status_code=500,
                    content={"error": "Forecast results log has an unexpected schema."},
                )
            history = list(reader)
        if any(None in row for row in history):
            return JSONResponse(
                status_code=500,
                content={"error": "Forecast results log contains malformed rows."},
            )
        history.sort(
            key=lambda row: row["Analysed Time (IST)"],
            reverse=True,
        )
        return [
            {field: row.get(field, "") or "" for field in FORECAST_RESULTS_FIELDS}
            for row in history
        ]
    except Exception as exc:
        print(f"[API] History read failed: {exc}")
        return JSONResponse(
            status_code=500,
            content={"error": "Prediction history could not be read.", "detail": str(exc)},
        )
