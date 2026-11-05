import json
import datetime
import math
import tempfile
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import mplfinance as mpf
from PIL import Image
from google import genai
from google.genai import types
from dotenv import load_dotenv
from .paths import ASSETS_DIR, PROJECT_ROOT
from .prediction_logs import (
    BTC_FORECAST_FIELDS,
    BTC_FORECAST_FILE,
    append_record,
    initialize_btc_forecast_log,
    to_utc_iso,
    utc_now,
)

load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

LOG_FILE = BTC_FORECAST_FILE
FORECAST_HORIZON_HOURS = 2
ALLOWED_DIRECTIONS = {"BULLISH", "BEARISH", "SIDEWAYS"}
REQUIRED_PREDICTION_FIELDS = {
    "predicted_direction",
    "confidence_score",
    "expected_high_next_2h",
    "expected_low_next_2h",
    "projected_price_in_2h",
    "predicted_scenario",
}


def validate_prediction(prediction: object) -> dict:
    if not isinstance(prediction, dict):
        raise ValueError("Gemini forecast must be a JSON object.")
    missing = REQUIRED_PREDICTION_FIELDS.difference(prediction)
    if missing:
        raise ValueError(
            "Gemini forecast is missing required fields: "
            + ", ".join(sorted(missing))
        )

    direction = prediction["predicted_direction"]
    if not isinstance(direction, str) or direction.strip().upper() not in ALLOWED_DIRECTIONS:
        raise ValueError("Gemini forecast has an unsupported predicted_direction.")
    confidence = prediction["confidence_score"]
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(confidence)
        or not 1 <= confidence <= 10
    ):
        raise ValueError("Gemini confidence_score must be a finite number from 1 to 10.")

    prices = {}
    for field in (
        "expected_high_next_2h",
        "expected_low_next_2h",
        "projected_price_in_2h",
    ):
        value = prediction[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"Gemini forecast field {field} must be a positive number.")
        prices[field] = float(value)
    if prices["expected_low_next_2h"] > prices["expected_high_next_2h"]:
        raise ValueError("Gemini forecast low price exceeds its high price.")
    if not isinstance(prediction["predicted_scenario"], str) or not prediction[
        "predicted_scenario"
    ].strip():
        raise ValueError("Gemini predicted_scenario must be a non-empty string.")
    validated = dict(prediction)
    validated["predicted_direction"] = direction.strip().upper()
    return validated

# ==========================================
# 1. CHART GENERATOR (The Eyes)
# ==========================================
def download_clean_df(ticker: str, interval: str, period: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.empty:
        raise RuntimeError(f"No {interval} candle data returned for {ticker}.")

    timestamps = pd.to_datetime(df.index, utc=True, errors="coerce")
    confirmed = timestamps.notna() & (
        timestamps + pd.Timedelta(interval) <= pd.Timestamp.now(tz="UTC")
    )
    df = df.loc[confirmed].copy()
    df.index = timestamps[confirmed]
    df = df.loc[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    ohlcv = df[["Open", "High", "Low", "Close", "Volume"]].to_numpy(dtype=float)
    valid_rows = (
        np.isfinite(ohlcv).all(axis=1)
        & (ohlcv[:, :4] > 0).all(axis=1)
        & (ohlcv[:, 1] >= ohlcv[:, 2])
        & (ohlcv[:, 4] >= 0)
    )
    df = df.loc[valid_rows]
    if df.empty:
        raise RuntimeError(f"No complete {interval} candles returned for {ticker}.")
    return df

def generate_dual_chart(
    ticker: str = "BTC-USD",
    output_path: str | Path = ASSETS_DIR / "prediction_chart.png",
):
    # Macro Trend (1-Hour candles) and Micro Structure (15-Minute candles)
    df_macro = download_clean_df(ticker, interval="1h", period="10d")
    df_micro = download_clean_df(ticker, interval="15m", period="3d")

    slice_macro = df_macro.tail(50).copy()
    slice_micro = df_micro.tail(60).copy()
    if len(slice_macro) < 50 or len(slice_micro) < 50:
        raise RuntimeError(
            "At least 50 complete hourly and 15-minute candles are required "
            "to generate the forecast charts."
        )

    custom_style = mpf.make_mpf_style(
        base_mpf_style='nightclouds',
        rc={'font.size': 7}
    )

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="forecast-", dir=ASSETS_DIR) as temp_dir:
        macro_path = Path(temp_dir) / "macro.png"
        micro_path = Path(temp_dir) / "micro.png"
        mpf.plot(
            slice_macro, type="candle", volume=True, mav=(20, 50),
            style=custom_style, title=f"{ticker} - Macro Trend (1H)",
            savefig=str(macro_path),
        )
        mpf.plot(
            slice_micro, type="candle", volume=True, mav=(20, 50),
            style=custom_style, title=f"{ticker} - Short-Term Structure (15m)",
            savefig=str(micro_path),
        )

        with Image.open(macro_path) as img_top, Image.open(micro_path) as img_bottom:
            width = max(img_top.width, img_bottom.width)
            height = img_top.height + img_bottom.height
            combined = Image.new("RGB", (width, height))
            combined.paste(img_top, (0, 0))
            combined.paste(img_bottom, (0, img_top.height))
            combined.save(output_file)

    return slice_macro, slice_micro, str(output_file)

# ==========================================
# 2. MATHEMATICAL INDICATORS (The Calculator)
# ==========================================
def calculate_metrics(df_micro: pd.DataFrame, df_macro: pd.DataFrame) -> dict:
    required = {"Close", "High", "Low"}
    if not required.issubset(df_micro.columns) or "Close" not in df_macro.columns:
        raise ValueError("Forecast candle data is missing required OHLC columns.")
    if len(df_micro) < 24 or df_macro.empty:
        raise ValueError("Insufficient confirmed candle data to calculate metrics.")

    close = df_micro['Close']
    high = df_micro['High']
    low = df_micro['Low']

    # 14-period RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))

    # 14-period ATR (Volatility)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()

    # Macro 50 EMA Trend
    ema50_macro = df_macro['Close'].ewm(span=50, adjust=False).mean()
    current_price = float(close.iloc[-1])
    if not math.isfinite(current_price) or current_price <= 0:
        raise ValueError("Latest confirmed BTC price must be finite and positive.")
    macro_bias = "BULLISH" if current_price > float(ema50_macro.iloc[-1]) else "BEARISH"

    return {
        "current_price": round(current_price, 2),
        "macro_trend": macro_bias,
        "rsi_15m": round(float(rsi.iloc[-1]), 2),
        "atr_15m": round(float(atr.iloc[-1]), 2),
        "recent_swing_high": round(float(high.tail(24).max()), 2),
        "recent_swing_low": round(float(low.tail(24).min()), 2),
        "source_candle_timestamp": (
            pd.Timestamp(df_micro.index[-1]) + pd.Timedelta(minutes=15)
        ).isoformat(),
    }

# ==========================================
# 3. AI MARKET FORECASTER (The Brain)
# ==========================================
def predict_next_2_hours(image_path: str, metrics: dict, ticker: str) -> dict:
    client = genai.Client()
    chart_image = Image.open(image_path)

    system_prompt = """
    You are an expert market analyst specializing in short-term price action forecasting.
    Do NOT recommend buying or selling. Your only job is to predict what the price will do over the NEXT 2 HOURS (the next 8 fifteen-minute candles).

    Analyze:
    1. Top Chart (1H Macro Trend) vs Bottom Chart (15m Micro Structure).
    2. Key support and resistance zones (recent swing highs and lows).
    3. Momentum and volatility using the provided RSI and ATR numbers.

    Respond ONLY in valid JSON matching this exact structure:
    {
        "predicted_direction": "BULLISH" | "BEARISH" | "SIDEWAYS",
        "confidence_score": 1-10,
        "expected_high_next_2h": float,
        "expected_low_next_2h": float,
        "projected_price_in_2h": float,
        "predicted_scenario": "Clear 2-3 sentence explanation of how price will move over the next 2 hours."
    }
    """

    user_prompt = f"""
    Asset: {ticker}
    Exact Market Metrics: {json.dumps(metrics, indent=2)}
    Examine the dual chart image and metrics. Forecast the next 2 hours of price action.
    """

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=[chart_image, user_prompt],
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            temperature=0.2
        )
    )

    try:
        return validate_prediction(json.loads(response.text))
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("Gemini response was not a valid JSON forecast.") from exc

# ==========================================
# 4. LOGGING & EXECUTION
# ==========================================
def log_prediction(
    ticker: str,
    metrics: dict,
    prediction: dict,
    log_file: Path = LOG_FILE,
    issued_at: datetime.datetime | None = None,
) -> None:
    prediction = validate_prediction(prediction)
    issue_time = issued_at or utc_now()
    if issue_time.tzinfo is None:
        raise ValueError("Forecast issue timestamp must include a timezone.")
    issue_timestamp = to_utc_iso(issue_time)
    source_timestamp = pd.Timestamp(metrics["source_candle_timestamp"])
    if source_timestamp.tzinfo is None:
        raise ValueError("Source candle timestamp must include a timezone.")
    source_timestamp = source_timestamp.tz_convert("UTC")
    if source_timestamp > pd.Timestamp(issue_time):
        raise ValueError("Source candle timestamp is later than the forecast issue time.")
    starting_price = metrics["current_price"]
    if (
        isinstance(starting_price, bool)
        or not isinstance(starting_price, (int, float))
        or not math.isfinite(starting_price)
        or starting_price <= 0
    ):
        raise ValueError("Starting BTC price must be finite and positive.")
    if not isinstance(ticker, str) or ticker.strip().upper() != "BTC-USD":
        raise ValueError("The 2-hour forecast log accepts only the BTC-USD ticker.")

    target_time = issue_time.astimezone(datetime.timezone.utc) + datetime.timedelta(
        hours=FORECAST_HORIZON_HOURS
    )
    record = {
        "forecast_id": str(uuid.uuid4()),
        "forecast_issue_timestamp": issue_timestamp,
        "ticker": ticker.strip().upper(),
        "starting_btc_price": starting_price,
        "source_candle_timestamp": source_timestamp.isoformat(),
        "forecast_horizon_hours": FORECAST_HORIZON_HOURS,
        "intended_evaluation_timestamp": to_utc_iso(target_time),
        "predicted_direction": prediction.get("predicted_direction"),
        "expected_high": prediction.get("expected_high_next_2h"),
        "expected_low": prediction.get("expected_low_next_2h"),
        "projected_price": prediction.get("projected_price_in_2h"),
        "confidence_score": prediction.get("confidence_score"),
        "scenario": prediction.get("predicted_scenario"),
        "evaluation_status": "PENDING",
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
        "evaluation_note": "",
        "legacy_source_file": "",
        "legacy_source_row": "",
    }

    initialize_btc_forecast_log(log_file)
    append_record(log_file, BTC_FORECAST_FIELDS, record)

def run_predictor(ticker: str = "BTC-USD"):
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 1. Generating charts for {ticker}...")
    df_macro, df_micro, chart_path = generate_dual_chart(ticker)

    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 2. Calculating market metrics...")
    metrics = calculate_metrics(df_micro, df_macro)

    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 3. Asking Gemini for 2-hour prediction...")
    prediction = predict_next_2_hours(chart_path, metrics, ticker)
    prediction = validate_prediction(prediction)

    print("\n================ 2-HOUR MARKET FORECAST ================")
    print(f"Asset:               {ticker}")
    print(f"Current Price:       ${metrics['current_price']}")
    print(f"Predicted Direction: {prediction['predicted_direction']} (Confidence: {prediction['confidence_score']}/10)")
    print(f"Expected 2H Range:   ${prediction['expected_low_next_2h']} to${prediction['expected_high_next_2h']}")
    print(f"Target Price (2H):   ${prediction['projected_price_in_2h']}")
    print(f"Scenario:            {prediction['predicted_scenario']}")
    print("========================================================\n")

    log_prediction(ticker, metrics, prediction)
    print(f"Saved forecast to {LOG_FILE}")

if __name__ == "__main__":
    run_predictor("BTC-USD")
