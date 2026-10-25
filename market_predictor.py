import os
import json
import datetime
import pandas as pd
import yfinance as yf
import mplfinance as mpf
from PIL import Image
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

LOG_FILE = "predictions_log.csv"

# ==========================================
# 1. CHART GENERATOR (The Eyes)
# ==========================================
def download_clean_df(ticker: str, interval: str, period: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()

def generate_dual_chart(ticker: str = "BTC-USD", output_path: str = "prediction_chart.png"):
    # Macro Trend (1-Hour candles) and Micro Structure (15-Minute candles)
    df_macro = download_clean_df(ticker, interval="1h", period="10d")
    df_micro = download_clean_df(ticker, interval="15m", period="3d")

    slice_macro = df_macro.tail(50).copy()
    slice_micro = df_micro.tail(60).copy()

    custom_style = mpf.make_mpf_style(
        base_mpf_style='nightclouds',
        rc={'font.size': 7}
    )

    mpf.plot(
        slice_macro, type='candle', volume=True, mav=(20, 50),
        style=custom_style, title=f"{ticker} - Macro Trend (1H)",
        savefig="temp_macro.png"
    )
    mpf.plot(
        slice_micro, type='candle', volume=True, mav=(20, 50),
        style=custom_style, title=f"{ticker} - Short-Term Structure (15m)",
        savefig="temp_micro.png"
    )

    # Combine both charts vertically into one image
    img_top = Image.open("temp_macro.png")
    img_bottom = Image.open("temp_micro.png")

    width = max(img_top.width, img_bottom.width)
    height = img_top.height + img_bottom.height
    combined = Image.new("RGB", (width, height))
    combined.paste(img_top, (0, 0))
    combined.paste(img_bottom, (0, img_top.height))
    combined.save(output_path)

    os.remove("temp_macro.png")
    os.remove("temp_micro.png")

    return slice_macro, slice_micro, output_path

# ==========================================
# 2. MATHEMATICAL INDICATORS (The Calculator)
# ==========================================
def calculate_metrics(df_micro: pd.DataFrame, df_macro: pd.DataFrame) -> dict:
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
    macro_bias = "BULLISH" if current_price > float(ema50_macro.iloc[-1]) else "BEARISH"

    return {
        "current_price": round(current_price, 2),
        "macro_trend": macro_bias,
        "rsi_15m": round(float(rsi.iloc[-1]), 2),
        "atr_15m": round(float(atr.iloc[-1]), 2),
        "recent_swing_high": round(float(high.tail(24).max()), 2),
        "recent_swing_low": round(float(low.tail(24).min()), 2)
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

    return json.loads(response.text)

# ==========================================
# 4. LOGGING & EXECUTION
# ==========================================
def log_prediction(ticker: str, metrics: dict, prediction: dict):
    record = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ticker": ticker,
        "current_price": metrics["current_price"],
        "predicted_direction": prediction.get("predicted_direction"),
        "confidence_score": prediction.get("confidence_score"),
        "expected_low_2h": prediction.get("expected_low_next_2h"),
        "expected_high_2h": prediction.get("expected_high_next_2h"),
        "projected_price_2h": prediction.get("projected_price_in_2h"),
        "scenario": prediction.get("predicted_scenario")
    }

    df = pd.DataFrame([record])
    if not os.path.exists(LOG_FILE):
        df.to_csv(LOG_FILE, index=False)
    else:
        df.to_csv(LOG_FILE, mode="a", header=False, index=False)

def run_predictor(ticker: str = "BTC-USD"):
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 1. Generating charts for {ticker}...")
    df_macro, df_micro, chart_path = generate_dual_chart(ticker)

    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 2. Calculating market metrics...")
    metrics = calculate_metrics(df_micro, df_macro)

    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 3. Asking Gemini for 2-hour prediction...")
    prediction = predict_next_2_hours(chart_path, metrics, ticker)

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
