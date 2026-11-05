import json
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

from evaluator import retrieve_past_lessons

load_dotenv()

PRO_TRADER_PROMPT = """You are an elite institutional quantitative cryptocurrency analyst. Your objective is to provide highly accurate, probabilistic market forecasts with strict risk management.
You have two inputs:
1. A visual chart displaying price action and institutional Volume Profile levels (Yellow = Point of Control, Cyan = Value Area High/Low).
2. A JSON payload containing exact spot prices, derivatives positioning (Open Interest, Funding Rates), and Macro conditions (DXY).
YOUR CONFLUENCE RUBRIC:
- Compare visual market structure against derivatives data. If price is rising but Open Interest is dropping, flag it as a low-volume short squeeze.
- If Funding Rates are highly elevated, be cautious of a long flush.
- If the DXY is spiking upward, heavily discount bullish Bitcoin setups.
- Use the exact Volume Profile prices from the data payload for your targets and invalidation levels.
OUTPUT RESTRICTION: Output YOUR ENTIRE RESPONSE as a single, valid JSON object. No markdown formatting.
SCHEMA: {"direction": "UP"|"DOWN"|"NEUTRAL"|"WAIT", "target_range_low": float, "target_range_high": float, "invalidation_price": float, "confidence_score": int, "market_regime": "string", "reasoning": "string"}"""


def generate_trading_advisory(
    market_context: dict[str, Any],
    chart_image_path: str = "prediction_chart.png",
) -> dict[str, Any]:
    image_path = Path(chart_image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Chart image not found: {image_path}")

    data_string = json.dumps(market_context, indent=2)
    regime_context = market_context.get("market_regime")
    if not isinstance(regime_context, str) or not regime_context.strip():
        regime_context = json.dumps(
            {
                "macro": market_context.get("macro", {}),
                "derivatives": market_context.get("derivatives", {}),
                "volume_profile": market_context.get("volume_profile", {}),
            },
            sort_keys=True,
        )
    past_lessons = retrieve_past_lessons(regime_context)
    client = genai.Client()

    with Image.open(image_path) as chart_image:
        for attempt in range(4):
            try:
                response = client.models.generate_content(
                    model="gemini-3.8-flash",
                    contents=[
                        PRO_TRADER_PROMPT,
                        data_string,
                        f"Relevant lessons from past trades:\n{past_lessons}",
                        chart_image,
                    ],
                    config=types.GenerateContentConfig(
                        temperature=0.2,
                        response_mime_type="application/json",
                    ),
                )
                break
            except Exception as exc:
                error_response = getattr(exc, "response", None)
                status_code = (
                    getattr(exc, "code", None)
                    or getattr(exc, "status_code", None)
                    or getattr(error_response, "status_code", None)
                )
                if str(status_code) != "503" or attempt == 3:
                    raise
                time.sleep(5)

    try:
        return json.loads(response.text)
    except json.JSONDecodeError as exc:
        print(f"[AI] Gemini returned invalid JSON: {exc}")
        raise ValueError("Gemini response was not valid JSON.") from exc


if __name__ == "__main__":
    import data_engine
    import chart_engine

    full_context = data_engine.compile_market_context()
    spot_data = data_engine.fetch_spot_data()
    profile_levels = chart_engine.generate_pro_chart(
        spot_data,
        filename="prediction_chart.png",
    )
    full_context["volume_profile"] = profile_levels
    advisory = generate_trading_advisory(
        full_context,
        chart_image_path="prediction_chart.png",
    )
    print(json.dumps(advisory, indent=2))
