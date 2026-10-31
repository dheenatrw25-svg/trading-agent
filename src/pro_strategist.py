import json
import math
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

from . import chart_engine, data_engine
from .evaluator import retrieve_past_lessons
from .paths import ASSETS_DIR, PROJECT_ROOT

load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

PRO_TRADER_PROMPT = """You are an elite institutional quantitative cryptocurrency analyst. Your objective is to provide highly accurate, probabilistic market forecasts with strict risk management.
You have two inputs:
1. A visual chart displaying price action and OHLCV-based Volume Profile levels (Yellow = Point of Control, Cyan = Value Area High/Low).
2. A JSON payload using the current_price, technical_indicators, and institutional_order_flow schema.
3. Supporting market details containing completed-candle context, volume profile, macro data, and derivative measurements.
4. The four-hour trading playbook. Apply its rules, thresholds, evidence requirements, and risk/reward test.
YOUR CONFLUENCE RUBRIC:
- Treat the advisory as a directional forecast, not an instruction to buy or sell and not an automatic trading system.
- The direction MUST be exactly UP or DOWN. When evidence is unclear or contradictory, choose the best-supported direction, normally use confidence 4/10 or lower when no meaningful edge exists, and explicitly state that this is not an actionable setup.
- Use only completed four-hour candles; a wick alone does not confirm breakout, reclaim, or rejection.
- Apply the supplied macro, volume, funding, open-interest, setup, and reward/risk rules. Do not infer open-interest change from a single snapshot or fabricate unavailable data.
- Treat order-book walls as displayed liquidity in one timestamped snapshot, not proof of persistent support/resistance or spoofing.
- Treat taker-volume pressure as a short-term flow proxy, not a guarantee of continued price direction.
- Liquidity awareness: consider a nearby sell wall as potential overhead resistance for an UP forecast and a nearby buy wall as potential support for a DOWN forecast. Do not set a target beyond a nearby opposing wall unless supplied price-action evidence shows it was absorbed or cleared. If a target is placed before a wall, use a level grounded in supplied market data; never invent a fixed offset from the wall. A wall is not a guaranteed barrier, and it must not override completed-candle confirmation or the playbook's risk/reward rules.
- Trap avoidance: compare taker-pressure status with completed-candle price action. If price rises while taker pressure is aggressively selling, or price falls while taker pressure is aggressively buying, explicitly describe the disagreement as conflicting evidence; do not claim that anyone is lying or that a trap is confirmed. Do not reverse direction based on this divergence alone. Seek confirmation from candle closes, volume, and validated open-interest context; if evidence remains contradictory with no clear edge, use confidence 4/10 or lower and state that the forecast is not actionable.
- Every target, invalidation, swing, and liquidity level must come from the supplied volume profile or actual candle data. Do not invent price levels.
OUTPUT RESTRICTION: Output YOUR ENTIRE RESPONSE as a single, valid JSON object. No markdown formatting.
The forecast horizon is the next 4 hours.
SCHEMA: {"direction": "UP"|"DOWN", "target_range_low": float, "target_range_high": float, "invalidation_price": float, "confidence_score": int, "market_regime": "string", "reasoning": "string"}"""

ADVISORY_FIELDS = {
    "direction",
    "target_range_low",
    "target_range_high",
    "invalidation_price",
    "confidence_score",
    "market_regime",
    "reasoning",
}
ADVISORY_DIRECTIONS = {"UP", "DOWN"}


def load_playbook() -> str:
    playbook_path = PROJECT_ROOT / "trading_playbook.md"
    try:
        playbook = playbook_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Trading playbook is required but was not found: {playbook_path}"
        ) from exc
    except OSError as exc:
        raise OSError(f"Could not read trading playbook at {playbook_path}: {exc}") from exc
    if not playbook.strip():
        raise ValueError(f"Trading playbook is empty: {playbook_path}")
    return playbook


def calculate_risk_reward(
    current_price: object,
    direction: str,
    target_range_low: object,
    target_range_high: object,
    invalidation_price: object,
) -> dict[str, Any]:
    values = {
        "current_price": current_price,
        "target_range_low": target_range_low,
        "target_range_high": target_range_high,
        "invalidation_price": invalidation_price,
    }
    parsed: dict[str, float] = {}
    for name, value in values.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            return {
                "actionable": False,
                "risk": None,
                "reward": None,
                "risk_reward_ratio": None,
                "reason": f"{name} is unavailable or invalid.",
            }
        parsed[name] = float(value)
    if direction not in ADVISORY_DIRECTIONS:
        raise ValueError("Risk/reward direction must be UP or DOWN.")
    if parsed["target_range_low"] > parsed["target_range_high"]:
        return {
            "actionable": False,
            "risk": None,
            "reward": None,
            "risk_reward_ratio": None,
            "reason": "Target range is inverted.",
        }

    entry = parsed["current_price"]
    target_midpoint = (
        parsed["target_range_low"] + parsed["target_range_high"]
    ) / 2
    if direction == "UP":
        risk = entry - parsed["invalidation_price"]
        reward = target_midpoint - entry
    else:
        risk = parsed["invalidation_price"] - entry
        reward = entry - target_midpoint

    if risk <= 0 or reward <= 0:
        return {
            "actionable": False,
            "risk": risk,
            "reward": reward,
            "risk_reward_ratio": None,
            "reason": "Target midpoint and invalidation must be on the correct sides of current price.",
        }
    ratio = reward / risk
    return {
        "actionable": ratio >= 1.5,
        "risk": risk,
        "reward": reward,
        "risk_reward_ratio": ratio,
        "reason": (
            "Reward/risk meets the 1.5 minimum."
            if ratio >= 1.5
            else "Reward/risk is below the 1.5 minimum."
        ),
    }


def _apply_python_strategy_checks(
    advisory: dict[str, Any],
    market_context: dict[str, Any],
) -> dict[str, Any]:
    result = dict(advisory)
    direction = result["direction"]
    confidence = int(result["confidence_score"])
    macro = market_context.get("macro", {})
    derivatives = market_context.get("derivatives", {})
    current_price = market_context.get("spot", {}).get("current_close")
    limitations: list[str] = []

    try:
        dxy_change = float(macro["dxy_daily_change_pct"])
    except (KeyError, TypeError, ValueError):
        dxy_change = math.nan
    if math.isfinite(dxy_change) and dxy_change >= 0.40 and direction == "UP":
        confidence = min(confidence, 5)
        limitations.append(
            "Strong-dollar headwind (DXY daily change at or above +0.40%) caps UP confidence at 5/10."
        )
    elif not math.isfinite(dxy_change):
        limitations.append(
            "DXY daily change is unavailable, so the macro filter could not be evaluated."
        )

    try:
        annualized_funding = float(derivatives["annualized_funding_rate_pct"])
    except (KeyError, TypeError, ValueError):
        annualized_funding = math.nan
    if (
        math.isfinite(annualized_funding)
        and annualized_funding > 25
        and direction == "UP"
    ):
        confidence = max(1, confidence - 2)
        limitations.append(
            "Crowded-long warning: annualized funding exceeds 25%; UP confidence reduced by at least two points."
        )
    elif not math.isfinite(annualized_funding):
        limitations.append(
            "Binance annualized funding is unavailable, so positioning could not be fully evaluated."
        )

    price_action = market_context.get("price_action")
    if not isinstance(price_action, dict):
        confidence = min(confidence, 4)
        limitations.append(
            "Completed four-hour candle and volume context is unavailable; this is not an actionable setup."
        )
    else:
        volume_comparison = price_action.get("volume_comparison", {})
        if (
            not isinstance(volume_comparison, dict)
            or volume_comparison.get("volume_state") == "unavailable"
        ):
            confidence = min(confidence, 4)
            limitations.append(
                "A reliable recent-volume comparison is unavailable; do not treat volume expansion or flatness as confirmed. This is not an actionable setup."
            )

    if derivatives.get("open_interest_trend") not in {"rising", "falling", "stagnant"}:
        limitations.append(
            "Historical open-interest trend is unavailable; the current snapshot is not treated as evidence of a trend."
        )

    if not isinstance(current_price, (int, float)) or isinstance(current_price, bool):
        risk_reward = calculate_risk_reward(
            None,
            direction,
            result["target_range_low"],
            result["target_range_high"],
            result["invalidation_price"],
        )
    else:
        risk_reward = calculate_risk_reward(
            current_price,
            direction,
            result["target_range_low"],
            result["target_range_high"],
            result["invalidation_price"],
        )
    if not risk_reward["actionable"]:
        confidence = min(confidence, 4)
        ratio = risk_reward["risk_reward_ratio"]
        rr_text = (
            f"Calculated reward/risk is {ratio:.2f}:1, below the required 1.5:1."
            if isinstance(ratio, (int, float))
            else f"Risk/reward could not be established: {risk_reward['reason']}"
        )
        limitations.append(f"{rr_text} This is not an actionable setup.")
    result["confidence_score"] = confidence
    if limitations:
        result["reasoning"] = (
            result["reasoning"].rstrip()
            + " "
            + " ".join(limitations)
        )
    return validate_advisory(result)


def validate_advisory(advisory: object) -> dict[str, Any]:
    if not isinstance(advisory, dict):
        raise ValueError("Gemini advisory must be a JSON object.")
    missing = ADVISORY_FIELDS.difference(advisory)
    if missing:
        raise ValueError(
            "Gemini advisory is missing required fields: "
            + ", ".join(sorted(missing))
        )
    if (
        not isinstance(advisory["direction"], str)
        or advisory["direction"] not in ADVISORY_DIRECTIONS
    ):
        raise ValueError("Gemini advisory has an unsupported direction.")

    confidence = advisory["confidence_score"]
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(confidence)
        or not 1 <= confidence <= 10
    ):
        raise ValueError("Gemini confidence_score must be a finite number from 1 to 10.")

    prices = {}
    for field in ("target_range_low", "target_range_high", "invalidation_price"):
        value = advisory[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"Gemini advisory field {field} must be a positive number.")
        prices[field] = float(value)
    if prices["target_range_low"] > prices["target_range_high"]:
        raise ValueError("Gemini advisory target low exceeds its target high.")
    for field in ("market_regime", "reasoning"):
        if not isinstance(advisory[field], str) or not advisory[field].strip():
            raise ValueError(f"Gemini advisory field {field} must be a non-empty string.")
    return advisory


def generate_trading_advisory(
    market_context: dict[str, Any],
    chart_image_path: str = str(ASSETS_DIR / "prediction_chart.png"),
) -> dict[str, Any]:
    image_path = Path(chart_image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Chart image not found: {image_path}")

    data_string = json.dumps(
        data_engine.build_ai_prompt_payload(market_context),
        indent=2,
    )
    supporting_context = {
        key: market_context[key]
        for key in (
            "price_action",
            "macro",
            "derivatives",
            "order_book",
            "volume_profile",
        )
        if key in market_context
    }
    supporting_data_string = json.dumps(supporting_context, indent=2)
    playbook = load_playbook()
    regime_context = market_context.get("market_regime")
    if not isinstance(regime_context, str) or not regime_context.strip():
        regime_context = json.dumps(
            {
                "macro": market_context.get("macro", {}),
                "derivatives": market_context.get("derivatives", {}),
                "price_action": market_context.get("price_action", {}),
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
                        f"Four-hour trading playbook:\n{playbook}",
                        data_string,
                        f"Supporting market details:\n{supporting_data_string}",
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
        advisory = validate_advisory(json.loads(response.text))
        return _apply_python_strategy_checks(advisory, market_context)
    except json.JSONDecodeError as exc:
        print(f"[AI] Gemini returned invalid JSON: {exc}")
        raise ValueError("Gemini response was not valid JSON.") from exc


if __name__ == "__main__":
    full_context = data_engine.compile_market_context()
    spot_data = data_engine.fetch_spot_data()
    profile_levels = chart_engine.generate_pro_chart(
        spot_data,
        filename=str(ASSETS_DIR / "prediction_chart.png"),
    )
    full_context["volume_profile"] = profile_levels
    advisory = generate_trading_advisory(
        full_context,
        chart_image_path=str(ASSETS_DIR / "prediction_chart.png"),
    )
    print(json.dumps(advisory, indent=2))
