"""
Phase 1: Multi-Timeframe Human-Like Paper Trading Agent
-------------------------------------------------------
Top-down visual chart analysis (4H + 15m) combined with exact mathematical
indicators and a deterministic risk filter. Market data via yfinance.
Reasoning via Gemini 2.5 Flash (API only — no local GPU).
"""

from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import matplotlib

# Headless backend so chart generation works without a display.
matplotlib.use("Agg")

import mplfinance as mpf
import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
STATE_PATH = BASE_DIR / "agent_state.json"
TRADES_PATH = BASE_DIR / "paper_trades.csv"
DUAL_CHART_PATH = BASE_DIR / "dual_chart.png"

STARTING_BALANCE = 10_000.00
DAILY_DRAWDOWN_PCT = 0.03  # lock if daily realized PnL < -3% of starting balance
ANTI_TILT_LOSSES = 3
ANTI_TILT_HOURS = 24
GRADE_A_RISK_PCT = 0.015  # 1.5% of current balance
GRADE_B_RISK_PCT = 0.0075  # 0.75% of current balance
MIN_RR = 1.5

GEMINI_MODEL = "gemini-3.8-flash"

TRADE_CSV_FIELDS = [
    "Timestamp",
    "Ticker",
    "Action",
    "Grade",
    "Entry",
    "SL",
    "TP",
    "Units",
    "Capital at Risk",
    "R:R",
    "Thesis",
]

# Strict JSON schema forced on the model so the risk filter can stay deterministic.
TRADE_DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["BUY", "SELL", "WAIT"],
            "description": "Only BUY or SELL when a valid A/B setup exists; otherwise WAIT.",
        },
        "setup_grade": {
            "type": "string",
            "enum": ["A", "B", "C"],
            "description": "A = textbook confluence. B = tradable but imperfect. C = no-trade.",
        },
        "confidence_score": {
            "type": "integer",
            "minimum": 1,
            "maximum": 10,
        },
        "entry_price": {"type": "number"},
        "stop_loss": {"type": "number"},
        "take_profit": {"type": "number"},
        "thesis": {
            "type": "string",
            "description": "Concise top-down thesis (4H context, 15m trigger, invalidation).",
        },
    },
    "required": [
        "action",
        "setup_grade",
        "confidence_score",
        "entry_price",
        "stop_loss",
        "take_profit",
        "thesis",
    ],
}

SYSTEM_INSTRUCTION = """
You are a disciplined proprietary-firm trader executing Phase-1 paper trades.
You MUST use top-down analysis: the 4-hour chart (top image) defines the only
permitted trade direction; the 15-minute chart (bottom image) is the execution
trigger. Never fade the 4H 50 EMA macro trend provided in the metrics JSON.

Rules:
- BUY only if macro_trend is Bullish AND 15m structure supports a long (pullback
  to rising EMAs, higher-low, RSI not extremely overbought into resistance).
- SELL only if macro_trend is Bearish AND 15m structure supports a short.
- If 4H and 15m disagree, or the chart is choppy/range-bound with no trigger,
  action MUST be WAIT and setup_grade MUST be C.
- Grade A: clear 4H trend, clean 15m trigger, obvious invalidation beyond a
  recent swing, and room to target at least 1.5R.
- Grade B: trend aligned but trigger is slightly late or structure is imperfect.
- Grade C: no trade. Use WAIT.
- Stops must be on the correct side of entry (long: SL < entry < TP;
  short: TP < entry < SL) and should sit beyond the relevant 15m swing using ATR
  as a sanity check (do not place stops inside normal noise).
- Targets must imply R:R >= 1.5 versus the stop distance.
- Do not invent levels that contradict the supplied exact numbers. Use them.
- You are not allowed to justify revenge trading, averaging, or "catching a knife".
Respond ONLY with the required JSON object. No markdown, no extra keys.
""".strip()

# ---------------------------------------------------------------------------
# 1. Risk & portfolio state (anti-tilt)
# ---------------------------------------------------------------------------


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def default_state() -> dict[str, Any]:
    today = _now_utc().date().isoformat()
    return {
        "starting_balance": STARTING_BALANCE,
        "current_balance": STARTING_BALANCE,
        "daily_realized_pnl": 0.0,
        "daily_pnl_date": today,
        "consecutive_losses": 0,
        "circuit_breaker_locked_until": None,
        "anti_tilt_locked_until": None,
        "open_positions": [],
    }


def load_state() -> dict[str, Any]:
    """Load agent_state.json, creating it with a $10,000 book if missing."""
    if not STATE_PATH.exists():
        state = default_state()
        save_state(state)
        print(f"[STATE] Initialized {STATE_PATH.name} with ${STARTING_BALANCE:,.2f}")
        return state

    with STATE_PATH.open("r", encoding="utf-8") as fh:
        state = json.load(fh)

    # Ensure newer keys exist if an older file is reused.
    merged = default_state()
    merged.update(state)
    return merged


def save_state(state: dict[str, Any]) -> None:
    with STATE_PATH.open("w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2)


def ensure_trades_csv() -> None:
    """Create the paper-trade audit log with headers if it does not exist."""
    if TRADES_PATH.exists():
        return
    with TRADES_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=TRADE_CSV_FIELDS)
        writer.writeheader()
    print(f"[STATE] Created empty audit log {TRADES_PATH.name}")


def rollover_daily_pnl(state: dict[str, Any]) -> dict[str, Any]:
    """Reset daily PnL at the UTC calendar-day boundary."""
    today = _now_utc().date().isoformat()
    if state.get("daily_pnl_date") != today:
        print(
            f"[RISK] New UTC day ({today}). Resetting daily realized PnL "
            f"(was ${state.get('daily_realized_pnl', 0):,.2f})."
        )
        state["daily_pnl_date"] = today
        state["daily_realized_pnl"] = 0.0
        # Daily circuit breaker expires at the next calendar day.
        lock = _parse_iso(state.get("circuit_breaker_locked_until"))
        if lock and _now_utc() >= lock:
            state["circuit_breaker_locked_until"] = None
            print("[RISK] Daily drawdown circuit breaker released.")
    return state


def lock_until_next_calendar_day() -> str:
    tomorrow = (_now_utc() + timedelta(days=1)).date()
    lock = datetime.combine(tomorrow, datetime.min.time(), tzinfo=timezone.utc)
    return _iso(lock)


def is_trading_locked(state: dict[str, Any]) -> tuple[bool, str]:
    """Return (locked, reason). Checks daily DD breaker and anti-tilt lockout."""
    now = _now_utc()
    dd_lock = _parse_iso(state.get("circuit_breaker_locked_until"))
    if dd_lock and now < dd_lock:
        return True, f"Daily drawdown circuit breaker active until {dd_lock.isoformat()}"

    tilt_lock = _parse_iso(state.get("anti_tilt_locked_until"))
    if tilt_lock and now < tilt_lock:
        return True, f"Anti-tilt lockout active until {tilt_lock.isoformat()}"

    # If a lock timestamp is in the past, clear it.
    if dd_lock and now >= dd_lock:
        state["circuit_breaker_locked_until"] = None
    if tilt_lock and now >= tilt_lock:
        state["anti_tilt_locked_until"] = None
        print("[RISK] Anti-tilt lockout expired.")

    return False, ""


def apply_realized_pnl(state: dict[str, Any], pnl: float) -> dict[str, Any]:
    """Update balance, daily PnL, consecutive losses, and fire circuit breakers."""
    state["current_balance"] = round(float(state["current_balance"]) + pnl, 2)
    state["daily_realized_pnl"] = round(float(state["daily_realized_pnl"]) + pnl, 2)

    if pnl < 0:
        state["consecutive_losses"] = int(state.get("consecutive_losses", 0)) + 1
        print(f"[RISK] Loss booked. Consecutive losses = {state['consecutive_losses']}")
    elif pnl > 0:
        state["consecutive_losses"] = 0
        print("[RISK] Win booked. Consecutive-loss streak reset to 0.")

    dd_limit = -DAILY_DRAWDOWN_PCT * float(state["starting_balance"])
    if float(state["daily_realized_pnl"]) < dd_limit:
        state["circuit_breaker_locked_until"] = lock_until_next_calendar_day()
        print(
            f"[RISK] CIRCUIT BREAKER: daily realized PnL "
            f"${state['daily_realized_pnl']:,.2f} is below {dd_limit:,.2f} "
            f"(-{DAILY_DRAWDOWN_PCT:.0%} of starting balance). "
            f"Locked until {state['circuit_breaker_locked_until']}."
        )

    if int(state.get("consecutive_losses", 0)) >= ANTI_TILT_LOSSES:
        until = _now_utc() + timedelta(hours=ANTI_TILT_HOURS)
        state["anti_tilt_locked_until"] = _iso(until)
        print(
            f"[RISK] ANTI-TILT: {ANTI_TILT_LOSSES} consecutive losses. "
            f"Paused until {state['anti_tilt_locked_until']}."
        )

    return state


# ---------------------------------------------------------------------------
# Open-position mark / exit (so circuit breakers can actually fire)
# ---------------------------------------------------------------------------


def _flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [str(c[0]).title() if isinstance(c, tuple) else str(c) for c in df.columns]
    else:
        df = df.copy()
        df.columns = [str(c).title() for c in df.columns]
    return df


def fetch_ohlcv(ticker: str, interval: str, period: str) -> pd.DataFrame:
    print(f"[DATA] Fetching {ticker}  interval={interval}  period={period}")
    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if df is None or df.empty:
        raise RuntimeError(f"No yfinance data returned for {ticker} ({interval}/{period}).")
    df = _flatten_yf_columns(df)
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"OHLCV missing columns {missing} for {ticker}.")
    df = df[list(required)].dropna()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df


def build_4h_frame(hourly: pd.DataFrame) -> pd.DataFrame:
    """yfinance has no native 4h interval — resample 60m bars into 4H candles."""
    agg = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }
    df_4h = hourly.resample("4h").agg(agg).dropna()
    return df_4h


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def atr_wilder(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["Close"].shift(1)
    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def resolve_open_positions(state: dict[str, Any], ticker: str, df_15m: pd.DataFrame) -> dict[str, Any]:
    """
    Conservative paper fill: walk 15m bars after entry.
    If SL and TP print in the same candle, assume SL is hit first (adverse selection).
    """
    remaining: list[dict[str, Any]] = []
    for pos in state.get("open_positions", []):
        if pos.get("ticker") != ticker:
            remaining.append(pos)
            continue

        entry_ts = _parse_iso(pos["timestamp"])
        action = pos["action"]
        sl = float(pos["stop_loss"])
        tp = float(pos["take_profit"])
        entry = float(pos["entry_price"])
        units = float(pos["units"])

        subsequent = df_15m[df_15m.index > entry_ts]
        exit_price: Optional[float] = None
        outcome = ""

        for ts, bar in subsequent.iterrows():
            high, low = float(bar["High"]), float(bar["Low"])
            if action == "BUY":
                sl_hit = low <= sl
                tp_hit = high >= tp
                if sl_hit and tp_hit:
                    exit_price, outcome = sl, "SL (same-bar, conservative)"
                    break
                if sl_hit:
                    exit_price, outcome = sl, "SL"
                    break
                if tp_hit:
                    exit_price, outcome = tp, "TP"
                    break
            else:  # SELL
                sl_hit = high >= sl
                tp_hit = low <= tp
                if sl_hit and tp_hit:
                    exit_price, outcome = sl, "SL (same-bar, conservative)"
                    break
                if sl_hit:
                    exit_price, outcome = sl, "SL"
                    break
                if tp_hit:
                    exit_price, outcome = tp, "TP"
                    break

        if exit_price is None:
            remaining.append(pos)
            print(f"[POS] {ticker} {action} still open @ {entry:.4f} | SL {sl:.4f} TP {tp:.4f}")
            continue

        if action == "BUY":
            pnl = (exit_price - entry) * units
        else:
            pnl = (entry - exit_price) * units
        pnl = round(pnl, 2)

        print(
            f"[POS] CLOSED {ticker} {action} via {outcome} @ {exit_price:.4f} | "
            f"PnL ${pnl:,.2f}"
        )
        state = apply_realized_pnl(state, pnl)

    state["open_positions"] = remaining
    return state


# ---------------------------------------------------------------------------
# 2. Dual-timeframe chart generator (the eyes)
# ---------------------------------------------------------------------------


def _plot_candles(df: pd.DataFrame, title: str, outfile: Path) -> None:
    """Render a nightclouds candle chart with 20/50 EMA overlays and volume."""
    plot_df = df.copy()
    # mplfinance wants a naive or tz-naive-looking DatetimeIndex for clean labels.
    plot_df.index = plot_df.index.tz_convert("UTC").tz_localize(None)

    ema20 = ema(plot_df["Close"], 20)
    ema50 = ema(plot_df["Close"], 50)
    addplots = [
        mpf.make_addplot(ema20, color="#f5a623", width=1.15),
        mpf.make_addplot(ema50, color="#4fc3f7", width=1.15),
    ]
    mpf.plot(
        plot_df,
        type="candle",
        style="nightclouds",
        addplot=addplots,
        volume=True,
        title=title,
        ylabel="Price",
        ylabel_lower="Volume",
        figratio=(16, 8),
        figscale=1.15,
        tight_layout=True,
        savefig=dict(fname=str(outfile), dpi=140, bbox_inches="tight"),
    )


def generate_dual_chart(ticker: str, df_4h: pd.DataFrame, df_15m: pd.DataFrame) -> Path:
    """
    Top: last 50 4H candles (macro). Bottom: last 60 15m candles (trigger).
    Stacked with Pillow into dual_chart.png. Temporary files are deleted.
    """
    print("[CHART] Generating dual-timeframe visual stack...")
    last_4h = df_4h.tail(50)
    last_15m = df_15m.tail(60)
    if len(last_4h) < 50:
        print(f"[CHART] Warning: only {len(last_4h)} 4H candles available (want 50).")
    if len(last_15m) < 60:
        print(f"[CHART] Warning: only {len(last_15m)} 15m candles available (want 60).")

    tmp_dir = Path(tempfile.mkdtemp(prefix="ta_charts_"))
    top_path = tmp_dir / "4h.png"
    bot_path = tmp_dir / "15m.png"

    try:
        _plot_candles(last_4h, f"{ticker}  4-Hour  |  Macro trend (50 candles)", top_path)
        _plot_candles(last_15m, f"{ticker}  15-Minute  |  Execution trigger (60 candles)", bot_path)

        top_img = Image.open(top_path).convert("RGB")
        bot_img = Image.open(bot_path).convert("RGB")
        width = max(top_img.width, bot_img.width)

        def _pad(img: Image.Image, width: int) -> Image.Image:
            if img.width == width:
                return img
            canvas = Image.new("RGB", (width, img.height), (10, 10, 14))
            canvas.paste(img, ((width - img.width) // 2, 0))
            return canvas

        top_img = _pad(top_img, width)
        bot_img = _pad(bot_img, width)
        stacked = Image.new("RGB", (width, top_img.height + bot_img.height), (10, 10, 14))
        stacked.paste(top_img, (0, 0))
        stacked.paste(bot_img, (0, top_img.height))
        stacked.save(DUAL_CHART_PATH, format="PNG")
        print(f"[CHART] Wrote {DUAL_CHART_PATH.name} ({stacked.size[0]}x{stacked.size[1]})")
    finally:
        for p in (top_path, bot_path):
            if p.exists():
                p.unlink()
        try:
            tmp_dir.rmdir()
        except OSError:
            pass

    return DUAL_CHART_PATH


# ---------------------------------------------------------------------------
# 3. Quantitative metrics engine
# ---------------------------------------------------------------------------


def compute_metrics(df_4h: pd.DataFrame, df_15m: pd.DataFrame) -> dict[str, Any]:
    """Exact numbers passed to Gemini so the model cannot hallucinate indicators."""
    print("[QTY] Computing RSI / ATR / swings / 4H 50 EMA macro trend...")
    close_15 = df_15m["Close"]
    rsi = rsi_wilder(close_15, 14)
    atr = atr_wilder(df_15m, 14)

    swing_window = df_15m.tail(24)
    swing_high = float(swing_window["High"].max())
    swing_low = float(swing_window["Low"].min())

    ema50_4h_series = ema(df_4h["Close"], 50)
    ema50_4h = float(ema50_4h_series.iloc[-1])
    last_4h_close = float(df_4h["Close"].iloc[-1])
    last_15m_close = float(close_15.iloc[-1])
    macro = "Bullish" if last_4h_close > ema50_4h else "Bearish"

    rsi_val = float(rsi.iloc[-1])
    atr_val = float(atr.iloc[-1])
    if np.isnan(rsi_val) or np.isnan(atr_val) or np.isnan(ema50_4h):
        raise RuntimeError("Insufficient history to compute RSI/ATR/50 EMA. Try a more liquid ticker.")

    metrics = {
        "last_15m_close": round(last_15m_close, 6),
        "last_4h_close": round(last_4h_close, 6),
        "rsi_14_15m": round(rsi_val, 4),
        "atr_14_15m": round(atr_val, 6),
        "swing_high_24_15m": round(swing_high, 6),
        "swing_low_24_15m": round(swing_low, 6),
        "ema50_4h": round(ema50_4h, 6),
        "macro_trend": macro,
        "asof_15m_utc": df_15m.index[-1].isoformat(),
        "asof_4h_utc": df_4h.index[-1].isoformat(),
        "bars_15m": int(len(df_15m)),
        "bars_4h": int(len(df_4h)),
    }
    print(f"[QTY] Metrics: {json.dumps(metrics)}")
    return metrics


# ---------------------------------------------------------------------------
# 4. Multimodal AI reasoner (Gemini API)
# ---------------------------------------------------------------------------


def get_genai_client() -> genai.Client:
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing API key. Set GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment."
        )
    return genai.Client(api_key=api_key)


def ask_gemini(chart_path: Path, metrics: dict[str, Any], ticker: str) -> dict[str, Any]:
    print(f"[AI] Sending dual chart + metrics to {GEMINI_MODEL}...")
    client = get_genai_client()
    image_bytes = chart_path.read_bytes()
    user_text = (
        f"Ticker: {ticker}\n"
        "Analyze the stacked chart (TOP = 4H macro, BOTTOM = 15m trigger) "
        "together with these exact quantitative metrics. "
        "Return a single trade decision object.\n\n"
        f"{json.dumps(metrics, indent=2)}"
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                    types.Part.from_text(text=user_text),
                ],
            )
        ],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.2,
            response_mime_type="application/json",
            response_json_schema=TRADE_DECISION_SCHEMA,
        ),
    )

    raw = (response.text or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    decision = json.loads(raw)
    print(f"[AI] Decision: {json.dumps(decision)}")
    return decision


# ---------------------------------------------------------------------------
# 5. Deterministic risk filter & execution
# ---------------------------------------------------------------------------


def risk_reward(entry: float, sl: float, tp: float) -> float:
    risk = abs(entry - sl)
    reward = abs(tp - entry)
    if risk <= 0:
        return 0.0
    return reward / risk


def sl_orientation_valid(action: str, entry: float, sl: float, tp: float) -> bool:
    if action == "BUY":
        return sl < entry < tp
    if action == "SELL":
        return tp < entry < sl
    return False


def filter_and_execute(
    state: dict[str, Any],
    ticker: str,
    decision: dict[str, Any],
    last_price: float,
) -> dict[str, Any]:
    action = str(decision.get("action", "WAIT")).upper()
    grade = str(decision.get("setup_grade", "C")).upper()
    thesis = str(decision.get("thesis", "")).strip()

    print(f"[FILTER] action={action} grade={grade} confidence={decision.get('confidence_score')}")

    if action == "WAIT" or grade == "C":
        print("[FILTER] Abort: WAIT and/or Grade C — no trade.")
        return state

    if action not in {"BUY", "SELL"} or grade not in {"A", "B"}:
        print("[FILTER] Abort: unrecognized action/grade after schema parse.")
        return state

    # One open book per ticker keeps the Phase-1 simulator deterministic.
    if any(p.get("ticker") == ticker for p in state.get("open_positions", [])):
        print(f"[FILTER] Abort: already have an open position in {ticker}.")
        return state

    try:
        entry = float(decision["entry_price"])
        sl = float(decision["stop_loss"])
        tp = float(decision["take_profit"])
    except (KeyError, TypeError, ValueError):
        print("[FILTER] Abort: missing/non-numeric entry, SL, or TP.")
        return state

    if not sl_orientation_valid(action, entry, sl, tp):
        print(
            f"[FILTER] Abort: invalid SL/TP orientation for {action} "
            f"(entry={entry}, SL={sl}, TP={tp})."
        )
        return state

    rr = risk_reward(entry, sl, tp)
    if rr < MIN_RR:
        print(f"[FILTER] Abort: R:R {rr:.3f} is strictly below {MIN_RR}.")
        return state

    stop_dist = abs(entry - sl)
    if stop_dist <= 0:
        print("[FILTER] Abort: zero stop distance.")
        return state

    balance = float(state["current_balance"])
    risk_pct = GRADE_A_RISK_PCT if grade == "A" else GRADE_B_RISK_PCT
    capital_at_risk = round(balance * risk_pct, 2)
    units = capital_at_risk / stop_dist

    # Soft sanity: reject if the model entry is wildly off last print (stale/hallucinated).
    if last_price > 0 and abs(entry - last_price) / last_price > 0.03:
        print(
            f"[FILTER] Abort: entry {entry} is >3% away from last 15m close {last_price}."
        )
        return state

    ts = _iso(_now_utc())
    row = {
        "Timestamp": ts,
        "Ticker": ticker,
        "Action": action,
        "Grade": grade,
        "Entry": f"{entry:.6f}",
        "SL": f"{sl:.6f}",
        "TP": f"{tp:.6f}",
        "Units": f"{units:.8f}",
        "Capital at Risk": f"{capital_at_risk:.2f}",
        "R:R": f"{rr:.4f}",
        "Thesis": thesis,
    }

    with TRADES_PATH.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=TRADE_CSV_FIELDS)
        writer.writerow(row)

    position = {
        "timestamp": ts,
        "ticker": ticker,
        "action": action,
        "grade": grade,
        "entry_price": entry,
        "stop_loss": sl,
        "take_profit": tp,
        "units": units,
        "capital_at_risk": capital_at_risk,
        "rr": rr,
        "thesis": thesis,
    }
    state.setdefault("open_positions", []).append(position)
    save_state(state)

    print(
        f"[EXEC] PAPER {action} {ticker}  grade={grade}  entry={entry:.4f}  "
        f"SL={sl:.4f}  TP={tp:.4f}  units={units:.6f}  "
        f"risk=${capital_at_risk:.2f}  R:R={rr:.2f}"
    )
    print(f"[EXEC] Thesis: {thesis}")
    return state


# ---------------------------------------------------------------------------
# 6. Orchestration
# ---------------------------------------------------------------------------


def run_agent(ticker: str) -> dict[str, Any]:
    """Execute the full Phase-1 pipeline for a single ticker."""
    ticker = ticker.strip().upper()
    print("=" * 72)
    print(f"[RUN] Phase 1 paper agent  |  {ticker}  |  {_iso(_now_utc())}")
    print("=" * 72)

    ensure_trades_csv()
    state = rollover_daily_pnl(load_state())
    save_state(state)

    locked, reason = is_trading_locked(state)
    if locked:
        print(f"[RUN] HALTED — {reason}")
        save_state(state)
        return {"status": "locked", "reason": reason, "state": state}

    print(
        f"[RUN] Balance ${float(state['current_balance']):,.2f}  |  "
        f"Daily PnL ${float(state['daily_realized_pnl']):,.2f}  |  "
        f"Consecutive losses {state.get('consecutive_losses', 0)}"
    )

    # Need extra 15m/60m history so EMA50 / RSI / ATR are fully warmed up
    # before we slice the visible window.
    df_15m = fetch_ohlcv(ticker, interval="15m", period="10d")
    df_1h = fetch_ohlcv(ticker, interval="60m", period="90d")
    df_4h = build_4h_frame(df_1h)
    print(f"[DATA] 15m bars={len(df_15m)}  4H bars={len(df_4h)}")

    if len(df_15m) < 30 or len(df_4h) < 55:
        raise RuntimeError(
            f"Not enough history for {ticker} (15m={len(df_15m)}, 4H={len(df_4h)})."
        )

    state = resolve_open_positions(state, ticker, df_15m)
    save_state(state)

    locked, reason = is_trading_locked(state)
    if locked:
        print(f"[RUN] HALTED after position resolve — {reason}")
        save_state(state)
        return {"status": "locked", "reason": reason, "state": state}

    chart_path = generate_dual_chart(ticker, df_4h, df_15m)
    metrics = compute_metrics(df_4h, df_15m)
    decision = ask_gemini(chart_path, metrics, ticker)
    state = filter_and_execute(state, ticker, decision, last_price=metrics["last_15m_close"])
    save_state(state)

    print("[RUN] Pipeline complete.")
    return {
        "status": "ok",
        "ticker": ticker,
        "metrics": metrics,
        "decision": decision,
        "state": {
            "current_balance": state["current_balance"],
            "daily_realized_pnl": state["daily_realized_pnl"],
            "consecutive_losses": state["consecutive_losses"],
            "open_positions": state.get("open_positions", []),
        },
    }


def main(argv: list[str]) -> int:
    ticker = argv[1] if len(argv) > 1 else "AAPL"
    try:
        run_agent(ticker)
        return 0
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
