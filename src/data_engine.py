import json
import math
from typing import Any

import numpy as np
import pandas as pd
import requests
import yfinance as yf


BINANCE_FUTURES_BASE_URL = "https://fapi.binance.com"
REQUEST_TIMEOUT_SECONDS = 10


def _confirmed_candles(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    if df.empty:
        raise RuntimeError("yfinance returned no BTC-USD spot candles.")
    required_columns = {"Open", "High", "Low", "Close", "Volume"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(
            "BTC-USD candle data is missing columns: "
            + ", ".join(sorted(missing_columns))
            + "."
        )

    interval_delta = pd.Timedelta(interval)
    timestamps = pd.to_datetime(df.index, utc=True, errors="coerce")
    now_utc = pd.Timestamp.now(tz="UTC")
    confirmed = timestamps.notna() & (timestamps + interval_delta <= now_utc)
    result = df.loc[confirmed].copy()
    result.index = timestamps[confirmed]
    result = result.loc[~result.index.duplicated(keep="last")].sort_index()
    result = result.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    try:
        result[["Open", "High", "Low", "Close", "Volume"]] = result[
            ["Open", "High", "Low", "Close", "Volume"]
        ].apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("BTC-USD OHLCV candle values must be numeric.") from exc
    ohlcv = result[["Open", "High", "Low", "Close", "Volume"]].to_numpy(dtype=float)
    valid_rows = (
        np.isfinite(ohlcv).all(axis=1)
        & (ohlcv[:, :4] > 0).all(axis=1)
        & (ohlcv[:, 1] >= ohlcv[:, 0])
        & (ohlcv[:, 1] >= ohlcv[:, 2])
        & (ohlcv[:, 1] >= ohlcv[:, 3])
        & (ohlcv[:, 2] <= ohlcv[:, 0])
        & (ohlcv[:, 2] <= ohlcv[:, 3])
        & (ohlcv[:, 4] >= 0)
    )
    result = result.loc[valid_rows]
    if result.empty:
        raise RuntimeError("No complete BTC-USD spot candles are available.")

    if len(result) < 2:
        raise RuntimeError("At least two complete BTC-USD spot candles are required.")
    return result


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def fetch_spot_data() -> pd.DataFrame:
    print("[DATA] Fetching BTC-USD spot candles (4h, 15d)...")
    df = yf.download(
        "BTC-USD",
        period="15d",
        interval="4h",
        progress=False,
    )
    df = _flatten_columns(df)
    df = _confirmed_candles(df, "4h")
    print(f"[DATA] Retrieved {len(df)} spot candles.")
    return df


def fetch_macro_context() -> dict[str, float]:
    print("[DATA] Fetching US Dollar Index (DX-Y.NYB)...")
    df = yf.download(
        "DX-Y.NYB",
        period="5d",
        interval="1d",
        progress=False,
    )
    df = _flatten_columns(df).dropna(subset=["Close"])
    timestamps = pd.to_datetime(df.index, utc=True, errors="coerce")
    now_utc = pd.Timestamp.now(tz="UTC")
    confirmed = timestamps.notna() & (
        timestamps + pd.Timedelta(days=1) <= now_utc
    )
    df = df.loc[confirmed].copy()
    df.index = timestamps[confirmed]
    df = df.loc[~df.index.duplicated(keep="last")].sort_index()
    if len(df) < 2:
        raise RuntimeError(
            "yfinance must return at least two completed daily DXY closing prices "
            "to calculate the daily percentage change."
        )

    previous_close = float(df["Close"].iloc[-2])
    latest_close = float(df["Close"].iloc[-1])
    if (
        not math.isfinite(previous_close)
        or not math.isfinite(latest_close)
        or previous_close <= 0
        or latest_close <= 0
    ):
        raise RuntimeError(
            "Cannot calculate DXY daily change from invalid completed closing prices."
        )

    daily_change_pct = ((latest_close - previous_close) / previous_close) * 100
    print(
        f"[DATA] DXY close={latest_close:.4f}, "
        f"daily change={daily_change_pct:.4f}%."
    )
    return {
        "dxy_latest_close": latest_close,
        "dxy_daily_change_pct": daily_change_pct,
    }


def _fetch_open_interest_trend(symbol: str) -> dict[str, Any]:
    unavailable = {
        "open_interest_change_pct": None,
        "open_interest_change_interval_hours": None,
        "open_interest_trend": "unavailable",
        "open_interest_trend_reason": "",
    }
    try:
        response = requests.get(
            f"{BINANCE_FUTURES_BASE_URL}/futures/data/openInterestHist",
            params={"symbol": symbol, "period": "4h", "limit": 2},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        history = response.json()
    except (requests.RequestException, ValueError) as exc:
        unavailable["open_interest_trend_reason"] = (
            f"Binance historical open-interest data unavailable: {exc}"
        )
        print(f"[DATA] {unavailable['open_interest_trend_reason']}")
        return unavailable

    if not isinstance(history, list) or len(history) < 2:
        unavailable["open_interest_trend_reason"] = (
            "Binance returned fewer than two historical open-interest observations."
        )
        print(f"[DATA] {unavailable['open_interest_trend_reason']}")
        return unavailable
    try:
        observations = sorted(
            (
                int(record["timestamp"]),
                float(record["sumOpenInterest"]),
            )
            for record in history
        )
    except (KeyError, TypeError, ValueError) as exc:
        unavailable["open_interest_trend_reason"] = (
            f"Binance historical open-interest data had an invalid shape: {exc}"
        )
        print(f"[DATA] {unavailable['open_interest_trend_reason']}")
        return unavailable

    previous_timestamp, previous_value = observations[-2]
    latest_timestamp, latest_value = observations[-1]
    interval_hours = (latest_timestamp - previous_timestamp) / 3_600_000
    now_milliseconds = pd.Timestamp.now(tz="UTC").timestamp() * 1000
    if (
        not math.isfinite(previous_value)
        or not math.isfinite(latest_value)
        or previous_value <= 0
        or latest_value <= 0
        or not 1 <= interval_hours <= 8
        or now_milliseconds - latest_timestamp > 8 * 3_600_000
    ):
        unavailable["open_interest_trend_reason"] = (
            "Historical open-interest observations were invalid, stale, or not "
            "separated by a reliable interval."
        )
        print(f"[DATA] {unavailable['open_interest_trend_reason']}")
        return unavailable

    change_pct = ((latest_value - previous_value) / previous_value) * 100
    if change_pct > 1:
        trend = "rising"
    elif change_pct < -1:
        trend = "falling"
    else:
        trend = "stagnant"
    result = {
        "open_interest_change_pct": change_pct,
        "open_interest_change_interval_hours": interval_hours,
        "open_interest_trend": trend,
        "open_interest_trend_reason": (
            "Compared the two latest valid Binance 4h-period observations; "
            "changes within +/-1% are operationally classified as stagnant."
        ),
    }
    print(
        f"[DATA] Historical open interest changed {change_pct:.4f}% "
        f"over {interval_hours:.2f} hours ({trend})."
    )
    return result


def fetch_derivatives_data(symbol: str = "BTCUSDT") -> dict[str, Any]:
    print(f"[DATA] Fetching Binance futures data for {symbol}...")
    endpoints = {
        "open_interest": f"{BINANCE_FUTURES_BASE_URL}/fapi/v1/openInterest",
        "premium_index": f"{BINANCE_FUTURES_BASE_URL}/fapi/v1/premiumIndex",
    }
    results: dict[str, dict[str, Any]] = {}

    for name, url in endpoints.items():
        try:
            response = requests.get(
                url,
                params={"symbol": symbol},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            results[name] = response.json()
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Binance {name.replace('_', ' ')} request failed for {symbol}: {exc}"
            ) from exc
        except ValueError as exc:
            raise RuntimeError(
                f"Binance returned invalid JSON for {name.replace('_', ' ')} "
                f"({symbol})."
            ) from exc

    try:
        open_interest = float(results["open_interest"]["openInterest"])
        funding_rate = float(results["premium_index"]["lastFundingRate"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Binance response for {symbol} is missing valid open interest or funding data."
        ) from exc
    if (
        not math.isfinite(open_interest)
        or open_interest <= 0
        or not math.isfinite(funding_rate)
    ):
        raise RuntimeError(
            f"Binance response for {symbol} contains invalid open interest or funding data."
        )

    annualized_funding_rate_pct = funding_rate * 3 * 365 * 100
    print(
        f"[DATA] Open interest={open_interest}, "
        f"annualized funding={annualized_funding_rate_pct:.4f}%."
    )
    return {
        "symbol": symbol,
        "open_interest": open_interest,
        "funding_rate": funding_rate,
        "annualized_funding_rate_pct": annualized_funding_rate_pct,
        **_fetch_open_interest_trend(symbol),
    }


def _build_price_action_context(spot_data: pd.DataFrame) -> dict[str, Any]:
    confirmed = _confirmed_candles(spot_data, "4h")
    latest = confirmed.iloc[-1]
    latest_timestamp = pd.Timestamp(confirmed.index[-1])
    latest_volume = float(latest["Volume"])
    baseline = confirmed["Volume"].iloc[-21:-1]
    if len(baseline) < 5:
        volume_context: dict[str, Any] = {
            "latest_volume": latest_volume,
            "baseline_median_volume": None,
            "latest_to_baseline_volume_ratio": None,
            "volume_state": "unavailable",
            "volume_state_note": (
                "At least five prior completed candles are required for a volume comparison."
            ),
        }
    else:
        baseline_median = float(baseline.median())
        if not math.isfinite(baseline_median) or baseline_median <= 0:
            ratio = None
            state = "unavailable"
            note = "Prior completed-candle median volume is not positive."
        else:
            ratio = latest_volume / baseline_median
            if ratio >= 1.5:
                state = "expanding"
            elif ratio < 0.8:
                state = "declining"
            elif 0.8 <= ratio <= 1.2:
                state = "flat"
            else:
                state = "moderately_above_baseline"
            note = (
                "Operational thresholds versus the median of up to 20 prior "
                "completed candles: expanding >=1.5x; flat 0.8x-1.2x; "
                "declining <0.8x. These are explicit heuristics, not validated "
                "trading thresholds."
            )
        volume_context = {
            "latest_volume": latest_volume,
            "baseline_median_volume": baseline_median,
            "latest_to_baseline_volume_ratio": ratio,
            "volume_state": state,
            "volume_state_note": note,
        }

    swing_window = confirmed.tail(20)
    swing_high_index = swing_window["High"].idxmax()
    swing_low_index = swing_window["Low"].idxmin()
    latest_candle = {
        "timestamp": latest_timestamp.isoformat(),
        "close_timestamp": (latest_timestamp + pd.Timedelta(hours=4)).isoformat(),
        "open": float(latest["Open"]),
        "high": float(latest["High"]),
        "low": float(latest["Low"]),
        "close": float(latest["Close"]),
        "volume": latest_volume,
    }
    recent_completed_candles = []
    for timestamp, candle in confirmed.tail(20).iterrows():
        candle_timestamp = pd.Timestamp(timestamp)
        recent_completed_candles.append(
            {
                "timestamp": candle_timestamp.isoformat(),
                "close_timestamp": (
                    candle_timestamp + pd.Timedelta(hours=4)
                ).isoformat(),
                "open": float(candle["Open"]),
                "high": float(candle["High"]),
                "low": float(candle["Low"]),
                "close": float(candle["Close"]),
                "volume": float(candle["Volume"]),
            }
        )
    return {
        "candle_interval": "4h",
        "latest_completed_candle": latest_candle,
        "recent_completed_candles": recent_completed_candles,
        "volume_comparison": volume_context,
        "recent_swing_high": {
            "price": float(swing_window.loc[swing_high_index, "High"]),
            "timestamp": pd.Timestamp(swing_high_index).isoformat(),
            "definition": "Highest actual candle high in the latest 20 completed candles.",
        },
        "recent_swing_low": {
            "price": float(swing_window.loc[swing_low_index, "Low"]),
            "timestamp": pd.Timestamp(swing_low_index).isoformat(),
            "definition": "Lowest actual candle low in the latest 20 completed candles.",
        },
    }


def compile_market_context(spot_data: pd.DataFrame | None = None) -> dict[str, Any]:
    print("[DATA] Compiling market context...")
    if spot_data is None:
        spot_data = fetch_spot_data()
    if spot_data.empty:
        raise RuntimeError("Cannot compile market context from empty BTC-USD candles.")
    price_action = _build_price_action_context(spot_data)
    confirmed_spot_data = _confirmed_candles(spot_data, "4h")
    macro_context = fetch_macro_context()
    derivatives_data = fetch_derivatives_data()

    current_spot_close = float(confirmed_spot_data["Close"].iloc[-1])
    context = {
        "spot": {
            "symbol": "BTC-USD",
            "current_close": current_spot_close,
        },
        "price_action": price_action,
        "macro": macro_context,
        "derivatives": derivatives_data,
    }
    print("[DATA] Market context compiled.")
    return context


if __name__ == "__main__":
    print(json.dumps(compile_market_context(), indent=2))
