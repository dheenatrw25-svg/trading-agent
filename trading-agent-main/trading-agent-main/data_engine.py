import json
from typing import Any

import pandas as pd
import requests
import yfinance as yf


BINANCE_FUTURES_BASE_URL = "https://fapi.binance.com"
REQUEST_TIMEOUT_SECONDS = 10


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
    df = _flatten_columns(df).dropna()
    if df.empty:
        raise RuntimeError("yfinance returned no BTC-USD spot candles.")
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
    if len(df) < 2:
        raise RuntimeError(
            "yfinance must return at least two daily DXY closing prices "
            "to calculate the daily percentage change."
        )

    previous_close = float(df["Close"].iloc[-2])
    latest_close = float(df["Close"].iloc[-1])
    if previous_close == 0:
        raise RuntimeError("Cannot calculate DXY daily change from a zero prior close.")

    daily_change_pct = ((latest_close - previous_close) / previous_close) * 100
    print(
        f"[DATA] DXY close={latest_close:.4f}, "
        f"daily change={daily_change_pct:.4f}%."
    )
    return {
        "dxy_latest_close": latest_close,
        "dxy_daily_change_pct": daily_change_pct,
    }


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
    }


def compile_market_context() -> dict[str, Any]:
    print("[DATA] Compiling market context...")
    spot_data = fetch_spot_data()
    macro_context = fetch_macro_context()
    derivatives_data = fetch_derivatives_data()

    current_spot_close = float(spot_data["Close"].iloc[-1])
    context = {
        "spot": {
            "symbol": "BTC-USD",
            "current_close": current_spot_close,
        },
        "macro": macro_context,
        "derivatives": derivatives_data,
    }
    print("[DATA] Market context compiled.")
    return context


if __name__ == "__main__":
    print(json.dumps(compile_market_context(), indent=2))
