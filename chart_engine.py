from typing import Any

import mplfinance as mpf
import numpy as np
import pandas as pd


def calculate_volume_profile(
    df: pd.DataFrame,
    bins: int = 50,
) -> dict[str, float]:
    required_columns = {"High", "Low", "Close", "Volume"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(
            f"Volume profile data is missing columns: {', '.join(sorted(missing_columns))}."
        )
    if not isinstance(bins, int) or isinstance(bins, bool) or bins < 1:
        raise ValueError("bins must be a positive integer.")

    profile_data = df[["High", "Low", "Close", "Volume"]].dropna()
    if profile_data.empty:
        raise ValueError("Cannot calculate a volume profile from empty OHLCV data.")

    typical_price = (
        profile_data["High"].to_numpy(dtype=float)
        + profile_data["Low"].to_numpy(dtype=float)
        + profile_data["Close"].to_numpy(dtype=float)
    ) / 3
    volume = profile_data["Volume"].to_numpy(dtype=float)
    if not np.isfinite(typical_price).all() or not np.isfinite(volume).all():
        raise ValueError("OHLCV data must contain only finite values.")
    if (volume < 0).any() or volume.sum() <= 0:
        raise ValueError("Volume data must be non-negative with a positive total.")

    volume_by_bin, bin_edges = np.histogram(
        typical_price,
        bins=bins,
        weights=volume,
    )
    if volume_by_bin.sum() <= 0:
        raise ValueError("Unable to allocate positive volume to price bins.")

    poc_index = int(np.argmax(volume_by_bin))
    value_area_target = volume_by_bin.sum() * 0.70
    value_area_volume = volume_by_bin[poc_index]
    low_index = high_index = poc_index

    while value_area_volume < value_area_target:
        next_low = low_index - 1
        next_high = high_index + 1
        low_volume = volume_by_bin[next_low] if next_low >= 0 else -1
        high_volume = volume_by_bin[next_high] if next_high < bins else -1

        if low_volume > high_volume:
            low_index = next_low
            value_area_volume += low_volume
        elif high_volume > low_volume:
            high_index = next_high
            value_area_volume += high_volume
        else:
            if next_low >= 0:
                low_index = next_low
                value_area_volume += low_volume
            if next_high < bins:
                high_index = next_high
                value_area_volume += high_volume

    return {
        "poc_price": float((bin_edges[poc_index] + bin_edges[poc_index + 1]) / 2),
        "value_area_high": float(bin_edges[high_index + 1]),
        "value_area_low": float(bin_edges[low_index]),
    }


def generate_pro_chart(
    df: pd.DataFrame,
    filename: str = "prediction_chart.png",
) -> dict[str, float]:
    levels = calculate_volume_profile(df)
    chart_data = df.dropna().copy()
    market_colors = mpf.make_marketcolors(
        up="#00b060",
        down="#ff3333",
        edge="inherit",
        wick="inherit",
        volume="inherit",
    )
    style = mpf.make_mpf_style(
        base_mpf_style="nightclouds",
        marketcolors=market_colors,
    )

    mpf.plot(
        chart_data,
        type="candle",
        style=style,
        volume=True,
        title="BTC-USD Candles with Volume Profile",
        ylabel="Price",
        ylabel_lower="Volume",
        hlines={
            "hlines": [
                levels["poc_price"],
                levels["value_area_high"],
                levels["value_area_low"],
            ],
            "colors": ["yellow", "cyan", "cyan"],
            "linestyle": "-.",
            "linewidths": 1.2,
        },
        tight_layout=True,
        savefig={"fname": filename, "dpi": 140, "bbox_inches": "tight"},
    )

    return {name: round(price, 2) for name, price in levels.items()}


if __name__ == "__main__":
    import data_engine

    market_data = data_engine.fetch_spot_data()
    profile_levels: dict[str, Any] = generate_pro_chart(market_data)
    print("[CHART] Volume Profile levels:")
    for level_name, price in profile_levels.items():
        print(f"  {level_name}: {price:.2f}")
