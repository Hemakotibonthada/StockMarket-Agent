"""Bar resampling utilities for tick/minute aggregation."""

from __future__ import annotations

import pandas as pd

from src.core.logging_utils import get_logger

logger = get_logger("data.resample")


def resample_bars(
    df: pd.DataFrame,
    interval: str = "5min",
    datetime_col: str = "datetime",
) -> pd.DataFrame:
    """Resample OHLCV data to a coarser bar interval.

    Args:
        df: DataFrame with OHLCV + datetime column.
        interval: Target interval (e.g., '1min', '5min', '15min', '1h', '1D').
        datetime_col: Name of the datetime column.

    Returns:
        Resampled OHLCV DataFrame.
    """
    df = df.copy()

    if datetime_col in df.columns:
        df = df.set_index(datetime_col)
    elif not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"DataFrame must have a datetime index or '{datetime_col}' column")

    # Group by symbol if present
    if "symbol" in df.columns:
        resampled = df.groupby("symbol").resample(interval).agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        resampled = resampled.dropna(subset=["open"]).reset_index()
    else:
        resampled = df.resample(interval).agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        resampled = resampled.dropna(subset=["open"]).reset_index()

    logger.info(f"Resampled to {interval}: {len(df)} -> {len(resampled)} bars")
    return resampled


def ticks_to_bars(
    ticks: pd.DataFrame,
    interval: str = "1min",
    price_col: str = "price",
    volume_col: str = "volume",
    datetime_col: str = "datetime",
) -> pd.DataFrame:
    """Convert tick data to OHLCV bars.

    Args:
        ticks: DataFrame with price, volume, datetime columns.
        interval: Bar interval.
        price_col: Column name for price.
        volume_col: Column name for volume.
        datetime_col: Column name for datetime.

    Returns:
        OHLCV DataFrame.
    """
    ticks = ticks.copy()
    if datetime_col in ticks.columns:
        ticks = ticks.set_index(datetime_col)

    bars = ticks.resample(interval).agg(
        {
            price_col: ["first", "max", "min", "last"],
            volume_col: "sum",
        }
    )
    bars.columns = ["open", "high", "low", "close", "volume"]
    bars = bars.dropna(subset=["open"]).reset_index()

    return bars
