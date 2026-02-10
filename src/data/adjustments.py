"""Split and dividend adjustments for historical price data."""

from __future__ import annotations

import pandas as pd

from src.core.logging_utils import get_logger

logger = get_logger("data.adjustments")


def adjust_for_splits(
    df: pd.DataFrame,
    splits: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Apply stock split adjustments to OHLCV data.

    If no splits DataFrame is provided, data is returned unchanged.
    Splits DataFrame should have columns: symbol, date, ratio (e.g., 2.0 for 2:1 split).
    """
    if splits is None or splits.empty:
        return df

    df = df.copy()
    for _, split in splits.iterrows():
        symbol = split["symbol"]
        split_date = pd.to_datetime(split["date"])
        ratio = float(split["ratio"])

        mask = (df["symbol"] == symbol) & (df["date"] < split_date)
        if mask.any():
            for col in ["open", "high", "low", "close"]:
                if col in df.columns:
                    df.loc[mask, col] = df.loc[mask, col] / ratio
            if "volume" in df.columns:
                df.loc[mask, "volume"] = (df.loc[mask, "volume"] * ratio).astype(int)
            logger.info(f"Applied split {ratio}:1 for {symbol} on {split_date.date()}")

    return df


def adjust_for_dividends(
    df: pd.DataFrame,
    dividends: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Apply dividend adjustments to price data.

    Dividends DataFrame should have columns: symbol, date, amount.
    """
    if dividends is None or dividends.empty:
        return df

    df = df.copy()
    for _, div in dividends.iterrows():
        symbol = div["symbol"]
        ex_date = pd.to_datetime(div["date"])
        amount = float(div["amount"])

        mask = (df["symbol"] == symbol) & (df["date"] < ex_date)
        if mask.any():
            # Get the close price on the day before ex-date for adjustment factor
            close_before = df.loc[
                (df["symbol"] == symbol) & (df["date"] < ex_date), "close"
            ].iloc[-1] if mask.any() else None

            if close_before and close_before > 0:
                adj_factor = (close_before - amount) / close_before
                for col in ["open", "high", "low", "close"]:
                    if col in df.columns:
                        df.loc[mask, col] = df.loc[mask, col] * adj_factor
                logger.info(
                    f"Applied dividend ₹{amount} for {symbol} on {ex_date.date()} "
                    f"(factor: {adj_factor:.4f})"
                )

    return df


def apply_adjustments(
    df: pd.DataFrame,
    splits: pd.DataFrame | None = None,
    dividends: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Apply all adjustments (splits then dividends) to price data."""
    df = adjust_for_splits(df, splits)
    df = adjust_for_dividends(df, dividends)
    return df
