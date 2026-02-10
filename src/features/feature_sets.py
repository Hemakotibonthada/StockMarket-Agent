"""Feature sets for model training and strategy evaluation."""

from __future__ import annotations

import pandas as pd

from src.features.indicators import (
    atr,
    bollinger_bands,
    ema,
    log_returns,
    macd,
    on_balance_volume,
    rate_of_change,
    realized_volatility,
    returns,
    rolling_max,
    rolling_mean,
    rolling_min,
    rolling_std,
    rsi,
    vwap,
    zscore,
)


def compute_base_features(df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """Compute a standard set of features for a single symbol's OHLCV data.

    Args:
        df: DataFrame with columns: open, high, low, close, volume.
            Must be sorted by time.
        params: Optional dict to override default periods.

    Returns:
        DataFrame with added feature columns.
    """
    params = params or {}
    df = df.copy()

    # Returns
    df["ret_1"] = returns(df["close"], 1)
    df["ret_5"] = returns(df["close"], 5)
    df["log_ret_1"] = log_returns(df["close"], 1)

    # Trend
    df["ema_10"] = ema(df["close"], span=10)
    df["ema_20"] = ema(df["close"], span=20)
    df["ema_50"] = ema(df["close"], span=50)
    df["sma_20"] = rolling_mean(df["close"], window=20)

    # Momentum
    df["rsi_14"] = rsi(df["close"], period=params.get("rsi_period", 14))
    df["roc_10"] = rate_of_change(df["close"], period=10)
    macd_line, signal_line, macd_hist = macd(df["close"])
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_hist"] = macd_hist

    # Volatility
    df["atr_14"] = atr(df["high"], df["low"], df["close"], period=params.get("atr_period", 14))
    df["realized_vol_20"] = realized_volatility(df["close"], window=20)
    bb_mid, bb_upper, bb_lower = bollinger_bands(df["close"], window=20)
    df["bb_mid"] = bb_mid
    df["bb_upper"] = bb_upper
    df["bb_lower"] = bb_lower

    # Z-score
    df["zscore_20"] = zscore(df["close"], window=params.get("zscore_window", 20))

    # Volume features
    df["obv"] = on_balance_volume(df["close"], df["volume"])
    df["vol_sma_20"] = rolling_mean(df["volume"], window=20)
    df["vol_ratio"] = df["volume"] / df["vol_sma_20"]

    # Range features
    df["high_low_range"] = df["high"] - df["low"]
    df["close_open_range"] = df["close"] - df["open"]
    df["upper_shadow"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_shadow"] = df[["open", "close"]].min(axis=1) - df["low"]

    # Rolling stats
    df["rolling_max_20"] = rolling_max(df["high"], window=20)
    df["rolling_min_20"] = rolling_min(df["low"], window=20)
    df["pct_from_high_20"] = (df["close"] - df["rolling_max_20"]) / df["rolling_max_20"]
    df["pct_from_low_20"] = (df["close"] - df["rolling_min_20"]) / df["rolling_min_20"]

    return df


def compute_intraday_features(
    df: pd.DataFrame,
    params: dict | None = None,
) -> pd.DataFrame:
    """Compute intraday-specific features (assumes bars within a session).

    Args:
        df: DataFrame with columns: open, high, low, close, volume, datetime.
        params: Optional parameter overrides.

    Returns:
        DataFrame with additional intraday features.
    """
    params = params or {}
    df = df.copy()

    # VWAP
    df["vwap"] = vwap(df["high"], df["low"], df["close"], df["volume"])
    df["vwap_dist"] = (df["close"] - df["vwap"]) / df["vwap"]

    # Intraday range
    if "datetime" in df.columns:
        df["bar_number"] = range(len(df))
        df["pct_session_elapsed"] = df["bar_number"] / max(len(df) - 1, 1)

    # Cumulative volume
    df["cum_volume"] = df["volume"].cumsum()

    return df


def get_feature_columns(include_intraday: bool = False) -> list[str]:
    """Return the list of feature column names for model training."""
    base_features = [
        "ret_1", "ret_5", "log_ret_1",
        "ema_10", "ema_20", "ema_50", "sma_20",
        "rsi_14", "roc_10", "macd", "macd_signal", "macd_hist",
        "atr_14", "realized_vol_20", "bb_mid", "bb_upper", "bb_lower",
        "zscore_20",
        "obv", "vol_sma_20", "vol_ratio",
        "high_low_range", "close_open_range", "upper_shadow", "lower_shadow",
        "rolling_max_20", "rolling_min_20", "pct_from_high_20", "pct_from_low_20",
    ]

    if include_intraday:
        base_features.extend(["vwap", "vwap_dist", "bar_number", "pct_session_elapsed", "cum_volume"])

    return base_features
