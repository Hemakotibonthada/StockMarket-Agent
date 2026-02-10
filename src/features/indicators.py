"""Technical indicators: ATR, RSI, z-score, VWAP, rolling stats, volatility."""

from __future__ import annotations

import numpy as np
import pandas as pd


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Average True Range (ATR)."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.rolling(window=period, min_periods=1).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (RSI)."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def zscore(series: pd.Series, window: int = 20) -> pd.Series:
    """Rolling z-score."""
    mean = series.rolling(window=window, min_periods=1).mean()
    std = series.rolling(window=window, min_periods=1).std()
    return (series - mean) / std.replace(0, np.nan)


def vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """Volume Weighted Average Price (intraday, cumulative within the series)."""
    typical_price = (high + low + close) / 3
    cum_tp_vol = (typical_price * volume).cumsum()
    cum_vol = volume.cumsum()
    return cum_tp_vol / cum_vol.replace(0, np.nan)


def rolling_mean(series: pd.Series, window: int = 20) -> pd.Series:
    """Simple rolling mean."""
    return series.rolling(window=window, min_periods=1).mean()


def rolling_std(series: pd.Series, window: int = 20) -> pd.Series:
    """Rolling standard deviation."""
    return series.rolling(window=window, min_periods=1).std()


def rolling_max(series: pd.Series, window: int = 20) -> pd.Series:
    """Rolling max."""
    return series.rolling(window=window, min_periods=1).max()


def rolling_min(series: pd.Series, window: int = 20) -> pd.Series:
    """Rolling min."""
    return series.rolling(window=window, min_periods=1).min()


def bollinger_bands(
    close: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands (middle, upper, lower)."""
    middle = rolling_mean(close, window)
    std = rolling_std(close, window)
    upper = middle + num_std * std
    lower = middle - num_std * std
    return middle, upper, lower


def ema(series: pd.Series, span: int = 20) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=span, min_periods=1, adjust=False).mean()


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, signal line, histogram."""
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def realized_volatility(close: pd.Series, window: int = 20) -> pd.Series:
    """Annualized realized volatility from log returns."""
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window=window, min_periods=1).std() * np.sqrt(252)


def returns(close: pd.Series, periods: int = 1) -> pd.Series:
    """Simple returns."""
    return close.pct_change(periods=periods)


def log_returns(close: pd.Series, periods: int = 1) -> pd.Series:
    """Log returns."""
    return np.log(close / close.shift(periods))


def rate_of_change(close: pd.Series, period: int = 10) -> pd.Series:
    """Rate of change (ROC) as percentage."""
    return (close / close.shift(period) - 1) * 100


def on_balance_volume(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume (OBV)."""
    direction = np.sign(close.diff())
    return (direction * volume).cumsum()
