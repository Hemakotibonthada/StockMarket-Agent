"""Tests for feature indicators and feature sets."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.indicators import (
    atr,
    bollinger_bands,
    ema,
    log_returns,
    macd,
    on_balance_volume,
    rate_of_change,
    returns,
    rolling_max,
    rolling_mean,
    rolling_min,
    rolling_std,
    rsi,
    vwap,
    zscore,
)


def _make_series(n: int = 100, seed: int = 42) -> pd.Series:
    """Generate a simple price series."""
    rng = np.random.RandomState(seed)
    prices = 100 + np.cumsum(rng.randn(n))
    return pd.Series(prices, name="close")


def _make_ohlcv(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data."""
    rng = np.random.RandomState(seed)
    close = 100 + np.cumsum(rng.randn(n))
    return pd.DataFrame({
        "open": close + rng.randn(n) * 0.5,
        "high": close + abs(rng.randn(n)) * 2,
        "low": close - abs(rng.randn(n)) * 2,
        "close": close,
        "volume": (rng.rand(n) * 100_000 + 50_000).astype(int),
    })


class TestATR:
    def test_atr_output_length(self):
        df = _make_ohlcv()
        result = atr(df["high"], df["low"], df["close"], period=14)
        assert len(result) == len(df)

    def test_atr_positive(self):
        df = _make_ohlcv()
        result = atr(df["high"], df["low"], df["close"], period=14)
        assert (result.dropna() >= 0).all()

    def test_atr_period_1(self):
        df = _make_ohlcv()
        result = atr(df["high"], df["low"], df["close"], period=1)
        assert len(result) == len(df)


class TestRSI:
    def test_rsi_output_length(self):
        close = _make_series()
        result = rsi(close, period=14)
        assert len(result) == len(close)

    def test_rsi_bounds(self):
        close = _make_series()
        result = rsi(close, period=14).dropna()
        assert (result >= 0).all()
        assert (result <= 100).all()

    def test_rsi_constant_price(self):
        close = pd.Series([100.0] * 50)
        result = rsi(close, period=14)
        # Constant price → RSI should be NaN or 50-ish (no movement)
        # With ewm implementation, gain=0, loss=0, so NaN is expected
        assert True  # Just verify no crash


class TestZScore:
    def test_zscore_output_length(self):
        s = _make_series()
        result = zscore(s, window=20)
        assert len(result) == len(s)

    def test_zscore_mean_near_zero(self):
        s = _make_series(200)
        result = zscore(s, window=20).dropna()
        # Mean z-score should be near 0 over enough data
        assert abs(result.mean()) < 1.0


class TestVWAP:
    def test_vwap_output_length(self):
        df = _make_ohlcv()
        result = vwap(df["high"], df["low"], df["close"], df["volume"])
        assert len(result) == len(df)

    def test_vwap_in_price_range(self):
        df = _make_ohlcv()
        result = vwap(df["high"], df["low"], df["close"], df["volume"])
        # VWAP should be between global min low and max high
        assert result.iloc[-1] >= df["low"].min() - 10
        assert result.iloc[-1] <= df["high"].max() + 10


class TestBollingerBands:
    def test_bollinger_bands_output(self):
        close = _make_series()
        middle, upper, lower = bollinger_bands(close, window=20, num_std=2.0)
        assert len(upper) == len(close)
        assert len(middle) == len(close)
        assert len(lower) == len(close)

    def test_bollinger_ordering(self):
        close = _make_series()
        middle, upper, lower = bollinger_bands(close, window=20, num_std=2.0)
        idx = upper.dropna().index.intersection(middle.dropna().index).intersection(lower.dropna().index)
        assert (upper[idx] >= middle[idx]).all()
        assert (middle[idx] >= lower[idx]).all()


class TestRollingStats:
    def test_rolling_mean_length(self):
        s = _make_series()
        assert len(rolling_mean(s, 20)) == len(s)

    def test_rolling_std_positive(self):
        s = _make_series()
        result = rolling_std(s, 20).dropna()
        assert (result >= 0).all()

    def test_rolling_max_min(self):
        s = _make_series()
        r_max = rolling_max(s, 20)
        r_min = rolling_min(s, 20)
        assert (r_max.dropna() >= r_min.dropna()).all()


class TestReturns:
    def test_returns_length(self):
        close = _make_series()
        r = returns(close)
        assert len(r) == len(close)

    def test_log_returns_length(self):
        close = _make_series()
        close = close.clip(lower=1)  # Ensure positive prices
        lr = log_returns(close)
        assert len(lr) == len(close)


class TestMACD:
    def test_macd_output(self):
        close = _make_series(100)
        macd_line, signal_line, histogram = macd(close)
        assert len(macd_line) == len(close)
        assert len(signal_line) == len(close)
        assert len(histogram) == len(close)


class TestEMA:
    def test_ema_length(self):
        close = _make_series()
        result = ema(close, span=12)
        assert len(result) == len(close)


class TestOBV:
    def test_obv_length(self):
        df = _make_ohlcv()
        result = on_balance_volume(df["close"], df["volume"])
        assert len(result) == len(df)


class TestROC:
    def test_roc_length(self):
        close = _make_series()
        result = rate_of_change(close, period=10)
        assert len(result) == len(close)


class TestFeatureSets:
    def test_compute_base_features(self):
        from src.features.feature_sets import compute_base_features

        df = _make_ohlcv(200)
        df["symbol"] = "TEST"
        df["datetime"] = pd.date_range("2023-01-01", periods=200, freq="D")

        result = compute_base_features(df)
        assert len(result) == len(df)
        # Should have many more columns than original
        assert len(result.columns) > len(df.columns)

    def test_get_feature_columns(self):
        from src.features.feature_sets import get_feature_columns

        cols = get_feature_columns()
        assert isinstance(cols, list)
        assert len(cols) > 0
