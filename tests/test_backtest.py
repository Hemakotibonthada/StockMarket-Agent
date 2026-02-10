"""Tests for the backtest engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.config import AppConfig, CostsConfig, SlippageConfig
from src.core.utils import set_seed


def _make_ohlcv(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    set_seed(seed)
    rng = np.random.RandomState(seed)

    dates = pd.date_range("2023-01-02", periods=n, freq="D")
    close = 100 + np.cumsum(rng.randn(n) * 2)
    close = np.maximum(close, 10)  # floor price

    df = pd.DataFrame({
        "datetime": dates,
        "symbol": "TEST",
        "open": close + rng.randn(n) * 0.5,
        "high": close + abs(rng.randn(n)) * 2,
        "low": close - abs(rng.randn(n)) * 2,
        "close": close,
        "volume": (rng.rand(n) * 100_000 + 50_000).astype(int),
    })
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)
    return df


class TestPortfolio:
    """Tests for Portfolio dataclass."""

    def test_initial_state(self):
        from src.backtest.engine import Portfolio

        p = Portfolio(initial_capital=500_000)
        assert p.cash == 500_000
        assert p.positions == {}
        assert p.trades == []

    def test_market_value_no_positions(self):
        from src.backtest.engine import Portfolio

        p = Portfolio(initial_capital=1_000_000)
        assert p.market_value({}) == 1_000_000

    def test_market_value_with_positions(self):
        from src.backtest.engine import Portfolio

        p = Portfolio(initial_capital=1_000_000)
        p.positions = {"TEST": 100}
        p.cash = 900_000
        mv = p.market_value({"TEST": 150.0})
        assert mv == 900_000 + 100 * 150.0

    def test_exposure(self):
        from src.backtest.engine import Portfolio

        p = Portfolio(initial_capital=1_000_000)
        p.positions = {"TEST": 100}
        p.cash = 900_000
        prices = {"TEST": 1000.0}
        # exposure = gross / market_value = 100*1000 / (900k + 100k) = 0.1
        exp = p.exposure(prices)
        assert 0.09 < exp < 0.11


class TestBacktestEngine:
    """Integration tests for the backtest engine."""

    def test_engine_runs(self):
        from src.backtest.engine import BacktestEngine
        from src.strategies.mean_reversion import MeanReversion
        from src.features.feature_sets import compute_base_features

        set_seed(42)
        df = _make_ohlcv(200)
        df = compute_base_features(df)

        from src.core.config import RiskConfig
        cfg = AppConfig(strategy="mean_reversion", risk=RiskConfig(initial_capital=1_000_000))
        strategy = MeanReversion(config={"zscore_entry": 1.5, "zscore_exit": 0.5})

        engine = BacktestEngine(strategy=strategy, config=cfg)
        result = engine.run(df)

        summary = result.summary()
        assert isinstance(summary, dict)

    def test_engine_deterministic(self):
        """Same seed should produce same results."""
        from src.backtest.engine import BacktestEngine
        from src.strategies.mean_reversion import MeanReversion
        from src.features.feature_sets import compute_base_features

        df = _make_ohlcv(200, seed=42)
        df = compute_base_features(df)

        from src.core.config import RiskConfig
        cfg = AppConfig(
            strategy="mean_reversion",
            slippage=SlippageConfig(mode="fixed", bps_mean=2.0, bps_std=0.0),
            risk=RiskConfig(initial_capital=1_000_000),
        )
        strategy1 = MeanReversion(config={"zscore_entry": 1.5, "zscore_exit": 0.5})
        strategy2 = MeanReversion(config={"zscore_entry": 1.5, "zscore_exit": 0.5})

        set_seed(42)
        engine1 = BacktestEngine(strategy=strategy1, config=cfg)
        result1 = engine1.run(df)

        set_seed(42)
        engine2 = BacktestEngine(strategy=strategy2, config=cfg)
        result2 = engine2.run(df)

        s1 = result1.summary()
        s2 = result2.summary()
        # Both runs should produce the same number of trades at minimum
        assert len(result1.trades) == len(result2.trades)


class TestTrade:
    def test_trade_creation(self):
        from src.backtest.engine import Trade

        t = Trade(
            symbol="TEST",
            side="BUY",
            entry_price=100.0,
            exit_price=110.0,
            quantity=10,
        )
        assert t.symbol == "TEST"
        assert t.quantity == 10
