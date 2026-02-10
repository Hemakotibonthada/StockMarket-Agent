"""Tests for the transaction cost model."""

from __future__ import annotations

import pytest

from src.backtest.costs import CostModel, TransactionCost
from src.core.config import CostsConfig, SlippageConfig


class TestTransactionCost:
    def test_total_auto_calculated(self):
        tc = TransactionCost(
            brokerage=10.0,
            stt=5.0,
            gst=1.8,
            stamp_duty=0.5,
            sebi_charges=0.1,
            slippage=3.0,
        )
        assert tc.total == pytest.approx(20.4)

    def test_zero_cost(self):
        tc = TransactionCost()
        assert tc.total == 0.0


class TestCostModel:
    def test_default_config(self):
        model = CostModel()
        cost = model.compute(100.0, 100, "BUY", seed=42)
        assert cost.total > 0
        assert cost.brokerage > 0
        assert cost.stt >= 0

    def test_zero_slippage_fixed(self):
        model = CostModel(
            slippage_config=SlippageConfig(mode="fixed", bps_mean=0.0, bps_std=0.0)
        )
        cost = model.compute(100.0, 100, "BUY")
        assert cost.slippage == 0.0

    def test_fixed_slippage(self):
        model = CostModel(
            slippage_config=SlippageConfig(mode="fixed", bps_mean=5.0, bps_std=0.0)
        )
        cost = model.compute(1000.0, 10, "BUY")
        # turnover = 10000, slippage = 10000 * 5/10000 = 5.0
        assert cost.slippage == pytest.approx(5.0)

    def test_random_slippage_deterministic(self):
        model = CostModel(
            slippage_config=SlippageConfig(mode="random", bps_mean=5.0, bps_std=2.0)
        )
        c1 = model.compute(100.0, 100, "BUY", seed=42)
        c2 = model.compute(100.0, 100, "BUY", seed=42)
        assert c1.slippage == pytest.approx(c2.slippage)

    def test_costs_proportional_to_turnover(self):
        model = CostModel(
            slippage_config=SlippageConfig(mode="fixed", bps_mean=0.0, bps_std=0.0)
        )
        c1 = model.compute(100.0, 100, "BUY")
        c2 = model.compute(100.0, 200, "BUY")
        # Double quantity should roughly double all proportional costs
        assert c2.brokerage == pytest.approx(c1.brokerage * 2)
        assert c2.stt == pytest.approx(c1.stt * 2)

    def test_apply_slippage_to_price_buy(self):
        model = CostModel(
            slippage_config=SlippageConfig(mode="fixed", bps_mean=5.0, bps_std=0.0)
        )
        slipped = model.apply_slippage_to_price(100.0, "BUY", seed=42)
        # Buy should get a worse (higher) price
        assert slipped >= 100.0

    def test_no_slippage_mode(self):
        model = CostModel(
            slippage_config=SlippageConfig(mode="none", bps_mean=0.0, bps_std=0.0)
        )
        cost = model.compute(100.0, 100, "BUY")
        assert cost.slippage == 0.0

    def test_large_order(self):
        """Costs should work for large orders without overflow."""
        model = CostModel(
            slippage_config=SlippageConfig(mode="fixed", bps_mean=1.0, bps_std=0.0)
        )
        cost = model.compute(5000.0, 10_000, "BUY")
        assert cost.total > 0
        assert cost.total < 5000.0 * 10_000  # Costs should be fraction of turnover
