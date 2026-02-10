"""Tests for risk management layer."""

from __future__ import annotations

import pytest

from src.core.config import RiskConfig
from src.risk.limits import RiskLimiter, RiskState
from src.risk.sizing import atr_position_size, fixed_fraction_size, kelly_fraction


class TestRiskLimiter:
    def _make_limiter(self, **kwargs) -> RiskLimiter:
        config = RiskConfig(**kwargs)
        limiter = RiskLimiter(config=config)
        limiter.initialize(1_000_000)
        return limiter

    def test_trade_allowed_normal(self):
        limiter = self._make_limiter(risk_per_trade_pct=1.0)
        allowed, reason = limiter.check_trade_allowed(5000, 1_000_000)
        assert allowed
        assert reason == "OK"

    def test_trade_rejected_excessive_risk(self):
        limiter = self._make_limiter(risk_per_trade_pct=0.5)
        # Risk 1% when limit is 0.5%
        allowed, reason = limiter.check_trade_allowed(10_000, 1_000_000)
        assert not allowed
        assert "exceeds limit" in reason

    def test_daily_loss_limit(self):
        limiter = self._make_limiter(daily_max_loss_pct=1.0)
        # Record a big loss
        limiter.record_trade_pnl(-15_000, 985_000)
        allowed, reason = limiter.check_trade_allowed(1000, 985_000)
        assert not allowed
        assert "Daily loss" in reason

    def test_kill_switch(self):
        limiter = self._make_limiter()
        limiter.kill("test reason")
        assert limiter.state.is_killed
        allowed, reason = limiter.check_trade_allowed(100, 1_000_000)
        assert not allowed
        assert "Kill switch" in reason

    def test_reset_kill(self):
        limiter = self._make_limiter()
        limiter.kill("test")
        limiter.reset_kill()
        assert not limiter.state.is_killed

    def test_consecutive_losses_tracked(self):
        limiter = self._make_limiter()
        limiter.record_trade_pnl(-100, 999_900)
        limiter.record_trade_pnl(-100, 999_800)
        limiter.record_trade_pnl(-100, 999_700)
        assert limiter.state.consecutive_losses == 3

        limiter.record_trade_pnl(100, 999_800)
        assert limiter.state.consecutive_losses == 0

    def test_peak_equity_tracking(self):
        limiter = self._make_limiter()
        limiter.record_trade_pnl(5000, 1_005_000)
        assert limiter.state.peak_equity == 1_005_000

        limiter.record_trade_pnl(-1000, 1_004_000)
        assert limiter.state.peak_equity == 1_005_000  # unchanged

    def test_summary(self):
        limiter = self._make_limiter()
        limiter.record_trade_pnl(-500, 999_500)
        s = limiter.summary()
        assert "daily_pnl" in s
        assert "is_killed" in s
        assert s["daily_pnl"] == -500

    def test_max_drawdown_kill(self):
        limiter = self._make_limiter(strategy_max_dd_pct=5.0)
        limiter.initialize(1_000_000)
        # Simulate drawdown exceeding 5%
        equity = 940_000  # 6% drawdown
        allowed, reason = limiter.check_trade_allowed(1000, equity)
        assert not allowed
        assert "drawdown" in reason.lower() or limiter.state.is_killed


class TestRiskState:
    def test_default_state(self):
        state = RiskState()
        assert state.daily_pnl == 0.0
        assert state.is_killed is False
        assert state.consecutive_losses == 0


class TestPositionSizing:
    def test_atr_position_size(self):
        size = atr_position_size(
            capital=1_000_000,
            risk_pct=0.01,
            atr_value=10.0,
            atr_multiplier=2.0,
        )
        # risk = 10000, atr_risk = 20, size = 10000/20 = 500
        assert size == 500

    def test_atr_position_size_zero_atr(self):
        size = atr_position_size(
            capital=1_000_000,
            risk_pct=0.01,
            atr_value=0.0,
            atr_multiplier=2.0,
        )
        assert size == 0

    def test_fixed_fraction(self):
        size = fixed_fraction_size(
            capital=1_000_000,
            risk_pct=0.01,
            entry_price=500.0,
            stop_loss=490.0,
        )
        # risk = 10000, risk_per_share = 10, size = 1000
        assert size == 1000

    def test_fixed_fraction_zero_price(self):
        size = fixed_fraction_size(
            capital=1_000_000,
            risk_pct=0.01,
            entry_price=500.0,
            stop_loss=500.0,
        )
        assert size == 0

    def test_kelly_fraction(self):
        k = kelly_fraction(
            win_rate=0.6,
            avg_win=100.0,
            avg_loss=80.0,
        )
        # b = 100/80 = 1.25, kelly = (1.25*0.6 - 0.4)/1.25 = (0.75-0.4)/1.25 = 0.28
        assert 0 < k <= 0.25  # capped at 0.25

    def test_kelly_negative(self):
        k = kelly_fraction(
            win_rate=0.3,
            avg_win=50.0,
            avg_loss=100.0,
        )
        # b=0.5, kelly = (0.5*0.3 - 0.7)/0.5 = (0.15-0.7)/0.5 = -1.1 → capped at 0
        assert k == 0
