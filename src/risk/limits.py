"""Risk limits: per-trade, daily, and weekly loss limits."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from src.core.config import RiskConfig
from src.core.logging_utils import AuditLogger, get_logger

logger = get_logger("risk.limits")


@dataclass
class RiskState:
    """Tracks cumulative risk metrics for limit enforcement."""

    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    current_date: date | None = None
    week_start: date | None = None
    trade_count_today: int = 0
    peak_equity: float = 0.0
    current_equity: float = 0.0
    consecutive_losses: int = 0
    is_killed: bool = False
    kill_reason: str = ""


class RiskLimiter:
    """Enforces hard risk limits on trading activity.

    Defaults:
        - risk_per_trade: 0.5% of equity
        - daily_max_loss: 1% of equity
        - weekly_max_loss: 2% of equity
        - strategy_max_dd: 5% from peak
    """

    def __init__(
        self,
        config: RiskConfig | None = None,
        audit_logger: AuditLogger | None = None,
    ):
        self.config = config or RiskConfig()
        self.state = RiskState()
        self.audit = audit_logger

    def initialize(self, equity: float) -> None:
        """Initialize risk state with current equity."""
        self.state.current_equity = equity
        self.state.peak_equity = equity
        self.state.current_date = date.today()
        self.state.week_start = date.today()

    def check_trade_allowed(
        self,
        risk_amount: float,
        current_equity: float,
    ) -> tuple[bool, str]:
        """Check if a new trade is allowed within risk limits.

        Args:
            risk_amount: Dollar amount at risk for the proposed trade.
            current_equity: Current portfolio equity.

        Returns:
            (allowed, reason) tuple.
        """
        if self.state.is_killed:
            return False, f"Kill switch active: {self.state.kill_reason}"

        self._update_dates()

        # Per-trade risk
        max_risk = current_equity * self.config.risk_per_trade_pct / 100
        if risk_amount > max_risk:
            reason = f"Trade risk {risk_amount:.0f} exceeds limit {max_risk:.0f}"
            self._log_rejection(reason)
            return False, reason

        # Daily loss limit
        daily_limit = current_equity * self.config.daily_max_loss_pct / 100
        if abs(self.state.daily_pnl) >= daily_limit and self.state.daily_pnl < 0:
            reason = f"Daily loss limit reached: {self.state.daily_pnl:.0f} vs limit {daily_limit:.0f}"
            self._log_rejection(reason)
            return False, reason

        # Weekly loss limit
        weekly_limit = current_equity * self.config.weekly_max_loss_pct / 100
        if abs(self.state.weekly_pnl) >= weekly_limit and self.state.weekly_pnl < 0:
            reason = f"Weekly loss limit reached: {self.state.weekly_pnl:.0f}"
            self._log_rejection(reason)
            return False, reason

        # Max drawdown
        drawdown_pct = (self.state.peak_equity - current_equity) / self.state.peak_equity * 100
        if drawdown_pct >= self.config.strategy_max_dd_pct:
            reason = f"Max drawdown {drawdown_pct:.1f}% exceeds limit {self.config.strategy_max_dd_pct}%"
            self._log_rejection(reason)
            self.kill(reason)
            return False, reason

        return True, "OK"

    def record_trade_pnl(self, pnl: float, equity: float) -> None:
        """Record a completed trade's PnL and update state."""
        self._update_dates()
        self.state.daily_pnl += pnl
        self.state.weekly_pnl += pnl
        self.state.trade_count_today += 1
        self.state.current_equity = equity
        self.state.peak_equity = max(self.state.peak_equity, equity)

        if pnl < 0:
            self.state.consecutive_losses += 1
        else:
            self.state.consecutive_losses = 0

        if self.audit:
            self.audit.log("trade_pnl", {
                "pnl": pnl,
                "daily_pnl": self.state.daily_pnl,
                "weekly_pnl": self.state.weekly_pnl,
                "equity": equity,
                "drawdown_pct": (self.state.peak_equity - equity) / self.state.peak_equity * 100,
            })

    def kill(self, reason: str) -> None:
        """Activate kill switch."""
        self.state.is_killed = True
        self.state.kill_reason = reason
        logger.critical(f"KILL SWITCH ACTIVATED: {reason}")
        if self.audit:
            self.audit.log("kill_switch", {"reason": reason})

    def reset_kill(self) -> None:
        """Reset kill switch (manual operation only)."""
        self.state.is_killed = False
        self.state.kill_reason = ""
        logger.warning("Kill switch reset manually")

    def _update_dates(self) -> None:
        """Reset daily/weekly counters on date change."""
        today = date.today()

        if self.state.current_date != today:
            self.state.daily_pnl = 0.0
            self.state.trade_count_today = 0
            self.state.current_date = today

        # Reset weekly on Monday
        if self.state.week_start is None or (today.weekday() == 0 and today != self.state.week_start):
            self.state.weekly_pnl = 0.0
            self.state.week_start = today

    def _log_rejection(self, reason: str) -> None:
        """Log a trade rejection."""
        logger.warning(f"Trade REJECTED: {reason}")
        if self.audit:
            self.audit.log("trade_rejected", {"reason": reason})

    def summary(self) -> dict[str, Any]:
        """Return current risk state summary."""
        dd = 0.0
        if self.state.peak_equity > 0:
            dd = (self.state.peak_equity - self.state.current_equity) / self.state.peak_equity * 100
        return {
            "daily_pnl": self.state.daily_pnl,
            "weekly_pnl": self.state.weekly_pnl,
            "drawdown_pct": dd,
            "consecutive_losses": self.state.consecutive_losses,
            "trade_count_today": self.state.trade_count_today,
            "is_killed": self.state.is_killed,
            "kill_reason": self.state.kill_reason,
        }
