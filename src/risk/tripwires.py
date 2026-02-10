"""Kill-switches and tripwires for safety monitoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.core.config import TripwireConfig
from src.core.logging_utils import AuditLogger, get_logger

logger = get_logger("risk.tripwires")


@dataclass
class TripwireState:
    """Tracked state for tripwire conditions."""

    consecutive_rejects: int = 0
    last_latency_ms: float = 0.0
    feed_seconds_since_update: float = 0.0
    exception_count: int = 0
    slippage_deviations: list[float] = field(default_factory=list)


class TripwireMonitor:
    """Monitor conditions that should trigger a kill-switch.

    Conditions:
        - Too many consecutive order rejects.
        - High latency exceeding threshold.
        - Data feed timeout.
        - Excessive exceptions.
        - Large slippage deviations.
    """

    def __init__(
        self,
        config: TripwireConfig | None = None,
        audit_logger: AuditLogger | None = None,
    ):
        self.config = config or TripwireConfig()
        self.state = TripwireState()
        self.audit = audit_logger
        self._tripped = False
        self._trip_reasons: list[str] = []

    @property
    def is_tripped(self) -> bool:
        return self._tripped

    @property
    def trip_reasons(self) -> list[str]:
        return self._trip_reasons.copy()

    def record_reject(self) -> None:
        """Record an order rejection."""
        self.state.consecutive_rejects += 1
        if self.state.consecutive_rejects >= self.config.max_consecutive_rejects:
            self._trip(
                f"Consecutive rejects: {self.state.consecutive_rejects} "
                f"(limit: {self.config.max_consecutive_rejects})"
            )

    def record_fill(self) -> None:
        """Record a successful fill (resets reject counter)."""
        self.state.consecutive_rejects = 0

    def record_latency(self, ms: float) -> None:
        """Record order/feed latency."""
        self.state.last_latency_ms = ms
        if ms > self.config.max_latency_ms:
            self._trip(f"Latency {ms:.0f}ms exceeds limit {self.config.max_latency_ms}ms")

    def record_feed_update(self) -> None:
        """Record that a data feed update was received (resets timeout)."""
        self.state.feed_seconds_since_update = 0.0

    def record_feed_gap(self, seconds: float) -> None:
        """Record elapsed time since last feed update."""
        self.state.feed_seconds_since_update = seconds
        if seconds > self.config.feed_timeout_seconds:
            self._trip(
                f"Feed timeout: {seconds:.0f}s since last update "
                f"(limit: {self.config.feed_timeout_seconds}s)"
            )

    def record_exception(self) -> None:
        """Record an unhandled exception."""
        self.state.exception_count += 1
        if self.state.exception_count >= 10:
            self._trip(f"Exception count: {self.state.exception_count}")

    def record_slippage(self, expected_bps: float, actual_bps: float) -> None:
        """Record slippage deviation from expected."""
        deviation = actual_bps - expected_bps
        self.state.slippage_deviations.append(deviation)

        # Check recent deviations
        recent = self.state.slippage_deviations[-20:]
        if len(recent) >= 5:
            import numpy as np
            avg_dev = np.mean(recent)
            if abs(avg_dev) > 20:  # 20 bps average deviation
                self._trip(f"Slippage deviation: avg {avg_dev:.1f} bps over last {len(recent)} trades")

    def check_drawdown(self, drawdown_pct: float) -> None:
        """Check if drawdown exceeds the tripwire threshold."""
        if drawdown_pct >= self.config.max_drawdown_pct:
            self._trip(f"Drawdown {drawdown_pct:.1f}% >= limit {self.config.max_drawdown_pct}%")

    def reset(self) -> None:
        """Reset all tripwire state."""
        self.state = TripwireState()
        self._tripped = False
        self._trip_reasons = []
        logger.info("Tripwire monitor reset")

    def _trip(self, reason: str) -> None:
        """Activate a tripwire."""
        self._tripped = True
        self._trip_reasons.append(reason)
        logger.critical(f"TRIPWIRE TRIPPED: {reason}")
        if self.audit:
            self.audit.log("tripwire", {"reason": reason, "state": self.summary()})

    def summary(self) -> dict[str, Any]:
        """Current tripwire state summary."""
        return {
            "is_tripped": self._tripped,
            "reasons": self._trip_reasons,
            "consecutive_rejects": self.state.consecutive_rejects,
            "last_latency_ms": self.state.last_latency_ms,
            "feed_gap_seconds": self.state.feed_seconds_since_update,
            "exception_count": self.state.exception_count,
        }
