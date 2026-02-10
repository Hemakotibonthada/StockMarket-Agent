"""Runtime health checks: feed latency, CPU/mem, exception counters."""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any

import psutil

from src.core.logging_utils import get_logger

logger = get_logger("live.health")


class HealthMonitor:
    """Monitor system and application health during live trading.

    Tracks:
    - Feed latency and dropped bars
    - CPU and memory usage
    - Exception counter
    - Reject rate
    - Slippage deviations
    """

    def __init__(self):
        self._start_time = time.time()
        self._last_feed_time: float | None = None
        self._bars_received = 0
        self._bars_dropped = 0
        self._exceptions = 0
        self._orders_total = 0
        self._orders_rejected = 0
        self._slippage_bps: list[float] = []

    def record_bar(self) -> None:
        """Record that a bar was received."""
        self._bars_received += 1
        self._last_feed_time = time.time()

    def record_dropped_bar(self) -> None:
        """Record that a bar was dropped."""
        self._bars_dropped += 1

    def record_exception(self) -> None:
        """Record an exception occurrence."""
        self._exceptions += 1

    def record_order(self, filled: bool) -> None:
        """Record an order submission."""
        self._orders_total += 1
        if not filled:
            self._orders_rejected += 1

    def record_slippage(self, bps: float) -> None:
        """Record a slippage observation (in bps)."""
        self._slippage_bps.append(bps)

    @property
    def feed_latency_seconds(self) -> float:
        """Seconds since last feed update."""
        if self._last_feed_time is None:
            return float("inf")
        return time.time() - self._last_feed_time

    @property
    def uptime_seconds(self) -> float:
        """Uptime of the health monitor."""
        return time.time() - self._start_time

    @property
    def reject_rate(self) -> float:
        """Fraction of orders rejected."""
        if self._orders_total == 0:
            return 0.0
        return self._orders_rejected / self._orders_total

    def system_stats(self) -> dict[str, Any]:
        """Get current system resource usage."""
        process = psutil.Process(os.getpid())
        return {
            "cpu_pct": psutil.cpu_percent(interval=0),
            "mem_rss_mb": process.memory_info().rss / (1024 * 1024),
            "mem_pct": process.memory_percent(),
            "n_threads": process.num_threads(),
        }

    def health_check(self) -> dict[str, Any]:
        """Run a full health check and return status dict."""
        import numpy as np

        system = self.system_stats()
        avg_slippage = np.mean(self._slippage_bps) if self._slippage_bps else 0.0
        std_slippage = np.std(self._slippage_bps) if len(self._slippage_bps) > 1 else 0.0

        status = "HEALTHY"
        issues = []

        if self.feed_latency_seconds > 120:
            status = "DEGRADED"
            issues.append(f"Feed latency: {self.feed_latency_seconds:.0f}s")

        if system["mem_pct"] > 80:
            status = "WARNING"
            issues.append(f"High memory: {system['mem_pct']:.1f}%")

        if self.reject_rate > 0.5 and self._orders_total > 5:
            status = "WARNING"
            issues.append(f"High reject rate: {self.reject_rate:.1%}")

        if self._exceptions > 10:
            status = "DEGRADED"
            issues.append(f"Exception count: {self._exceptions}")

        return {
            "status": status,
            "issues": issues,
            "uptime_seconds": self.uptime_seconds,
            "feed_latency_seconds": self.feed_latency_seconds,
            "bars_received": self._bars_received,
            "bars_dropped": self._bars_dropped,
            "orders_total": self._orders_total,
            "orders_rejected": self._orders_rejected,
            "reject_rate": self.reject_rate,
            "avg_slippage_bps": avg_slippage,
            "std_slippage_bps": std_slippage,
            "exceptions": self._exceptions,
            **system,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def print_dashboard(self) -> str:
        """Generate a terminal dashboard string."""
        h = self.health_check()
        lines = [
            "╔══════════════════════════════════════════════════╗",
            "║          STOCK AGENT HEALTH DASHBOARD            ║",
            "╠══════════════════════════════════════════════════╣",
            f"║ Status: {h['status']:<42}║",
            f"║ Uptime: {h['uptime_seconds']:.0f}s{'':<37}║",
            f"║ Feed Latency: {h['feed_latency_seconds']:.1f}s{'':<32}║",
            f"║ Bars: {h['bars_received']} received, {h['bars_dropped']} dropped{'':<23}║",
            f"║ Orders: {h['orders_total']} total, {h['orders_rejected']} rejected ({h['reject_rate']:.1%}){'':<8}║",
            f"║ Avg Slippage: {h['avg_slippage_bps']:.1f} bps{'':<30}║",
            f"║ CPU: {h['cpu_pct']:.1f}%  RAM: {h['mem_rss_mb']:.0f}MB ({h['mem_pct']:.1f}%){'':<17}║",
            f"║ Exceptions: {h['exceptions']}{'':<35}║",
        ]
        if h["issues"]:
            lines.append("║ Issues:                                          ║")
            for issue in h["issues"]:
                lines.append(f"║   ⚠ {issue:<44}║")
        lines.append("╚══════════════════════════════════════════════════╝")
        return "\n".join(lines)
