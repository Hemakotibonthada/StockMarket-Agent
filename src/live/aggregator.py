"""Tick-to-bar aggregator and websocket stubs for live data feeds."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Callable

import pandas as pd

from src.core.logging_utils import get_logger

logger = get_logger("live.aggregator")


class BarAggregator:
    """Aggregate incoming ticks into OHLCV bars.

    Supports multiple symbols and configurable intervals.
    """

    def __init__(
        self,
        interval_seconds: int = 300,
        on_bar: Callable[[dict[str, Any]], None] | None = None,
    ):
        self.interval_seconds = interval_seconds
        self.on_bar = on_bar

        # Per-symbol tick buffers
        self._buffers: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._bar_start: dict[str, datetime] = {}

    def on_tick(self, tick: dict[str, Any]) -> dict[str, Any] | None:
        """Process an incoming tick.

        Args:
            tick: Dict with keys: symbol, price, volume, timestamp.

        Returns:
            Completed bar dict if a bar was closed, else None.
        """
        symbol = tick["symbol"]
        timestamp = tick.get("timestamp", datetime.now(timezone.utc))

        if isinstance(timestamp, str):
            timestamp = pd.Timestamp(timestamp)

        # Initialize bar start
        if symbol not in self._bar_start:
            self._bar_start[symbol] = timestamp

        self._buffers[symbol].append(tick)

        # Check if bar is complete
        elapsed = (timestamp - self._bar_start[symbol]).total_seconds()
        if elapsed >= self.interval_seconds:
            bar = self._close_bar(symbol, timestamp)
            if self.on_bar:
                self.on_bar(bar)
            return bar

        return None

    def _close_bar(self, symbol: str, end_time: datetime) -> dict[str, Any]:
        """Close the current bar for a symbol."""
        ticks = self._buffers[symbol]

        if not ticks:
            return {
                "symbol": symbol,
                "datetime": end_time,
                "open": 0, "high": 0, "low": 0, "close": 0, "volume": 0,
            }

        prices = [t["price"] for t in ticks]
        volumes = [t.get("volume", 0) for t in ticks]

        bar = {
            "symbol": symbol,
            "datetime": self._bar_start[symbol],
            "open": prices[0],
            "high": max(prices),
            "low": min(prices),
            "close": prices[-1],
            "volume": sum(volumes),
        }

        # Reset buffer
        self._buffers[symbol] = []
        self._bar_start[symbol] = end_time

        return bar

    def flush_all(self) -> list[dict[str, Any]]:
        """Flush all open bars (e.g., at end of session)."""
        bars = []
        now = datetime.now(timezone.utc)
        for symbol in list(self._buffers.keys()):
            if self._buffers[symbol]:
                bars.append(self._close_bar(symbol, now))
        return bars


class WebSocketStub:
    """Stub websocket handler for live data feeds.

    In production, this would connect to NSE/broker websocket feeds.
    For paper trading, it can replay stored tick data.
    """

    def __init__(self, aggregator: BarAggregator):
        self.aggregator = aggregator
        self._connected = False

    async def connect(self, url: str = "") -> None:
        """Connect to websocket feed (stub)."""
        logger.info(f"WebSocket stub connected (url={url})")
        self._connected = True

    async def disconnect(self) -> None:
        """Disconnect from websocket feed."""
        self._connected = False
        bars = self.aggregator.flush_all()
        logger.info(f"WebSocket disconnected, flushed {len(bars)} bars")

    def simulate_tick(self, tick: dict[str, Any]) -> dict[str, Any] | None:
        """Simulate an incoming tick for testing."""
        return self.aggregator.on_tick(tick)
