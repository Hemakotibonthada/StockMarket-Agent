"""Base strategy interface and common types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class SignalType(str, Enum):
    ENTRY_LONG = "ENTRY_LONG"
    ENTRY_SHORT = "ENTRY_SHORT"
    EXIT_LONG = "EXIT_LONG"
    EXIT_SHORT = "EXIT_SHORT"
    HOLD = "HOLD"


@dataclass
class Signal:
    """A single trading signal from a strategy."""

    symbol: str
    signal_type: SignalType
    price: float
    timestamp: pd.Timestamp | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyState:
    """Mutable state carried across bars for a strategy."""

    positions: dict[str, float] = field(default_factory=dict)  # symbol -> quantity
    entry_prices: dict[str, float] = field(default_factory=dict)
    entry_times: dict[str, pd.Timestamp] = field(default_factory=dict)
    bar_count: int = 0
    daily_pnl: float = 0.0
    custom: dict[str, Any] = field(default_factory=dict)


class BaseStrategy(ABC):
    """Abstract base class for all strategies.

    Every strategy must implement `generate_signals`.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.name: str = self.__class__.__name__

    @abstractmethod
    def generate_signals(
        self,
        bars: pd.DataFrame,
        state: StrategyState,
        config: dict[str, Any] | None = None,
    ) -> list[Signal]:
        """Generate trading signals from current bars and state.

        Args:
            bars: OHLCV DataFrame (may be multi-symbol).
            state: Current strategy state.
            config: Optional runtime config overrides.

        Returns:
            List of trading signals.
        """
        ...

    def on_fill(self, symbol: str, side: Side, price: float, quantity: float, state: StrategyState) -> None:
        """Called when an order is filled. Update state accordingly."""
        if side == Side.BUY:
            state.positions[symbol] = state.positions.get(symbol, 0) + quantity
            state.entry_prices[symbol] = price
        elif side == Side.SELL:
            state.positions[symbol] = state.positions.get(symbol, 0) - quantity
            if state.positions.get(symbol, 0) <= 0:
                state.positions.pop(symbol, None)
                state.entry_prices.pop(symbol, None)
                state.entry_times.pop(symbol, None)

    def reset(self) -> StrategyState:
        """Return a fresh state."""
        return StrategyState()
