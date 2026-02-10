"""Abstract broker interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"


@dataclass
class Order:
    """Represents a trading order."""

    order_id: str = ""
    symbol: str = ""
    side: str = "BUY"  # BUY or SELL
    quantity: int = 0
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    status: OrderStatus = OrderStatus.PENDING
    filled_price: float = 0.0
    filled_quantity: int = 0
    timestamp: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    """Represents a current position."""

    symbol: str = ""
    quantity: int = 0
    avg_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


class BrokerBase(ABC):
    """Abstract base class for broker implementations."""

    @abstractmethod
    def place_order(self, order: Order) -> Order:
        """Place an order. Returns updated order with fill info."""
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        ...

    @abstractmethod
    def get_positions(self) -> list[Position]:
        """Get all current positions."""
        ...

    @abstractmethod
    def get_order_status(self, order_id: str) -> Order:
        """Get current status of an order."""
        ...

    @abstractmethod
    def get_balance(self) -> float:
        """Get available cash balance."""
        ...
