"""Paper broker: deterministic fills + randomized slippage for simulation."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import numpy as np

from src.backtest.costs import CostModel
from src.core.config import CostsConfig, SlippageConfig
from src.core.logging_utils import AuditLogger, get_logger
from src.exec.broker_base import (
    BrokerBase,
    Order,
    OrderStatus,
    OrderType,
    Position,
)

logger = get_logger("exec.paper_broker")


class PaperBroker(BrokerBase):
    """Simulated broker for paper trading.

    Provides deterministic fills with optional randomized slippage.
    Maintains a virtual portfolio and tracks all orders.
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        costs_config: CostsConfig | None = None,
        slippage_config: SlippageConfig | None = None,
        audit_logger: AuditLogger | None = None,
        seed: int = 42,
    ):
        self.cash = initial_capital
        self.initial_capital = initial_capital
        self.positions: dict[str, Position] = {}  # symbol -> Position
        self.orders: dict[str, Order] = {}  # order_id -> Order
        self.cost_model = CostModel(costs_config, slippage_config)
        self.audit = audit_logger
        self.rng = np.random.RandomState(seed)

        # Current market prices (set externally)
        self.market_prices: dict[str, float] = {}

    def set_market_prices(self, prices: dict[str, float]) -> None:
        """Update current market prices for simulation."""
        self.market_prices = prices

    def place_order(self, order: Order) -> Order:
        """Place and immediately fill an order (paper mode).

        Applies slippage and transaction costs.
        """
        order.order_id = str(uuid.uuid4())[:8]
        order.timestamp = datetime.now(timezone.utc)

        # Get base price
        base_price = self.market_prices.get(order.symbol, order.limit_price or 0)
        if base_price <= 0:
            order.status = OrderStatus.REJECTED
            order.metadata["reject_reason"] = "No market price available"
            logger.warning(f"Order REJECTED for {order.symbol}: no price")
            return order

        # Apply slippage
        fill_price = self.cost_model.apply_slippage_to_price(
            base_price, order.side, seed=self.rng.randint(0, 100000)
        )

        # Compute costs
        cost = self.cost_model.compute(fill_price, order.quantity, order.side)

        # Check cash for buys
        if order.side == "BUY":
            required = fill_price * order.quantity + cost.total
            if required > self.cash:
                # Reduce quantity to fit
                affordable = int((self.cash * 0.95 - cost.total) / fill_price)
                if affordable <= 0:
                    order.status = OrderStatus.REJECTED
                    order.metadata["reject_reason"] = "Insufficient funds"
                    logger.warning(f"Order REJECTED for {order.symbol}: insufficient funds")
                    return order
                order.quantity = affordable
                cost = self.cost_model.compute(fill_price, order.quantity, order.side)

        # Execute fill
        order.filled_price = fill_price
        order.filled_quantity = order.quantity
        order.status = OrderStatus.FILLED
        order.metadata["costs"] = cost.total
        order.metadata["slippage_bps"] = (
            abs(fill_price - base_price) / base_price * 10000
        )

        # Update positions and cash
        self._update_position(order, cost.total)

        # Store order
        self.orders[order.order_id] = order

        # Audit log
        if self.audit:
            self.audit.log("order_filled", {
                "order_id": order.order_id,
                "symbol": order.symbol,
                "side": order.side,
                "quantity": order.quantity,
                "fill_price": fill_price,
                "base_price": base_price,
                "costs": cost.total,
                "cash_after": self.cash,
            })

        logger.info(
            f"FILLED: {order.side} {order.quantity} {order.symbol} "
            f"@ {fill_price:.2f} (cost: {cost.total:.2f})"
        )
        return order

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order (no-op in paper mode since fills are instant)."""
        if order_id in self.orders:
            self.orders[order_id].status = OrderStatus.CANCELLED
            return True
        return False

    def get_positions(self) -> list[Position]:
        """Get all current positions."""
        result = []
        for pos in self.positions.values():
            if pos.quantity != 0:
                # Update unrealized PnL
                current_price = self.market_prices.get(pos.symbol, pos.avg_price)
                pos.unrealized_pnl = (current_price - pos.avg_price) * pos.quantity
                result.append(pos)
        return result

    def get_order_status(self, order_id: str) -> Order:
        """Get order status."""
        return self.orders.get(order_id, Order(status=OrderStatus.REJECTED))

    def get_balance(self) -> float:
        """Get available cash."""
        return self.cash

    def get_equity(self) -> float:
        """Get total equity (cash + positions value)."""
        pos_value = sum(
            pos.quantity * self.market_prices.get(pos.symbol, pos.avg_price)
            for pos in self.positions.values()
        )
        return self.cash + pos_value

    def _update_position(self, order: Order, costs: float) -> None:
        """Update position tracking after a fill."""
        symbol = order.symbol
        qty = order.quantity
        price = order.filled_price

        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)

        pos = self.positions[symbol]

        if order.side == "BUY":
            # Update average price
            total_cost = pos.avg_price * pos.quantity + price * qty
            pos.quantity += qty
            pos.avg_price = total_cost / pos.quantity if pos.quantity > 0 else 0
            self.cash -= price * qty + costs
        else:
            # Realize PnL
            pnl = (price - pos.avg_price) * qty
            pos.realized_pnl += pnl
            pos.quantity -= qty
            self.cash += price * qty - costs

            if pos.quantity <= 0:
                pos.quantity = 0
                pos.avg_price = 0

    def summary(self) -> dict[str, Any]:
        """Portfolio summary."""
        equity = self.get_equity()
        return {
            "cash": self.cash,
            "equity": equity,
            "n_positions": len([p for p in self.positions.values() if p.quantity > 0]),
            "total_return_pct": (equity / self.initial_capital - 1) * 100,
            "n_orders": len(self.orders),
        }
