"""OPTIONAL Zerodha Kite adapter — disabled by default.

Requires .env with ZERODHA_API_KEY, ZERODHA_API_SECRET, ZERODHA_ACCESS_TOKEN
and explicit --confirm-live true CLI flag.
"""

from __future__ import annotations

import os
from typing import Any

from src.core.logging_utils import get_logger
from src.exec.broker_base import BrokerBase, Order, OrderStatus, Position

logger = get_logger("exec.zerodha")


class ZerodhaAdapter(BrokerBase):
    """Zerodha Kite Connect API adapter.

    DISABLED by default. Only activates when:
    1. Environment variables are set (ZERODHA_API_KEY, etc.)
    2. --confirm-live true is passed via CLI
    """

    def __init__(self) -> None:
        self.api_key = os.getenv("ZERODHA_API_KEY", "")
        self.api_secret = os.getenv("ZERODHA_API_SECRET", "")
        self.access_token = os.getenv("ZERODHA_ACCESS_TOKEN", "")
        self.kite: Any = None

        if not all([self.api_key, self.api_secret, self.access_token]):
            logger.warning(
                "Zerodha credentials not configured. "
                "Set ZERODHA_API_KEY, ZERODHA_API_SECRET, ZERODHA_ACCESS_TOKEN in .env"
            )

    def connect(self) -> bool:
        """Initialize Kite Connect session."""
        try:
            from kiteconnect import KiteConnect

            self.kite = KiteConnect(api_key=self.api_key)
            self.kite.set_access_token(self.access_token)
            logger.info("Connected to Zerodha Kite API")
            return True
        except ImportError:
            logger.error("kiteconnect package not installed. pip install kiteconnect")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Zerodha: {e}")
            return False

    def place_order(self, order: Order) -> Order:
        """Place order via Kite Connect API."""
        if self.kite is None:
            order.status = OrderStatus.REJECTED
            order.metadata["reject_reason"] = "Not connected to Zerodha"
            return order

        try:
            variety = "regular"
            exchange = "NSE"
            tradingsymbol = order.symbol
            transaction_type = order.side
            quantity = order.quantity
            order_type = order.order_type.value
            product = "MIS"  # Intraday

            kite_order_id = self.kite.place_order(
                variety=variety,
                exchange=exchange,
                tradingsymbol=tradingsymbol,
                transaction_type=transaction_type,
                quantity=quantity,
                order_type=order_type,
                product=product,
                price=order.limit_price,
            )

            order.order_id = str(kite_order_id)
            order.status = OrderStatus.PENDING
            logger.info(f"Zerodha order placed: {order.order_id}")

        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.metadata["reject_reason"] = str(e)
            logger.error(f"Zerodha order failed: {e}")

        return order

    def cancel_order(self, order_id: str) -> bool:
        """Cancel order via Kite Connect."""
        if self.kite is None:
            return False
        try:
            self.kite.cancel_order(variety="regular", order_id=order_id)
            return True
        except Exception as e:
            logger.error(f"Cancel failed: {e}")
            return False

    def get_positions(self) -> list[Position]:
        """Get current positions from Zerodha."""
        if self.kite is None:
            return []
        try:
            positions = self.kite.positions()
            result = []
            for p in positions.get("net", []):
                result.append(Position(
                    symbol=p["tradingsymbol"],
                    quantity=p["quantity"],
                    avg_price=p["average_price"],
                    unrealized_pnl=p["unrealised"],
                    realized_pnl=p["realised"],
                ))
            return result
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []

    def get_order_status(self, order_id: str) -> Order:
        """Get order status from Zerodha."""
        if self.kite is None:
            return Order(status=OrderStatus.REJECTED)
        try:
            history = self.kite.order_history(order_id)
            if history:
                latest = history[-1]
                order = Order(
                    order_id=order_id,
                    symbol=latest.get("tradingsymbol", ""),
                    side=latest.get("transaction_type", ""),
                    quantity=latest.get("quantity", 0),
                    filled_price=latest.get("average_price", 0),
                    filled_quantity=latest.get("filled_quantity", 0),
                )
                status_map = {
                    "COMPLETE": OrderStatus.FILLED,
                    "REJECTED": OrderStatus.REJECTED,
                    "CANCELLED": OrderStatus.CANCELLED,
                }
                order.status = status_map.get(latest.get("status", ""), OrderStatus.PENDING)
                return order
        except Exception as e:
            logger.error(f"Failed to get order status: {e}")
        return Order(order_id=order_id, status=OrderStatus.REJECTED)

    def get_balance(self) -> float:
        """Get available margin from Zerodha."""
        if self.kite is None:
            return 0.0
        try:
            margins = self.kite.margins()
            return float(margins.get("equity", {}).get("available", {}).get("cash", 0))
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return 0.0
