"""OPTIONAL Upstox adapter — disabled by default.

Requires .env with UPSTOX_API_KEY, UPSTOX_API_SECRET, UPSTOX_ACCESS_TOKEN
and explicit --confirm-live true CLI flag.
"""

from __future__ import annotations

import os

from src.core.logging_utils import get_logger
from src.exec.broker_base import BrokerBase, Order, OrderStatus, Position

logger = get_logger("exec.upstox")


class UpstoxAdapter(BrokerBase):
    """Upstox API adapter.

    DISABLED by default. Only activates when:
    1. Environment variables are set (UPSTOX_API_KEY, etc.)
    2. --confirm-live true is passed via CLI
    """

    def __init__(self) -> None:
        self.api_key = os.getenv("UPSTOX_API_KEY", "")
        self.api_secret = os.getenv("UPSTOX_API_SECRET", "")
        self.access_token = os.getenv("UPSTOX_ACCESS_TOKEN", "")

        if not all([self.api_key, self.api_secret, self.access_token]):
            logger.warning(
                "Upstox credentials not configured. "
                "Set UPSTOX_API_KEY, UPSTOX_API_SECRET, UPSTOX_ACCESS_TOKEN in .env"
            )

    def place_order(self, order: Order) -> Order:
        """Place order via Upstox API (stub)."""
        # TODO: Implement with upstox-python-sdk when available
        order.status = OrderStatus.REJECTED
        order.metadata["reject_reason"] = "Upstox adapter not fully implemented"
        logger.warning("Upstox adapter: place_order not implemented")
        return order

    def cancel_order(self, order_id: str) -> bool:
        logger.warning("Upstox adapter: cancel_order not implemented")
        return False

    def get_positions(self) -> list[Position]:
        logger.warning("Upstox adapter: get_positions not implemented")
        return []

    def get_order_status(self, order_id: str) -> Order:
        return Order(order_id=order_id, status=OrderStatus.REJECTED)

    def get_balance(self) -> float:
        return 0.0
