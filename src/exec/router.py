"""Order router with rate limiting, price protection, retries, and audit logging."""

from __future__ import annotations

import json
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any

from src.core.config import OrderRouterConfig
from src.core.logging_utils import AuditLogger, get_logger
from src.exec.broker_base import BrokerBase, Order, OrderStatus

logger = get_logger("exec.router")


class OrderRouter:
    """Routes orders through safety checks before sending to broker.

    Implements:
    - Rate limiting (per-second, per-minute)
    - Price protection (reject orders far from market price)
    - Smart retries with backoff
    - Circuit breaker on consecutive failures
    - JSONL audit logging of every order event
    """

    def __init__(
        self,
        broker: BrokerBase,
        config: OrderRouterConfig | None = None,
        audit_logger: AuditLogger | None = None,
        market_prices: dict[str, float] | None = None,
    ):
        self.broker = broker
        self.config = config or OrderRouterConfig()
        self.audit = audit_logger or AuditLogger()
        self.market_prices = market_prices or {}

        # Rate limiting
        self._second_timestamps: deque[float] = deque()
        self._minute_timestamps: deque[float] = deque()

        # Circuit breaker
        self._consecutive_failures = 0
        self._circuit_open = False
        self._circuit_open_until: float = 0.0

    def submit_order(self, order: Order) -> Order:
        """Submit an order through all safety checks.

        Checks: circuit breaker → rate limit → price protection → place order.
        Retries on transient failures.
        """
        # Circuit breaker check
        if self._circuit_open:
            if time.time() < self._circuit_open_until:
                order.status = OrderStatus.REJECTED
                order.metadata["reject_reason"] = "Circuit breaker open"
                self._log_order_event(order, "circuit_breaker_reject")
                return order
            else:
                self._circuit_open = False
                self._consecutive_failures = 0
                logger.info("Circuit breaker reset")

        # Rate limit check
        if not self._check_rate_limit():
            order.status = OrderStatus.REJECTED
            order.metadata["reject_reason"] = "Rate limit exceeded"
            self._log_order_event(order, "rate_limit_reject")
            return order

        # Price protection check
        if not self._check_price_protection(order):
            order.status = OrderStatus.REJECTED
            self._log_order_event(order, "price_protection_reject")
            return order

        # Place order with retries
        for attempt in range(self.config.max_retries + 1):
            try:
                self._log_order_event(order, f"submit_attempt_{attempt + 1}")
                result = self.broker.place_order(order)

                if result.status == OrderStatus.FILLED:
                    self._consecutive_failures = 0
                    self._record_rate(time.time())
                    self._log_order_event(result, "filled")
                    return result
                elif result.status == OrderStatus.REJECTED:
                    self._consecutive_failures += 1
                    self._check_circuit_breaker()
                    self._log_order_event(result, "rejected")

                    if attempt < self.config.max_retries:
                        delay = self.config.retry_delay_ms / 1000 * (attempt + 1)
                        logger.info(f"Retrying in {delay:.1f}s (attempt {attempt + 2})")
                        time.sleep(delay)
                    else:
                        return result
                else:
                    self._log_order_event(result, result.status.value.lower())
                    return result

            except Exception as e:
                self._consecutive_failures += 1
                logger.error(f"Order submission error: {e}")
                self._log_order_event(order, "error", {"error": str(e)})
                self._check_circuit_breaker()

                if attempt < self.config.max_retries:
                    delay = self.config.retry_delay_ms / 1000 * (attempt + 1)
                    time.sleep(delay)

        order.status = OrderStatus.REJECTED
        order.metadata["reject_reason"] = "Max retries exhausted"
        return order

    def update_market_prices(self, prices: dict[str, float]) -> None:
        """Update market prices for price protection checks."""
        self.market_prices.update(prices)

    def _check_rate_limit(self) -> bool:
        """Check if order rate is within limits."""
        now = time.time()

        # Clean old timestamps
        while self._second_timestamps and now - self._second_timestamps[0] > 1:
            self._second_timestamps.popleft()
        while self._minute_timestamps and now - self._minute_timestamps[0] > 60:
            self._minute_timestamps.popleft()

        if len(self._second_timestamps) >= self.config.max_orders_per_second:
            logger.warning("Rate limit: per-second limit reached")
            return False

        if len(self._minute_timestamps) >= self.config.max_orders_per_minute:
            logger.warning("Rate limit: per-minute limit reached")
            return False

        return True

    def _record_rate(self, timestamp: float) -> None:
        """Record a successful order timestamp for rate limiting."""
        self._second_timestamps.append(timestamp)
        self._minute_timestamps.append(timestamp)

    def _check_price_protection(self, order: Order) -> bool:
        """Check if the order price is within acceptable range of market price."""
        market_price = self.market_prices.get(order.symbol, 0)
        if market_price <= 0:
            return True  # No market price available, allow

        if order.limit_price is not None and order.limit_price > 0:
            deviation_pct = abs(order.limit_price - market_price) / market_price * 100
            if deviation_pct > self.config.price_protection_pct:
                order.metadata["reject_reason"] = (
                    f"Price protection: limit {order.limit_price:.2f} deviates "
                    f"{deviation_pct:.1f}% from market {market_price:.2f}"
                )
                logger.warning(order.metadata["reject_reason"])
                return False

        return True

    def _check_circuit_breaker(self) -> None:
        """Open circuit breaker if too many consecutive failures."""
        if self._consecutive_failures >= 5:
            self._circuit_open = True
            self._circuit_open_until = time.time() + 60  # 1 minute cooldown
            logger.critical(
                f"CIRCUIT BREAKER OPEN: {self._consecutive_failures} consecutive failures"
            )

    def _log_order_event(
        self,
        order: Order,
        event_type: str,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Log an order event to the audit trail."""
        data = {
            "order_id": order.order_id,
            "symbol": order.symbol,
            "side": order.side,
            "quantity": order.quantity,
            "order_type": order.order_type.value,
            "status": order.status.value,
            "filled_price": order.filled_price,
            **(extra or {}),
        }
        self.audit.log(f"order_{event_type}", data)
