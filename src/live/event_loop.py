"""Asyncio event loop for live/paper trading sessions."""

from __future__ import annotations

import asyncio
import signal
import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from src.core.clocks import is_market_open, now_ist, seconds_to_close
from src.core.config import AppConfig
from src.core.logging_utils import AuditLogger, get_logger
from src.exec.broker_base import BrokerBase, Order, OrderType
from src.exec.router import OrderRouter
from src.risk.limits import RiskLimiter
from src.risk.tripwires import TripwireMonitor
from src.strategies.base_strategy import BaseStrategy, Signal, SignalType, StrategyState

logger = get_logger("live.event_loop")


class TradingEventLoop:
    """Asyncio-based event loop for live/paper trading.

    Processes bars at each interval, generates signals, routes orders,
    and enforces risk limits with health monitoring.
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        broker: BrokerBase,
        config: AppConfig,
        router: OrderRouter | None = None,
        risk_limiter: RiskLimiter | None = None,
        tripwire: TripwireMonitor | None = None,
        audit_logger: AuditLogger | None = None,
    ):
        self.strategy = strategy
        self.broker = broker
        self.config = config
        self.audit = audit_logger or AuditLogger(config.logging.get("audit_file", "./logs/audit.jsonl"))
        self.router = router or OrderRouter(broker, config.order_router, self.audit)
        self.risk_limiter = risk_limiter or RiskLimiter(config.risk, self.audit)
        self.tripwire = tripwire or TripwireMonitor(config.tripwires, self.audit)
        self.state = strategy.reset()

        self._running = False
        self._bar_buffer: list[dict[str, Any]] = []
        self._current_prices: dict[str, float] = {}

    async def run(self, bar_source: Any = None) -> None:
        """Start the trading event loop.

        Args:
            bar_source: Optional async iterator of bar dicts.
                        If None, uses simulated bars at the configured interval.
        """
        logger.info(f"Starting trading loop: strategy={self.strategy.name}")
        self._running = True

        # Initialize risk limiter
        equity = self.broker.get_balance()
        self.risk_limiter.initialize(equity)

        # Register shutdown handler
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._shutdown)
            except NotImplementedError:
                pass  # Windows doesn't support signal handlers in event loops

        self.audit.log("session_start", {
            "strategy": self.strategy.name,
            "broker": type(self.broker).__name__,
            "initial_equity": equity,
        })

        try:
            if bar_source:
                async for bar in bar_source:
                    if not self._running:
                        break
                    await self._process_bar(bar)
            else:
                # Simulated interval loop
                interval_seconds = self._parse_interval(self.config.bar_interval)
                while self._running:
                    if not is_market_open():
                        logger.info("Market closed. Waiting...")
                        await asyncio.sleep(60)
                        continue

                    if seconds_to_close() < interval_seconds:
                        logger.info("Near market close. Running EOD procedures.")
                        await self._end_of_day()
                        break

                    start_time = time.time()

                    # Generate simulated bar (in paper mode, bars come from stored data)
                    bar = self._generate_simulated_bar()
                    if bar:
                        await self._process_bar(bar)

                    elapsed = time.time() - start_time
                    sleep_time = max(0, interval_seconds - elapsed)
                    await asyncio.sleep(sleep_time)

        except Exception as e:
            logger.error(f"Event loop error: {e}")
            self.tripwire.record_exception()
        finally:
            await self._end_of_day()
            self.audit.log("session_end", {
                "strategy": self.strategy.name,
                "final_equity": self.broker.get_balance(),
            })

        logger.info("Trading event loop stopped")

    async def _process_bar(self, bar: dict[str, Any]) -> None:
        """Process a single bar: update prices, generate signals, route orders."""
        # Update current prices
        symbol = bar.get("symbol", "DEFAULT")
        price = bar.get("close", 0)
        self._current_prices[symbol] = price

        if hasattr(self.broker, "set_market_prices"):
            self.broker.set_market_prices(self._current_prices)

        self.router.update_market_prices(self._current_prices)

        # Record feed update for tripwire
        self.tripwire.record_feed_update()

        # Build bar DataFrame for strategy
        self._bar_buffer.append(bar)
        bars_df = pd.DataFrame(self._bar_buffer[-500:])  # Keep last 500 bars

        # Check tripwires
        if self.tripwire.is_tripped:
            logger.warning("Tripwire tripped — skipping signal generation")
            return

        # Generate signals
        try:
            start = time.time()
            signals = self.strategy.generate_signals(bars_df, self.state)
            latency_ms = (time.time() - start) * 1000
            self.tripwire.record_latency(latency_ms)
        except Exception as e:
            logger.error(f"Strategy error: {e}")
            self.tripwire.record_exception()
            return

        # Process signals
        for sig in signals:
            await self._process_signal(sig)

        self.state.bar_count += 1

    async def _process_signal(self, sig: Signal) -> None:
        """Convert a signal into an order and route it."""
        # Check risk limits
        equity = self.broker.get_balance()
        risk_amount = equity * self.config.risk.risk_per_trade_pct / 100

        allowed, reason = self.risk_limiter.check_trade_allowed(risk_amount, equity)
        if not allowed:
            logger.warning(f"Signal rejected by risk: {reason}")
            self.tripwire.record_reject()
            return

        # Determine side and quantity
        if sig.signal_type in (SignalType.ENTRY_LONG, SignalType.EXIT_SHORT):
            side = "BUY"
        else:
            side = "SELL"

        # Compute quantity
        from src.risk.sizing import fixed_fraction_size
        risk_pct = self.config.risk.risk_per_trade_pct / 100
        stop = sig.stop_loss or sig.price * 0.97
        quantity = fixed_fraction_size(equity, risk_pct, sig.price, stop)

        if quantity <= 0:
            return

        order = Order(
            symbol=sig.symbol,
            side=side,
            quantity=quantity,
            order_type=OrderType.MARKET,
        )

        # Route through safety checks
        result = self.router.submit_order(order)

        if result.status.value == "FILLED":
            self.tripwire.record_fill()
            # Update strategy state
            from src.strategies.base_strategy import Side
            self.strategy.on_fill(
                sig.symbol,
                Side.BUY if side == "BUY" else Side.SELL,
                result.filled_price,
                result.filled_quantity,
                self.state,
            )
            logger.info(f"Order filled: {side} {quantity} {sig.symbol} @ {result.filled_price:.2f}")
        else:
            self.tripwire.record_reject()
            logger.warning(f"Order not filled: {result.metadata}")

    async def _end_of_day(self) -> None:
        """End-of-day procedures: flatten positions, log summary."""
        logger.info("Running end-of-day procedures")
        # In a real implementation, you'd flatten intraday positions here
        equity = self.broker.get_balance()
        self.audit.log("end_of_day", {
            "equity": equity,
            "bar_count": self.state.bar_count,
            "risk_summary": self.risk_limiter.summary(),
            "tripwire_summary": self.tripwire.summary(),
        })

    def _shutdown(self) -> None:
        """Handle graceful shutdown."""
        logger.info("Shutdown signal received")
        self._running = False

    def _generate_simulated_bar(self) -> dict[str, Any] | None:
        """Generate a simulated bar for paper trading (placeholder)."""
        # In practice, this would be fed by a data source
        return None

    @staticmethod
    def _parse_interval(interval: str) -> int:
        """Parse bar interval string to seconds."""
        unit_map = {"s": 1, "min": 60, "h": 3600, "D": 86400}
        for unit, secs in unit_map.items():
            if interval.endswith(unit):
                try:
                    return int(interval[: -len(unit)]) * secs
                except ValueError:
                    pass
        return 300  # Default 5 minutes
