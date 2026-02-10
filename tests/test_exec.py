"""Tests for execution layer: paper broker, order routing."""

from __future__ import annotations

import pytest

from src.core.config import CostsConfig, SlippageConfig
from src.exec.broker_base import Order, OrderStatus, OrderType, Position
from src.exec.paper_broker import PaperBroker


class TestOrderDataclass:
    def test_default_order(self):
        o = Order()
        assert o.status == OrderStatus.PENDING
        assert o.side == "BUY"
        assert o.quantity == 0

    def test_order_with_values(self):
        o = Order(symbol="RELIANCE", side="SELL", quantity=100, order_type=OrderType.LIMIT, limit_price=2500.0)
        assert o.symbol == "RELIANCE"
        assert o.order_type == OrderType.LIMIT
        assert o.limit_price == 2500.0


class TestPosition:
    def test_default_position(self):
        p = Position()
        assert p.quantity == 0
        assert p.avg_price == 0.0


class TestPaperBroker:
    def _make_broker(self, capital: float = 1_000_000, seed: int = 42) -> PaperBroker:
        return PaperBroker(
            initial_capital=capital,
            costs_config=CostsConfig(brokerage_bps=3, stt_bps=2, gst_bps=1.8, stamp_bps=1, sebi_bps=0.1),
            slippage_config=SlippageConfig(mode="fixed", bps_mean=0.0, bps_std=0.0),
            seed=seed,
        )

    def test_initial_state(self):
        broker = self._make_broker()
        assert broker.cash == 1_000_000
        assert broker.get_balance() == 1_000_000
        assert broker.get_positions() == []

    def test_buy_order_fills(self):
        broker = self._make_broker()
        broker.set_market_prices({"TEST": 100.0})

        order = Order(symbol="TEST", side="BUY", quantity=100)
        filled = broker.place_order(order)

        assert filled.status == OrderStatus.FILLED
        assert filled.filled_quantity == 100
        assert filled.filled_price > 0

    def test_buy_reduces_cash(self):
        broker = self._make_broker()
        broker.set_market_prices({"TEST": 100.0})

        order = Order(symbol="TEST", side="BUY", quantity=100)
        broker.place_order(order)

        assert broker.cash < 1_000_000

    def test_sell_increases_cash(self):
        broker = self._make_broker()
        broker.set_market_prices({"TEST": 100.0})

        # Buy first
        buy = Order(symbol="TEST", side="BUY", quantity=100)
        broker.place_order(buy)
        cash_after_buy = broker.cash

        # Sell
        sell = Order(symbol="TEST", side="SELL", quantity=100)
        broker.place_order(sell)

        assert broker.cash > cash_after_buy

    def test_reject_no_price(self):
        broker = self._make_broker()
        # No market price set
        order = Order(symbol="UNKNOWN", side="BUY", quantity=100)
        filled = broker.place_order(order)
        assert filled.status == OrderStatus.REJECTED

    def test_reject_insufficient_funds(self):
        broker = self._make_broker(capital=100)
        broker.set_market_prices({"EXPENSIVE": 10_000.0})

        order = Order(symbol="EXPENSIVE", side="BUY", quantity=100)
        filled = broker.place_order(order)
        # Should either reject or reduce quantity
        assert filled.status in (OrderStatus.REJECTED, OrderStatus.FILLED)
        if filled.status == OrderStatus.FILLED:
            assert filled.filled_quantity < 100

    def test_position_tracking(self):
        broker = self._make_broker()
        broker.set_market_prices({"TEST": 100.0})

        order = Order(symbol="TEST", side="BUY", quantity=50)
        broker.place_order(order)

        positions = broker.get_positions()
        assert len(positions) == 1
        assert positions[0].symbol == "TEST"
        assert positions[0].quantity == 50

    def test_buy_then_sell_closes_position(self):
        broker = self._make_broker()
        broker.set_market_prices({"TEST": 100.0})

        broker.place_order(Order(symbol="TEST", side="BUY", quantity=100))
        broker.place_order(Order(symbol="TEST", side="SELL", quantity=100))

        positions = broker.get_positions()
        assert len(positions) == 0  # Position should be closed

    def test_equity_calculation(self):
        broker = self._make_broker()
        broker.set_market_prices({"TEST": 100.0})

        # With no positions, equity should equal cash
        assert broker.get_equity() == broker.cash

        # After buying, equity should include position value
        broker.place_order(Order(symbol="TEST", side="BUY", quantity=100))
        equity = broker.get_equity()
        # equity = cash + position_value, should be close to initial capital
        assert abs(equity - 1_000_000) < 5000  # Within transaction costs

    def test_cancel_order(self):
        broker = self._make_broker()
        broker.set_market_prices({"TEST": 100.0})

        order = Order(symbol="TEST", side="BUY", quantity=100)
        filled = broker.place_order(order)

        # Cancel should work (even if already filled in paper mode)
        result = broker.cancel_order(filled.order_id)
        assert result is True

    def test_cancel_unknown_order(self):
        broker = self._make_broker()
        result = broker.cancel_order("nonexistent")
        assert result is False

    def test_summary(self):
        broker = self._make_broker()
        broker.set_market_prices({"TEST": 100.0})
        broker.place_order(Order(symbol="TEST", side="BUY", quantity=100))

        s = broker.summary()
        assert "cash" in s
        assert "equity" in s
        assert "n_positions" in s
        assert s["n_orders"] == 1

    def test_multiple_symbols(self):
        broker = self._make_broker()
        broker.set_market_prices({"A": 100.0, "B": 200.0})

        broker.place_order(Order(symbol="A", side="BUY", quantity=50))
        broker.place_order(Order(symbol="B", side="BUY", quantity=25))

        positions = broker.get_positions()
        symbols = {p.symbol for p in positions}
        assert symbols == {"A", "B"}
