"""Backtesting engine: vectorized + event-driven hybrid with portfolio accounting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.backtest.costs import CostModel, TransactionCost
from src.backtest.metrics import PerformanceMetrics, compute_all_metrics, compute_drawdown
from src.core.config import AppConfig
from src.core.logging_utils import get_logger
from src.strategies.base_strategy import BaseStrategy, Signal, SignalType, StrategyState

logger = get_logger("backtest.engine")


@dataclass
class Trade:
    """Record of a completed trade."""

    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: int
    entry_time: Any = None
    exit_time: Any = None
    pnl: float = 0.0
    costs: float = 0.0
    net_pnl: float = 0.0
    holding_bars: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Portfolio:
    """Tracks cash, positions, and equity over time."""

    initial_capital: float = 1_000_000.0
    cash: float = 0.0
    positions: dict[str, int] = field(default_factory=dict)  # symbol -> qty
    entry_prices: dict[str, float] = field(default_factory=dict)
    entry_times: dict[str, Any] = field(default_factory=dict)
    entry_bars: dict[str, int] = field(default_factory=dict)
    equity_curve: list[float] = field(default_factory=list)
    trades: list[Trade] = field(default_factory=list)
    total_costs: float = 0.0

    def __post_init__(self) -> None:
        self.cash = self.initial_capital

    def market_value(self, prices: dict[str, float]) -> float:
        """Current total portfolio value."""
        positions_value = sum(
            qty * prices.get(sym, 0) for sym, qty in self.positions.items()
        )
        return self.cash + positions_value

    def exposure(self, prices: dict[str, float]) -> float:
        """Current gross exposure as fraction of equity."""
        mv = self.market_value(prices)
        if mv <= 0:
            return 0.0
        gross = sum(abs(qty) * prices.get(sym, 0) for sym, qty in self.positions.items())
        return gross / mv


class BacktestEngine:
    """Event-driven backtesting engine with portfolio accounting.

    Processes bars chronologically, generates signals via the strategy,
    applies costs/slippage, and tracks portfolio state.
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        config: AppConfig,
        cost_model: CostModel | None = None,
    ):
        self.strategy = strategy
        self.config = config
        self.cost_model = cost_model or CostModel(config.costs, config.slippage)
        self.portfolio = Portfolio(initial_capital=config.risk.initial_capital)
        self.state = strategy.reset()

    def run(self, data: pd.DataFrame) -> BacktestResult:
        """Run the backtest on the provided data.

        Args:
            data: OHLCV DataFrame sorted by time, with optional 'symbol' column.
                  Must have columns: open, high, low, close, volume, and a time
                  column (date or datetime).

        Returns:
            BacktestResult with equity curve, trades, and metrics.
        """
        logger.info(f"Starting backtest: strategy={self.strategy.name}, rows={len(data)}")

        time_col = "datetime" if "datetime" in data.columns else "date"
        timestamps = data[time_col].unique() if time_col in data.columns else range(len(data))

        bar_index = 0
        for ts in timestamps:
            if time_col in data.columns:
                current_bars = data[data[time_col] == ts]
            else:
                current_bars = data.iloc[[bar_index]]

            # Get current prices for portfolio valuation
            prices = {}
            if "symbol" in current_bars.columns:
                for _, row in current_bars.iterrows():
                    prices[row["symbol"]] = row["close"]
            else:
                prices["DEFAULT"] = current_bars["close"].iloc[0]

            # Check stop losses
            self._check_stops(prices, ts, bar_index)

            # Generate signals
            # Build up lookback window for the strategy
            if time_col in data.columns:
                lookback = data[data[time_col] <= ts]
            else:
                lookback = data.iloc[:bar_index + 1]

            signals = self.strategy.generate_signals(lookback, self.state)

            # Process signals
            for signal in signals:
                self._process_signal(signal, bar_index)

            # Record equity
            self.portfolio.equity_curve.append(self.portfolio.market_value(prices))
            bar_index += 1

        # Close all remaining positions at the last known prices
        self._close_all_positions(prices, timestamps[-1] if len(timestamps) > 0 else None, bar_index)

        return self._build_result(data)

    def _process_signal(self, signal: Signal, bar_index: int) -> None:
        """Process a single signal into a portfolio action."""
        symbol = signal.symbol
        price = signal.price

        if signal.signal_type in (SignalType.ENTRY_LONG, SignalType.ENTRY_SHORT):
            # Compute position size (simple: use risk-based sizing)
            quantity = self._compute_position_size(signal)
            if quantity <= 0:
                return

            # Apply slippage
            side = "BUY" if signal.signal_type == SignalType.ENTRY_LONG else "SELL"
            exec_price = self.cost_model.apply_slippage_to_price(price, side)

            # Compute costs
            cost = self.cost_model.compute(exec_price, quantity, side)

            # Check if we have enough cash
            required = exec_price * quantity + cost.total
            if required > self.portfolio.cash:
                quantity = int((self.portfolio.cash * 0.95 - cost.total) / exec_price)
                if quantity <= 0:
                    return
                cost = self.cost_model.compute(exec_price, quantity, side)

            # Execute
            self.portfolio.cash -= exec_price * quantity + cost.total
            self.portfolio.positions[symbol] = self.portfolio.positions.get(symbol, 0) + (
                quantity if side == "BUY" else -quantity
            )
            self.portfolio.entry_prices[symbol] = exec_price
            self.portfolio.entry_times[symbol] = signal.timestamp
            self.portfolio.entry_bars[symbol] = bar_index
            self.portfolio.total_costs += cost.total

            # Update strategy state
            self.state.positions[symbol] = self.portfolio.positions[symbol]
            self.state.entry_prices[symbol] = exec_price
            self.state.entry_times[symbol] = signal.timestamp

            logger.debug(
                f"{side} {quantity} {symbol} @ {exec_price:.2f} "
                f"(cost: {cost.total:.2f})"
            )

        elif signal.signal_type in (SignalType.EXIT_LONG, SignalType.EXIT_SHORT):
            if symbol not in self.portfolio.positions:
                return

            quantity = abs(self.portfolio.positions[symbol])
            side = "SELL" if signal.signal_type == SignalType.EXIT_LONG else "BUY"
            exec_price = self.cost_model.apply_slippage_to_price(price, side)
            cost = self.cost_model.compute(exec_price, quantity, side)

            # Record trade
            entry_price = self.portfolio.entry_prices.get(symbol, price)
            if signal.signal_type == SignalType.EXIT_LONG:
                pnl = (exec_price - entry_price) * quantity
            else:
                pnl = (entry_price - exec_price) * quantity

            net_pnl = pnl - cost.total
            holding_bars = bar_index - self.portfolio.entry_bars.get(symbol, bar_index)

            trade = Trade(
                symbol=symbol,
                side="LONG" if signal.signal_type == SignalType.EXIT_LONG else "SHORT",
                entry_price=entry_price,
                exit_price=exec_price,
                quantity=quantity,
                entry_time=self.portfolio.entry_times.get(symbol),
                exit_time=signal.timestamp,
                pnl=pnl,
                costs=cost.total,
                net_pnl=net_pnl,
                holding_bars=holding_bars,
                metadata=signal.metadata,
            )
            self.portfolio.trades.append(trade)

            # Update portfolio
            self.portfolio.cash += exec_price * quantity - cost.total
            self.portfolio.total_costs += cost.total
            del self.portfolio.positions[symbol]
            self.portfolio.entry_prices.pop(symbol, None)
            self.portfolio.entry_times.pop(symbol, None)
            self.portfolio.entry_bars.pop(symbol, None)

            # Update strategy state
            self.state.positions.pop(symbol, None)
            self.state.entry_prices.pop(symbol, None)
            self.state.entry_times.pop(symbol, None)

            logger.debug(
                f"EXIT {trade.side} {quantity} {symbol} @ {exec_price:.2f} "
                f"PnL={net_pnl:.2f}"
            )

    def _compute_position_size(self, signal: Signal) -> int:
        """Compute position size based on risk config."""
        risk_pct = self.config.risk.risk_per_trade_pct / 100
        portfolio_value = sum(self.portfolio.equity_curve[-1:]) or self.portfolio.initial_capital
        risk_amount = portfolio_value * risk_pct

        if signal.stop_loss and signal.price != signal.stop_loss:
            risk_per_share = abs(signal.price - signal.stop_loss)
            quantity = int(risk_amount / risk_per_share)
        else:
            # Default: allocate up to risk_amount worth of shares
            quantity = int(risk_amount / signal.price) if signal.price > 0 else 0

        return max(0, quantity)

    def _check_stops(self, prices: dict[str, float], timestamp: Any, bar_index: int) -> None:
        """Check and execute stop-loss exits."""
        for symbol in list(self.portfolio.positions.keys()):
            if symbol not in prices:
                continue
            current_price = prices[symbol]
            entry_price = self.portfolio.entry_prices.get(symbol)
            if entry_price is None:
                continue

            pos = self.portfolio.positions[symbol]
            # Check hard stop (5% from entry)
            if pos > 0 and current_price < entry_price * 0.95:
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.EXIT_LONG,
                    price=current_price,
                    timestamp=timestamp,
                    metadata={"reason": "stop_loss"},
                )
                self._process_signal(signal, bar_index)
            elif pos < 0 and current_price > entry_price * 1.05:
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.EXIT_SHORT,
                    price=current_price,
                    timestamp=timestamp,
                    metadata={"reason": "stop_loss"},
                )
                self._process_signal(signal, bar_index)

    def _close_all_positions(self, prices: dict[str, float], timestamp: Any, bar_index: int) -> None:
        """Close all open positions at end of backtest."""
        for symbol in list(self.portfolio.positions.keys()):
            price = prices.get(symbol, 0)
            if price <= 0:
                continue
            pos = self.portfolio.positions[symbol]
            signal_type = SignalType.EXIT_LONG if pos > 0 else SignalType.EXIT_SHORT
            signal = Signal(
                symbol=symbol,
                signal_type=signal_type,
                price=price,
                timestamp=timestamp,
                metadata={"reason": "end_of_backtest"},
            )
            self._process_signal(signal, bar_index)

    def _build_result(self, data: pd.DataFrame) -> "BacktestResult":
        """Build the final backtest result."""
        equity = pd.Series(self.portfolio.equity_curve)
        trade_returns = pd.Series([t.net_pnl / (t.entry_price * t.quantity) for t in self.portfolio.trades if t.quantity > 0])

        metrics = compute_all_metrics(equity, trade_returns)
        metrics.total_trades = len(self.portfolio.trades)
        if self.portfolio.trades:
            metrics.avg_holding_bars = np.mean([t.holding_bars for t in self.portfolio.trades])

        return BacktestResult(
            equity_curve=equity,
            drawdown=compute_drawdown(equity),
            trades=self.portfolio.trades,
            metrics=metrics,
            total_costs=self.portfolio.total_costs,
            config=self.config,
        )


@dataclass
class BacktestResult:
    """Container for backtest results."""

    equity_curve: pd.Series
    drawdown: pd.Series
    trades: list[Trade]
    metrics: PerformanceMetrics
    total_costs: float
    config: AppConfig

    def summary(self) -> dict[str, Any]:
        """Return a summary dict of key metrics."""
        return {
            "total_return_pct": f"{self.metrics.total_return_pct:.2f}%",
            "cagr_pct": f"{self.metrics.cagr_pct:.2f}%",
            "max_drawdown_pct": f"{self.metrics.max_drawdown_pct:.2f}%",
            "sharpe_ratio": f"{self.metrics.sharpe_ratio:.2f}",
            "sortino_ratio": f"{self.metrics.sortino_ratio:.2f}",
            "calmar_ratio": f"{self.metrics.calmar_ratio:.2f}",
            "profit_factor": f"{self.metrics.profit_factor:.2f}",
            "win_rate_pct": f"{self.metrics.win_rate_pct:.1f}%",
            "total_trades": self.metrics.total_trades,
            "total_costs": f"₹{self.total_costs:,.2f}",
            "var_95_pct": f"{self.metrics.var_95_pct:.2f}%",
            "es_95_pct": f"{self.metrics.es_95_pct:.2f}%",
        }
