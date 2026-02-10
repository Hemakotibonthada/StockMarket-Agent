"""Opening Range Breakout (ORB) Momentum strategy.

Trades breakouts of the opening range (first N minutes) with ATR-based
filters and time-based stops.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.features.indicators import atr
from src.strategies.base_strategy import BaseStrategy, Signal, SignalType, StrategyState


class ORBMomentum(BaseStrategy):
    """ORB Momentum strategy for intraday trading.

    Parameters (via config):
        orb_window_minutes: Minutes for the opening range (default: 15).
        atr_period: ATR lookback (default: 14).
        atr_multiplier: Minimum ATR ratio for valid breakout (default: 1.5).
        time_stop_minutes: Max time to hold a position (default: 180).
        max_positions: Maximum concurrent positions (default: 5).
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.name = "ORBMomentum"
        self.orb_window = self.config.get("orb_window_minutes", 15)
        self.atr_period = self.config.get("atr_period", 14)
        self.atr_multiplier = self.config.get("atr_multiplier", 1.5)
        self.time_stop_minutes = self.config.get("time_stop_minutes", 180)
        self.max_positions = self.config.get("max_positions", 5)

    def generate_signals(
        self,
        bars: pd.DataFrame,
        state: StrategyState,
        config: dict[str, Any] | None = None,
    ) -> list[Signal]:
        """Generate ORB breakout signals.

        Args:
            bars: Intraday OHLCV bars with 'datetime' and 'symbol' columns.
            state: Current strategy state.
            config: Optional runtime config overrides.

        Returns:
            List of signals.
        """
        signals: list[Signal] = []
        symbols = bars["symbol"].unique() if "symbol" in bars.columns else ["UNKNOWN"]

        for symbol in symbols:
            sym_bars = bars[bars["symbol"] == symbol] if "symbol" in bars.columns else bars

            if len(sym_bars) < self.orb_window:
                continue

            # Check time-stop exits first
            if symbol in state.positions and symbol in state.entry_times:
                entry_time = state.entry_times[symbol]
                current_time = sym_bars["datetime"].iloc[-1] if "datetime" in sym_bars.columns else None
                if current_time and entry_time:
                    elapsed = (current_time - entry_time).total_seconds() / 60
                    if elapsed >= self.time_stop_minutes:
                        signals.append(Signal(
                            symbol=symbol,
                            signal_type=SignalType.EXIT_LONG,
                            price=sym_bars["close"].iloc[-1],
                            timestamp=current_time,
                            metadata={"reason": "time_stop"},
                        ))
                        continue

            # Skip if already in position
            if symbol in state.positions:
                continue

            # Skip if max positions reached
            if len(state.positions) >= self.max_positions:
                continue

            # Compute opening range
            orb_bars = sym_bars.iloc[:self.orb_window]
            orb_high = orb_bars["high"].max()
            orb_low = orb_bars["low"].min()
            orb_range = orb_high - orb_low

            # Current ATR for filter
            current_atr = atr(
                sym_bars["high"], sym_bars["low"], sym_bars["close"],
                period=self.atr_period,
            ).iloc[-1]

            # Filter: ORB range should be meaningful relative to ATR
            if current_atr > 0 and orb_range < current_atr * 0.5:
                continue

            # Check for breakout on the latest bar
            latest = sym_bars.iloc[-1]
            current_time = latest.get("datetime", None)

            # Long breakout
            if latest["close"] > orb_high and latest["volume"] > sym_bars["volume"].mean():
                stop_loss = orb_low
                take_profit = latest["close"] + self.atr_multiplier * current_atr

                signals.append(Signal(
                    symbol=symbol,
                    signal_type=SignalType.ENTRY_LONG,
                    price=latest["close"],
                    timestamp=current_time,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    confidence=min(1.0, orb_range / current_atr) if current_atr > 0 else 0.5,
                    metadata={
                        "orb_high": orb_high,
                        "orb_low": orb_low,
                        "atr": current_atr,
                    },
                ))

            # Short breakout
            elif latest["close"] < orb_low and latest["volume"] > sym_bars["volume"].mean():
                stop_loss = orb_high
                take_profit = latest["close"] - self.atr_multiplier * current_atr

                signals.append(Signal(
                    symbol=symbol,
                    signal_type=SignalType.ENTRY_SHORT,
                    price=latest["close"],
                    timestamp=current_time,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    confidence=min(1.0, orb_range / current_atr) if current_atr > 0 else 0.5,
                    metadata={
                        "orb_high": orb_high,
                        "orb_low": orb_low,
                        "atr": current_atr,
                    },
                ))

        state.bar_count += 1
        return signals
