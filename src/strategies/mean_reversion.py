"""Mean Reversion strategy using z-score and RSI filters."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.features.indicators import rsi, zscore
from src.strategies.base_strategy import BaseStrategy, Signal, SignalType, StrategyState


class MeanReversion(BaseStrategy):
    """Mean Reversion strategy (EOD or 15-min bars).

    Parameters (via config):
        zscore_window: Lookback for z-score (default: 20).
        zscore_entry: Z-score threshold for entry (default: 2.0).
        zscore_exit: Z-score threshold for exit (default: 0.5).
        rsi_period: RSI period (default: 14).
        rsi_oversold: RSI oversold threshold (default: 30).
        rsi_overbought: RSI overbought threshold (default: 70).
        max_positions: Maximum concurrent positions (default: 5).
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.name = "MeanReversion"
        self.zscore_window = self.config.get("zscore_window", 20)
        self.zscore_entry = self.config.get("zscore_entry", 2.0)
        self.zscore_exit = self.config.get("zscore_exit", 0.5)
        self.rsi_period = self.config.get("rsi_period", 14)
        self.rsi_oversold = self.config.get("rsi_oversold", 30)
        self.rsi_overbought = self.config.get("rsi_overbought", 70)
        self.max_positions = self.config.get("max_positions", 5)

    def generate_signals(
        self,
        bars: pd.DataFrame,
        state: StrategyState,
        config: dict[str, Any] | None = None,
    ) -> list[Signal]:
        """Generate mean reversion signals based on z-score and RSI.

        Args:
            bars: OHLCV DataFrame with 'symbol' column.
            state: Current strategy state.
            config: Optional runtime config overrides.

        Returns:
            List of signals.
        """
        signals: list[Signal] = []
        symbols = bars["symbol"].unique() if "symbol" in bars.columns else ["UNKNOWN"]

        for symbol in symbols:
            sym_bars = bars[bars["symbol"] == symbol] if "symbol" in bars.columns else bars

            if len(sym_bars) < self.zscore_window + 5:
                continue

            # Compute indicators
            z = zscore(sym_bars["close"], window=self.zscore_window)
            r = rsi(sym_bars["close"], period=self.rsi_period)

            current_z = z.iloc[-1]
            current_rsi = r.iloc[-1]
            current_price = sym_bars["close"].iloc[-1]
            current_time = sym_bars["datetime"].iloc[-1] if "datetime" in sym_bars.columns else None

            # Exit logic
            if symbol in state.positions:
                pos = state.positions[symbol]
                if pos > 0 and current_z >= -self.zscore_exit:
                    # Exit long: z-score has reverted
                    signals.append(Signal(
                        symbol=symbol,
                        signal_type=SignalType.EXIT_LONG,
                        price=current_price,
                        timestamp=current_time,
                        metadata={"zscore": current_z, "rsi": current_rsi, "reason": "mean_revert"},
                    ))
                elif pos < 0 and current_z <= self.zscore_exit:
                    # Exit short: z-score has reverted
                    signals.append(Signal(
                        symbol=symbol,
                        signal_type=SignalType.EXIT_SHORT,
                        price=current_price,
                        timestamp=current_time,
                        metadata={"zscore": current_z, "rsi": current_rsi, "reason": "mean_revert"},
                    ))
                continue

            # Entry logic – skip if at max positions
            if len(state.positions) >= self.max_positions:
                continue

            # Long entry: deeply oversold
            if current_z <= -self.zscore_entry and current_rsi <= self.rsi_oversold:
                stop_loss = current_price * 0.97  # 3% stop
                signals.append(Signal(
                    symbol=symbol,
                    signal_type=SignalType.ENTRY_LONG,
                    price=current_price,
                    timestamp=current_time,
                    stop_loss=stop_loss,
                    confidence=min(1.0, abs(current_z) / 3.0),
                    metadata={"zscore": current_z, "rsi": current_rsi},
                ))

            # Short entry: deeply overbought
            elif current_z >= self.zscore_entry and current_rsi >= self.rsi_overbought:
                stop_loss = current_price * 1.03  # 3% stop
                signals.append(Signal(
                    symbol=symbol,
                    signal_type=SignalType.ENTRY_SHORT,
                    price=current_price,
                    timestamp=current_time,
                    stop_loss=stop_loss,
                    confidence=min(1.0, abs(current_z) / 3.0),
                    metadata={"zscore": current_z, "rsi": current_rsi},
                ))

        state.bar_count += 1
        return signals
