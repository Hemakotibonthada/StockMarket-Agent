"""Pairs Trading strategy with cointegration check and spread normalization."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.features.indicators import zscore
from src.strategies.base_strategy import BaseStrategy, Signal, SignalType, StrategyState


def test_cointegration(series_a: pd.Series, series_b: pd.Series) -> tuple[float, float]:
    """Simple Engle-Granger cointegration test (OLS + ADF on residuals).

    Returns:
        (adf_stat, p_value) from the ADF test on the spread residuals.
    """
    from statsmodels.tsa.stattools import adfuller

    # OLS regression: a = alpha + beta * b + epsilon
    a = series_a.dropna().values
    b = series_b.dropna().values
    min_len = min(len(a), len(b))
    a, b = a[:min_len], b[:min_len]

    if min_len < 30:
        return 0.0, 1.0

    slope, intercept, _, _, _ = stats.linregress(b, a)
    spread = a - (slope * b + intercept)

    result = adfuller(spread, maxlag=10, autolag="AIC")
    return float(result[0]), float(result[1])


class PairsTrading(BaseStrategy):
    """Pairs Trading strategy.

    Identifies cointegrated pairs, normalizes the spread, and trades
    mean reversion of the spread.

    Parameters (via config):
        pairs: List of [symbol_a, symbol_b] pairs to trade.
        coint_pvalue: Max p-value for cointegration (default: 0.05).
        zscore_entry: Z-score entry threshold (default: 2.0).
        zscore_exit: Z-score exit threshold (default: 0.5).
        lookback: Lookback window for spread z-score (default: 60).
        max_positions: Maximum number of active pair positions (default: 3).
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.name = "PairsTrading"
        self.pairs: list[list[str]] = self.config.get("pairs", [])
        self.coint_pvalue = self.config.get("coint_pvalue", 0.05)
        self.zscore_entry = self.config.get("zscore_entry", 2.0)
        self.zscore_exit = self.config.get("zscore_exit", 0.5)
        self.lookback = self.config.get("lookback", 60)
        self.max_positions = self.config.get("max_positions", 3)

    def _compute_spread(
        self,
        series_a: pd.Series,
        series_b: pd.Series,
    ) -> tuple[pd.Series, float, float]:
        """Compute the spread and hedge ratio between two series."""
        a = series_a.values
        b = series_b.values
        min_len = min(len(a), len(b))
        a, b = a[:min_len], b[:min_len]

        slope, intercept, _, _, _ = stats.linregress(b, a)
        spread = pd.Series(a - (slope * b + intercept), index=series_a.index[:min_len])
        return spread, slope, intercept

    def generate_signals(
        self,
        bars: pd.DataFrame,
        state: StrategyState,
        config: dict[str, Any] | None = None,
    ) -> list[Signal]:
        """Generate pairs trading signals.

        Args:
            bars: OHLCV DataFrame with 'symbol' and time columns.
            state: Current strategy state.
            config: Optional runtime overrides.

        Returns:
            List of signals (may include hedged entry/exit pairs).
        """
        signals: list[Signal] = []

        if "symbol" not in bars.columns:
            return signals

        for pair in self.pairs:
            if len(pair) != 2:
                continue

            sym_a, sym_b = pair
            a_bars = bars[bars["symbol"] == sym_a]
            b_bars = bars[bars["symbol"] == sym_b]

            if len(a_bars) < self.lookback or len(b_bars) < self.lookback:
                continue

            a_close = a_bars["close"].reset_index(drop=True)
            b_close = b_bars["close"].reset_index(drop=True)

            # Cointegration check
            _, pvalue = test_cointegration(a_close, b_close)
            if pvalue > self.coint_pvalue:
                continue

            # Compute spread and z-score
            spread, hedge_ratio, intercept = self._compute_spread(a_close, b_close)
            z = zscore(spread, window=self.lookback)
            current_z = z.iloc[-1]

            pair_key = f"{sym_a}/{sym_b}"
            current_price_a = a_close.iloc[-1]
            current_price_b = b_close.iloc[-1]
            current_time = (
                a_bars["datetime"].iloc[-1] if "datetime" in a_bars.columns else None
            )

            # Exit logic
            if pair_key in state.positions:
                if abs(current_z) <= self.zscore_exit:
                    # Close both legs
                    pos = state.positions[pair_key]
                    if pos > 0:
                        signals.append(Signal(
                            symbol=sym_a, signal_type=SignalType.EXIT_LONG,
                            price=current_price_a, timestamp=current_time,
                            metadata={"pair": pair_key, "zscore": current_z},
                        ))
                        signals.append(Signal(
                            symbol=sym_b, signal_type=SignalType.EXIT_SHORT,
                            price=current_price_b, timestamp=current_time,
                            metadata={"pair": pair_key, "zscore": current_z, "hedge_ratio": hedge_ratio},
                        ))
                    else:
                        signals.append(Signal(
                            symbol=sym_a, signal_type=SignalType.EXIT_SHORT,
                            price=current_price_a, timestamp=current_time,
                            metadata={"pair": pair_key, "zscore": current_z},
                        ))
                        signals.append(Signal(
                            symbol=sym_b, signal_type=SignalType.EXIT_LONG,
                            price=current_price_b, timestamp=current_time,
                            metadata={"pair": pair_key, "zscore": current_z, "hedge_ratio": hedge_ratio},
                        ))
                continue

            # Entry logic
            active_pairs = sum(1 for k in state.positions if "/" in k)
            if active_pairs >= self.max_positions:
                continue

            if current_z <= -self.zscore_entry:
                # Spread is low: buy A, sell B
                signals.append(Signal(
                    symbol=sym_a, signal_type=SignalType.ENTRY_LONG,
                    price=current_price_a, timestamp=current_time,
                    confidence=min(1.0, abs(current_z) / 3.0),
                    metadata={"pair": pair_key, "zscore": current_z, "hedge_ratio": hedge_ratio},
                ))
                signals.append(Signal(
                    symbol=sym_b, signal_type=SignalType.ENTRY_SHORT,
                    price=current_price_b, timestamp=current_time,
                    confidence=min(1.0, abs(current_z) / 3.0),
                    metadata={"pair": pair_key, "zscore": current_z, "hedge_ratio": hedge_ratio},
                ))

            elif current_z >= self.zscore_entry:
                # Spread is high: sell A, buy B
                signals.append(Signal(
                    symbol=sym_a, signal_type=SignalType.ENTRY_SHORT,
                    price=current_price_a, timestamp=current_time,
                    confidence=min(1.0, abs(current_z) / 3.0),
                    metadata={"pair": pair_key, "zscore": current_z, "hedge_ratio": hedge_ratio},
                ))
                signals.append(Signal(
                    symbol=sym_b, signal_type=SignalType.ENTRY_LONG,
                    price=current_price_b, timestamp=current_time,
                    confidence=min(1.0, abs(current_z) / 3.0),
                    metadata={"pair": pair_key, "zscore": current_z, "hedge_ratio": hedge_ratio},
                ))

        state.bar_count += 1
        return signals
