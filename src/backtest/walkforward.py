"""Walk-forward validation and robustness testing."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestEngine, BacktestResult
from src.core.config import AppConfig
from src.core.logging_utils import get_logger
from src.strategies.base_strategy import BaseStrategy

logger = get_logger("backtest.walkforward")


def walk_forward_split(
    data: pd.DataFrame,
    train_years: list[int],
    validate_years: list[int],
    test_years: list[int],
    date_col: str = "date",
) -> dict[str, pd.DataFrame]:
    """Split data into train/validate/test sets by year.

    Returns:
        Dict with keys 'train', 'validate', 'test'.
    """
    data = data.copy()
    if date_col in data.columns:
        data[date_col] = pd.to_datetime(data[date_col])
        year = data[date_col].dt.year
    else:
        raise ValueError(f"Date column '{date_col}' not found")

    return {
        "train": data[year.isin(train_years)].copy(),
        "validate": data[year.isin(validate_years)].copy(),
        "test": data[year.isin(test_years)].copy(),
    }


def run_walk_forward(
    strategy: BaseStrategy,
    data: pd.DataFrame,
    config: AppConfig,
    date_col: str = "date",
) -> dict[str, BacktestResult]:
    """Run walk-forward backtest across train/validate/test periods.

    Returns:
        Dict with BacktestResult for each period.
    """
    splits = walk_forward_split(
        data,
        config.walkforward.train_years,
        config.walkforward.validate_years,
        config.walkforward.test_years,
        date_col,
    )

    results = {}
    for period, period_data in splits.items():
        if period_data.empty:
            logger.warning(f"No data for {period} period")
            continue

        logger.info(f"Running {period} backtest: {len(period_data)} bars")
        engine = BacktestEngine(strategy, config)
        results[period] = engine.run(period_data)
        logger.info(
            f"{period.upper()} results: "
            f"return={results[period].metrics.total_return_pct:.2f}%, "
            f"sharpe={results[period].metrics.sharpe_ratio:.2f}"
        )

    return results


def monte_carlo_trades(
    trades: list,
    n_simulations: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Monte Carlo simulation by reordering trades.

    Shuffles trade sequences to estimate distribution of outcomes.

    Returns:
        Dict with percentile statistics for final PnL.
    """
    if not trades:
        return {"p5": 0, "p25": 0, "p50": 0, "p75": 0, "p95": 0}

    rng = np.random.RandomState(seed)
    pnls = [t.net_pnl for t in trades]
    final_values = []

    for _ in range(n_simulations):
        shuffled = rng.permutation(pnls)
        cumulative = np.cumsum(shuffled)
        final_values.append(cumulative[-1])

    final_values = np.array(final_values)
    return {
        "p5": float(np.percentile(final_values, 5)),
        "p25": float(np.percentile(final_values, 25)),
        "p50": float(np.percentile(final_values, 50)),
        "p75": float(np.percentile(final_values, 75)),
        "p95": float(np.percentile(final_values, 95)),
        "mean": float(final_values.mean()),
        "std": float(final_values.std()),
    }


def parameter_sweep(
    strategy_class: type,
    data: pd.DataFrame,
    config: AppConfig,
    param_grid: dict[str, list[Any]],
) -> pd.DataFrame:
    """Run backtests across parameter combinations.

    Args:
        strategy_class: Strategy class to instantiate.
        data: OHLCV data.
        config: Base config.
        param_grid: Dict of param_name -> list of values to try.

    Returns:
        DataFrame of results with columns for each param + metrics.
    """
    import itertools

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    results_list = []

    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        logger.info(f"Sweep: {params}")

        # Override strategy params
        strategy_config = {**config.strategy_params, **params}
        strategy = strategy_class(config=strategy_config)

        engine = BacktestEngine(strategy, config)
        result = engine.run(data)

        row = {**params}
        row["total_return_pct"] = result.metrics.total_return_pct
        row["sharpe_ratio"] = result.metrics.sharpe_ratio
        row["max_drawdown_pct"] = result.metrics.max_drawdown_pct
        row["total_trades"] = result.metrics.total_trades
        row["profit_factor"] = result.metrics.profit_factor
        results_list.append(row)

    return pd.DataFrame(results_list)
