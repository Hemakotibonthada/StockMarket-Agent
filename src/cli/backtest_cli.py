"""Backtest CLI.

Usage:
    sa-backtest run --config configs/backtest.yaml
    sa-backtest walkforward --config configs/backtest.yaml
    sa-backtest sweep --config configs/backtest.yaml --param lookback --values 10,20,30
    sa-backtest report --results results/backtest_latest.json
"""

from __future__ import annotations

from pathlib import Path
import json

import typer
from rich.console import Console
from rich.table import Table

from src.core.config import load_config
from src.core.logging_utils import get_logger
from src.core.utils import set_seed

app = typer.Typer(name="sa-backtest", help="Backtesting engine CLI.")
console = Console()
logger = get_logger("cli.backtest")


def _load_data(cfg):
    """Load data for backtesting."""
    from src.core.io import read_parquet
    from src.features.feature_sets import compute_base_features
    
    data_path = Path(cfg.data_dir) / "processed"
    if not data_path.exists():
        console.print("[bold red]✗[/] No processed data found. Run `sa-data ingest` first.")
        raise typer.Exit(1)
    
    parquet_files = list(data_path.glob("*.parquet"))
    if not parquet_files:
        console.print("[bold red]✗[/] No parquet files found in processed data.")
        raise typer.Exit(1)
    
    import pandas as pd
    dfs = [read_parquet(f) for f in parquet_files]
    df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
    
    df = compute_base_features(df)
    return df


def _get_strategy(cfg):
    """Instantiate a strategy from config."""
    from src.strategies.orb_momentum import ORBMomentum
    from src.strategies.mean_reversion import MeanReversion
    from src.strategies.pairs_trading import PairsTrading
    
    strategy_name = cfg.strategy
    params = cfg.strategy_params or {}
    
    strategies = {
        "orb_momentum": ORBMomentum,
        "mean_reversion": MeanReversion,
        "pairs_trading": PairsTrading,
    }
    
    cls = strategies.get(strategy_name)
    if cls is None:
        console.print(f"[bold red]✗[/] Unknown strategy: {strategy_name}")
        console.print(f"Available: {list(strategies.keys())}")
        raise typer.Exit(1)
    
    return cls(**params)


@app.command()
def run(
    config: Path = typer.Option("configs/backtest.yaml", help="Backtest config file"),
    output_dir: Path = typer.Option("results", help="Output directory"),
):
    """Run a single backtest."""
    from src.backtest.engine import BacktestEngine

    cfg = load_config(config)
    set_seed(cfg.seed)

    console.print("[bold blue]Running backtest...[/]")
    console.print(f"  Strategy: {cfg.strategy}")
    console.print(f"  Capital:  ₹{cfg.initial_capital:,.0f}")

    df = _load_data(cfg)
    strategy = _get_strategy(cfg)

    engine = BacktestEngine(
        config=cfg,
        strategy=strategy,
    )
    result = engine.run(df)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = result.summary()
    summary_path = output_dir / "backtest_latest.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Print results table
    table = Table(title="Backtest Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    for k, v in summary.items():
        if isinstance(v, float):
            table.add_row(k, f"{v:.4f}")
        else:
            table.add_row(k, str(v))

    console.print(table)
    console.print(f"[bold green]✓[/] Results saved: {summary_path}")

    # Generate HTML report
    from src.backtest.reports import generate_html_report
    report_path = output_dir / "backtest_report.html"
    generate_html_report(result, output_path=report_path)
    console.print(f"[bold green]✓[/] Report: {report_path}")


@app.command()
def walkforward(
    config: Path = typer.Option("configs/backtest.yaml", help="Config file"),
    output_dir: Path = typer.Option("results", help="Output directory"),
):
    """Run walk-forward analysis."""
    from src.backtest.walkforward import run_walk_forward

    cfg = load_config(config)
    set_seed(cfg.seed)

    console.print("[bold blue]Running walk-forward analysis...[/]")

    df = _load_data(cfg)
    strategy = _get_strategy(cfg)

    results = run_walk_forward(
        df=df,
        config=cfg,
        strategy=strategy,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    table = Table(title="Walk-Forward Results")
    table.add_column("Window", style="cyan")
    table.add_column("Trades", style="white")
    table.add_column("Return", style="green")
    table.add_column("Sharpe", style="yellow")

    for i, r in enumerate(results):
        s = r.summary()
        table.add_row(
            f"Window {i + 1}",
            str(s.get("total_trades", 0)),
            f"{s.get('total_return', 0):.2%}",
            f"{s.get('sharpe_ratio', 0):.2f}",
        )

    console.print(table)


@app.command()
def sweep(
    config: Path = typer.Option("configs/backtest.yaml", help="Config file"),
    param: str = typer.Option(..., help="Parameter name to sweep"),
    values: str = typer.Option(..., help="Comma-separated values"),
):
    """Parameter sweep over a strategy parameter."""
    from src.backtest.walkforward import parameter_sweep

    cfg = load_config(config)
    set_seed(cfg.seed)

    value_list = [float(v) if "." in v else int(v) for v in values.split(",")]
    param_grid = {param: value_list}

    console.print(f"[bold blue]Sweeping {param}:[/] {value_list}")

    df = _load_data(cfg)
    strategy = _get_strategy(cfg)

    results = parameter_sweep(
        df=df,
        config=cfg,
        strategy_class=type(strategy),
        param_grid=param_grid,
    )

    table = Table(title=f"Parameter Sweep: {param}")
    table.add_column("Value", style="cyan")
    table.add_column("Return", style="green")
    table.add_column("Sharpe", style="yellow")
    table.add_column("Max DD", style="red")

    for params, r in results:
        s = r.summary()
        table.add_row(
            str(params.get(param)),
            f"{s.get('total_return', 0):.2%}",
            f"{s.get('sharpe_ratio', 0):.2f}",
            f"{s.get('max_drawdown', 0):.2%}",
        )

    console.print(table)


@app.command()
def report(
    results_file: Path = typer.Argument("results/backtest_latest.json", help="Results JSON"),
):
    """Display a saved backtest result."""
    if not results_file.exists():
        console.print(f"[bold red]✗[/] File not found: {results_file}")
        raise typer.Exit(1)

    with open(results_file) as f:
        summary = json.load(f)

    table = Table(title="Backtest Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    for k, v in summary.items():
        if isinstance(v, float):
            table.add_row(k, f"{v:.4f}")
        else:
            table.add_row(k, str(v))

    console.print(table)


def main():
    app()


if __name__ == "__main__":
    main()
