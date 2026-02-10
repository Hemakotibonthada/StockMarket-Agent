"""Paper trading CLI.

Usage:
    sa-paper --config configs/paper_trade.yaml
    sa-paper --config configs/paper_trade.yaml --data data/processed
"""

from __future__ import annotations

from pathlib import Path
import asyncio

import typer
from rich.console import Console
from rich.live import Live
from rich.table import Table

from src.core.config import load_config
from src.core.logging_utils import get_logger
from src.core.utils import set_seed

app = typer.Typer(name="sa-paper", help="Paper trading simulation.")
console = Console()
logger = get_logger("cli.paper")


@app.command()
def run(
    config: Path = typer.Option("configs/paper_trade.yaml", help="Paper trade config"),
    data_dir: Path = typer.Option("data/processed", help="Processed data directory"),
    speed: float = typer.Option(1.0, help="Replay speed multiplier (higher = faster)"),
    max_bars: int = typer.Option(0, help="Max bars to process (0 = all)"),
):
    """Run paper trading simulation with simulated market data."""
    import pandas as pd
    from src.core.io import read_parquet
    from src.exec.paper_broker import PaperBroker
    from src.risk.limits import RiskLimiter
    from src.risk.tripwires import TripwireMonitor
    from src.features.feature_sets import compute_base_features
    from src.live.event_loop import TradingEventLoop

    cfg = load_config(config)
    set_seed(cfg.seed)

    console.print("[bold blue]Starting paper trading session[/]")
    console.print(f"  Strategy: {cfg.strategy}")
    console.print(f"  Capital:  ₹{cfg.initial_capital:,.0f}")
    console.print(f"  Broker:   paper")

    # Load data
    parquet_files = list(Path(data_dir).glob("*.parquet"))
    if not parquet_files:
        console.print("[bold red]✗[/] No data found. Run `sa-data ingest` first.")
        raise typer.Exit(1)

    dfs = [read_parquet(f) for f in parquet_files]
    df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
    df = compute_base_features(df)

    if max_bars > 0:
        df = df.head(max_bars)

    console.print(f"  Data:     {len(df)} bars across {df['symbol'].nunique()} symbols")

    # Instantiate components
    from src.cli.backtest_cli import _get_strategy
    strategy = _get_strategy(cfg)

    broker = PaperBroker(
        initial_capital=cfg.initial_capital,
        cost_config=cfg.costs,
        slippage_config=cfg.slippage,
    )

    risk_limiter = RiskLimiter(
        max_loss_per_trade=cfg.risk.max_loss_per_trade,
        max_daily_loss=cfg.risk.max_daily_loss,
        max_weekly_loss=cfg.risk.max_weekly_loss,
        max_drawdown=cfg.risk.max_drawdown,
    )

    tripwire = TripwireMonitor(config=cfg.tripwire) if cfg.tripwire else None

    # Run event loop
    event_loop = TradingEventLoop(
        strategy=strategy,
        broker=broker,
        risk_limiter=risk_limiter,
        tripwire=tripwire,
        config=cfg,
    )

    try:
        asyncio.run(event_loop.run_replay(df, speed=speed))
    except KeyboardInterrupt:
        console.print("\n[bold yellow]⚠ Paper trading stopped by user[/]")

    # Print summary
    positions = broker.get_positions()
    table = Table(title="Paper Trading Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Final Capital", f"₹{broker.cash:,.2f}")
    table.add_row("Open Positions", str(len(positions)))
    table.add_row("Total Orders", str(len(broker.order_history)))
    table.add_row("P&L", f"₹{broker.cash - cfg.initial_capital:,.2f}")
    console.print(table)

    console.print("[bold green]✓[/] Paper trading complete.")


@app.command()
def status(
    config: Path = typer.Option("configs/paper_trade.yaml", help="Config file"),
):
    """Show current paper trading configuration."""
    cfg = load_config(config)

    table = Table(title="Paper Trading Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Strategy", cfg.strategy)
    table.add_row("Capital", f"₹{cfg.initial_capital:,.0f}")
    table.add_row("Broker", cfg.broker)
    table.add_row("Max Daily Loss", f"₹{cfg.risk.max_daily_loss:,.0f}")
    table.add_row("Max Drawdown", f"{cfg.risk.max_drawdown:.0%}")

    console.print(table)


def main():
    app()


if __name__ == "__main__":
    main()
