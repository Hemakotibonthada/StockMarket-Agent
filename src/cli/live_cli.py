"""Live trading CLI.

Usage:
    sa-live --config configs/live_trade.yaml --confirm-live true
    
IMPORTANT: Live trading requires explicit confirmation via --confirm-live flag.
"""

from __future__ import annotations

from pathlib import Path
import asyncio

import typer
from rich.console import Console
from rich.panel import Panel

from src.core.config import load_config
from src.core.logging_utils import get_logger
from src.core.utils import set_seed

app = typer.Typer(name="sa-live", help="Live trading (requires --confirm-live).")
console = Console()
logger = get_logger("cli.live")


LIVE_WARNING = """
[bold red]⚠  LIVE TRADING WARNING  ⚠[/bold red]

You are about to start LIVE TRADING with REAL MONEY.

This will:
  • Connect to your broker account
  • Place REAL orders on the exchange
  • Risk REAL capital

Make sure you have:
  ✓ Tested thoroughly in paper mode
  ✓ Set appropriate risk limits
  ✓ Configured your broker API credentials in .env
  ✓ Reviewed RISK_POLICY.md and COMPLIANCE.md

[bold yellow]The authors accept NO LIABILITY for trading losses.[/bold yellow]
"""


@app.command()
def start(
    config: Path = typer.Option("configs/live_trade.yaml", help="Live trading config"),
    confirm_live: bool = typer.Option(False, "--confirm-live", help="REQUIRED: Confirm live trading"),
    dry_run: bool = typer.Option(False, help="Connect but don't place orders"),
):
    """Start live trading session."""
    cfg = load_config(config)
    set_seed(cfg.seed)

    # Safety gate: require explicit confirmation
    if not confirm_live:
        console.print(Panel(LIVE_WARNING, title="LIVE TRADING", border_style="red"))
        console.print("[bold red]✗[/] You must pass --confirm-live true to start live trading.")
        console.print("  For paper trading, use: [cyan]sa-paper --config configs/paper_trade.yaml[/]")
        raise typer.Exit(1)

    # Double-check with interactive confirmation
    console.print(Panel(LIVE_WARNING, title="LIVE TRADING", border_style="red"))
    if not typer.confirm("Are you ABSOLUTELY sure you want to trade with real money?"):
        console.print("[yellow]Aborted.[/]")
        raise typer.Exit(0)

    console.print("[bold blue]Initializing live trading...[/]")
    console.print(f"  Strategy: {cfg.strategy}")
    console.print(f"  Broker:   {cfg.broker}")
    console.print(f"  Dry Run:  {dry_run}")

    # Load broker adapter
    broker = _get_broker(cfg)
    if broker is None:
        raise typer.Exit(1)

    # Instantiate components
    from src.cli.backtest_cli import _get_strategy
    from src.risk.limits import RiskLimiter
    from src.risk.tripwires import TripwireMonitor
    from src.live.event_loop import TradingEventLoop
    from src.live.health import HealthMonitor

    strategy = _get_strategy(cfg)

    risk_limiter = RiskLimiter(
        max_loss_per_trade=cfg.risk.max_loss_per_trade,
        max_daily_loss=cfg.risk.max_daily_loss,
        max_weekly_loss=cfg.risk.max_weekly_loss,
        max_drawdown=cfg.risk.max_drawdown,
    )

    tripwire = TripwireMonitor(config=cfg.tripwire) if cfg.tripwire else None
    health = HealthMonitor()

    event_loop = TradingEventLoop(
        strategy=strategy,
        broker=broker,
        risk_limiter=risk_limiter,
        tripwire=tripwire,
        config=cfg,
        health_monitor=health,
        dry_run=dry_run,
    )

    console.print("[bold green]✓[/] All components initialized.")
    console.print("[bold blue]Starting live event loop...[/]")

    try:
        asyncio.run(event_loop.run_live())
    except KeyboardInterrupt:
        console.print("\n[bold yellow]⚠ Live trading stopped by user[/]")
        logger.warning("Live trading stopped by user interrupt")
    except Exception as e:
        console.print(f"\n[bold red]✗ Fatal error: {e}[/]")
        logger.exception("Fatal error in live trading")
        raise typer.Exit(1)

    # Print health dashboard
    console.print(health.print_dashboard())
    console.print("[bold green]✓[/] Live trading session ended.")


@app.command()
def health(
    config: Path = typer.Option("configs/live_trade.yaml", help="Config file"),
):
    """Show health status of the system."""
    from src.live.health import HealthMonitor

    monitor = HealthMonitor()
    console.print(monitor.print_dashboard())


@app.command()
def reconcile(
    config: Path = typer.Option("configs/live_trade.yaml", help="Config file"),
):
    """Reconcile local state with broker positions."""
    cfg = load_config(config)

    broker = _get_broker(cfg)
    if broker is None:
        raise typer.Exit(1)

    from src.exec.reconcile import Reconciler

    reconciler = Reconciler(broker=broker)
    discrepancies = reconciler.check_positions()

    if not discrepancies:
        console.print("[bold green]✓[/] Positions reconciled — no discrepancies.")
    else:
        console.print("[bold yellow]⚠ Discrepancies found:[/]")
        for d in discrepancies:
            console.print(f"  {d}")


def _get_broker(cfg):
    """Instantiate broker adapter from config."""
    broker_name = cfg.broker

    if broker_name == "paper":
        from src.exec.paper_broker import PaperBroker
        return PaperBroker(
            initial_capital=cfg.initial_capital,
            cost_config=cfg.costs,
            slippage_config=cfg.slippage,
        )
    elif broker_name == "zerodha":
        try:
            from src.exec.zerodha_adapter import ZerodhaAdapter
            return ZerodhaAdapter()
        except ImportError:
            console.print("[bold red]✗[/] kiteconnect not installed. Run: pip install kiteconnect")
            return None
        except Exception as e:
            console.print(f"[bold red]✗[/] Failed to init Zerodha: {e}")
            return None
    elif broker_name == "upstox":
        try:
            from src.exec.upstox_adapter import UpstoxAdapter
            return UpstoxAdapter()
        except Exception as e:
            console.print(f"[bold red]✗[/] Failed to init Upstox: {e}")
            return None
    else:
        console.print(f"[bold red]✗[/] Unknown broker: {broker_name}")
        return None


def main():
    app()


if __name__ == "__main__":
    main()
