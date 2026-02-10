"""Data ingestion and processing CLI.

Usage:
    sa-data ingest --input-dir data/raw --output-dir data/processed
    sa-data resample --input data/processed/RELIANCE.parquet --freq 15min
    sa-data universe --config configs/universe_nifty50.yaml
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from src.core.config import load_config
from src.core.logging_utils import get_logger

app = typer.Typer(name="sa-data", help="Data ingestion and processing.")
console = Console()
logger = get_logger("cli.data")


@app.command()
def ingest(
    input_dir: Path = typer.Option("data/raw", help="Raw CSV directory"),
    output_dir: Path = typer.Option("data/processed", help="Output parquet directory"),
    config: Path = typer.Option("configs/base.yaml", help="Config file"),
    bar_interval: str = typer.Option("5min", help="Bar interval for resampling"),
):
    """Ingest raw CSV data, apply adjustments, and save as parquet."""
    from src.data.loaders import ingest_and_process

    console.print(f"[bold blue]Ingesting data from:[/] {input_dir}")
    cfg = load_config(config)

    output_dir.mkdir(parents=True, exist_ok=True)

    df = ingest_and_process(
        raw_dir=input_dir,
        output_dir=output_dir,
        bar_interval=bar_interval,
    )

    console.print(f"[bold green]✓[/] Processed {len(df)} rows")
    console.print(f"[bold green]✓[/] Symbols: {df['symbol'].nunique() if 'symbol' in df.columns else 'N/A'}")
    console.print(f"[bold green]✓[/] Output: {output_dir}")


@app.command()
def resample(
    input_file: Path = typer.Argument(..., help="Input parquet file"),
    freq: str = typer.Option("15min", help="Target frequency"),
    output: Path | None = typer.Option(None, help="Output file (default: auto-named)"),
):
    """Resample OHLCV data to a different frequency."""
    from src.data.resample import resample_bars
    from src.core.io import read_parquet, write_parquet

    console.print(f"[bold blue]Resampling:[/] {input_file} → {freq}")

    df = read_parquet(input_file)
    resampled = resample_bars(df, freq)

    if output is None:
        output = input_file.parent / f"{input_file.stem}_{freq}.parquet"

    write_parquet(resampled, output)
    console.print(f"[bold green]✓[/] Output: {output} ({len(resampled)} bars)")


@app.command()
def universe(
    config: Path = typer.Option("configs/universe_nifty50.yaml", help="Universe config"),
):
    """Show the trading universe from config."""
    from src.data.universe import Universe

    u = Universe.from_yaml(config)

    table = Table(title="Trading Universe")
    table.add_column("Symbol", style="cyan")

    for sym in u.symbols:
        table.add_row(sym)

    console.print(table)
    console.print(f"Total: {len(u.symbols)} symbols")


@app.command()
def info(
    file: Path = typer.Argument(..., help="Parquet file to inspect"),
):
    """Show info about a parquet data file."""
    from src.core.io import read_parquet

    df = read_parquet(file)

    table = Table(title=f"Data Info: {file.name}")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Rows", str(len(df)))
    table.add_row("Columns", ", ".join(df.columns.tolist()))
    if "symbol" in df.columns:
        table.add_row("Symbols", str(df["symbol"].nunique()))
    if "datetime" in df.columns or df.index.name == "datetime":
        idx = df.index if df.index.name == "datetime" else df["datetime"]
        table.add_row("Start", str(idx.min()))
        table.add_row("End", str(idx.max()))
    table.add_row("Memory (MB)", f"{df.memory_usage(deep=True).sum() / 1e6:.2f}")

    console.print(table)


def main():
    app()


if __name__ == "__main__":
    main()
