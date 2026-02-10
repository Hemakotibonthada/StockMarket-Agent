"""Data loaders for bhavcopy CSVs, local Parquet files, and sample data."""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Any

import pandas as pd

from src.core.io import read_csv, read_parquet, write_parquet
from src.core.logging_utils import get_logger

logger = get_logger("data.loaders")

# Expected columns in bhavcopy CSV
BHAVCOPY_COLUMNS = {
    "SYMBOL": "symbol",
    "SERIES": "series",
    "OPEN": "open",
    "HIGH": "high",
    "LOW": "low",
    "CLOSE": "close",
    "LAST": "last",
    "PREVCLOSE": "prev_close",
    "TOTTRDQTY": "volume",
    "TOTTRDVAL": "turnover",
    "TIMESTAMP": "date",
    "TOTALTRADES": "trades",
}

INTRADAY_COLUMNS = {
    "symbol": "symbol",
    "date": "date",
    "time": "time",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
}


def load_bhavcopy(path: str | Path, series: str = "EQ") -> pd.DataFrame:
    """Load a single NSE bhavcopy CSV and normalize columns.

    Args:
        path: Path to the bhavcopy CSV file.
        series: Filter to this series (default: 'EQ' for equity).

    Returns:
        DataFrame with standardized columns.
    """
    df = read_csv(path)

    # Normalize column names (strip whitespace)
    df.columns = df.columns.str.strip()

    # Rename known columns
    rename_map = {k: v for k, v in BHAVCOPY_COLUMNS.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    # Filter series
    if "series" in df.columns:
        df = df[df["series"] == series].copy()

    # Parse date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    # Ensure numeric columns
    for col in ["open", "high", "low", "close", "volume", "turnover"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info(f"Loaded bhavcopy: {path} ({len(df)} rows)")
    return df


def load_bhavcopy_dir(
    input_dir: str | Path,
    series: str = "EQ",
    pattern: str = "*.csv",
) -> pd.DataFrame:
    """Load all bhavcopy CSVs from a directory and concatenate.

    Args:
        input_dir: Directory containing bhavcopy CSVs.
        series: Filter to this series.
        pattern: Glob pattern for CSV files.

    Returns:
        Concatenated DataFrame sorted by (symbol, date).
    """
    input_dir = Path(input_dir)
    files = sorted(input_dir.glob(pattern))
    if not files:
        logger.warning(f"No CSV files found in {input_dir}")
        return pd.DataFrame()

    frames = []
    for f in files:
        try:
            df = load_bhavcopy(f, series=series)
            frames.append(df)
        except Exception as e:
            logger.error(f"Error loading {f}: {e}")

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values(["symbol", "date"]).reset_index(drop=True)
    logger.info(f"Loaded {len(files)} files -> {len(result)} total rows")
    return result


def load_intraday_csv(path: str | Path) -> pd.DataFrame:
    """Load intraday CSV data with datetime parsing."""
    df = read_csv(path)
    df.columns = df.columns.str.strip().str.lower()

    # Combine date + time if separate columns
    if "date" in df.columns and "time" in df.columns:
        df["datetime"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str))
    elif "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])

    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def load_parquet_data(path: str | Path) -> pd.DataFrame:
    """Load processed Parquet data."""
    return read_parquet(path)


def save_processed(df: pd.DataFrame, output_dir: str | Path, symbol: str | None = None) -> Path:
    """Save processed DataFrame to Parquet, optionally per-symbol."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if symbol:
        out_path = output_dir / f"{symbol}.parquet"
        symbol_df = df[df["symbol"] == symbol] if "symbol" in df.columns else df
        write_parquet(symbol_df, out_path)
    else:
        out_path = output_dir / "all_data.parquet"
        write_parquet(df, out_path)

    logger.info(f"Saved processed data to {out_path}")
    return out_path


def ingest_and_process(
    input_dir: str | Path,
    output_dir: str | Path,
    adjust: bool = True,
    per_symbol: bool = True,
) -> None:
    """Full ingestion pipeline: load CSVs, optionally adjust, save as Parquet."""
    from src.data.adjustments import apply_adjustments

    df = load_bhavcopy_dir(input_dir)
    if df.empty:
        logger.warning("No data to process")
        return

    if adjust:
        df = apply_adjustments(df)

    output_dir = Path(output_dir)
    if per_symbol and "symbol" in df.columns:
        for symbol in df["symbol"].unique():
            save_processed(df, output_dir, symbol=symbol)
    else:
        save_processed(df, output_dir)

    logger.info(f"Ingestion complete: {len(df)} rows processed")
