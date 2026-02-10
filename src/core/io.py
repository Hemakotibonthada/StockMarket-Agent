"""I/O helpers for CSV, Parquet, DuckDB, and SQLite."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import duckdb

    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False


def read_csv(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Read a CSV file with sensible defaults for market data."""
    defaults = {"parse_dates": True, "index_col": None}
    defaults.update(kwargs)
    return pd.read_csv(path, **defaults)


def write_csv(df: pd.DataFrame, path: str | Path, **kwargs: Any) -> None:
    """Write a DataFrame to CSV."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, **kwargs)


def read_parquet(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Read a Parquet file."""
    return pd.read_parquet(path, **kwargs)


def write_parquet(df: pd.DataFrame, path: str | Path, **kwargs: Any) -> None:
    """Write a DataFrame to Parquet with snappy compression."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    defaults = {"compression": "snappy"}
    defaults.update(kwargs)
    df.to_parquet(path, index=False, **defaults)


class DuckDBStore:
    """Lightweight DuckDB wrapper for metadata and analytics queries."""

    def __init__(self, db_path: str | Path = ":memory:"):
        if not HAS_DUCKDB:
            raise ImportError("duckdb is required for DuckDBStore")
        self.db_path = str(db_path)
        self.conn = duckdb.connect(self.db_path)

    def execute(self, query: str, params: list | None = None) -> Any:
        if params:
            return self.conn.execute(query, params)
        return self.conn.execute(query)

    def query_df(self, query: str) -> pd.DataFrame:
        return self.conn.execute(query).fetchdf()

    def register_df(self, name: str, df: pd.DataFrame) -> None:
        self.conn.register(name, df)

    def load_parquet(self, table_name: str, path: str | Path) -> None:
        self.conn.execute(
            f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_parquet('{path}')"
        )

    def close(self) -> None:
        self.conn.close()


class SQLiteStore:
    """Lightweight SQLite wrapper as fallback metadata store."""

    def __init__(self, db_path: str | Path = ":memory:"):
        self.db_path = str(db_path)
        self.conn = sqlite3.connect(self.db_path)

    def execute(self, query: str, params: tuple = ()) -> sqlite3.Cursor:
        return self.conn.execute(query, params)

    def query_df(self, query: str) -> pd.DataFrame:
        return pd.read_sql_query(query, self.conn)

    def commit(self) -> None:
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()


def get_store(db_type: str = "duckdb", db_path: str | Path = ":memory:") -> DuckDBStore | SQLiteStore:
    """Factory to get the configured metadata store."""
    if db_type == "duckdb" and HAS_DUCKDB:
        return DuckDBStore(db_path)
    return SQLiteStore(db_path)
