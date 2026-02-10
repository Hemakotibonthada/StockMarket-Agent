"""Real market data provider using yfinance for Indian NSE/BSE stocks.

Provides caching, error handling, and fallback mechanisms for reliable data access.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
import streamlit as st

try:
    import yfinance as yf
    _HAS_YFINANCE = True
except ImportError:
    _HAS_YFINANCE = False

from src.core.logging_utils import get_logger

logger = get_logger("data.market_data")

# ── NSE symbols → Yahoo Finance suffix mapping ────────────────────────────────
# Yahoo Finance uses .NS for NSE, .BO for BSE
NSE_SUFFIX = ".NS"
BSE_SUFFIX = ".BO"

# Commonly used Indian stock symbols and their full names
INDIAN_STOCKS = {
    "RELIANCE": "Reliance Industries",
    "TCS": "Tata Consultancy Services",
    "INFY": "Infosys",
    "HDFCBANK": "HDFC Bank",
    "ICICIBANK": "ICICI Bank",
    "HINDUNILVR": "Hindustan Unilever",
    "ITC": "ITC Limited",
    "SBIN": "State Bank of India",
    "BHARTIARTL": "Bharti Airtel",
    "KOTAKBANK": "Kotak Mahindra Bank",
    "LT": "Larsen & Toubro",
    "AXISBANK": "Axis Bank",
    "WIPRO": "Wipro",
    "HCLTECH": "HCL Technologies",
    "MARUTI": "Maruti Suzuki",
    "TATAMOTORS": "Tata Motors",
    "SUNPHARMA": "Sun Pharma",
    "ONGC": "ONGC",
    "NTPC": "NTPC",
    "POWERGRID": "Power Grid",
    "TATASTEEL": "Tata Steel",
    "BAJFINANCE": "Bajaj Finance",
    "ADANIENT": "Adani Enterprises",
    "ASIANPAINT": "Asian Paints",
    "TITAN": "Titan Company",
    "ULTRACEMCO": "UltraTech Cement",
    "NESTLEIND": "Nestle India",
    "BAJAJFINSV": "Bajaj Finserv",
    "TECHM": "Tech Mahindra",
    "INDUSINDBK": "IndusInd Bank",
}

# Default watchlist for dashboard
DEFAULT_WATCHLIST = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]

# Index tickers
INDICES = {
    "NIFTY 50": "^NSEI",
    "SENSEX": "^BSESN",
    "NIFTY BANK": "^NSEBANK",
    "NIFTY IT": "^CNXIT",
}


def _to_yahoo_ticker(symbol: str) -> str:
    """Convert NSE symbol to Yahoo Finance ticker."""
    symbol = symbol.strip().upper()
    # Already has suffix
    if symbol.endswith((".NS", ".BO")):
        return symbol
    # Index symbols
    if symbol.startswith("^"):
        return symbol
    if symbol in INDICES:
        return INDICES[symbol]
    # Default to NSE
    return f"{symbol}{NSE_SUFFIX}"


def _normalize_ohlcv(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Normalize yfinance output to standard OHLCV columns."""
    if df is None or df.empty:
        return pd.DataFrame()

    result = pd.DataFrame()
    result["date"] = df.index
    result["symbol"] = symbol
    result["open"] = df["Open"].values
    result["high"] = df["High"].values
    result["low"] = df["Low"].values
    result["close"] = df["Close"].values
    result["volume"] = df["Volume"].values

    # Drop NaN rows
    result = result.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)

    # Ensure numeric types
    for col in ["open", "high", "low", "close"]:
        result[col] = pd.to_numeric(result[col], errors="coerce")
    result["volume"] = pd.to_numeric(result["volume"], errors="coerce").fillna(0).astype(int)

    return result


@st.cache_data(ttl=300, show_spinner=False)
def fetch_stock_data(
    symbol: str,
    period: str = "2y",
    interval: str = "1d",
) -> pd.DataFrame:
    """Fetch historical OHLCV data for a single stock from Yahoo Finance.

    Args:
        symbol: NSE stock symbol (e.g., 'RELIANCE', 'TCS').
        period: Data period ('1mo', '3mo', '6mo', '1y', '2y', '5y', 'max').
        interval: Bar interval ('1d', '1wk', '1mo', '5m', '15m', '1h').

    Returns:
        DataFrame with columns: date, symbol, open, high, low, close, volume.
    """
    if not _HAS_YFINANCE:
        logger.error("yfinance not installed. Install with: pip install yfinance")
        return pd.DataFrame()

    ticker = _to_yahoo_ticker(symbol)
    try:
        tk = yf.Ticker(ticker)
        df = tk.history(period=period, interval=interval)
        if df.empty:
            logger.warning(f"No data from yfinance for {ticker}")
            return pd.DataFrame()
        return _normalize_ohlcv(df, symbol.split(".")[0].replace(NSE_SUFFIX, ""))
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def fetch_stock_data_daterange(
    symbol: str,
    start: str,
    end: str | None = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """Fetch historical data between specific dates.

    Args:
        symbol: NSE stock symbol.
        start: Start date string ('YYYY-MM-DD').
        end: End date string (defaults to today).
        interval: Bar interval.

    Returns:
        Normalized OHLCV DataFrame.
    """
    if not _HAS_YFINANCE:
        return pd.DataFrame()

    ticker = _to_yahoo_ticker(symbol)
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")
    try:
        tk = yf.Ticker(ticker)
        df = tk.history(start=start, end=end, interval=interval)
        if df.empty:
            return pd.DataFrame()
        return _normalize_ohlcv(df, symbol.split(".")[0])
    except Exception as e:
        logger.error(f"Error fetching {ticker} [{start}–{end}]: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def fetch_multiple_stocks(
    symbols: list[str],
    period: str = "2y",
    interval: str = "1d",
) -> dict[str, pd.DataFrame]:
    """Fetch data for multiple symbols.

    Returns:
        Dict[symbol, DataFrame]. Skips symbols that fail to download.
    """
    result = {}
    for sym in symbols:
        df = fetch_stock_data(sym, period=period, interval=interval)
        if not df.empty:
            result[sym] = df
    return result


@st.cache_data(ttl=60, show_spinner=False)
def fetch_live_quote(symbol: str) -> dict[str, Any]:
    """Fetch current quote / latest price info for a symbol.

    Returns:
        Dict with keys: symbol, price, change, change_pct, volume,
        day_high, day_low, prev_close, market_cap, pe_ratio, etc.
    """
    if not _HAS_YFINANCE:
        return {}

    ticker = _to_yahoo_ticker(symbol)
    try:
        tk = yf.Ticker(ticker)
        info = tk.info
        fast = tk.fast_info if hasattr(tk, "fast_info") else {}

        price = info.get("currentPrice") or info.get("regularMarketPrice", 0)
        prev_close = info.get("previousClose") or info.get("regularMarketPreviousClose", 0)
        change = price - prev_close if price and prev_close else 0
        change_pct = (change / prev_close * 100) if prev_close else 0

        return {
            "symbol": symbol,
            "name": info.get("shortName", symbol),
            "price": price,
            "prev_close": prev_close,
            "change": change,
            "change_pct": change_pct,
            "volume": info.get("volume", 0),
            "day_high": info.get("dayHigh", 0),
            "day_low": info.get("dayLow", 0),
            "week_52_high": info.get("fiftyTwoWeekHigh", 0),
            "week_52_low": info.get("fiftyTwoWeekLow", 0),
            "market_cap": info.get("marketCap", 0),
            "pe_ratio": info.get("trailingPE", 0),
            "sector": info.get("sector", ""),
            "industry": info.get("industry", ""),
        }
    except Exception as e:
        logger.error(f"Error fetching quote for {ticker}: {e}")
        return {"symbol": symbol, "price": 0, "change": 0, "change_pct": 0}


@st.cache_data(ttl=120, show_spinner=False)
def fetch_watchlist_quotes(symbols: list[str]) -> list[dict[str, Any]]:
    """Fetch live quotes for a list of symbols."""
    quotes = []
    for sym in symbols:
        q = fetch_live_quote(sym)
        if q:
            quotes.append(q)
    return quotes


@st.cache_data(ttl=600, show_spinner=False)
def fetch_index_data(index_name: str = "NIFTY 50", period: str = "1y") -> pd.DataFrame:
    """Fetch index data (NIFTY, SENSEX, etc.)."""
    ticker = INDICES.get(index_name, index_name)
    if not _HAS_YFINANCE:
        return pd.DataFrame()
    try:
        tk = yf.Ticker(ticker)
        df = tk.history(period=period)
        if df.empty:
            return pd.DataFrame()
        return _normalize_ohlcv(df, index_name)
    except Exception as e:
        logger.error(f"Error fetching index {index_name}: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=1800, show_spinner=False)
def get_stock_info(symbol: str) -> dict[str, Any]:
    """Get detailed stock information (sector, industry, description, etc.)."""
    if not _HAS_YFINANCE:
        return {}
    ticker = _to_yahoo_ticker(symbol)
    try:
        tk = yf.Ticker(ticker)
        return tk.info
    except Exception:
        return {}


def search_stocks(query: str) -> list[dict[str, str]]:
    """Search for stocks matching a query string.

    Returns list of dicts with 'symbol' and 'name' keys.
    """
    query = query.strip().upper()
    results = []
    for sym, name in INDIAN_STOCKS.items():
        if query in sym or query in name.upper():
            results.append({"symbol": sym, "name": name})
    return results[:10]


def get_available_symbols() -> list[str]:
    """Return list of commonly-used Indian stock symbols."""
    return sorted(INDIAN_STOCKS.keys())


def get_default_watchlist() -> list[str]:
    """Return the default dashboard watchlist."""
    return list(DEFAULT_WATCHLIST)


def validate_data(df: pd.DataFrame) -> tuple[bool, str]:
    """Validate that a DataFrame has required OHLCV columns and sufficient rows.

    Returns:
        (is_valid, message) tuple.
    """
    required = {"open", "high", "low", "close", "volume"}
    if df.empty:
        return False, "DataFrame is empty. No data available."

    missing = required - set(df.columns)
    if missing:
        return False, f"Missing required columns: {missing}"

    if len(df) < 30:
        return False, f"Insufficient data: {len(df)} rows (minimum 30 required)"

    # Check for excessive NaN
    nan_pct = df[["open", "high", "low", "close"]].isna().mean().max() * 100
    if nan_pct > 10:
        return False, f"Too many missing values: {nan_pct:.0f}% NaN"

    return True, f"{len(df)} bars loaded successfully"
