"""Clock utilities for IST-aware time handling and trading session management."""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

# NSE trading hours (IST)
MARKET_OPEN = dt.time(9, 15)
MARKET_CLOSE = dt.time(15, 30)
PRE_OPEN_START = dt.time(9, 0)
PRE_OPEN_END = dt.time(9, 8)


def now_ist() -> dt.datetime:
    """Return current datetime in IST."""
    return dt.datetime.now(tz=IST)


def to_ist(timestamp: dt.datetime) -> dt.datetime:
    """Convert any timezone-aware datetime to IST."""
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=IST)
    return timestamp.astimezone(IST)


def is_market_open(timestamp: dt.datetime | None = None) -> bool:
    """Check if the given time falls within NSE trading hours."""
    ts = to_ist(timestamp) if timestamp else now_ist()
    return MARKET_OPEN <= ts.time() <= MARKET_CLOSE


def market_open_today() -> dt.datetime:
    """Return today's market open timestamp in IST."""
    today = now_ist().date()
    return dt.datetime.combine(today, MARKET_OPEN, tzinfo=IST)


def market_close_today() -> dt.datetime:
    """Return today's market close timestamp in IST."""
    today = now_ist().date()
    return dt.datetime.combine(today, MARKET_CLOSE, tzinfo=IST)


def seconds_to_close(timestamp: dt.datetime | None = None) -> float:
    """Seconds remaining until market close."""
    ts = to_ist(timestamp) if timestamp else now_ist()
    close = dt.datetime.combine(ts.date(), MARKET_CLOSE, tzinfo=IST)
    return max(0.0, (close - ts).total_seconds())
