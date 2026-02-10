"""IST trading calendar with NSE holidays and session management."""

from __future__ import annotations

import datetime as dt
from typing import Sequence

import pandas as pd

from src.core.clocks import IST, MARKET_CLOSE, MARKET_OPEN

# NSE holidays (sample set — extend as needed)
NSE_HOLIDAYS_2023_2024 = [
    # 2023
    "2023-01-26", "2023-03-07", "2023-03-30", "2023-04-04", "2023-04-07",
    "2023-04-14", "2023-04-22", "2023-05-01", "2023-06-29", "2023-08-15",
    "2023-09-19", "2023-10-02", "2023-10-24", "2023-11-14", "2023-11-27",
    "2023-12-25",
    # 2024
    "2024-01-26", "2024-03-08", "2024-03-25", "2024-03-29", "2024-04-11",
    "2024-04-14", "2024-04-17", "2024-04-21", "2024-05-01", "2024-05-23",
    "2024-06-17", "2024-07-17", "2024-08-15", "2024-10-02", "2024-10-12",
    "2024-11-01", "2024-11-15", "2024-12-25",
    # 2025
    "2025-01-26", "2025-02-26", "2025-03-14", "2025-03-31", "2025-04-10",
    "2025-04-14", "2025-04-18", "2025-05-01", "2025-08-15", "2025-08-27",
    "2025-10-02", "2025-10-21", "2025-10-22", "2025-11-05", "2025-11-26",
    "2025-12-25",
]


class TradingCalendar:
    """NSE trading calendar with holiday awareness."""

    def __init__(self, holidays: Sequence[str] | None = None):
        holidays = holidays or NSE_HOLIDAYS_2023_2024
        self.holidays: set[dt.date] = {
            pd.Timestamp(h).date() for h in holidays
        }

    def is_trading_day(self, date: dt.date) -> bool:
        """Check if a given date is a trading day (weekday + not holiday)."""
        if date.weekday() >= 5:  # Saturday=5, Sunday=6
            return False
        return date not in self.holidays

    def trading_days(
        self,
        start: dt.date | str,
        end: dt.date | str,
    ) -> list[dt.date]:
        """Return list of trading days in the given range (inclusive)."""
        start = pd.Timestamp(start).date() if isinstance(start, str) else start
        end = pd.Timestamp(end).date() if isinstance(end, str) else end

        days = []
        current = start
        while current <= end:
            if self.is_trading_day(current):
                days.append(current)
            current += dt.timedelta(days=1)
        return days

    def next_trading_day(self, date: dt.date) -> dt.date:
        """Return the next trading day after the given date."""
        current = date + dt.timedelta(days=1)
        while not self.is_trading_day(current):
            current += dt.timedelta(days=1)
        return current

    def prev_trading_day(self, date: dt.date) -> dt.date:
        """Return the previous trading day before the given date."""
        current = date - dt.timedelta(days=1)
        while not self.is_trading_day(current):
            current -= dt.timedelta(days=1)
        return current

    def session_open(self, date: dt.date) -> dt.datetime:
        """Market open time for the given date."""
        return dt.datetime.combine(date, MARKET_OPEN, tzinfo=IST)

    def session_close(self, date: dt.date) -> dt.datetime:
        """Market close time for the given date."""
        return dt.datetime.combine(date, MARKET_CLOSE, tzinfo=IST)

    def add_holidays(self, holidays: Sequence[str]) -> None:
        """Add additional holidays."""
        for h in holidays:
            self.holidays.add(pd.Timestamp(h).date())


# Module-level singleton
calendar = TradingCalendar()
