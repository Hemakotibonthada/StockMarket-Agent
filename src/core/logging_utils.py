"""Structured logging with Rich console and rotating file handlers."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.logging import RichHandler

console = Console()

_LOG_FORMAT = "%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: str = "INFO",
    log_dir: str | Path = "./logs",
    app_name: str = "stock-agent",
) -> logging.Logger:
    """Configure root logger with Rich console + rotating file handlers."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger(app_name)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    if root.handlers:
        return root

    # Rich console handler
    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
    )
    rich_handler.setLevel(logging.DEBUG)
    root.addHandler(rich_handler)

    # Rotating file handler
    file_handler = RotatingFileHandler(
        log_dir / f"{app_name}.log",
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
    )
    file_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    file_handler.setLevel(logging.DEBUG)
    root.addHandler(file_handler)

    return root


def get_logger(name: str) -> logging.Logger:
    """Get a child logger under the stock-agent namespace."""
    return logging.getLogger(f"stock-agent.{name}")


class AuditLogger:
    """Append-only JSONL audit log for trade events."""

    def __init__(self, path: str | Path = "./logs/audit.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event_type: str, data: dict[str, Any]) -> None:
        """Write a single audit event as a JSON line."""
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event_type,
            **data,
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def read_all(self) -> list[dict[str, Any]]:
        """Read all audit events (for reconciliation)."""
        if not self.path.exists():
            return []
        events = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
        return events
