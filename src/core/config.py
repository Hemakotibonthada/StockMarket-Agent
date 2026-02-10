"""Configuration management with YAML loading and Pydantic validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class CostsConfig(BaseModel):
    brokerage_bps: float = 5.0
    stt_bps: float = 10.0
    gst_bps: float = 1.8
    stamp_bps: float = 0.003
    sebi_bps: float = 0.0001


class SlippageConfig(BaseModel):
    mode: str = "random"
    bps_mean: float = 4.0
    bps_std: float = 3.0


class RiskConfig(BaseModel):
    risk_per_trade_pct: float = 0.5
    daily_max_loss_pct: float = 1.0
    weekly_max_loss_pct: float = 2.0
    strategy_max_dd_pct: float = 5.0
    initial_capital: float = 1_000_000.0


class TripwireConfig(BaseModel):
    max_consecutive_rejects: int = 3
    max_latency_ms: int = 2000
    max_drawdown_pct: float = 5.0
    feed_timeout_seconds: int = 60


class WalkforwardConfig(BaseModel):
    train_years: list[int] = Field(default_factory=lambda: [2018, 2019, 2020])
    validate_years: list[int] = Field(default_factory=lambda: [2021])
    test_years: list[int] = Field(default_factory=lambda: [2022, 2023])


class OrderRouterConfig(BaseModel):
    max_orders_per_second: int = 5
    max_orders_per_minute: int = 50
    price_protection_pct: float = 1.0
    max_retries: int = 3
    retry_delay_ms: int = 500


class AppConfig(BaseModel):
    """Master application configuration."""

    # Base
    data_dir: str = "./data"
    timezone: str = "Asia/Kolkata"
    db: str = "duckdb"
    universe_file: str = "./configs/universe_nifty50.yaml"
    bar_interval: str = "5min"
    seed: int = 42
    log_level: str = "INFO"
    log_dir: str = "./logs"
    models_dir: str = "./models_registry"
    reports_dir: str = "./reports"

    # Strategy
    strategy: str = "orb_momentum"
    strategy_params: dict[str, Any] = Field(default_factory=dict)

    # Broker
    broker: str = "paper"
    confirm_live: bool = False

    # Sub-configs
    costs: CostsConfig = Field(default_factory=CostsConfig)
    slippage: SlippageConfig = Field(default_factory=SlippageConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    tripwires: TripwireConfig = Field(default_factory=TripwireConfig)
    walkforward: WalkforwardConfig = Field(default_factory=WalkforwardConfig)
    order_router: OrderRouterConfig = Field(default_factory=OrderRouterConfig)

    # Report
    report: dict[str, str] = Field(default_factory=lambda: {"format": "html", "output_dir": "./reports"})

    # Logging
    logging: dict[str, str] = Field(default_factory=lambda: {"audit_file": "./logs/audit.jsonl", "level": "INFO"})


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return as dict."""
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_config(config_path: str | Path) -> AppConfig:
    """Load config from YAML, supporting 'inherits' for base config merging."""
    path = Path(config_path)
    raw = load_yaml(path)

    # Handle inheritance
    if "inherits" in raw:
        base_path = path.parent / raw.pop("inherits")
        base = load_yaml(base_path)
        # Deep merge: raw overrides base
        merged = _deep_merge(base, raw)
    else:
        merged = raw

    return AppConfig(**merged)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
