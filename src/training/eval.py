"""Offline model evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

from src.core.config import AppConfig, load_config
from src.core.logging_utils import get_logger, setup_logging
from src.core.utils import set_seed
from src.data.loaders import load_parquet_data
from src.models.selection import ModelRegistry
from src.training.dataset import prepare_training_data

logger = get_logger("training.eval")


def evaluate_model(
    config: AppConfig,
    model_name: str | None = None,
    version: int | None = None,
    data: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Evaluate a model from the registry on test data.

    Args:
        config: Application configuration.
        model_name: Model name in registry. Defaults to strategy + model type.
        version: Model version (None = latest).
        data: Optional test data. If None, loads from processed dir.

    Returns:
        Evaluation metrics.
    """
    set_seed(config.seed)

    # Load data
    if data is None:
        processed_dir = Path(config.data_dir) / "processed"
        parquet_files = list(processed_dir.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files in {processed_dir}")
        frames = [load_parquet_data(f) for f in parquet_files]
        data = pd.concat(frames, ignore_index=True)

    # Prepare features
    X, y, feature_cols = prepare_training_data(data, feature_params=config.strategy_params)

    # Use last 20% as test
    split_idx = int(len(X) * 0.8)
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]

    # Load model
    registry = ModelRegistry(config.models_dir)
    model_name = model_name or f"{config.strategy}_random_forest"
    model, metrics, scaler = registry.load_model(model_name, version)

    # Predict
    X_arr = np.nan_to_num(np.asarray(X_test, dtype=np.float64))
    if scaler is not None:
        X_arr = scaler.transform(X_arr)
    y_pred = model.predict(X_arr)

    # Metrics
    eval_metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "n_samples": len(X_test),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
    }

    logger.info(f"Evaluation results: accuracy={eval_metrics['accuracy']:.4f}")
    return eval_metrics


def main(config_path: str = "./configs/backtest.yaml") -> None:
    """CLI entry point for evaluation."""
    config = load_config(config_path)
    setup_logging(config.log_level, config.log_dir)

    logger.info("Starting model evaluation...")
    metrics = evaluate_model(config)

    from rich.console import Console
    console = Console()
    console.print(f"\n[bold]Accuracy:[/bold] {metrics['accuracy']:.4f}")
    console.print(f"[bold]F1 Score:[/bold] {metrics['f1_score']:.4f}")
    console.print(f"\n{metrics['classification_report']}")


if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "./configs/backtest.yaml"
    main(config_path)
