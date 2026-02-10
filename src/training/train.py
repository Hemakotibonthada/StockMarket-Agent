"""Offline model training pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.core.config import AppConfig, load_config
from src.core.logging_utils import get_logger, setup_logging
from src.core.utils import set_seed
from src.data.loaders import load_parquet_data
from src.models.classic import ClassicModel
from src.models.selection import ModelRegistry
from src.training.dataset import prepare_training_data

logger = get_logger("training.train")


def train_model(
    config: AppConfig,
    data: pd.DataFrame | None = None,
    model_type: str = "random_forest",
    model_params: dict[str, Any] | None = None,
) -> tuple[ClassicModel, dict[str, float]]:
    """Train a classification model on prepared data.

    Args:
        config: Application config.
        data: Optional pre-loaded DataFrame. If None, loads from processed dir.
        model_type: Type of classic model to train.
        model_params: Optional model hyperparameters.

    Returns:
        (trained_model, metrics) tuple.
    """
    set_seed(config.seed)

    # Load data if not provided
    if data is None:
        processed_dir = Path(config.data_dir) / "processed"
        parquet_files = list(processed_dir.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {processed_dir}")

        frames = []
        for f in parquet_files:
            frames.append(load_parquet_data(f))
        data = pd.concat(frames, ignore_index=True)
        logger.info(f"Loaded {len(data)} rows from {len(parquet_files)} files")

    # Prepare features and target
    X, y, feature_cols = prepare_training_data(
        data,
        feature_params=config.strategy_params,
    )

    if len(X) == 0:
        raise ValueError("No training samples after feature preparation")

    # Split train/test by time (last 20% for test)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Train model
    model = ClassicModel(model_type=model_type, params=model_params or {"seed": config.seed})
    train_metrics = model.fit(X_train, y_train)

    # Evaluate on test set
    from sklearn.metrics import accuracy_score, f1_score

    y_pred = model.predict(X_test)
    test_metrics = {
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }

    all_metrics = {**train_metrics, **test_metrics}
    logger.info(f"Test metrics: {test_metrics}")

    # Save to registry
    registry = ModelRegistry(config.models_dir)
    registry.save_model(
        model=model.model,
        name=f"{config.strategy}_{model_type}",
        metrics=all_metrics,
        metadata={
            "model_type": model_type,
            "feature_columns": feature_cols,
            "n_train": len(X_train),
            "n_test": len(X_test),
        },
        scaler=model.scaler,
    )

    return model, all_metrics


def main(config_path: str = "./configs/backtest.yaml") -> None:
    """CLI entry point for training."""
    config = load_config(config_path)
    setup_logging(config.log_level, config.log_dir)

    logger.info("Starting model training...")
    model, metrics = train_model(config)

    # Print results
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title="Training Results")
    table.add_column("Metric")
    table.add_column("Value")
    for k, v in metrics.items():
        table.add_row(k, f"{v:.4f}")
    console.print(table)


if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "./configs/backtest.yaml"
    main(config_path)
