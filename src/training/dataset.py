"""Dataset preparation for model training."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.logging_utils import get_logger
from src.features.feature_sets import compute_base_features, get_feature_columns

logger = get_logger("training.dataset")


def create_classification_target(
    df: pd.DataFrame,
    forward_period: int = 1,
    threshold_pct: float = 0.0,
    close_col: str = "close",
) -> pd.Series:
    """Create a binary classification target: 1 if forward return > threshold, else 0.

    Args:
        df: DataFrame with close prices.
        forward_period: Number of bars to look forward.
        threshold_pct: Minimum return percentage for label 1.
        close_col: Name of the close price column.

    Returns:
        Series of binary labels.
    """
    forward_return = df[close_col].pct_change(forward_period).shift(-forward_period)
    target = (forward_return > threshold_pct / 100).astype(int)
    return target


def prepare_training_data(
    df: pd.DataFrame,
    feature_params: dict | None = None,
    target_params: dict | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Prepare features and target for model training.

    Args:
        df: Raw OHLCV DataFrame (single symbol, sorted by time).
        feature_params: Parameters for feature computation.
        target_params: Parameters for target creation.

    Returns:
        (X, y, feature_columns) tuple with NaN rows dropped.
    """
    target_params = target_params or {}

    # Compute features
    df_feat = compute_base_features(df, feature_params)

    # Create target
    y = create_classification_target(
        df_feat,
        forward_period=target_params.get("forward_period", 1),
        threshold_pct=target_params.get("threshold_pct", 0.0),
    )
    df_feat["target"] = y

    # Select feature columns
    feature_cols = get_feature_columns()
    available_cols = [c for c in feature_cols if c in df_feat.columns]

    # Drop NaN
    df_clean = df_feat[available_cols + ["target"]].dropna()

    X = df_clean[available_cols]
    y = df_clean["target"]

    logger.info(f"Prepared dataset: {len(X)} samples, {len(available_cols)} features")
    return X, y, available_cols


def create_sequence_data(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "target",
    seq_len: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Create sequence data for LSTM/GRU models.

    Args:
        df: DataFrame with features and target.
        feature_cols: Feature column names.
        target_col: Target column name.
        seq_len: Sequence length.

    Returns:
        (X, y) tuple where X has shape (n, seq_len, n_features).
    """
    data = df[feature_cols].values
    targets = df[target_col].values

    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i])
        y.append(targets[i])

    return np.array(X), np.array(y)
