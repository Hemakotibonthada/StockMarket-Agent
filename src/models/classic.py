"""Classic ML models: LogReg, RandomForest, GBM (XGBoost/LightGBM)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler

from src.core.logging_utils import get_logger

logger = get_logger("models.classic")


class ClassicModel:
    """Wrapper for classic ML classification models.

    Supports: logistic_regression, random_forest, xgboost, lightgbm.
    """

    SUPPORTED_MODELS = {"logistic_regression", "random_forest", "xgboost", "lightgbm"}

    def __init__(self, model_type: str = "random_forest", params: dict[str, Any] | None = None):
        self.model_type = model_type
        self.params = params or {}
        self.model: Any = None
        self.scaler = StandardScaler()
        self.feature_names: list[str] = []

        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unknown model type: {model_type}. Supported: {self.SUPPORTED_MODELS}")

    def _create_model(self) -> Any:
        """Instantiate the underlying model."""
        if self.model_type == "logistic_regression":
            return LogisticRegression(
                max_iter=self.params.get("max_iter", 1000),
                C=self.params.get("C", 1.0),
                random_state=self.params.get("seed", 42),
            )
        elif self.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=self.params.get("n_estimators", 200),
                max_depth=self.params.get("max_depth", 10),
                min_samples_leaf=self.params.get("min_samples_leaf", 5),
                random_state=self.params.get("seed", 42),
                n_jobs=-1,
            )
        elif self.model_type == "xgboost":
            import xgboost as xgb
            return xgb.XGBClassifier(
                n_estimators=self.params.get("n_estimators", 200),
                max_depth=self.params.get("max_depth", 6),
                learning_rate=self.params.get("learning_rate", 0.1),
                subsample=self.params.get("subsample", 0.8),
                colsample_bytree=self.params.get("colsample_bytree", 0.8),
                random_state=self.params.get("seed", 42),
                use_label_encoder=False,
                eval_metric="logloss",
            )
        elif self.model_type == "lightgbm":
            import lightgbm as lgb
            return lgb.LGBMClassifier(
                n_estimators=self.params.get("n_estimators", 200),
                max_depth=self.params.get("max_depth", 6),
                learning_rate=self.params.get("learning_rate", 0.1),
                subsample=self.params.get("subsample", 0.8),
                colsample_bytree=self.params.get("colsample_bytree", 0.8),
                random_state=self.params.get("seed", 42),
                verbose=-1,
            )

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> dict[str, float]:
        """Train the model.

        Args:
            X: Feature matrix.
            y: Target labels (0/1 or -1/0/1).

        Returns:
            Dict with training metrics.
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)

        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y)

        # Handle NaN/Inf
        X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)

        # Scale features
        X_scaled = self.scaler.fit_transform(X_arr)

        self.model = self._create_model()
        self.model.fit(X_scaled, y_arr)

        # Training metrics
        y_pred = self.model.predict(X_scaled)
        acc = accuracy_score(y_arr, y_pred)
        f1 = f1_score(y_arr, y_pred, average="weighted", zero_division=0)

        logger.info(f"Model trained: {self.model_type}, accuracy={acc:.4f}, f1={f1:.4f}")
        return {"accuracy": acc, "f1_score": f1}

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if self.model is None:
            raise RuntimeError("Model not trained yet")
        X_arr = np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = self.scaler.transform(X_arr)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if self.model is None:
            raise RuntimeError("Model not trained yet")
        X_arr = np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = self.scaler.transform(X_arr)
        return self.model.predict_proba(X_scaled)

    def feature_importance(self) -> pd.Series | None:
        """Return feature importances (if available)."""
        if self.model is None:
            return None
        if hasattr(self.model, "feature_importances_"):
            return pd.Series(
                self.model.feature_importances_,
                index=self.feature_names or range(len(self.model.feature_importances_)),
            ).sort_values(ascending=False)
        return None
