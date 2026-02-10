"""Model selection, persistence, and registry management."""

from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.core.logging_utils import get_logger

logger = get_logger("models.selection")


class ModelRegistry:
    """Local model registry with versioned directories.

    Structure:
        models_registry/
            model_name/
                v1/
                    model.pkl
                    metrics.yaml
                    metadata.json
                v2/
                    ...
    """

    def __init__(self, registry_dir: str | Path = "./models_registry"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)

    def save_model(
        self,
        model: Any,
        name: str,
        metrics: dict[str, float],
        metadata: dict[str, Any] | None = None,
        scaler: Any | None = None,
    ) -> Path:
        """Save a model to the registry with versioning.

        Args:
            model: The trained model object.
            name: Model name (used as directory name).
            metrics: Performance metrics dict.
            metadata: Additional metadata.
            scaler: Optional scaler to persist.

        Returns:
            Path to the versioned model directory.
        """
        model_dir = self.registry_dir / name
        model_dir.mkdir(parents=True, exist_ok=True)

        # Determine next version
        existing = [d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith("v")]
        version = len(existing) + 1
        version_dir = model_dir / f"v{version}"
        version_dir.mkdir()

        # Save model
        model_path = version_dir / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # Save scaler
        if scaler is not None:
            scaler_path = version_dir / "scaler.pkl"
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler, f)

        # Save metrics
        metrics_path = version_dir / "metrics.yaml"
        with open(metrics_path, "w") as f:
            yaml.dump(metrics, f, default_flow_style=False)

        # Save metadata
        meta = {
            "name": name,
            "version": version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics_summary": metrics,
            **(metadata or {}),
        }
        meta_path = version_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)

        logger.info(f"Saved model {name} v{version} to {version_dir}")
        return version_dir

    def load_model(
        self,
        name: str,
        version: int | None = None,
    ) -> tuple[Any, dict[str, float], Any | None]:
        """Load a model from the registry.

        Args:
            name: Model name.
            version: Specific version to load (None = latest).

        Returns:
            (model, metrics, scaler) tuple.
        """
        model_dir = self.registry_dir / name

        if version is None:
            # Load latest version
            versions = sorted(
                [d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith("v")],
                key=lambda d: int(d.name[1:]),
            )
            if not versions:
                raise FileNotFoundError(f"No versions found for model '{name}'")
            version_dir = versions[-1]
        else:
            version_dir = model_dir / f"v{version}"

        if not version_dir.exists():
            raise FileNotFoundError(f"Version directory not found: {version_dir}")

        # Load model
        with open(version_dir / "model.pkl", "rb") as f:
            model = pickle.load(f)

        # Load metrics
        metrics_path = version_dir / "metrics.yaml"
        metrics = {}
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = yaml.safe_load(f) or {}

        # Load scaler
        scaler = None
        scaler_path = version_dir / "scaler.pkl"
        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)

        logger.info(f"Loaded model {name} from {version_dir}")
        return model, metrics, scaler

    def list_models(self) -> list[dict[str, Any]]:
        """List all models and their versions."""
        models = []
        for model_dir in sorted(self.registry_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            versions = sorted(
                [d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith("v")],
                key=lambda d: int(d.name[1:]),
            )
            for v_dir in versions:
                meta_path = v_dir / "metadata.json"
                meta = {}
                if meta_path.exists():
                    with open(meta_path) as f:
                        meta = json.load(f)
                models.append({
                    "name": model_dir.name,
                    "version": v_dir.name,
                    "path": str(v_dir),
                    **meta,
                })
        return models

    def best_model(
        self,
        name: str,
        metric: str = "f1_score",
        higher_is_better: bool = True,
    ) -> tuple[Any, dict[str, float], Any | None]:
        """Load the best model version by a given metric.

        Args:
            name: Model name.
            metric: Metric to compare.
            higher_is_better: Whether higher values are better.

        Returns:
            (model, metrics, scaler) tuple for the best version.
        """
        model_dir = self.registry_dir / name
        versions = [d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith("v")]

        best_version = None
        best_value = float("-inf") if higher_is_better else float("inf")

        for v_dir in versions:
            metrics_path = v_dir / "metrics.yaml"
            if not metrics_path.exists():
                continue
            with open(metrics_path) as f:
                metrics = yaml.safe_load(f) or {}
            value = metrics.get(metric, float("-inf") if higher_is_better else float("inf"))
            if (higher_is_better and value > best_value) or (not higher_is_better and value < best_value):
                best_value = value
                best_version = int(v_dir.name[1:])

        if best_version is None:
            raise FileNotFoundError(f"No models found for '{name}' with metric '{metric}'")

        return self.load_model(name, best_version)
