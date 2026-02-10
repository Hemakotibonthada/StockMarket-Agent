"""Optional deep learning models (PyTorch LSTM/GRU) — guarded import."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.core.logging_utils import get_logger

logger = get_logger("models.deep")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.info("PyTorch not available; deep learning models disabled")


def _check_torch() -> None:
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch is required for deep learning models. "
            "Install with: pip install torch"
        )


class LSTMClassifier(nn.Module if HAS_TORCH else object):
    """LSTM classifier for sequence-based signal prediction."""

    def __init__(
        self,
        input_size: int = 30,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.2,
    ):
        _check_torch()
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.fc(last_hidden)


class DeepModel:
    """Wrapper for training/inference with PyTorch sequence models."""

    def __init__(
        self,
        model_type: str = "lstm",
        input_size: int = 30,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_classes: int = 2,
        params: dict[str, Any] | None = None,
    ):
        _check_torch()
        self.params = params or {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_type == "lstm":
            self.model = LSTMClassifier(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_classes=num_classes,
                dropout=self.params.get("dropout", 0.2),
            ).to(self.device)
        else:
            raise ValueError(f"Unknown deep model type: {model_type}")

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.params.get("learning_rate", 0.001),
        )
        self.criterion = nn.CrossEntropyLoss()

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 50,
        batch_size: int = 64,
        val_X: np.ndarray | None = None,
        val_y: np.ndarray | None = None,
    ) -> dict[str, list[float]]:
        """Train the model on sequence data.

        Args:
            X: Shape (n_samples, seq_len, n_features).
            y: Shape (n_samples,) integer labels.
            epochs: Number of training epochs.
            batch_size: Mini-batch size.
            val_X: Optional validation features.
            val_y: Optional validation labels.

        Returns:
            Training history with loss and accuracy per epoch.
        """
        X_t = torch.FloatTensor(X).to(self.device)
        y_t = torch.LongTensor(y).to(self.device)

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        history: dict[str, list[float]] = {"train_loss": [], "train_acc": []}
        if val_X is not None:
            history["val_loss"] = []
            history["val_acc"] = []

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            correct = 0
            total = 0

            for batch_X, batch_y in loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * len(batch_y)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == batch_y).sum().item()
                total += len(batch_y)

            epoch_loss = total_loss / total
            epoch_acc = correct / total
            history["train_loss"].append(epoch_loss)
            history["train_acc"].append(epoch_acc)

            # Validation
            if val_X is not None and val_y is not None:
                val_loss, val_acc = self._evaluate(val_X, val_y, batch_size)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

            if (epoch + 1) % 10 == 0:
                msg = f"Epoch {epoch+1}/{epochs}: loss={epoch_loss:.4f}, acc={epoch_acc:.4f}"
                if val_X is not None:
                    msg += f", val_loss={history['val_loss'][-1]:.4f}, val_acc={history['val_acc'][-1]:.4f}"
                logger.info(msg)

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_t)
            _, predicted = torch.max(outputs, 1)
            return predicted.cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_t)
            probs = torch.softmax(outputs, dim=1)
            return probs.cpu().numpy()

    def _evaluate(self, X: np.ndarray, y: np.ndarray, batch_size: int) -> tuple[float, float]:
        """Evaluate on a dataset."""
        self.model.eval()
        X_t = torch.FloatTensor(X).to(self.device)
        y_t = torch.LongTensor(y).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_t)
            loss = self.criterion(outputs, y_t).item()
            _, predicted = torch.max(outputs, 1)
            acc = (predicted == y_t).float().mean().item()

        return loss, acc
