"""
fl/models/logistic_regression.py
==================================
Logistic Regression implemented as a 1-layer PyTorch model.

Federation mechanism: FedAvg on model weights (weight vector + bias).
What crosses the bank boundary: weight tensor only.

Compatible with Opacus DP-SGD.
"""

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class LogisticRegressionModel(nn.Module):
    """
    Binary logistic regression as a single linear layer.

    Parameters
    ----------
    n_features : int
        Number of input features (from preprocessing schema).
    """

    def __init__(self, n_features: int):
        super().__init__()
        self.n_features = n_features
        self.linear = nn.Linear(n_features, 1)
        # Initialise near zero for stable training
        nn.init.xavier_uniform_(self.linear.weight.unsqueeze(0))
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)  # (batch,)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)

    # ------------------------------------------------------------------
    # Flower / FedAvg parameter interface
    # ------------------------------------------------------------------

    def get_parameters(self) -> list[np.ndarray]:
        """Return model weights as a list of numpy arrays (FedAvg compatible)."""
        return [p.detach().cpu().numpy() for p in self.parameters()]

    def set_parameters(self, params: list[np.ndarray]) -> None:
        """Set model weights from a list of numpy arrays."""
        for p, new_val in zip(self.parameters(), params):
            p.data = torch.tensor(new_val, dtype=p.dtype)

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_flat_weights(self) -> np.ndarray:
        """Flatten all parameters into a single 1-D numpy array."""
        return np.concatenate([p.detach().cpu().numpy().ravel() for p in self.parameters()])

    def set_flat_weights(self, flat: np.ndarray) -> None:
        """Set parameters from a flattened 1-D array."""
        offset = 0
        for p in self.parameters():
            size = p.numel()
            p.data = torch.tensor(
                flat[offset: offset + size].reshape(p.shape), dtype=p.dtype
            )
            offset += size


def train_epoch(
    model: LogisticRegressionModel,
    X: np.ndarray,
    y: np.ndarray,
    optimizer: torch.optim.Optimizer,
    batch_size: int = 256,
    device: str = "cpu",
) -> float:
    """
    Train for one epoch. Returns mean BCE loss.
    Does NOT apply DP; use fl/dp/dp_trainer.py for DP-SGD.
    """
    model.train()
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(xb)

    return total_loss / len(dataset)
