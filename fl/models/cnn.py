"""
fl/models/cnn.py
================
1-D CNN over tabular feature vector for fraud detection.

Architecture:
  Input: (batch, n_features)  →  reshape to (batch, 1, n_features)
  Conv1d(1, 32, kernel=5, padding=2) → ReLU → MaxPool(2)
  Conv1d(32, 64, kernel=3, padding=1) → ReLU → MaxPool(2)
  Flatten → Linear(64 * (n_features//4), 128) → ReLU → Dropout(0.3)
  Linear(128, 1) → output logit

Federation mechanism: FedAvg on all model weights.
What crosses the bank boundary: weight tensors only.

Compatible with Opacus DP-SGD.
"""

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class CNNModel(nn.Module):
    """
    1-D CNN for binary classification on tabular fraud features.

    Parameters
    ----------
    n_features : int
        Number of input features.
    dropout : float
        Dropout probability in the classifier head.
    """

    def __init__(self, n_features: int, dropout: float = 0.3):
        super().__init__()
        self.n_features = n_features

        # Convolutional backbone
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.relu = nn.ReLU()

        # After two maxpool(2): n_features → n_features // 4
        # Guard against very small feature counts
        reduced = max(n_features // 4, 1)
        flat_dim = 64 * reduced

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_features) → (batch, 1, n_features)
        x = x.unsqueeze(1)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.classifier(x)
        return x.squeeze(-1)  # (batch,)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))

    # ------------------------------------------------------------------
    # FedAvg parameter interface
    # ------------------------------------------------------------------

    def get_parameters(self) -> list[np.ndarray]:
        return [p.detach().cpu().numpy() for p in self.parameters()]

    def set_parameters(self, params: list[np.ndarray]) -> None:
        for p, new_val in zip(self.parameters(), params):
            p.data = torch.tensor(new_val, dtype=p.dtype)

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_flat_weights(self) -> np.ndarray:
        return np.concatenate([p.detach().cpu().numpy().ravel() for p in self.parameters()])

    def set_flat_weights(self, flat: np.ndarray) -> None:
        offset = 0
        for p in self.parameters():
            size = p.numel()
            p.data = torch.tensor(
                flat[offset: offset + size].reshape(p.shape), dtype=p.dtype
            )
            offset += size


def train_epoch(
    model: CNNModel,
    X: np.ndarray,
    y: np.ndarray,
    optimizer: torch.optim.Optimizer,
    batch_size: int = 256,
    device: str = "cpu",
    class_weights: torch.Tensor | None = None,
) -> float:
    """Train for one epoch; return mean BCE loss."""
    model.train()
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    total_loss = 0.0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        if class_weights is not None:
            # Per-sample weight: w1*y + w0*(1-y)
            w = class_weights[1] * yb + class_weights[0] * (1.0 - yb)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, yb, weight=w)
        else:
            loss = nn.functional.binary_cross_entropy_with_logits(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(xb)

    return total_loss / len(dataset)
