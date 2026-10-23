"""
fl/dp/dp_trainer.py
====================
DP-SGD training wrapper using Opacus.

Applies to: Logistic Regression and CNN (gradient-based models only).
Does NOT apply to tree models — see docs/DEVIATIONS.md: DEV-03.

Opacus enforces:
  - Per-sample gradient clipping (max_grad_norm = C)
  - Gaussian noise addition (noise_multiplier = σ)
  - RDP (Rényi Differential Privacy) accounting

Privacy guarantee per round:
  ε is computed by the RDP accountant for the given (σ, C, batch_size, n_samples, n_steps).

Usage:
    trainer = DPTrainer(
        model=lr_model,
        max_grad_norm=1.0,
        noise_multiplier=1.1,
        batch_size=256,
        delta=1e-5,
    )
    for epoch in range(n_epochs):
        loss = trainer.train_epoch(X_train, y_train)
    eps = trainer.get_epsilon(delta=1e-5)
"""

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class DPTrainer:
    """
    Wraps a PyTorch model with Opacus PrivacyEngine for DP-SGD training.

    Parameters
    ----------
    model : nn.Module
        A LogisticRegressionModel or CNNModel instance.
    max_grad_norm : float
        Per-sample gradient clipping bound (C in the DP-SGD algorithm).
    noise_multiplier : float
        Ratio of noise std to clipping bound (σ/C). Higher = more privacy.
        Suggested values for (ε, δ=1e-5):
          noise_multiplier=0.3 → ε≈∞ (essentially no DP)
          noise_multiplier=0.5 → ε≈8
          noise_multiplier=0.8 → ε≈3
          noise_multiplier=1.1 → ε≈1
    batch_size : int
    delta : float
        Target δ for the (ε, δ)-DP guarantee. Typically 1/n_samples.
    lr : float
        Learning rate for SGD (Opacus replaces Adam with DP-compatible SGD).
    device : str
    """

    def __init__(
        self,
        model: nn.Module,
        max_grad_norm: float = 1.0,
        noise_multiplier: float = 1.1,
        batch_size: int = 256,
        delta: float = 1e-5,
        lr: float = 0.01,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier
        self.batch_size = batch_size
        self.delta = delta
        self.device = device
        self._dp_enabled = False
        self._privacy_engine = None
        self._optimizer = None
        self._n_samples = None

        # We defer Opacus attachment to the first train call (need dataset size)
        self._lr = lr

    def _attach_opacus(self, n_samples: int, n_epochs: int) -> None:
        """Attach Opacus PrivacyEngine to the model and optimizer."""
        try:
            from opacus import PrivacyEngine
        except ImportError:
            raise ImportError(
                "Opacus is required for DP training. "
                "Install with: pip install opacus==1.4.0"
            )

        self._optimizer = torch.optim.SGD(self.model.parameters(), lr=self._lr)
        self._privacy_engine = PrivacyEngine()

        # Opacus requires a DataLoader to set up correctly; we create a minimal one
        dummy_dataset = TensorDataset(
            torch.zeros(n_samples, self.model.n_features),
            torch.zeros(n_samples),
        )
        dummy_loader = DataLoader(dummy_dataset, batch_size=self.batch_size, shuffle=True)

        self.model, self._optimizer, self._dp_loader_template = (
            self._privacy_engine.make_private(
                module=self.model,
                optimizer=self._optimizer,
                data_loader=dummy_loader,
                noise_multiplier=self.noise_multiplier,
                max_grad_norm=self.max_grad_norm,
            )
        )
        self._dp_enabled = True
        self._n_samples = n_samples
        logger.info(
            f"[DP] Opacus attached. noise_multiplier={self.noise_multiplier}, "
            f"max_grad_norm={self.max_grad_norm}, batch_size={self.batch_size}, "
            f"n_samples={n_samples}"
        )

    def train_epoch(
        self,
        X: np.ndarray,
        y: np.ndarray,
        class_weight_tensor: Optional[torch.Tensor] = None,
    ) -> float:
        """
        Train for one epoch with DP-SGD.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)
        class_weight_tensor : optional torch.Tensor, shape (2,)
            Class weights for weighted BCE loss.

        Returns
        -------
        mean_loss : float
        """
        n_samples = len(X)

        if not self._dp_enabled:
            self._attach_opacus(n_samples, n_epochs=1)

        self.model.train()
        dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        criterion = nn.BCEWithLogitsLoss()
        total_loss = 0.0

        for xb, yb in loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            self._optimizer.zero_grad()
            logits = self.model(xb)

            if class_weight_tensor is not None:
                w = class_weight_tensor[1] * yb + class_weight_tensor[0] * (1.0 - yb)
                loss = nn.functional.binary_cross_entropy_with_logits(logits, yb, weight=w)
            else:
                loss = criterion(logits, yb)

            loss.backward()
            self._optimizer.step()
            total_loss += loss.item() * len(xb)

        return total_loss / n_samples

    def get_epsilon(self, delta: Optional[float] = None) -> float:
        """
        Compute the current (ε, δ) privacy budget spent so far.

        Returns ε for the given δ using the RDP accountant.
        Returns float('inf') if DP is not enabled.
        """
        if not self._dp_enabled or self._privacy_engine is None:
            logger.warning("[DP] Privacy engine not attached; returning ε=inf")
            return float("inf")

        delta = delta or self.delta
        eps = self._privacy_engine.get_epsilon(delta=delta)
        logger.info(f"[DP] Current privacy budget: ε={eps:.4f}, δ={delta}")
        return float(eps)

    def get_privacy_spent(self) -> dict:
        """Return a dict with current privacy budget."""
        eps = self.get_epsilon()
        return {
            "epsilon": eps,
            "delta": self.delta,
            "noise_multiplier": self.noise_multiplier,
            "max_grad_norm": self.max_grad_norm,
        }

    @property
    def underlying_model(self) -> nn.Module:
        """Return the underlying (possibly Opacus-wrapped) model."""
        return self.model
