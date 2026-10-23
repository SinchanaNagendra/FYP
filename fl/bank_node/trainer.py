"""
fl/bank_node/trainer.py
========================
Local training loop for a bank node.

Supports:
  - Standard training (all models)
  - DP-SGD training (LR, CNN via Opacus)
  - FedProx proximal term (LR, CNN)
  - All seven balancing methods

Privacy invariant: this module operates only on the bank's local data.
It never sends data out; it only produces model parameters (or sufficient stats).
"""

import logging
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from fl.balancing.methods import BalancingMethod, apply_balancing, compute_class_weights

logger = logging.getLogger(__name__)


class LocalTrainer:
    """
    Manages local training for one bank across all model types.

    Parameters
    ----------
    model_type : str
        'logistic_regression' | 'cnn' | 'gaussian_nb' |
        'decision_tree' | 'random_forest' | 'knn'
    model : model instance
    balancing : BalancingMethod
    n_epochs : int
    batch_size : int
    lr : float
    device : str
    use_dp : bool
        If True, wrap with Opacus DP-SGD (only for LR/CNN).
    dp_max_grad_norm : float
    dp_noise_multiplier : float
    dp_delta : float
    fedprox_mu : float
        If > 0, apply FedProx proximal term. Only for LR/CNN.
    random_state : int
    """

    TORCH_MODELS = {"logistic_regression", "cnn"}
    SKLEARN_MODELS = {"gaussian_nb", "decision_tree", "random_forest"}
    LOCAL_ONLY = {"knn"}

    def __init__(
        self,
        model_type: str,
        model,
        balancing: BalancingMethod = BalancingMethod.NONE,
        n_epochs: int = 5,
        batch_size: int = 256,
        lr: float = 0.01,
        device: str = "cpu",
        use_dp: bool = False,
        dp_max_grad_norm: float = 1.0,
        dp_noise_multiplier: float = 1.1,
        dp_delta: float = 1e-5,
        fedprox_mu: float = 0.0,
        random_state: int = 42,
    ):
        self.model_type = model_type.lower()
        self.model = model
        self.balancing = balancing
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.use_dp = use_dp
        self.dp_max_grad_norm = dp_max_grad_norm
        self.dp_noise_multiplier = dp_noise_multiplier
        self.dp_delta = dp_delta
        self.fedprox_mu = fedprox_mu
        self.random_state = random_state
        self._dp_trainer = None
        self._privacy_spent = None

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        global_params: Optional[list] = None,  # for FedProx
    ) -> Dict[str, Any]:
        """
        Train the model on the bank's local training data.

        Parameters
        ----------
        X_train : np.ndarray, shape (n_samples, n_features)
        y_train : np.ndarray, shape (n_samples,)
        global_params : optional list of np.ndarray
            Current global model parameters (for FedProx proximal term).

        Returns
        -------
        result : dict with keys:
            - 'train_time': float (seconds)
            - 'n_samples': int
            - 'losses': list of float (per epoch, torch models only)
            - 'privacy_spent': dict or None
        """
        start = time.time()

        # Apply balancing (training data only, never val/test)
        X_bal, y_bal = apply_balancing(
            X_train, y_train, self.balancing, random_state=self.random_state
        )
        n_samples = len(X_bal)

        losses = []

        if self.model_type in self.TORCH_MODELS:
            losses = self._train_torch(X_bal, y_bal, global_params)
        elif self.model_type in self.SKLEARN_MODELS:
            self._train_sklearn(X_bal, y_bal)
        elif self.model_type in self.LOCAL_ONLY:
            # KNN: local fit only
            self.model.fit(X_bal, y_bal)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        elapsed = time.time() - start
        logger.info(
            f"[Trainer] model={self.model_type}, balancing={self.balancing.value}, "
            f"n_samples={n_samples}, time={elapsed:.2f}s"
        )

        return {
            "train_time": elapsed,
            "n_samples": int(n_samples),
            "losses": losses,
            "privacy_spent": self._privacy_spent,
        }

    def _train_torch(
        self,
        X: np.ndarray,
        y: np.ndarray,
        global_params: Optional[list],
    ) -> list:
        """Train a PyTorch model (LR or CNN) for n_epochs."""
        losses = []

        # Class weights for weighted BCE
        class_weight_tensor = None
        if self.balancing == BalancingMethod.WEIGHTED:
            cw = compute_class_weights(y)
            class_weight_tensor = torch.tensor(
                [cw.get(0, 1.0), cw.get(1, 1.0)], dtype=torch.float32
            ).to(self.device)

        if self.use_dp:
            # Initialise DP trainer
            from fl.dp.dp_trainer import DPTrainer
            if self._dp_trainer is None:
                self._dp_trainer = DPTrainer(
                    model=self.model,
                    max_grad_norm=self.dp_max_grad_norm,
                    noise_multiplier=self.dp_noise_multiplier,
                    batch_size=self.batch_size,
                    delta=self.dp_delta,
                    lr=self.lr,
                    device=self.device,
                )
            for epoch in range(self.n_epochs):
                loss = self._dp_trainer.train_epoch(X, y, class_weight_tensor)
                losses.append(loss)
            self._privacy_spent = self._dp_trainer.get_privacy_spent()
        else:
            # Standard SGD
            self.model.to(self.device)
            self.model.train()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            dataset = TensorDataset(
                torch.tensor(X, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32),
            )
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            for epoch in range(self.n_epochs):
                epoch_loss = 0.0
                for xb, yb in loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    optimizer.zero_grad()
                    logits = self.model(xb)

                    if class_weight_tensor is not None:
                        w = (class_weight_tensor[1] * yb
                             + class_weight_tensor[0] * (1.0 - yb))
                        loss = nn.functional.binary_cross_entropy_with_logits(
                            logits, yb, weight=w
                        )
                    else:
                        loss = nn.functional.binary_cross_entropy_with_logits(logits, yb)

                    # FedProx proximal term
                    if self.fedprox_mu > 0 and global_params is not None:
                        from fl.strategies.fabric_fedprox import FabricFedProx
                        strategy = FabricFedProx(n_banks=1, mu=self.fedprox_mu)
                        prox = strategy.proximal_loss(self.model, global_params, self.device)
                        loss = loss + prox

                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item() * len(xb)

                losses.append(epoch_loss / len(dataset))

        return losses

    def _train_sklearn(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train a scikit-learn or custom sklearn-compatible model."""
        if self.model_type == "gaussian_nb":
            self.model.fit(X, y)
        elif self.model_type in ("decision_tree", "random_forest"):
            self.model.fit(X, y)
        else:
            raise ValueError(f"Unknown sklearn model: {self.model_type}")

    def get_parameters(self) -> list:
        """Get model parameters for transmission."""
        if self.model_type in self.LOCAL_ONLY:
            raise RuntimeError(
                f"Model type '{self.model_type}' cannot be federated. "
                "Use it only as a local/centralized baseline."
            )
        return self.model.get_parameters()
