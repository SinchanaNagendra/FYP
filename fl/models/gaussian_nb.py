"""
fl/models/gaussian_nb.py
=========================
Gaussian Naive Bayes with federated aggregation via exact sufficient statistics.

Federation mechanism: per-class sufficient statistics (count, mean, variance)
are aggregated via secure sum across banks. No raw data is transmitted.

What crosses the bank boundary:
  - Per-class: (n_k, sum_k, sum_sq_k) for each feature dimension
  - These aggregate exactly into global (mu, sigma^2) per class

Privacy note: per-class counts are transmitted. For very small classes,
this could reveal presence of rare samples. Enforce min_class_count ≥ k
(default k=5) and document this in DEVIATIONS.md.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.naive_bayes import GaussianNB


# Minimum samples per class to transmit stats (privacy guard)
MIN_CLASS_COUNT = 5


class GaussianNBModel:
    """
    Gaussian NB with exact federated aggregation.

    Local training computes per-class sufficient statistics.
    Global model is reconstructed from the aggregate of all banks' statistics.
    """

    def __init__(self, n_features: int, classes: list[int] = None):
        self.n_features = n_features
        self.classes = classes or [0, 1]
        self.n_classes = len(self.classes)

        # Per-class sufficient statistics (accumulated locally)
        self._counts = np.zeros(self.n_classes, dtype=np.int64)
        self._sum = np.zeros((self.n_classes, n_features), dtype=np.float64)
        self._sum_sq = np.zeros((self.n_classes, n_features), dtype=np.float64)

        # Global model parameters (set after aggregation)
        self._priors = None
        self._means = None
        self._vars = None
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianNBModel":
        """
        Train locally: accumulate sufficient statistics.
        Does NOT expose raw data; only stats are computed.
        """
        for k, cls in enumerate(self.classes):
            mask = (y == cls)
            count = int(mask.sum())
            if count == 0:
                continue
            Xk = X[mask]
            self._counts[k] += count
            self._sum[k] += Xk.sum(axis=0)
            self._sum_sq[k] += (Xk ** 2).sum(axis=0)

        self._rebuild_from_local_stats()
        return self

    def _rebuild_from_local_stats(self) -> None:
        """Rebuild the GNB parameters from accumulated sufficient statistics."""
        total = self._counts.sum()
        if total == 0:
            return
        self._priors = self._counts / total
        self._means = np.zeros_like(self._sum)
        self._vars = np.ones_like(self._sum)  # default var=1 to avoid division by zero

        for k in range(self.n_classes):
            n = self._counts[k]
            if n == 0:
                continue
            mu = self._sum[k] / n
            self._means[k] = mu
            var = self._sum_sq[k] / n - mu ** 2
            self._vars[k] = np.clip(var, 1e-9, None)

        self._fitted = True

    def get_sufficient_stats(self) -> dict:
        """
        Return per-class sufficient statistics for transmission.
        These are the ONLY values that leave the bank.
        """
        # Privacy guard: refuse to transmit if any class has fewer than MIN_CLASS_COUNT
        for k, cls in enumerate(self.classes):
            if 0 < self._counts[k] < MIN_CLASS_COUNT:
                raise ValueError(
                    f"Class {cls} has only {self._counts[k]} samples. "
                    f"Minimum is {MIN_CLASS_COUNT} to prevent individual identification."
                )

        return {
            "counts": self._counts.tolist(),
            "sum": self._sum.tolist(),
            "sum_sq": self._sum_sq.tolist(),
            "n_features": self.n_features,
            "classes": self.classes,
        }

    def set_aggregate_stats(self, agg_stats: dict) -> None:
        """
        Set model parameters from globally aggregated sufficient statistics.
        Called after the secure-sum round.
        """
        self._counts = np.array(agg_stats["counts"], dtype=np.int64)
        self._sum = np.array(agg_stats["sum"], dtype=np.float64)
        self._sum_sq = np.array(agg_stats["sum_sq"], dtype=np.float64)
        self._rebuild_from_local_stats()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return (n_samples, n_classes) probability array."""
        if not self._fitted:
            raise RuntimeError("GaussianNBModel is not fitted yet.")

        n_samples = len(X)
        log_probs = np.zeros((n_samples, self.n_classes))

        for k in range(self.n_classes):
            # Log prior
            lp = np.log(self._priors[k] + 1e-10)
            # Log likelihood: Gaussian per feature
            mu = self._means[k]
            var = self._vars[k]
            ll = -0.5 * np.sum(np.log(2 * np.pi * var + 1e-10), axis=0)
            ll += -0.5 * np.sum(((X - mu) ** 2) / (var + 1e-10), axis=1)
            log_probs[:, k] = lp + ll

        # Softmax
        log_probs -= log_probs.max(axis=1, keepdims=True)
        probs = np.exp(log_probs)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return self.classes[np.argmax(probs, axis=1)]

    # ------------------------------------------------------------------
    # FedAvg-compatible parameter interface
    # ------------------------------------------------------------------

    def get_parameters(self) -> list[np.ndarray]:
        """
        For FedAvg: return sufficient stats as parameter arrays.
        NOTE: GNB uses exact aggregation, not FedAvg; this interface is for
        compatibility with the Flower strategy dispatcher only.
        """
        return [
            self._counts.astype(np.float64),
            self._sum,
            self._sum_sq,
        ]

    def set_parameters(self, params: list[np.ndarray]) -> None:
        self._counts = params[0].astype(np.int64)
        self._sum = params[1].astype(np.float64)
        self._sum_sq = params[2].astype(np.float64)
        self._rebuild_from_local_stats()


def aggregate_gnb_params(bank_params: list[list[np.ndarray]]) -> list[np.ndarray]:
    """
    Aggregate GNB sufficient statistics from multiple banks.
    This is called once all banks have submitted their stats (via secure sum).

    Parameters
    ----------
    bank_params : list of [counts, sum, sum_sq] per bank

    Returns
    -------
    [agg_counts, agg_sum, agg_sum_sq]
    """
    agg_counts = sum(p[0] for p in bank_params)
    agg_sum = sum(p[1] for p in bank_params)
    agg_sum_sq = sum(p[2] for p in bank_params)
    return [agg_counts, agg_sum, agg_sum_sq]
