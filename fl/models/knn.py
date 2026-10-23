"""
fl/models/knn.py
================
K-Nearest Neighbours — LOCAL AND CENTRALIZED BASELINES ONLY.

KNN is a non-parametric model whose "parameters" ARE the training data.
Federating KNN would require transmitting raw training records to other
banks, directly violating the privacy invariant.

See docs/DEVIATIONS.md: DEV-01.

This class raises PrivacyViolationError if federation is attempted.
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier


class PrivacyViolationError(Exception):
    """Raised when an operation would violate the privacy invariant."""
    pass


class KNNModel:
    """
    KNN classifier for local-only and centralised baselines.

    Parameters
    ----------
    n_neighbors : int
    metric : str
    n_jobs : int
    """

    def __init__(
        self,
        n_features: int,
        n_neighbors: int = 5,
        metric: str = "minkowski",
        n_jobs: int = -1,
    ):
        self.n_features = n_features
        self.n_neighbors = n_neighbors
        self.classes = [0, 1]
        self._clf = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            metric=metric,
            n_jobs=n_jobs,
        )
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNModel":
        self._clf.fit(X, y)
        self._fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("KNNModel not fitted.")
        return self._clf.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("KNNModel not fitted.")
        return self._clf.predict(X)

    # ------------------------------------------------------------------
    # Federation BLOCKED — intentional PrivacyViolationError
    # ------------------------------------------------------------------

    def get_parameters(self) -> list[np.ndarray]:
        """
        RAISES PrivacyViolationError.

        KNN's "parameters" are the training data themselves.
        Calling this would expose raw records, violating the privacy invariant.
        See docs/DEVIATIONS.md: DEV-01.
        """
        raise PrivacyViolationError(
            "KNN cannot be federated without transmitting raw training data. "
            "This violates the privacy invariant. "
            "KNN is only supported as a local-only or centralized baseline. "
            "See docs/DEVIATIONS.md: DEV-01."
        )

    def set_parameters(self, params: list[np.ndarray]) -> None:
        raise PrivacyViolationError(
            "KNN cannot receive federated parameters. See docs/DEVIATIONS.md: DEV-01."
        )
