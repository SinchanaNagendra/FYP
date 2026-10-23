"""
fl/models/random_forest.py
==========================
Random Forest for federated learning.

Federation mechanism: federated bagging — each bank trains a local forest;
the global model is the union of all banks' forests (all individual trees
from all banks merged into one ensemble).

What crosses the bank boundary:
  - Serialised individual decision trees (leaf probabilities only, not counts)
  - Same privacy guarantees as decision_tree.py (k-leaf rule)

Privacy: same as DecisionTreeModel — k-leaf rule, probabilities not counts.
No formal (ε,δ)-DP. See docs/DEVIATIONS.md: DEV-03.
"""

import io
import pickle
from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from .decision_tree import _tree_to_proba_only, _verify_min_leaf, MIN_SAMPLES_LEAF


class RandomForestModel:
    """
    Federated Random Forest via federated bagging (union of local forests).

    Parameters
    ----------
    n_features : int
    n_estimators : int
        Number of trees in THIS bank's local forest.
    max_depth : int, optional
    min_samples_leaf : int
        Must be >= MIN_SAMPLES_LEAF.
    class_weight : str or dict, optional
    n_jobs : int
    random_state : int
    """

    def __init__(
        self,
        n_features: int,
        n_estimators: int = 50,
        max_depth: Optional[int] = 10,
        min_samples_leaf: int = MIN_SAMPLES_LEAF,
        class_weight: Optional[str] = "balanced",
        n_jobs: int = -1,
        random_state: int = 42,
    ):
        if min_samples_leaf < MIN_SAMPLES_LEAF:
            raise ValueError(
                f"min_samples_leaf={min_samples_leaf} < {MIN_SAMPLES_LEAF}. "
                f"Violates k-leaf privacy rule."
            )

        self.n_features = n_features
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.classes = [0, 1]

        self._local_forest = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        self._global_trees: list[DecisionTreeClassifier] = []
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestModel":
        """Train local forest on this bank's data."""
        self._local_forest.fit(X, y)
        self._global_trees = list(self._local_forest.estimators_)
        self._fitted = True
        return self

    def get_parameters(self) -> list[np.ndarray]:
        """
        Serialise the local forest's individual trees for transmission.
        Each tree has leaf counts replaced with probabilities.
        """
        if not self._fitted:
            raise RuntimeError("RandomForestModel not fitted.")

        trees_proba = [_tree_to_proba_only(t) for t in self._local_forest.estimators_]

        buf = io.BytesIO()
        n = len(trees_proba)
        buf.write(n.to_bytes(4, "big"))
        for tree in trees_proba:
            tree_bytes = pickle.dumps(tree)
            buf.write(len(tree_bytes).to_bytes(8, "big"))
            buf.write(tree_bytes)

        return [np.frombuffer(buf.getvalue(), dtype=np.uint8)]

    def set_parameters(self, params: list[np.ndarray]) -> None:
        """
        Receive and add peer banks' trees to global ensemble.
        Called by the federation strategy after collecting all bank trees.
        """
        peer_trees = _deserialise_forest(params[0])
        self._global_trees = peer_trees
        self._fitted = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Soft vote over all trees in the global ensemble."""
        if not self._fitted or not self._global_trees:
            raise RuntimeError("RandomForestModel: no trees in global ensemble.")

        proba_sum = sum(t.predict_proba(X) for t in self._global_trees)
        return proba_sum / len(self._global_trees)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def verify_min_samples_leaf(self) -> bool:
        """Verify k-leaf rule across all local trees."""
        if not self._fitted:
            return True
        return all(
            _verify_min_leaf(t, MIN_SAMPLES_LEAF)
            for t in self._local_forest.estimators_
        )


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _deserialise_forest(data: np.ndarray) -> list[DecisionTreeClassifier]:
    buf = io.BytesIO(data.tobytes())
    n = int.from_bytes(buf.read(4), "big")
    trees = []
    for _ in range(n):
        size = int.from_bytes(buf.read(8), "big")
        tree_bytes = buf.read(size)
        trees.append(pickle.loads(tree_bytes))
    return trees


def aggregate_forests(bank_params: list[list[np.ndarray]]) -> list[np.ndarray]:
    """
    Aggregate all banks' forest parameters into one serialised global forest.
    Called by the federation strategy.

    Parameters
    ----------
    bank_params : list of [serialised_trees] per bank

    Returns
    -------
    [serialised_all_trees]
    """
    all_trees = []
    for params in bank_params:
        all_trees.extend(_deserialise_forest(params[0]))

    buf = io.BytesIO()
    n = len(all_trees)
    buf.write(n.to_bytes(4, "big"))
    for tree in all_trees:
        tree_bytes = pickle.dumps(tree)
        buf.write(len(tree_bytes).to_bytes(8, "big"))
        buf.write(tree_bytes)

    return [np.frombuffer(buf.getvalue(), dtype=np.uint8)]
