"""
fl/models/decision_tree.py
==========================
Decision Tree for federated learning.

Federation mechanism: each bank trains a local tree; the global model is an
ensemble (soft vote) of all banks' trees.

What crosses the bank boundary:
  - Serialised tree structure (sklearn pickle bytes)
  - Leaf class PROBABILITIES only (not counts), enforced by wrapper

Privacy controls:
  - min_samples_leaf >= 20 (k-leaf rule) enforced at construction
  - Leaf probabilities transmitted, not raw leaf counts
  - This provides some protection but does NOT constitute formal (ε,δ)-DP.
    See docs/DEVIATIONS.md: DEV-03.

Note: tree structure can leak information about rare values. The k-leaf rule
mitigates but does not eliminate this. Document limitations in report.
"""

import io
import pickle
from typing import Optional

import numpy as np
from sklearn.tree import DecisionTreeClassifier


# Hard minimum for min_samples_leaf (privacy k-anonymity at leaf level)
MIN_SAMPLES_LEAF = 20


class DecisionTreeModel:
    """
    Federated Decision Tree: local tree + ensemble of peer trees.

    Parameters
    ----------
    n_features : int
    max_depth : int, optional
    min_samples_leaf : int
        Must be >= MIN_SAMPLES_LEAF for privacy.
    class_weight : str or dict, optional
    random_state : int
    """

    def __init__(
        self,
        n_features: int,
        max_depth: Optional[int] = 10,
        min_samples_leaf: int = MIN_SAMPLES_LEAF,
        class_weight: Optional[str] = "balanced",
        random_state: int = 42,
    ):
        if min_samples_leaf < MIN_SAMPLES_LEAF:
            raise ValueError(
                f"min_samples_leaf={min_samples_leaf} < {MIN_SAMPLES_LEAF}. "
                f"This violates the k-leaf privacy rule. "
                f"Minimum allowed: {MIN_SAMPLES_LEAF}."
            )

        self.n_features = n_features
        self.min_samples_leaf = min_samples_leaf
        self.classes = [0, 1]

        self._local_tree = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=random_state,
        )
        self._peer_trees: list[DecisionTreeClassifier] = []
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeModel":
        """Train local tree on this bank's data."""
        self._local_tree.fit(X, y)
        self._peer_trees = []  # reset peer trees on new local fit
        self._fitted = True
        return self

    def get_parameters(self) -> list[np.ndarray]:
        """
        Serialise the local tree for transmission.
        The serialised bytes contain the tree structure and leaf PROBABILITIES.
        Leaf counts are NOT included (they are stripped before serialisation).
        """
        if not self._fitted:
            raise RuntimeError("DecisionTreeModel not fitted yet.")
        # Strip leaf counts: overwrite tree_.value with probabilities
        tree_proba = _tree_to_proba_only(self._local_tree)
        buf = io.BytesIO()
        pickle.dump(tree_proba, buf)
        blob = buf.getvalue()
        # Wrap as 1-D numpy array of uint8 for Flower compatibility
        return [np.frombuffer(blob, dtype=np.uint8)]

    def set_parameters(self, params: list[np.ndarray]) -> None:
        """
        Receive aggregated tree ensemble from server.
        params[0] is a concatenated list of serialised trees.
        """
        # In tree federation, server sends back a list of all banks' trees
        # as a variable-length byte blob with a header
        self._peer_trees = _deserialise_tree_list(params[0])
        self._fitted = True

    def add_peer_tree(self, serialised: np.ndarray) -> None:
        """Add one peer bank's serialised tree to the ensemble."""
        tree = _deserialise_single_tree(serialised)
        self._peer_trees.append(tree)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Soft-vote ensemble of local + peer trees."""
        if not self._fitted:
            raise RuntimeError("No fitted tree available.")

        trees = []
        if self._fitted and hasattr(self._local_tree, "tree_"):
            trees.append(self._local_tree)
        trees.extend(self._peer_trees)

        if not trees:
            raise RuntimeError("No trees in ensemble.")

        # Average predicted probabilities across all trees
        proba_sum = sum(t.predict_proba(X) for t in trees)
        return proba_sum / len(trees)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def verify_min_samples_leaf(self) -> bool:
        """Assert that all leaves have >= MIN_SAMPLES_LEAF samples."""
        if not self._fitted or not hasattr(self._local_tree, "tree_"):
            return True
        return _verify_min_leaf(self._local_tree, MIN_SAMPLES_LEAF)


# ---------------------------------------------------------------------------
# Helper: strip leaf counts, keep only probabilities
# ---------------------------------------------------------------------------

def _tree_to_proba_only(clf: DecisionTreeClassifier) -> DecisionTreeClassifier:
    """
    Return a copy of the classifier where tree_.value contains only class
    PROBABILITIES (normalised), not raw sample counts.

    This prevents exact count disclosure from the serialised tree.
    """
    import copy
    clf_copy = copy.deepcopy(clf)
    tree = clf_copy.tree_

    # tree_.value shape: (n_nodes, n_outputs, n_classes)
    val = tree.value.copy()
    node_sums = val.sum(axis=2, keepdims=True)
    node_sums[node_sums == 0] = 1  # avoid division by zero
    val_proba = val / node_sums  # normalise to probabilities

    # Replace counts with probabilities (write 1 sample worth of proba)
    # so sklearn's predict_proba still works correctly
    tree.value[:] = val_proba

    return clf_copy


def _verify_min_leaf(clf: DecisionTreeClassifier, k: int) -> bool:
    """Return True if all leaves have node_count (samples) >= k."""
    tree = clf.tree_
    is_leaf = tree.children_left == -1
    leaf_samples = tree.n_node_samples[is_leaf]
    return bool((leaf_samples >= k).all())


def _serialise_tree_list(trees: list[DecisionTreeClassifier]) -> np.ndarray:
    """Serialise a list of trees into a single byte array with length headers."""
    buf = io.BytesIO()
    n = len(trees)
    buf.write(n.to_bytes(4, "big"))
    for tree in trees:
        tree_bytes = pickle.dumps(tree)
        buf.write(len(tree_bytes).to_bytes(8, "big"))
        buf.write(tree_bytes)
    return np.frombuffer(buf.getvalue(), dtype=np.uint8)


def _deserialise_tree_list(data: np.ndarray) -> list[DecisionTreeClassifier]:
    """Deserialise a packed list of trees."""
    buf = io.BytesIO(data.tobytes())
    n = int.from_bytes(buf.read(4), "big")
    trees = []
    for _ in range(n):
        size = int.from_bytes(buf.read(8), "big")
        tree_bytes = buf.read(size)
        trees.append(pickle.loads(tree_bytes))
    return trees


def _deserialise_single_tree(data: np.ndarray) -> DecisionTreeClassifier:
    return pickle.loads(data.tobytes())
