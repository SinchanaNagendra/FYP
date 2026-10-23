"""
tests/unit/test_models.py
==========================
Unit tests for all six model classes.

Tests:
  - LR: forward pass, get/set parameters, FedAvg weight roundtrip
  - CNN: forward pass, get/set parameters
  - GNB: fit, sufficient stats, set_aggregate_stats, predict_proba
  - DT: fit, k-leaf rule enforced, leaf probabilities not counts
  - RF: fit, k-leaf rule enforced, aggregate_forests
  - KNN: raises PrivacyViolationError on get_parameters/set_parameters

Run with: pytest tests/unit/test_models.py -v
"""

import numpy as np
import pytest
import torch

from fl.models import (
    LogisticRegressionModel,
    CNNModel,
    GaussianNBModel,
    DecisionTreeModel,
    RandomForestModel,
    KNNModel,
)
from fl.models.knn import PrivacyViolationError
from fl.models.decision_tree import MIN_SAMPLES_LEAF, _verify_min_leaf
from fl.models.gaussian_nb import aggregate_gnb_params
from fl.models.random_forest import aggregate_forests


N_FEATURES = 64
N_SAMPLES = 500
RNG = np.random.default_rng(42)


def _make_data(n=N_SAMPLES, n_features=N_FEATURES, fraud_rate=0.1):
    X = RNG.standard_normal((n, n_features)).astype(np.float32)
    y = (RNG.random(n) < fraud_rate).astype(np.int32)
    return X, y


# ---------------------------------------------------------------------------
# Logistic Regression
# ---------------------------------------------------------------------------

class TestLogisticRegression:

    def test_forward_pass(self):
        model = LogisticRegressionModel(N_FEATURES)
        X, _ = _make_data()
        xb = torch.tensor(X[:10])
        out = model(xb)
        assert out.shape == (10,)

    def test_predict_proba_range(self):
        model = LogisticRegressionModel(N_FEATURES)
        X, _ = _make_data()
        probs = model.predict_proba(torch.tensor(X))
        assert probs.min() >= 0.0 and probs.max() <= 1.0

    def test_get_set_parameters_roundtrip(self):
        model = LogisticRegressionModel(N_FEATURES)
        params = model.get_parameters()
        # Modify first param
        params_mod = [p + 1.0 for p in params]
        model.set_parameters(params_mod)
        params_new = model.get_parameters()
        for p_mod, p_new in zip(params_mod, params_new):
            np.testing.assert_allclose(p_mod, p_new, atol=1e-5)

    def test_flat_weights_roundtrip(self):
        model = LogisticRegressionModel(N_FEATURES)
        flat = model.get_flat_weights()
        model.set_flat_weights(flat + 0.5)
        flat2 = model.get_flat_weights()
        np.testing.assert_allclose(flat2, flat + 0.5, atol=1e-5)

    def test_parameter_count(self):
        model = LogisticRegressionModel(N_FEATURES)
        assert model.get_parameter_count() == N_FEATURES + 1  # weights + bias


# ---------------------------------------------------------------------------
# CNN
# ---------------------------------------------------------------------------

class TestCNN:

    def test_forward_pass(self):
        model = CNNModel(N_FEATURES)
        X, _ = _make_data()
        xb = torch.tensor(X[:10])
        out = model(xb)
        assert out.shape == (10,)

    def test_predict_proba_range(self):
        model = CNNModel(N_FEATURES)
        X, _ = _make_data()
        probs = model.predict_proba(torch.tensor(X))
        assert probs.min() >= 0.0 and probs.max() <= 1.0

    def test_get_set_parameters_roundtrip(self):
        model = CNNModel(N_FEATURES)
        params_orig = model.get_parameters()
        model_2 = CNNModel(N_FEATURES)
        model_2.set_parameters(params_orig)
        params_new = model_2.get_parameters()
        for p1, p2 in zip(params_orig, params_new):
            np.testing.assert_allclose(p1, p2, atol=1e-5)


# ---------------------------------------------------------------------------
# Gaussian NB
# ---------------------------------------------------------------------------

class TestGaussianNB:

    def test_fit_and_predict(self):
        X, y = _make_data(n=300)
        model = GaussianNBModel(N_FEATURES)
        model.fit(X, y)
        proba = model.predict_proba(X[:10])
        assert proba.shape == (10, 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_sufficient_stats_schema(self):
        X, y = _make_data(n=300, fraud_rate=0.3)
        model = GaussianNBModel(N_FEATURES)
        model.fit(X, y)
        stats = model.get_sufficient_stats()
        assert "counts" in stats
        assert "sum" in stats
        assert "sum_sq" in stats
        assert len(stats["counts"]) == 2
        assert len(stats["sum"]) == 2
        assert len(stats["sum"][0]) == N_FEATURES

    def test_aggregate_stats_matches_pooled(self):
        """Aggregating two banks' stats should equal fitting on pooled data."""
        X1, y1 = _make_data(n=200, fraud_rate=0.1)
        X2, y2 = _make_data(n=200, fraud_rate=0.2)
        X_all = np.concatenate([X1, X2])
        y_all = np.concatenate([y1, y2])

        m1 = GaussianNBModel(N_FEATURES)
        m1.fit(X1, y1)
        m2 = GaussianNBModel(N_FEATURES)
        m2.fit(X2, y2)

        # Aggregate
        agg_params = aggregate_gnb_params(
            [m1.get_parameters(), m2.get_parameters()])
        m_agg = GaussianNBModel(N_FEATURES)
        m_agg.set_parameters(agg_params)

        # Pooled
        m_pool = GaussianNBModel(N_FEATURES)
        m_pool.fit(X_all, y_all)

        # Predicted probabilities should be similar (not exact due to float ops)
        proba_agg = m_agg.predict_proba(X_all[:20])
        proba_pool = m_pool.predict_proba(X_all[:20])
        np.testing.assert_allclose(proba_agg, proba_pool, atol=1e-4)

    def test_min_class_count_guard(self):
        """Should raise if minority class has fewer than MIN_CLASS_COUNT samples."""
        X = RNG.standard_normal((100, N_FEATURES)).astype(np.float32)
        y = np.array([0] * 98 + [1, 1])  # only 2 fraud samples
        model = GaussianNBModel(N_FEATURES)
        model.fit(X, y)
        with pytest.raises(ValueError, match="Minimum is"):
            model.get_sufficient_stats()


# ---------------------------------------------------------------------------
# Decision Tree
# ---------------------------------------------------------------------------

class TestDecisionTree:

    def test_fit_and_predict(self):
        X, y = _make_data(n=400, fraud_rate=0.2)
        model = DecisionTreeModel(N_FEATURES)
        model.fit(X, y)
        proba = model.predict_proba(X[:10])
        assert proba.shape == (10, 2)

    def test_min_samples_leaf_enforced_at_construction(self):
        with pytest.raises(ValueError, match="k-leaf"):
            DecisionTreeModel(N_FEATURES, min_samples_leaf=5)

    def test_min_samples_leaf_satisfied_after_fit(self):
        X, y = _make_data(n=1000, fraud_rate=0.3)
        model = DecisionTreeModel(
            N_FEATURES, min_samples_leaf=MIN_SAMPLES_LEAF)
        model.fit(X, y)
        assert model.verify_min_samples_leaf(), (
            "Some leaves have fewer than MIN_SAMPLES_LEAF samples!"
        )

    def test_leaf_proba_not_counts(self):
        """Transmitted parameters must be leaf probabilities, not raw counts."""
        X, y = _make_data(n=500, fraud_rate=0.2)
        model = DecisionTreeModel(N_FEATURES)
        model.fit(X, y)
        # get_parameters() should not raise
        params = model.get_parameters()
        assert params is not None
        assert len(params) == 1  # single byte blob

        # Reconstruct: must still predict correctly (prob sums to 1)
        model2 = DecisionTreeModel(N_FEATURES)
        # Simulate receiving params from a peer
        model2._peer_trees = []
        model2._local_tree.fit(X[:100], y[:100])  # minimal local tree
        model2._fitted = True
        # Verify tree_ values sum to 1 in leaves (probabilities, not counts)
        import sklearn.tree as skt
        tree_clf = model._local_tree
        tree = tree_clf.tree_
        is_leaf = tree.children_left == -1
        leaf_vals = tree.value[is_leaf]  # shape: (n_leaves, 1, 2)
        leaf_sums = leaf_vals.sum(axis=2)  # sum over classes
        # After _tree_to_proba_only: each leaf should sum to 1
        from fl.models.decision_tree import _tree_to_proba_only
        proba_tree = _tree_to_proba_only(tree_clf)
        proba_leaf_vals = proba_tree.tree_.value[is_leaf]
        proba_sums = proba_leaf_vals.sum(axis=2)
        np.testing.assert_allclose(
            proba_sums, np.ones_like(proba_sums), atol=1e-5)


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------

class TestRandomForest:

    def test_fit_and_predict(self):
        X, y = _make_data(n=500, fraud_rate=0.2)
        model = RandomForestModel(N_FEATURES, n_estimators=10)
        model.fit(X, y)
        proba = model.predict_proba(X[:10])
        assert proba.shape == (10, 2)

    def test_min_samples_leaf_enforced(self):
        with pytest.raises(ValueError, match="k-leaf"):
            RandomForestModel(N_FEATURES, min_samples_leaf=3)

    def test_min_samples_leaf_verified_after_fit(self):
        X, y = _make_data(n=1000, fraud_rate=0.3)
        model = RandomForestModel(
            N_FEATURES, n_estimators=5, min_samples_leaf=MIN_SAMPLES_LEAF)
        model.fit(X, y)
        assert model.verify_min_samples_leaf()

    def test_aggregate_forests_merges_trees(self):
        X, y = _make_data(n=500, fraud_rate=0.3)
        model1 = RandomForestModel(N_FEATURES, n_estimators=5)
        model2 = RandomForestModel(N_FEATURES, n_estimators=5)
        model1.fit(X[:250], y[:250])
        model2.fit(X[250:], y[250:])

        params1 = model1.get_parameters()
        params2 = model2.get_parameters()

        agg_params = aggregate_forests([params1, params2])
        model_global = RandomForestModel(N_FEATURES, n_estimators=5)
        model_global.set_parameters(agg_params)

        # Should have 10 trees total (5 from each bank)
        assert len(model_global._global_trees) == 10


# ---------------------------------------------------------------------------
# KNN
# ---------------------------------------------------------------------------

class TestKNN:

    def test_local_training_and_predict(self):
        """KNN should work for local-only baseline."""
        X, y = _make_data(n=200, fraud_rate=0.2)
        model = KNNModel(N_FEATURES, n_neighbors=3)
        model.fit(X[:150], y[:150])
        proba = model.predict_proba(X[150:])
        assert proba.shape == (50, 2)

    def test_get_parameters_raises_privacy_violation(self):
        """Calling get_parameters on KNN must raise PrivacyViolationError."""
        model = KNNModel(N_FEATURES)
        with pytest.raises(PrivacyViolationError):
            model.get_parameters()

    def test_set_parameters_raises_privacy_violation(self):
        """Calling set_parameters on KNN must raise PrivacyViolationError."""
        model = KNNModel(N_FEATURES)
        with pytest.raises(PrivacyViolationError):
            model.set_parameters([np.zeros(10)])
