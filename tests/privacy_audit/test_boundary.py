"""
tests/privacy_audit/test_boundary.py
======================================
Privacy Audit Test 2 – Data Boundary Test.

Validates that all outbound payloads from the FL layer contain ONLY
allowed data types:
  - Parameter tensors (float32/float64 numpy arrays)
  - Hashes (64-char hex strings)
  - Metadata (round_id, bank_id, norm, size, timestamps)

Any payload containing row-shaped data (n_features × n_samples matrices,
TransactionID lists, raw feature values) FAILS this test.

Run: pytest tests/privacy_audit/test_boundary.py -v
"""

import io
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# Maximum allowed dimensions for any transmitted tensor.
# A CNN FC layer may legitimately have > 1000 elements in one dimension
# (e.g., 64*channels = 1600 for Conv1d(32,64)→flatten).
# The key invariant is that NO dimension equals n_samples (e.g. 500).
MAX_BATCH_SIZE_THRESHOLD = 100_000  # pathologically large = likely data batch


class TestDataBoundary:

    def test_logistic_regression_parameters_are_1d_per_layer(self):
        """
        LR parameters: weight vector (1D) + bias (scalar).
        Must NOT be a 2D matrix with many rows (which would suggest batched data).
        """
        from fl.models.logistic_regression import LogisticRegressionModel
        n_features = 100
        model = LogisticRegressionModel(n_features)
        params = model.get_parameters()

        for i, p in enumerate(params):
            assert isinstance(p, np.ndarray), f"Param {i} is not np.ndarray"
            # LR has weight (n_features,) and bias (1,) — no 2D data matrices
            total_elements = p.size
            assert total_elements <= n_features + 1, (
                f"LR param {i} has {total_elements} elements — "
                f"suspicious (may contain row data)"
            )

    def test_cnn_parameters_are_weight_tensors(self):
        """
        CNN parameters must all be convolution filters or FC weights/biases.
        No parameter should have a shape matching (n_samples, n_features) —
        i.e., no dimension should equal the number of training samples.
        CNN models legitimately have large flat dimensions (conv channels × reduced_features).
        """
        from fl.models.cnn import CNNModel
        n_features = 100
        n_samples = 400  # typical batch/dataset size to check against
        model = CNNModel(n_features)
        params = model.get_parameters()

        assert len(params) > 0, "CNN should have parameters"

        for i, p in enumerate(params):
            assert isinstance(p, np.ndarray)
            # No dimension should equal n_samples (that would be a data batch)
            for dim_size in p.shape:
                assert dim_size != n_samples, (
                    f"CNN param {i} has dimension {dim_size} = n_samples={n_samples}. "
                    "This may indicate a data batch is being transmitted."
                )
        # Confirm parameters are float arrays (weights), not uint8 raw data
        for i, p in enumerate(params):
            assert p.dtype in (np.float32, np.float64), (
                f"CNN param {i} has dtype {p.dtype}; expected float32/float64"
            )

    def test_gnb_sufficient_stats_are_aggregates_not_rows(self):
        """
        GNB transmitted stats: [counts, sum, sum_sq] with shape (n_classes, n_features).
        Must NOT be the raw training data (shape would be (n_samples, n_features)).
        """
        from fl.models.gaussian_nb import GaussianNBModel
        rng = np.random.default_rng(42)
        n_features = 50
        n_samples = 500
        X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
        y = (rng.random(n_samples) < 0.3).astype(np.int32)

        # Ensure enough minority samples for get_sufficient_stats
        y[:50] = 1  # at least 50 positive

        model = GaussianNBModel(n_features)
        model.fit(X, y)
        params = model.get_parameters()

        # params[0] = counts (n_classes,)
        # params[1] = sum (n_classes, n_features)
        # params[2] = sum_sq (n_classes, n_features)
        counts = params[0]
        sum_arr = params[1]
        sum_sq_arr = params[2]

        assert counts.shape == (
            2,), f"counts shape should be (2,), got {counts.shape}"
        assert sum_arr.shape == (2, n_features), (
            f"sum shape should be (2, {n_features}), got {sum_arr.shape}"
        )
        # Most importantly: n_samples is NOT in any dimension
        for arr in params:
            for dim_size in arr.shape:
                assert dim_size != n_samples, (
                    f"GNB param has dimension {dim_size} = n_samples={n_samples}. "
                    "Raw training data may be leaking!"
                )

    def test_decision_tree_params_are_serialised_structure(self):
        """
        Decision Tree parameters are a serialised tree structure (bytes blob).
        Check they are a uint8 byte array, not a float matrix of training data.
        """
        from fl.models.decision_tree import DecisionTreeModel
        rng = np.random.default_rng(42)
        n_features = 50
        n_samples = 300
        X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
        y = (rng.random(n_samples) < 0.3).astype(np.int32)
        y[:30] = 1

        model = DecisionTreeModel(n_features)
        model.fit(X, y)
        params = model.get_parameters()

        assert len(params) == 1, "DT should transmit a single byte blob"
        blob = params[0]
        assert blob.dtype == np.uint8, f"DT params should be uint8, got {blob.dtype}"
        # Byte blob cannot be interpreted as a float data matrix
        # (the dtype difference is itself a boundary check)

    def test_knn_raises_on_get_parameters(self):
        """
        KNN must RAISE PrivacyViolationError when get_parameters is called.
        This ensures the boundary is enforced by code, not just convention.
        """
        from fl.models.knn import KNNModel, PrivacyViolationError
        model = KNNModel(n_features=50)
        with pytest.raises(PrivacyViolationError):
            model.get_parameters()

    def test_ipfs_artifact_is_opaque_bytes_not_csv(self):
        """
        IPFS upload should encrypt the artifact before transmission.
        The encrypted bytes should not be parseable as CSV/parquet/JSON.
        """
        from fl.bank_node.ipfs_client import IPFSClient
        import os

        key = os.urandom(32)
        client = IPFSClient(
            ipfs_api_url="http://localhost:5001", encryption_key=key)

        # Simulate a model parameter payload
        fake_params = [np.random.randn(100).astype(np.float32)]
        payload = IPFSClient.serialize_parameters(fake_params)

        # Encrypt it
        ciphertext = client._encrypt(payload)

        # Verify: ciphertext is NOT parseable as JSON
        with pytest.raises(Exception):
            import json
            json.loads(ciphertext)

        # Verify: ciphertext does NOT contain the original payload bytes
        # (first 20 bytes of plaintext should not appear verbatim in ciphertext)
        assert payload[:20] not in ciphertext, (
            "Plaintext appears in ciphertext — encryption not applied!"
        )

    def test_masked_update_dtype_is_uint64_not_float_data(self):
        """
        Masked updates must be uint64 arrays (ring arithmetic).
        Float32 data matrices would have dtype float32, not uint64.
        This provides a dtype-level boundary check.
        """
        from fl.secure_agg.keys import KeyPair, derive_shared_seed
        from fl.secure_agg.aggregator import SecureAggregator

        n_elements = 200
        weights = np.random.randn(n_elements).astype(np.float32)

        kp0 = KeyPair.generate()
        kp1 = KeyPair.generate()
        seed = derive_shared_seed(kp0, kp1.public_bytes, 0, 1, "test")
        seeds = {(0, 1): seed}

        masked = SecureAggregator.apply_mask(0, weights, seeds, n_elements)

        assert masked.dtype == np.uint64, (
            f"Masked update dtype should be uint64, got {masked.dtype}. "
            "Non-uint64 dtype may indicate plaintext data is being transmitted."
        )
        assert masked.shape == (n_elements,)
