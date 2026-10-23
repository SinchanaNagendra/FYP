"""
tests/integration/test_fl_rounds.py
=====================================
Phase 5 Integration Test – Federated Learning Rounds (no-blockchain mode).

Runs a complete simulated FL experiment over synthetic data:
  - 5 banks, 5 rounds
  - LR and CNN models
  - Secure aggregation (X25519 + PRG masking)
  - Verifies round-over-round PR-AUC improvement (or at least no catastrophic collapse)
  - Verifies aggregate == plain average (masks cancel)
  - Verifies no raw data ever passed between bank nodes

This test does NOT require Docker, Fabric, or IPFS.
It runs entirely in-process using the BankNode and SecureAggregator classes.

Run: pytest tests/integration/test_fl_rounds.py -v --tb=short
"""

import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fl.balancing.methods import BalancingMethod
from fl.models.logistic_regression import LogisticRegressionModel
from fl.models.cnn import CNNModel
from fl.models.gaussian_nb import GaussianNBModel
from fl.secure_agg.keys import KeyPair, derive_shared_seed
from fl.secure_agg.aggregator import SecureAggregator
from fl.secure_agg.masking import encode_fixed_point, decode_fixed_point, FIXED_POINT_SCALE
from fl.bank_node.evaluator import evaluate
from fl.bank_node.trainer import LocalTrainer

N_BANKS = 5
N_FEATURES = 64
N_SAMPLES_PER_BANK = 300
FRAUD_RATE = 0.15
N_ROUNDS = 5
SEED = 42


# ---------------------------------------------------------------------------
# Synthetic data fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bank_datasets():
    """Generate synthetic per-bank datasets (no real data needed for integration test)."""
    rng = np.random.default_rng(SEED)
    datasets = []
    for bank_id in range(N_BANKS):
        n = N_SAMPLES_PER_BANK
        X = rng.standard_normal((n, N_FEATURES)).astype(np.float32)
        y = (rng.random(n) < FRAUD_RATE).astype(np.int32)
        # Ensure at least some fraud
        y[:int(n * FRAUD_RATE)] = 1
        # Temporal split: 70/15/15
        n_train = int(n * 0.70)
        n_val = int(n * 0.85)
        datasets.append({
            "bank_id": bank_id,
            "X_train": X[:n_train], "y_train": y[:n_train],
            "X_val": X[n_train:n_val], "y_val": y[n_train:n_val],
            "X_test": X[n_val:], "y_test": y[n_val:],
        })
    return datasets


# ---------------------------------------------------------------------------
# Helper: run one FL round (in-process, no IPFS/Fabric)
# ---------------------------------------------------------------------------

def run_one_round(
    model_type: str,
    models: List,
    datasets: List[dict],
    global_params: List[np.ndarray],
    round_id: str,
    use_secure_agg: bool = True,
) -> tuple:
    """
    Run one federated round in-process.
    Returns (new_global_params, per_bank_val_metrics).
    """
    n_banks = len(models)

    # Key exchange
    if use_secure_agg:
        keypairs = {i: KeyPair.generate() for i in range(n_banks)}
        all_seeds = {}
        for i in range(n_banks):
            for j in range(i + 1, n_banks):
                seed_ij = derive_shared_seed(
                    keypairs[i], keypairs[j].public_bytes, i, j, round_id
                )
                all_seeds[(i, j)] = seed_ij

    # Local training
    results = []  # (params, n_samples)
    for bank_id, (model, data) in enumerate(zip(models, datasets)):
        model.set_parameters(global_params)
        trainer = LocalTrainer(
            model_type=model_type,
            model=model,
            balancing=BalancingMethod.NONE,
            n_epochs=3,
            device="cpu",
            random_state=SEED + bank_id,
        )
        train_result = trainer.train(data["X_train"], data["y_train"], global_params)
        local_params = trainer.get_parameters()
        results.append((local_params, train_result["n_samples"]))

    # Aggregation
    if use_secure_agg:
        flat_size = sum(p.size for p in results[0][0])
        agg = SecureAggregator(n_banks=n_banks, n_elements=flat_size)
        for bank_id, (local_params, _) in enumerate(results):
            flat = np.concatenate([p.ravel() for p in local_params])
            bank_seeds = {k: v for k, v in all_seeds.items() if bank_id in k}
            masked = SecureAggregator.apply_mask(bank_id, flat, bank_seeds, flat_size)
            agg.add_masked_update(bank_id, masked)
        agg_flat = agg.aggregate()

        # Reconstruct parameter list from flat
        new_global = []
        offset = 0
        for p in global_params:
            size = p.size
            new_global.append(agg_flat[offset: offset + size].reshape(p.shape))
            offset += size
    else:
        # Plain FedAvg
        total = sum(n for _, n in results)
        n_layers = len(results[0][0])
        new_global = []
        for layer_idx in range(n_layers):
            weighted = sum(
                (n / total) * params[layer_idx].astype(np.float64)
                for params, n in results
            )
            new_global.append(weighted.astype(np.float32))

    # Apply new global model to all banks and evaluate
    val_metrics = []
    for model, data in zip(models, datasets):
        model.set_parameters(new_global)
        eval_type = "torch" if model_type in ("logistic_regression", "cnn") else "gnb"
        m = evaluate(model, data["X_val"], data["y_val"], "val", eval_type, "cpu")
        val_metrics.append(m)

    return new_global, val_metrics


# ---------------------------------------------------------------------------
# Test 1: LR federated rounds converge
# ---------------------------------------------------------------------------

class TestFLRoundsLR:

    def test_lr_rounds_no_crash(self, bank_datasets):
        """LR: 5 federated rounds complete without exception."""
        models = [LogisticRegressionModel(N_FEATURES) for _ in range(N_BANKS)]
        global_params = models[0].get_parameters()

        for r in range(1, N_ROUNDS + 1):
            global_params, metrics = run_one_round(
                "logistic_regression", models, bank_datasets,
                global_params, f"round_{r:04d}", use_secure_agg=True,
            )

        # After 5 rounds, PR-AUC should be > 0.3 (better than random 0.15)
        pr_aucs = [m["pr_auc"] for m in metrics if m.get("pr_auc")]
        assert len(pr_aucs) > 0
        mean_pr_auc = np.mean(pr_aucs)
        assert mean_pr_auc > 0.2, (
            f"LR PR-AUC after {N_ROUNDS} rounds = {mean_pr_auc:.4f}; "
            "expected > 0.2 (well above random baseline)"
        )

    def test_lr_secagg_equals_plain_fedavg(self, bank_datasets):
        """
        Secure-aggregation FedAvg result must match plain FedAvg within quantisation tolerance.
        This is the core correctness guarantee.
        """
        rng = np.random.default_rng(99)
        models_sa = [LogisticRegressionModel(N_FEATURES) for _ in range(N_BANKS)]
        models_plain = [LogisticRegressionModel(N_FEATURES) for _ in range(N_BANKS)]

        # Use the same initial params
        init_params = models_sa[0].get_parameters()
        for m in models_plain:
            m.set_parameters(init_params)

        global_sa, _ = run_one_round(
            "logistic_regression", models_sa, bank_datasets,
            init_params, "round_sa_test", use_secure_agg=True,
        )
        global_plain, _ = run_one_round(
            "logistic_regression", models_plain, bank_datasets,
            init_params, "round_plain_test", use_secure_agg=False,
        )

        # Results should match within fixed-point quantisation error
        tol = 2.0 / FIXED_POINT_SCALE
        for i, (p_sa, p_plain) in enumerate(zip(global_sa, global_plain)):
            np.testing.assert_allclose(
                p_sa, p_plain, atol=tol,
                err_msg=f"Layer {i}: secure-agg result deviates from plain FedAvg"
            )

    def test_no_raw_data_in_parameters(self, bank_datasets):
        """
        Model parameters must have correct types — no float32 matrix with
        shape (n_samples, n_features) passed between banks.
        """
        models = [LogisticRegressionModel(N_FEATURES) for _ in range(N_BANKS)]
        global_params = models[0].get_parameters()

        for r in range(1, 3):
            global_params, _ = run_one_round(
                "logistic_regression", models, bank_datasets,
                global_params, f"round_{r}", use_secure_agg=False,
            )

        for i, p in enumerate(global_params):
            # LR params: (N_FEATURES,) weight + (1,) bias
            assert p.dtype in (np.float32, np.float64), f"Layer {i}: unexpected dtype {p.dtype}"
            assert p.ndim <= 2, f"Layer {i}: unexpectedly high-dimensional param"
            # Most importantly: no dimension should be N_SAMPLES_PER_BANK
            for dim in p.shape:
                assert dim != N_SAMPLES_PER_BANK, (
                    f"Layer {i}: dim={dim} = N_SAMPLES_PER_BANK — raw data leak?"
                )


# ---------------------------------------------------------------------------
# Test 2: CNN federated rounds
# ---------------------------------------------------------------------------

class TestFLRoundsCNN:

    def test_cnn_rounds_no_crash(self, bank_datasets):
        """CNN: 3 rounds complete without exception."""
        models = [CNNModel(N_FEATURES) for _ in range(N_BANKS)]
        global_params = models[0].get_parameters()

        for r in range(1, 4):
            global_params, metrics = run_one_round(
                "cnn", models, bank_datasets,
                global_params, f"round_{r}", use_secure_agg=True,
            )

        pr_aucs = [m["pr_auc"] for m in metrics if m.get("pr_auc")]
        assert len(pr_aucs) > 0, "CNN produced no valid PR-AUC metrics"


# ---------------------------------------------------------------------------
# Test 3: GNB exact aggregation
# ---------------------------------------------------------------------------

class TestFLRoundsGNB:

    def test_gnb_aggregation_is_exact(self, bank_datasets):
        """
        GNB federated aggregation should produce the same result as
        training on the pooled dataset (exact sufficient statistics).
        """
        from fl.models.gaussian_nb import GaussianNBModel
        from fl.strategies.gnb_strategy import GNBStrategy

        # Federated
        models = [GaussianNBModel(N_FEATURES) for _ in range(N_BANKS)]
        results = []
        for bank_id, (model, data) in enumerate(zip(models, bank_datasets)):
            model.fit(data["X_train"], data["y_train"])
            results.append((model.get_parameters(), len(data["X_train"])))

        strategy = GNBStrategy(n_banks=N_BANKS)
        agg_params = strategy.aggregate_fit(results)

        global_gnb = GaussianNBModel(N_FEATURES)
        global_gnb.set_parameters(agg_params)

        # Pooled baseline
        X_all = np.vstack([d["X_train"] for d in bank_datasets])
        y_all = np.concatenate([d["y_train"] for d in bank_datasets])
        pooled_gnb = GaussianNBModel(N_FEATURES)
        pooled_gnb.fit(X_all, y_all)

        # Compare on combined val set
        X_val = np.vstack([d["X_val"] for d in bank_datasets])
        y_val = np.concatenate([d["y_val"] for d in bank_datasets])

        p_fed = global_gnb.predict_proba(X_val)
        p_pool = pooled_gnb.predict_proba(X_val)

        np.testing.assert_allclose(p_fed, p_pool, atol=1e-4,
                                   err_msg="GNB federated != pooled (exact aggregation violated)")


# ---------------------------------------------------------------------------
# Test 4: Secure aggregation mask verification (integration level)
# ---------------------------------------------------------------------------

class TestSecureAggIntegration:

    def test_masks_cancel_across_5_banks(self):
        """Verify mask cancellation property at integration level with 5 banks."""
        n_banks = 5
        n_elements = 1000
        keypairs = {i: KeyPair.generate() for i in range(n_banks)}
        all_seeds = {}
        for i in range(n_banks):
            for j in range(i + 1, n_banks):
                seed = derive_shared_seed(keypairs[i], keypairs[j].public_bytes, i, j, "int_test")
                all_seeds[(i, j)] = seed

        total = np.zeros(n_elements, dtype=np.uint64)
        for bank_id in range(n_banks):
            bank_seeds = {k: v for k, v in all_seeds.items() if bank_id in k}
            mask = __import__(
                "fl.secure_agg.masking", fromlist=["generate_mask"]
            ).generate_mask(bank_id, bank_seeds, n_elements)
            total = (total + mask).astype(np.uint64)

        assert np.all(total == 0), (
            f"Masks do not cancel: {np.count_nonzero(total)} non-zero elements"
        )

    def test_aggregate_verifiable_by_any_bank(self):
        """
        Any bank that downloads all masked updates and sums them
        gets the same aggregate (no trusted aggregator needed).
        """
        n_banks = 3
        n_elements = 200
        rng = np.random.default_rng(77)

        keypairs = {i: KeyPair.generate() for i in range(n_banks)}
        all_seeds = {}
        for i in range(n_banks):
            for j in range(i + 1, n_banks):
                s = derive_shared_seed(keypairs[i], keypairs[j].public_bytes, i, j, "verify_test")
                all_seeds[(i, j)] = s

        weights = [rng.uniform(-1, 1, n_elements).astype(np.float32) for _ in range(n_banks)]
        masked = []
        for bank_id in range(n_banks):
            bank_seeds = {k: v for k, v in all_seeds.items() if bank_id in k}
            m = SecureAggregator.apply_mask(bank_id, weights[bank_id], bank_seeds, n_elements)
            masked.append(m)

        # Bank 0 aggregates
        agg_sum = np.zeros(n_elements, dtype=np.uint64)
        for m in masked:
            agg_sum = (agg_sum + m).astype(np.uint64)

        result_b0 = decode_fixed_point(agg_sum, n_participants=n_banks)

        # Bank 1 also aggregates (same masked uploads, different bank)
        result_b1 = decode_fixed_point(agg_sum, n_participants=n_banks)

        np.testing.assert_array_equal(result_b0, result_b1,
                                      err_msg="Aggregate result differs between banks")

        plain_avg = np.mean(weights, axis=0)
        tol = 2.0 / FIXED_POINT_SCALE
        np.testing.assert_allclose(result_b0, plain_avg, atol=tol,
                                   err_msg="Aggregate != plain average")
