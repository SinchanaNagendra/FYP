"""
experiments/run_dp_sweep.py
=============================
Phase 6: Privacy–utility curve.

Sweeps noise_multiplier over a range that maps to ε ≈ {1, 3, 8, ∞}.
For each setting, runs federated training (LR + CNN) and records:
  - Per-round and cumulative ε
  - Val PR-AUC, F1, ROC-AUC per round
  - Final test metrics

This produces the data for the privacy–utility trade-off figure.

Usage:
    python experiments/run_dp_sweep.py --n-banks 5 --seed 42 --output results/dp_sweep
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.utils import set_global_seed, write_results, mean_std

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Noise multiplier → approximate ε mapping (for n=1000 samples, batch=256, 5 rounds×3 epochs)
# Actual ε computed by RDP accountant at runtime.
NOISE_SETTINGS = [
    {"label": "no_dp",  "noise_multiplier": 0.0,  "use_dp": False},
    {"label": "eps_8",  "noise_multiplier": 0.5,  "use_dp": True},
    {"label": "eps_3",  "noise_multiplier": 0.8,  "use_dp": True},
    {"label": "eps_1",  "noise_multiplier": 1.1,  "use_dp": True},
    {"label": "eps_0.5","noise_multiplier": 1.5,  "use_dp": True},
]

N_ROUNDS = 10
N_EPOCHS = 5
BATCH_SIZE = 256
MAX_GRAD_NORM = 1.0
DELTA = 1e-5


def run_dp_experiment(
    model_type: str,
    datasets: list,
    noise_setting: dict,
    n_rounds: int,
    n_epochs: int,
    batch_size: int,
    max_grad_norm: float,
    delta: float,
    seed: int,
    device: str,
) -> dict:
    """Run federated training for one (model_type, noise_setting) combination."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    from fl.dp.accountant import PrivacyAccountant
    from fl.models.logistic_regression import LogisticRegressionModel
    from fl.models.cnn import CNNModel
    from fl.secure_agg.keys import KeyPair, derive_shared_seed
    from fl.secure_agg.aggregator import SecureAggregator
    from fl.secure_agg.masking import decode_fixed_point, FIXED_POINT_SCALE
    from fl.bank_node.evaluator import evaluate

    set_global_seed(seed)
    n_banks = len(datasets)
    use_dp = noise_setting["use_dp"]
    noise_mult = noise_setting["noise_multiplier"]

    # Build models
    n_features = datasets[0]["X_train"].shape[1]
    if model_type == "logistic_regression":
        models = [LogisticRegressionModel(n_features) for _ in range(n_banks)]
    elif model_type == "cnn":
        models = [CNNModel(n_features) for _ in range(n_banks)]
    else:
        raise ValueError(f"DP sweep only supports LR and CNN, got: {model_type}")

    global_params = models[0].get_parameters()
    accountant = PrivacyAccountant(delta=delta)
    round_records = []

    for round_num in range(1, n_rounds + 1):
        round_id = f"r{round_num:03d}"
        keypairs = {i: KeyPair.generate() for i in range(n_banks)}
        all_seeds = {}
        for i in range(n_banks):
            for j in range(i + 1, n_banks):
                s = derive_shared_seed(keypairs[i], keypairs[j].public_bytes, i, j, round_id)
                all_seeds[(i, j)] = s

        # Local training
        local_results = []
        eps_this_round = float("inf")

        for bank_id, (model, data) in enumerate(zip(models, datasets)):
            model.set_parameters(global_params)
            X, y = data["X_train"], data["y_train"]
            n_samples = len(X)

            if use_dp:
                from fl.dp.dp_trainer import DPTrainer
                dp_trainer = DPTrainer(
                    model=model,
                    max_grad_norm=max_grad_norm,
                    noise_multiplier=noise_mult,
                    batch_size=batch_size,
                    delta=delta,
                    lr=0.01,
                    device=device,
                )
                for epoch in range(n_epochs):
                    dp_trainer.train_epoch(X, y)
                eps_this_round = min(eps_this_round, dp_trainer.get_epsilon(delta))
            else:
                model.to(device)
                model.train()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                dataset = TensorDataset(
                    torch.tensor(X, dtype=torch.float32),
                    torch.tensor(y, dtype=torch.float32),
                )
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                for epoch in range(n_epochs):
                    for xb, yb in loader:
                        xb, yb = xb.to(device), yb.to(device)
                        optimizer.zero_grad()
                        logits = model(xb)
                        loss = nn.functional.binary_cross_entropy_with_logits(logits, yb)
                        loss.backward()
                        optimizer.step()
                eps_this_round = float("inf")

            local_results.append((model.get_parameters(), n_samples))

        # Record privacy
        if use_dp:
            n_steps = n_epochs * max(1, n_samples // batch_size)
            accountant.record_round(
                round_id=round_id,
                epsilon=eps_this_round,
                noise_multiplier=noise_mult,
                max_grad_norm=max_grad_norm,
                n_steps=n_steps,
                n_samples=n_samples,
            )

        # Secure aggregation
        flat_size = sum(p.size for p in global_params)
        agg = SecureAggregator(n_banks=n_banks, n_elements=flat_size)
        for bank_id, (local_params, _) in enumerate(local_results):
            flat = np.concatenate([p.ravel() for p in local_params])
            bank_seeds = {k: v for k, v in all_seeds.items() if bank_id in k}
            masked = SecureAggregator.apply_mask(bank_id, flat, bank_seeds, flat_size)
            agg.add_masked_update(bank_id, masked)

        agg_flat = agg.aggregate()
        new_global = []
        offset = 0
        for p in global_params:
            new_global.append(agg_flat[offset: offset + p.size].reshape(p.shape))
            offset += p.size
        global_params = new_global

        # Evaluate
        for model in models:
            model.set_parameters(global_params)

        val_pr_aucs = []
        for model, data in zip(models, datasets):
            m = evaluate(model, data["X_val"], data["y_val"], "val", "torch", device)
            val_pr_aucs.append(m.get("pr_auc"))

        valid_pr_aucs = [x for x in val_pr_aucs if x is not None]
        round_records.append({
            "round": round_num,
            "epsilon": eps_this_round if use_dp else None,
            "cumulative_epsilon": accountant.cumulative_epsilon if use_dp else None,
            "val_pr_auc_mean": float(np.mean(valid_pr_aucs)) if valid_pr_aucs else None,
            "val_pr_auc_std": float(np.std(valid_pr_aucs)) if valid_pr_aucs else None,
        })
        logger.info(
            f"[DPSweep] model={model_type}, noise={noise_mult}, "
            f"round={round_num}/{n_rounds}, "
            f"ε={eps_this_round:.3f if eps_this_round != float('inf') else 'inf'}, "
            f"PR-AUC={round_records[-1]['val_pr_auc_mean']:.4f}"
        )

    # Final test eval
    test_metrics = []
    for model, data in zip(models, datasets):
        m = evaluate(model, data["X_test"], data["y_test"], "test", "torch", device)
        test_metrics.append(m)

    privacy_summary = accountant.get_summary() if use_dp else {"n_rounds": 0, "cumulative_epsilon": None}

    return {
        "model": model_type,
        "noise_label": noise_setting["label"],
        "noise_multiplier": noise_mult,
        "use_dp": use_dp,
        "rounds": round_records,
        "final_test_pr_auc_mean": mean_std([m.get("pr_auc") for m in test_metrics])[0],
        "final_test_pr_auc_std": mean_std([m.get("pr_auc") for m in test_metrics])[1],
        "final_test_roc_auc_mean": mean_std([m.get("roc_auc") for m in test_metrics])[0],
        "privacy_summary": privacy_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="DP privacy-utility sweep")
    parser.add_argument("--n-banks", type=int, default=5)
    parser.add_argument("--scenario", default="S1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-rounds", type=int, default=N_ROUNDS)
    parser.add_argument("--n-epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--partitions", default="data/partitions")
    parser.add_argument("--output", default="results/dp_sweep")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    import pandas as pd
    set_global_seed(args.seed)

    # Check partitions exist
    partitions_base = Path(args.partitions) / f"N{args.n_banks}" / args.scenario
    if not partitions_base.exists():
        logger.error(f"Partitions not found: {partitions_base}. Run: make data-partition")
        sys.exit(1)

    from fl.preprocessing import FraudPreprocessor
    preprocessor = FraudPreprocessor()

    # Load all bank data
    datasets = []
    for bank_id in range(args.n_banks):
        bank_dir = partitions_base / f"bank{bank_id}"
        df_train = pd.read_parquet(bank_dir / "train.parquet")
        df_val = pd.read_parquet(bank_dir / "val.parquet")
        df_test = pd.read_parquet(bank_dir / "test.parquet")
        X_train, y_train = preprocessor.transform(df_train)
        X_val, y_val = preprocessor.transform(df_val)
        X_test, y_test = preprocessor.transform(df_test)
        datasets.append({
            "bank_id": bank_id,
            "X_train": X_train, "y_train": y_train,
            "X_val": X_val, "y_val": y_val,
            "X_test": X_test, "y_test": y_test,
        })

    output_dir = Path(args.output)
    all_results = []

    for model_type in ["logistic_regression", "cnn"]:
        for noise_setting in NOISE_SETTINGS:
            logger.info(
                f"\n[DPSweep] === model={model_type}, noise={noise_setting['label']} ==="
            )
            try:
                result = run_dp_experiment(
                    model_type=model_type,
                    datasets=datasets,
                    noise_setting=noise_setting,
                    n_rounds=args.n_rounds,
                    n_epochs=args.n_epochs,
                    batch_size=256,
                    max_grad_norm=1.0,
                    delta=DELTA,
                    seed=args.seed,
                    device=args.device,
                )
                all_results.append(result)
            except Exception as e:
                logger.error(f"[DPSweep] FAILED: {e}")
                all_results.append({"model": model_type, "noise_label": noise_setting["label"], "error": str(e)})

    config = vars(args)
    write_results({"results": all_results}, output_dir, f"dp_sweep_seed{args.seed}", config)

    # Print summary table
    print("\n=== DP Sweep Summary ===")
    print(f"{'Model':<20} {'Noise':<10} {'Cum-ε':<10} {'PR-AUC':<10} {'ROC-AUC':<10}")
    print("-" * 60)
    for r in all_results:
        if "error" in r:
            continue
        cum_eps = r.get("privacy_summary", {}).get("cumulative_epsilon")
        cum_eps_str = f"{cum_eps:.2f}" if cum_eps is not None else "∞"
        print(
            f"{r['model']:<20} {r['noise_label']:<10} {cum_eps_str:<10} "
            f"{r.get('final_test_pr_auc_mean', 'N/A')!s:<10} "
            f"{r.get('final_test_roc_auc_mean', 'N/A')!s:<10}"
        )

    print(f"\nResults written to: {output_dir}/dp_sweep_seed{args.seed}.json")


if __name__ == "__main__":
    main()
