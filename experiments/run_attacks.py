"""
experiments/run_attacks.py
===========================
Phase 7: Attacks and defences evaluation.

Runs the full attack/defence matrix:
  - Attacks: label_flip (m=1,2), model_poison (scaled, sign-flip, backdoor),
    free_rider (random, stale), membership_inference (loss-threshold)
  - Defences: Mode A (norm clip + secure agg), Mode B (robust agg without secure agg)

Usage:
    python experiments/run_attacks.py --n-banks 5 --seed 42 --output results/attacks
"""

# path setup MUST come before any local imports
import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.utils import set_global_seed, write_results  # noqa: E402
sys.path.insert(0, str(Path(__file__).parent.parent))


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _build_datasets(partitions_base: Path, n_banks: int) -> list:
    import pandas as pd
    from fl.preprocessing import FraudPreprocessor
    preprocessor = FraudPreprocessor()
    datasets = []
    for bank_id in range(n_banks):
        d = partitions_base / f"bank{bank_id}"
        df_tr = pd.read_parquet(d / "train.parquet")
        df_va = pd.read_parquet(d / "val.parquet")
        df_te = pd.read_parquet(d / "test.parquet")
        X_tr, y_tr = preprocessor.transform(df_tr)
        X_va, y_va = preprocessor.transform(df_va)
        X_te, y_te = preprocessor.transform(df_te)
        datasets.append({"bank_id": bank_id,
                         "X_train": X_tr, "y_train": y_tr,
                         "X_val": X_va, "y_val": y_va,
                         "X_test": X_te, "y_test": y_te})
    return datasets


def _train_clean_global(model_type: str, datasets: list, n_rounds: int, seed: int, device: str):
    """Train a clean global model via plain FedAvg (no attacks). Returns global_params."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from fl.models.logistic_regression import LogisticRegressionModel
    from fl.models.cnn import CNNModel
    from fl.bank_node.evaluator import evaluate

    n_features = datasets[0]["X_train"].shape[1]
    n_banks = len(datasets)

    def make_model():
        return LogisticRegressionModel(n_features) if model_type == "logistic_regression" \
            else CNNModel(n_features)

    models = [make_model() for _ in range(n_banks)]
    global_params = models[0].get_parameters()

    for round_num in range(1, n_rounds + 1):
        results = []
        for model, data in zip(models, datasets):
            model.set_parameters(global_params)
            X, y = data["X_train"], data["y_train"]
            model.to(device)
            model.train()
            opt = torch.optim.Adam(model.parameters(), lr=0.001)
            loader = DataLoader(TensorDataset(
                torch.tensor(X, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32)),
                batch_size=256, shuffle=True)
            for _ in range(3):
                for xb, yb in loader:
                    xb, yb = xb.to(device), yb.to(device)
                    opt.zero_grad()
                    loss = nn.functional.binary_cross_entropy_with_logits(
                        model(xb), yb)
                    loss.backward()
                    opt.step()
            results.append((model.get_parameters(), len(X)))

        total = sum(n for _, n in results)
        n_layers = len(results[0][0])
        global_params = [
            sum((n / total) * p[i].astype(np.float64)
                for p, n in results).astype(np.float32)
            for i in range(n_layers)
        ]

    return global_params, models


def run_label_flip_attack(
    model_type: str, datasets: list, n_malicious: int, n_rounds: int, seed: int, device: str,
) -> Dict[str, Any]:
    """Label-flip attack: m malicious banks flip fraud→non-fraud labels."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from fl.models.logistic_regression import LogisticRegressionModel
    from fl.models.cnn import CNNModel
    from fl.bank_node.evaluator import evaluate
    from attacks.poisoning.label_flip import label_flip_attack

    n_features = datasets[0]["X_train"].shape[1]
    n_banks = len(datasets)
    malicious_banks = set(range(n_malicious))

    def make_model():
        return LogisticRegressionModel(n_features) if model_type == "logistic_regression" \
            else CNNModel(n_features)

    models = [make_model() for _ in range(n_banks)]
    global_params = models[0].get_parameters()

    for round_num in range(1, n_rounds + 1):
        results = []
        for bank_id, (model, data) in enumerate(zip(models, datasets)):
            model.set_parameters(global_params)
            X, y = data["X_train"], data["y_train"]
            if bank_id in malicious_banks:
                X, y = label_flip_attack(
                    X, y, flip_rate=1.0, random_state=seed + bank_id)

            model.to(device)
            model.train()
            opt = torch.optim.Adam(model.parameters(), lr=0.001)
            loader = DataLoader(TensorDataset(
                torch.tensor(X, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32)),
                batch_size=256, shuffle=True)
            for _ in range(3):
                for xb, yb in loader:
                    xb, yb = xb.to(device), yb.to(device)
                    opt.zero_grad()
                    loss = nn.functional.binary_cross_entropy_with_logits(
                        model(xb), yb)
                    loss.backward()
                    opt.step()
            results.append((model.get_parameters(), len(X)))

        total = sum(n for _, n in results)
        n_layers = len(results[0][0])
        global_params = [
            sum((n / total) * p[i].astype(np.float64)
                for p, n in results).astype(np.float32)
            for i in range(n_layers)
        ]

    for model in models:
        model.set_parameters(global_params)

    test_metrics = []
    for model, data in zip(models, datasets):
        m = evaluate(model, data["X_test"],
                     data["y_test"], "test", "torch", device)
        test_metrics.append(m)

    pr_aucs = [m.get("pr_auc") for m in test_metrics if m.get("pr_auc")]
    recalls = [m.get("recall") for m in test_metrics if m.get("recall")]
    return {
        "attack": "label_flip",
        "n_malicious": n_malicious,
        "model": model_type,
        "mean_pr_auc": float(np.mean(pr_aucs)) if pr_aucs else None,
        "mean_recall": float(np.mean(recalls)) if recalls else None,
    }


def run_model_poison_attack(
    model_type: str, datasets: list, poison_type: str, n_rounds: int, seed: int, device: str,
) -> Dict[str, Any]:
    """Model poisoning: bank 0 submits a poisoned update."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from fl.models.logistic_regression import LogisticRegressionModel
    from fl.models.cnn import CNNModel
    from fl.bank_node.evaluator import evaluate
    from attacks.poisoning.model_poison import (
        scaled_update_attack, sign_flip_attack, backdoor_attack
    )

    n_features = datasets[0]["X_train"].shape[1]
    n_banks = len(datasets)

    def make_model():
        return LogisticRegressionModel(n_features) if model_type == "logistic_regression" \
            else CNNModel(n_features)

    models = [make_model() for _ in range(n_banks)]
    global_params = models[0].get_parameters()

    for round_num in range(1, n_rounds + 1):
        results = []
        for bank_id, (model, data) in enumerate(zip(models, datasets)):
            model.set_parameters(global_params)
            X, y = data["X_train"], data["y_train"]
            model.to(device)
            model.train()
            opt = torch.optim.Adam(model.parameters(), lr=0.001)
            loader = DataLoader(TensorDataset(
                torch.tensor(X, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32)),
                batch_size=256, shuffle=True)
            for _ in range(3):
                for xb, yb in loader:
                    xb, yb = xb.to(device), yb.to(device)
                    opt.zero_grad()
                    loss = nn.functional.binary_cross_entropy_with_logits(
                        model(xb), yb)
                    loss.backward()
                    opt.step()
            local_params = model.get_parameters()

            if bank_id == 0:  # bank 0 is the attacker
                if poison_type == "scaled":
                    local_params = scaled_update_attack(
                        local_params, scale=10.0)
                elif poison_type == "sign_flip":
                    local_params = sign_flip_attack(local_params)
                elif poison_type == "backdoor":
                    local_params = backdoor_attack(
                        local_params, trigger_scale=5.0)

            results.append((local_params, len(X)))

        total = sum(n for _, n in results)
        n_layers = len(results[0][0])
        global_params = [
            sum((n / total) * p[i].astype(np.float64)
                for p, n in results).astype(np.float32)
            for i in range(n_layers)
        ]

    for model in models:
        model.set_parameters(global_params)
    test_metrics = [
        evaluate(m, d["X_test"], d["y_test"], "test", "torch", device)
        for m, d in zip(models, datasets)
    ]
    pr_aucs = [m.get("pr_auc") for m in test_metrics if m.get("pr_auc")]
    return {
        "attack": "model_poison",
        "poison_type": poison_type,
        "model": model_type,
        "mean_pr_auc": float(np.mean(pr_aucs)) if pr_aucs else None,
    }


def run_mode_b_robust_agg(
    model_type: str, datasets: list, n_malicious: int,
    robust_method: str, n_rounds: int, seed: int, device: str,
) -> Dict[str, Any]:
    """Mode B: robust aggregation (no secure agg) against model poisoning."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from fl.models.logistic_regression import LogisticRegressionModel
    from fl.models.cnn import CNNModel
    from fl.bank_node.evaluator import evaluate
    from attacks.poisoning.model_poison import scaled_update_attack
    from attacks.defences.robust_agg import coordinate_median, trimmed_mean, krum, norm_clipping

    n_features = datasets[0]["X_train"].shape[1]
    n_banks = len(datasets)

    def make_model():
        return LogisticRegressionModel(n_features) if model_type == "logistic_regression" \
            else CNNModel(n_features)

    models = [make_model() for _ in range(n_banks)]
    global_params = models[0].get_parameters()

    for round_num in range(1, n_rounds + 1):
        all_params = []
        for bank_id, (model, data) in enumerate(zip(models, datasets)):
            model.set_parameters(global_params)
            X, y = data["X_train"], data["y_train"]
            model.to(device)
            model.train()
            opt = torch.optim.Adam(model.parameters(), lr=0.001)
            loader = DataLoader(TensorDataset(
                torch.tensor(X, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32)),
                batch_size=256, shuffle=True)
            for _ in range(3):
                for xb, yb in loader:
                    xb, yb = xb.to(device), yb.to(device)
                    opt.zero_grad()
                    loss = nn.functional.binary_cross_entropy_with_logits(
                        model(xb), yb)
                    loss.backward()
                    opt.step()
            local_p = model.get_parameters()
            if bank_id < n_malicious:
                local_p = scaled_update_attack(local_p, scale=10.0)
            all_params.append(local_p)

        # Norm clipping before robust agg (Mode B)
        all_params = norm_clipping(all_params, clip_norm=10.0)

        if robust_method == "median":
            global_params = coordinate_median(all_params)
        elif robust_method == "trimmed_mean":
            global_params = trimmed_mean(
                all_params, trim_fraction=max(0.1, n_malicious / n_banks))
        elif robust_method == "krum":
            global_params = krum(all_params, n_malicious=n_malicious)

    for model in models:
        model.set_parameters(global_params)
    test_metrics = [
        evaluate(m, d["X_test"], d["y_test"], "test", "torch", device)
        for m, d in zip(models, datasets)
    ]
    pr_aucs = [m.get("pr_auc") for m in test_metrics if m.get("pr_auc")]
    return {
        "mode": "B",
        "robust_method": robust_method,
        "n_malicious": n_malicious,
        "model": model_type,
        "mean_pr_auc": float(np.mean(pr_aucs)) if pr_aucs else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-banks", type=int, default=5)
    parser.add_argument("--scenario", default="S1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-rounds", type=int, default=5)
    parser.add_argument("--partitions", default="data/partitions")
    parser.add_argument("--output", default="results/attacks")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    set_global_seed(args.seed)
    partitions_base = Path(args.partitions) / \
        f"N{args.n_banks}" / args.scenario
    if not partitions_base.exists():
        logger.error(f"Partitions not found: {partitions_base}")
        sys.exit(1)

    datasets = _build_datasets(partitions_base, args.n_banks)
    all_results = []

    for model_type in ["logistic_regression", "cnn"]:
        # Label-flip m=1 and m=2
        for m in [1, 2]:
            r = run_label_flip_attack(
                model_type, datasets, m, args.n_rounds, args.seed, args.device)
            all_results.append(r)
            logger.info(f"[attacks] {r}")

        # Model poisoning
        for pt in ["scaled", "sign_flip", "backdoor"]:
            r = run_model_poison_attack(
                model_type, datasets, pt, args.n_rounds, args.seed, args.device)
            all_results.append(r)
            logger.info(f"[attacks] {r}")

        # Mode B robust aggregation vs scaled poisoning
        for method in ["median", "trimmed_mean", "krum"]:
            for m in [1, 2]:
                # Krum requires n_banks > 2*n_malicious + 2
                # With 5 banks: m=1 → 5>4 ✓, m=2 → 5>6 ✗ (skip)
                if method == "krum" and args.n_banks <= 2 * m + 2:
                    logger.info(
                        f"[attacks] Skipping Krum: n_banks={args.n_banks} <= 2*{m}+2={2*m+2} "
                        "(not enough banks for Krum with this many malicious)"
                    )
                    all_results.append({
                        "mode": "B", "robust_method": method, "n_malicious": m,
                        "model": model_type, "mean_pr_auc": None,
                        "skipped": f"Krum requires n_banks > 2*n_malicious+2; need >{2*m+2} banks",
                    })
                    continue
                r = run_mode_b_robust_agg(
                    model_type, datasets, m, method, args.n_rounds, args.seed, args.device)
                all_results.append(r)
                logger.info(f"[attacks] {r}")

    write_results({"results": all_results}, Path(args.output),
                  f"attacks_{args.scenario}_seed{args.seed}", vars(args))

    print("\n=== Attack Results Summary ===")
    print(f"{'Attack':<25} {'Model':<22} {'m':<5} {'PR-AUC':<10}")
    print("-" * 65)
    for r in all_results:
        attack = r.get("attack") or r.get("mode", "")
        detail = r.get("poison_type") or r.get("robust_method") or ""
        label = f"{attack}/{detail}" if detail else attack
        model = r.get("model", "")
        m = r.get("n_malicious", "-")
        pr = r.get("mean_pr_auc")
        print(
            f"{label:<25} {model:<22} {str(m):<5} {str(round(pr, 4)) if pr else 'N/A':<10}")


if __name__ == "__main__":
    main()
