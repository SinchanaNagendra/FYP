"""
experiments/run_matrix.py
==========================
Phase 10: Full experiment matrix.

Runs all combinations of:
  - seeds:     42, 123, 777
  - n_banks:   3, 5, 8
  - scenarios: S1, S2, S3_a0.1, S3_a0.5, S3_a1.0
  - models:    LR, CNN, GNB, DT, RF
  - balancing: none, weighted, ros, rus, smote, adasyn, hybrid
  - mode:      centralized, local, federated (no-blockchain)

Results written to results/full_matrix/ with one JSON per (seed, n_banks, scenario).
Aggregated summary tables written to results/full_matrix/summary_*.csv.

Usage:
    python experiments/run_matrix.py --seeds 42 123 777 --n-banks 3 5 8 --output results/full_matrix
    python experiments/run_matrix.py --seeds 42 --n-banks 5 --scenarios S1  # quick test
"""

import argparse
import itertools
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.utils import set_global_seed, write_results, get_git_hash, mean_std

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ALL_MODELS = ["logistic_regression", "cnn", "gaussian_nb", "decision_tree", "random_forest"]
ALL_BALANCING = ["none", "weighted", "ros", "rus", "smote", "adasyn", "hybrid"]
MODES = ["centralized", "local", "federated"]


def run_cell(
    seed: int,
    n_banks: int,
    scenario: str,
    model_type: str,
    balancing: str,
    mode: str,
    partitions_dir: Path,
    output_dir: Path,
    n_epochs: int,
    n_rounds: int,
    device: str,
) -> dict:
    """Run one cell of the matrix. Returns a result dict."""
    from experiments.run_baselines import run_centralized, run_local
    from fl.balancing.methods import BalancingMethod

    set_global_seed(seed)
    scenario_path = partitions_dir / f"N{n_banks}" / scenario
    if not scenario_path.exists():
        return {"status": "skipped", "reason": f"partition not found: {scenario_path}"}

    try:
        bal = BalancingMethod(balancing)
    except ValueError:
        return {"status": "error", "reason": f"unknown balancing: {balancing}"}

    try:
        if mode == "centralized":
            rows = run_centralized(
                partitions_dir, n_banks, scenario, n_epochs, seed, output_dir / "centralized", device
            )
            # Filter to just this model + balancing
            matching = [r for r in rows if r.get("model") == model_type and r.get("balancing") == balancing]
            return matching[0] if matching else {"status": "error", "reason": "no matching row"}

        elif mode == "local":
            rows = run_local(
                partitions_dir, n_banks, scenario, n_epochs, seed, output_dir / "local", device
            )
            matching = [r for r in rows if r.get("model") == model_type and r.get("balancing") == balancing]
            if not matching:
                return {"status": "error", "reason": "no matching row"}
            pr_aucs = [r.get("pr_auc") for r in matching if r.get("pr_auc")]
            return {
                "model": model_type, "balancing": balancing, "mode": mode,
                "n_banks": n_banks, "scenario": scenario, "seed": seed,
                "mean_pr_auc": float(np.mean(pr_aucs)) if pr_aucs else None,
                "mean_roc_auc": float(np.mean([r.get("roc_auc") for r in matching if r.get("roc_auc")])) if matching else None,
            }

        elif mode == "federated":
            if model_type == "knn":
                return {"status": "skipped", "reason": "KNN cannot be federated (see DEVIATIONS.md)"}

            # Run FL rounds in-process (no-blockchain)
            import pandas as pd
            from fl.preprocessing import FraudPreprocessor
            from fl.balancing.methods import BalancingMethod
            from fl.models.logistic_regression import LogisticRegressionModel
            from fl.models.cnn import CNNModel
            from fl.models.gaussian_nb import GaussianNBModel
            from fl.models.decision_tree import DecisionTreeModel
            from fl.models.random_forest import RandomForestModel
            from fl.bank_node.evaluator import evaluate
            from fl.bank_node.trainer import LocalTrainer
            from fl.strategies.gnb_strategy import GNBStrategy
            from fl.strategies.tree_strategy import TreeStrategy
            from fl.strategies.fabric_fedavg import FabricFedAvg
            from fl.secure_agg.keys import KeyPair, derive_shared_seed
            from fl.secure_agg.aggregator import SecureAggregator

            preprocessor = FraudPreprocessor()
            datasets = []
            for bank_id in range(n_banks):
                d = scenario_path / f"bank{bank_id}"
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

            n_features = preprocessor.n_features

            def make_model():
                kw = {"random_state": seed}
                if model_type == "logistic_regression": return LogisticRegressionModel(n_features)
                elif model_type == "cnn": return CNNModel(n_features)
                elif model_type == "gaussian_nb": return GaussianNBModel(n_features)
                elif model_type == "decision_tree": return DecisionTreeModel(n_features, **kw)
                elif model_type == "random_forest": return RandomForestModel(n_features, n_estimators=20, **kw)

            models = [make_model() for _ in range(n_banks)]
            global_params = models[0].get_parameters()
            is_torch = model_type in ("logistic_regression", "cnn")
            eval_type = "torch" if is_torch else ("gnb" if model_type == "gaussian_nb" else "tree")

            for round_num in range(1, n_rounds + 1):
                round_id = f"r{round_num}"
                keypairs = {i: KeyPair.generate() for i in range(n_banks)}
                all_seeds_round = {}
                for i in range(n_banks):
                    for j in range(i + 1, n_banks):
                        s = derive_shared_seed(keypairs[i], keypairs[j].public_bytes, i, j, round_id)
                        all_seeds_round[(i, j)] = s

                local_results = []
                for bank_id, (model, data) in enumerate(zip(models, datasets)):
                    model.set_parameters(global_params)
                    trainer = LocalTrainer(
                        model_type=model_type, model=model, balancing=bal,
                        n_epochs=n_epochs, device=device, random_state=seed + bank_id,
                    )
                    train_result = trainer.train(data["X_train"], data["y_train"], global_params)
                    local_results.append((model.get_parameters(), train_result["n_samples"]))

                # Secure aggregation
                flat_size = sum(p.size for p in global_params)
                agg = SecureAggregator(n_banks=n_banks, n_elements=flat_size)
                for bank_id, (lp, _) in enumerate(local_results):
                    flat = np.concatenate([p.ravel() for p in lp])
                    bank_seeds = {k: v for k, v in all_seeds_round.items() if bank_id in k}
                    masked = SecureAggregator.apply_mask(bank_id, flat, bank_seeds, flat_size)
                    agg.add_masked_update(bank_id, masked)
                agg_flat = agg.aggregate()
                offset, new_global = 0, []
                for p in global_params:
                    new_global.append(agg_flat[offset: offset + p.size].reshape(p.shape))
                    offset += p.size
                global_params = new_global

            for model in models:
                model.set_parameters(global_params)

            test_pr_aucs, test_roc_aucs, test_f1s = [], [], []
            for model, data in zip(models, datasets):
                m = evaluate(model, data["X_test"], data["y_test"], "test", eval_type, device)
                if m.get("pr_auc"): test_pr_aucs.append(m["pr_auc"])
                if m.get("roc_auc"): test_roc_aucs.append(m["roc_auc"])
                if m.get("f1"): test_f1s.append(m["f1"])

            return {
                "model": model_type, "balancing": balancing, "mode": mode,
                "n_banks": n_banks, "scenario": scenario, "seed": seed,
                "mean_pr_auc": float(np.mean(test_pr_aucs)) if test_pr_aucs else None,
                "mean_roc_auc": float(np.mean(test_roc_aucs)) if test_roc_aucs else None,
                "mean_f1": float(np.mean(test_f1s)) if test_f1s else None,
            }

    except Exception as e:
        logger.error(f"[matrix] FAILED seed={seed} n={n_banks} scenario={scenario} "
                     f"model={model_type} bal={balancing} mode={mode}: {e}")
        return {"status": "error", "reason": str(e), "model": model_type,
                "balancing": balancing, "mode": mode, "n_banks": n_banks,
                "scenario": scenario, "seed": seed}


def aggregate_seeds(rows: list) -> dict:
    """Aggregate results across 3 seeds: mean ± std."""
    pr_aucs = [r["mean_pr_auc"] for r in rows if r.get("mean_pr_auc")]
    roc_aucs = [r["mean_roc_auc"] for r in rows if r.get("mean_roc_auc")]
    f1s = [r["mean_f1"] for r in rows if r.get("mean_f1")]
    mu_pr, std_pr = mean_std(pr_aucs)
    mu_roc, std_roc = mean_std(roc_aucs)
    mu_f1, std_f1 = mean_std(f1s)
    base = {k: v for k, v in rows[0].items() if k not in ("seed", "mean_pr_auc", "mean_roc_auc", "mean_f1")}
    return {
        **base,
        "pr_auc_mean": round(mu_pr, 4) if mu_pr else None,
        "pr_auc_std": round(std_pr, 4) if std_pr else None,
        "roc_auc_mean": round(mu_roc, 4) if mu_roc else None,
        "roc_auc_std": round(std_roc, 4) if std_roc else None,
        "f1_mean": round(mu_f1, 4) if mu_f1 else None,
        "f1_std": round(std_f1, 4) if std_f1 else None,
        "n_seeds": len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Full experiment matrix")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 777])
    parser.add_argument("--n-banks", type=int, nargs="+", default=[3, 5, 8])
    parser.add_argument("--scenarios", nargs="+",
                        default=["S1", "S2", "S3_a0.1", "S3_a0.5", "S3_a1.0"])
    parser.add_argument("--models", nargs="+", default=ALL_MODELS)
    parser.add_argument("--balancing", nargs="+", default=ALL_BALANCING)
    parser.add_argument("--modes", nargs="+", default=MODES)
    parser.add_argument("--n-epochs", type=int, default=5)
    parser.add_argument("--n-rounds", type=int, default=5)
    parser.add_argument("--partitions", default="data/partitions")
    parser.add_argument("--output", default="results/full_matrix")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    partitions_dir = Path(args.partitions)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    git = get_git_hash()
    total_cells = (len(args.seeds) * len(args.n_banks) * len(args.scenarios)
                   * len(args.models) * len(args.balancing) * len(args.modes))
    logger.info(f"[matrix] Running {total_cells} cells (git={git})")

    all_raw = []
    done = 0
    t_start = time.time()

    for n_banks, scenario in itertools.product(args.n_banks, args.scenarios):
        scenario_rows = []
        for seed, model_type, balancing, mode in itertools.product(
            args.seeds, args.models, args.balancing, args.modes
        ):
            result = run_cell(
                seed, n_banks, scenario, model_type, balancing, mode,
                partitions_dir, output_dir, args.n_epochs, args.n_rounds, args.device
            )
            result["git_hash"] = git
            scenario_rows.append(result)
            all_raw.append(result)
            done += 1
            if done % 20 == 0:
                elapsed = time.time() - t_start
                eta = elapsed / done * (total_cells - done)
                logger.info(f"[matrix] {done}/{total_cells} cells done, ETA {eta/60:.1f}min")

        # Write per-scenario results
        write_results(
            {"rows": scenario_rows},
            output_dir / "raw",
            f"N{n_banks}_{scenario}",
            {"n_banks": n_banks, "scenario": scenario, "git": git},
        )

    # Aggregate over seeds and write summary
    # Group by (n_banks, scenario, model, balancing, mode)
    from collections import defaultdict
    groups = defaultdict(list)
    for r in all_raw:
        if r.get("status") in ("skipped", "error"):
            continue
        key = (r.get("n_banks"), r.get("scenario"), r.get("model"),
               r.get("balancing"), r.get("mode"))
        groups[key].append(r)

    summary_rows = [aggregate_seeds(rows) for rows in groups.values() if rows]

    # Write CSV summary
    try:
        import csv
        csv_path = output_dir / "summary.csv"
        if summary_rows:
            cols = list(summary_rows[0].keys())
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=cols)
                writer.writeheader()
                writer.writerows(summary_rows)
            logger.info(f"[matrix] Summary CSV: {csv_path}")
    except Exception as e:
        logger.warning(f"[matrix] Could not write CSV: {e}")

    # Write JSON summary
    write_results(
        {"summary": summary_rows, "raw_count": len(all_raw)},
        output_dir, "summary", vars(args),
    )

    logger.info(f"\n[matrix] Done. {len(all_raw)} cells in {(time.time()-t_start)/60:.1f}min")
    logger.info(f"[matrix] Results: {output_dir}/summary.json and summary.csv")


if __name__ == "__main__":
    main()
