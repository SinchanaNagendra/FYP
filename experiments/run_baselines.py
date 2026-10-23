"""
experiments/run_baselines.py
=============================
Phase 2: Centralized and local baselines for all 6 models x balancing methods.

Modes:
  centralized - Train on pooled data from all banks, evaluate on pooled test.
  local       - Train and evaluate separately for each bank.

Results are written to results/baselines/{mode}/.
"""

# --- path setup MUST come before any fl.* imports ---
import sys
import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Insert project root so `fl` and `experiments` packages are importable
# regardless of the working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fl.preprocessing import FraudPreprocessor           # noqa: E402
from fl.bank_node.evaluator import evaluate               # noqa: E402
from fl.balancing.methods import (                        # noqa: E402
    BalancingMethod,
    apply_balancing,
    compute_class_weights,
)
from experiments.utils import (                           # noqa: E402
    set_global_seed,
    write_results,
    format_results_table,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


ALL_MODELS = [
    "logistic_regression",
    "cnn",
    "gaussian_nb",
    "decision_tree",
    "random_forest",
    "knn",
]

ALL_BALANCING = [
    BalancingMethod.NONE,
    BalancingMethod.WEIGHTED,
    BalancingMethod.ROS,
    BalancingMethod.RUS,
    BalancingMethod.SMOTE,
    BalancingMethod.ADASYN,
    BalancingMethod.HYBRID,
]

# For the centralized run on 413K rows only use a reduced set of
# balancing methods to keep runtime reasonable.
CENTRALIZED_BALANCING = [
    BalancingMethod.NONE,
    BalancingMethod.WEIGHTED,
    BalancingMethod.RUS,
]


def build_model(model_type: str, n_features: int, seed: int):
    """Create a fresh model instance."""
    from fl.models import (
        LogisticRegressionModel,
        CNNModel,
        GaussianNBModel,
        DecisionTreeModel,
        RandomForestModel,
        KNNModel,
    )

    tree_kwargs = {"random_state": seed}

    models = {
        "logistic_regression": lambda: LogisticRegressionModel(n_features),
        "cnn": lambda: CNNModel(n_features),
        "gaussian_nb": lambda: GaussianNBModel(n_features),
        "decision_tree": lambda: DecisionTreeModel(n_features, **tree_kwargs),
        "random_forest": lambda: RandomForestModel(n_features, n_estimators=50, **tree_kwargs),
        "knn": lambda: KNNModel(n_features, n_neighbors=5),
    }

    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}")

    return models[model_type]()


def save_checkpoint(rows: list, output_dir: Path, mode: str, scenario: str, seed: int) -> None:
    """Persist progress to CSV after every completed run."""
    checkpoint_dir = output_dir / mode
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = checkpoint_dir / f"{scenario}_seed{seed}_progress.csv"
    pd.DataFrame(rows).to_csv(checkpoint_file, index=False)


def train_model(
    model_type: str,
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    balancing: BalancingMethod,
    n_epochs: int = 2,
    batch_size: int = 256,
    device: str = "cpu",
    seed: int = 42,
) -> dict:
    """Apply balancing then train the model. Returns timing metadata."""
    start_time = time.time()

    X_bal, y_bal = apply_balancing(
        X_train, y_train, balancing, random_state=seed)

    if model_type in ("logistic_regression", "cnn"):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        model.to(device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        class_weight_tensor = None
        if balancing == BalancingMethod.WEIGHTED:
            cw = compute_class_weights(y_bal)
            class_weight_tensor = torch.tensor(
                [cw.get(0, 1.0), cw.get(1, 1.0)],
                dtype=torch.float32,
                device=device,
            )

        dataset = TensorDataset(
            torch.as_tensor(X_bal, dtype=torch.float32),
            torch.as_tensor(y_bal, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(n_epochs):
            epoch_start = time.time()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = model(xb)
                if class_weight_tensor is not None:
                    w = class_weight_tensor[1] * yb + \
                        class_weight_tensor[0] * (1.0 - yb)
                    loss = nn.functional.binary_cross_entropy_with_logits(
                        logits, yb, weight=w)
                else:
                    loss = nn.functional.binary_cross_entropy_with_logits(
                        logits, yb)
                loss.backward()
                optimizer.step()
            logger.info(
                "[%s] epoch %d/%d completed in %.1f seconds",
                model_type, epoch + 1, n_epochs, time.time() - epoch_start,
            )

    elif model_type in ("gaussian_nb", "decision_tree", "random_forest", "knn"):
        model.fit(X_bal, y_bal)

    return {
        "train_time": time.time() - start_time,
        "n_samples": len(X_bal),
        "n_original": len(X_train),
    }


def get_model_type_for_eval(model_type: str) -> str:
    """Map model name to evaluator's model_type string."""
    if model_type in ("logistic_regression", "cnn"):
        return "torch"
    if model_type == "gaussian_nb":
        return "gnb"
    if model_type in ("decision_tree", "random_forest"):
        return "tree"
    return "sklearn"


def _stratified_subsample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Return a stratified subsample of n rows from df (by isFraud label)."""
    rng = np.random.default_rng(seed)
    fraud_mask = df["isFraud"] == 1
    n_fraud = min(int(n * 0.035) + 1, int(fraud_mask.sum()))
    n_legit = n - n_fraud
    idx_fraud = rng.choice(np.where(fraud_mask)[
                           0],   size=n_fraud, replace=False)
    idx_legit = rng.choice(np.where(~fraud_mask)[
                           0],  size=n_legit, replace=False)
    idx = np.sort(np.concatenate([idx_fraud, idx_legit]))
    return df.iloc[idx].reset_index(drop=True)


def run_centralized(
    partitions_dir: Path,
    n_banks: int,
    scenario: str,
    n_epochs: int,
    seed: int,
    output_dir: Path,
    device: str,
    selected_models: list = None,
    quick_subsample: int = None,
) -> list:
    """Train on pooled data from all banks; evaluate on pooled test set."""
    set_global_seed(seed)
    preprocessor = FraudPreprocessor()
    selected_models = selected_models or ALL_MODELS

    logger.info(
        "[centralized] Loading data: %d banks, scenario=%s", n_banks, scenario)

    train_dfs, val_dfs, test_dfs = [], [], []
    for bank_id in range(n_banks):
        base = partitions_dir / f"N{n_banks}" / scenario / f"bank{bank_id}"
        train_dfs.append(pd.read_parquet(base / "train.parquet"))
        val_dfs.append(pd.read_parquet(base / "val.parquet"))
        test_dfs.append(pd.read_parquet(base / "test.parquet"))

    df_train = pd.concat(train_dfs, ignore_index=True)
    df_val = pd.concat(val_dfs,   ignore_index=True)
    df_test = pd.concat(test_dfs,  ignore_index=True)

    if quick_subsample and len(df_train) > quick_subsample:
        df_train = _stratified_subsample(df_train, quick_subsample, seed)
        logger.info("[centralized] Quick subsample → %d rows", len(df_train))

    X_train, y_train = preprocessor.transform(df_train)
    X_val,   y_val = preprocessor.transform(df_val)
    X_test,  y_test = preprocessor.transform(df_test)
    n_features = preprocessor.n_features

    logger.info(
        "[centralized] Pooled: train=%s, val=%s, test=%s, fraud=%.3f%%",
        f"{len(X_train):,}", f"{len(X_val):,}", f"{len(X_test):,}",
        y_train.mean() * 100,
    )

    all_rows = []
    for model_type in selected_models:
        for balancing in CENTRALIZED_BALANCING:
            if model_type == "knn" and balancing not in (
                BalancingMethod.NONE, BalancingMethod.WEIGHTED
            ):
                continue

            logger.info("[centralized] model=%s, balancing=%s",
                        model_type, balancing.value)
            try:
                model = build_model(model_type, n_features, seed)
                train_info = train_model(
                    model_type, model, X_train, y_train,
                    balancing, n_epochs=n_epochs, device=device, seed=seed,
                )
                test_metrics = evaluate(
                    model, X_test, y_test, "test",
                    get_model_type_for_eval(model_type), device,
                )
                all_rows.append({
                    "mode": "centralized",
                    "model": model_type,
                    "balancing": balancing.value,
                    "scenario": scenario,
                    "seed": seed,
                    "train_time": round(train_info["train_time"], 3),
                    "n_train": train_info["n_samples"],
                    **{k: v for k, v in test_metrics.items() if k != "split"},
                })
            except Exception:
                logger.exception(
                    "[centralized] FAILED model=%s balancing=%s", model_type, balancing.value)
                all_rows.append({
                    "mode": "centralized", "model": model_type,
                    "balancing": balancing.value, "scenario": scenario,
                    "seed": seed, "error": "see log",
                })

            save_checkpoint(all_rows, output_dir,
                            "centralized", scenario, seed)

    write_results(
        {"rows": all_rows},
        output_dir / "centralized",
        f"{scenario}_seed{seed}",
        {"mode": "centralized", "n_banks": n_banks, "scenario": scenario,
         "seed": seed, "n_epochs": n_epochs},
    )
    return all_rows


def run_local(
    partitions_dir: Path,
    n_banks: int,
    scenario: str,
    n_epochs: int,
    seed: int,
    output_dir: Path,
    device: str,
    selected_models: list = None,
    quick_subsample: int = None,
) -> list:
    """Train and evaluate independently on each bank's local data."""
    set_global_seed(seed)
    preprocessor = FraudPreprocessor()
    all_rows = []
    selected_models = selected_models or ALL_MODELS

    for bank_id in range(n_banks):
        base = partitions_dir / f"N{n_banks}" / scenario / f"bank{bank_id}"
        df_train = pd.read_parquet(base / "train.parquet")
        df_test = pd.read_parquet(base / "test.parquet")

        if quick_subsample and len(df_train) > quick_subsample:
            df_train = _stratified_subsample(
                df_train, quick_subsample, seed + bank_id)

        X_train, y_train = preprocessor.transform(df_train)
        X_test,  y_test = preprocessor.transform(df_test)
        n_features = preprocessor.n_features

        logger.info(
            "[local] bank%d: train=%s, fraud=%.3f%%",
            bank_id, f"{len(X_train):,}", y_train.mean() * 100,
        )

        for model_type in selected_models:
            for balancing in ALL_BALANCING:
                if model_type == "knn" and balancing not in (
                    BalancingMethod.NONE, BalancingMethod.WEIGHTED
                ):
                    continue

                logger.info(
                    "[local] bank%d model=%s balancing=%s",
                    bank_id, model_type, balancing.value,
                )
                try:
                    model = build_model(model_type, n_features, seed)
                    train_info = train_model(
                        model_type, model, X_train, y_train,
                        balancing, n_epochs=n_epochs, device=device, seed=seed,
                    )
                    test_metrics = evaluate(
                        model, X_test, y_test, "test",
                        get_model_type_for_eval(model_type), device,
                    )
                    all_rows.append({
                        "mode": "local", "bank_id": bank_id,
                        "model": model_type, "balancing": balancing.value,
                        "scenario": scenario, "seed": seed,
                        "train_time": round(train_info["train_time"], 3),
                        "n_train": train_info["n_samples"],
                        **{k: v for k, v in test_metrics.items() if k != "split"},
                    })
                except Exception:
                    logger.exception(
                        "[local] bank%d FAILED model=%s balancing=%s",
                        bank_id, model_type, balancing.value,
                    )
                    all_rows.append({
                        "mode": "local", "bank_id": bank_id,
                        "model": model_type, "balancing": balancing.value,
                        "scenario": scenario, "seed": seed, "error": "see log",
                    })

                save_checkpoint(all_rows, output_dir, "local", scenario, seed)

    write_results(
        {"rows": all_rows},
        output_dir / "local",
        f"{scenario}_seed{seed}",
        {"mode": "local", "n_banks": n_banks, "scenario": scenario,
         "seed": seed, "n_epochs": n_epochs},
    )
    return all_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run centralized and local baselines")
    parser.add_argument(
        "--mode", choices=["centralized", "local", "both"], default="both")
    parser.add_argument("--n-banks", type=int, default=5)
    parser.add_argument("--scenario", default="S1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-epochs", type=int, default=2)
    parser.add_argument("--partitions", default="data/partitions")
    parser.add_argument("--output", default="results/baselines")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--models", nargs="+", default=None, choices=ALL_MODELS,
        help="Only run these models. E.g.: --models logistic_regression gaussian_nb",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Subsample to 20K rows, 1 epoch, skip CNN — for a fast smoke test.",
    )
    args = parser.parse_args()

    partitions_dir = Path(args.partitions)
    output_dir = Path(args.output)
    scenario_dir = partitions_dir / f"N{args.n_banks}" / args.scenario

    if not scenario_dir.exists():
        print(f"[ERROR] Partitions not found: {scenario_dir}")
        print("Run: python data/scripts/partition.py --n-banks 5 --seed 42")
        sys.exit(1)

    # Apply --quick overrides
    n_epochs = args.n_epochs
    selected_models = args.models or ALL_MODELS
    quick_subsample = None

    if args.quick:
        n_epochs = 1
        quick_subsample = 20_000
        selected_models = [m for m in selected_models if m != "cnn"]
        logger.info(
            "[quick] subsample=%d, epochs=1, models=%s",
            quick_subsample, selected_models,
        )

    if args.mode in ("centralized", "both"):
        rows = run_centralized(
            partitions_dir, args.n_banks, args.scenario,
            n_epochs, args.seed, output_dir, args.device,
            selected_models=selected_models,
            quick_subsample=quick_subsample,
        )
        cols = ["model", "balancing", "pr_auc", "roc_auc", "f1",
                "accuracy", "majority_class_accuracy", "train_time"]
        print("\n=== CENTRALIZED RESULTS ===")
        print(format_results_table(
            [{k: v for k, v in r.items() if k in cols}
             for r in rows if "error" not in r],
            cols,
        ))

    if args.mode in ("local", "both"):
        rows = run_local(
            partitions_dir, args.n_banks, args.scenario,
            n_epochs, args.seed, output_dir, args.device,
            selected_models=selected_models,
            quick_subsample=quick_subsample,
        )
        print(f"\n[local] {len(rows)} rows written to {output_dir / 'local'}")

    print("\n[baselines] Done. Results in:", output_dir)


if __name__ == "__main__":
    main()
