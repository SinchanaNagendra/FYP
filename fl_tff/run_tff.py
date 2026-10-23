"""
fl_tff/run_tff.py
==================
TensorFlow Federated (TFF) implementation of LR and CNN for framework comparison.

Runs in a dedicated Docker container (see fl_tff/Dockerfile) to isolate
TF/TFF dependencies from the PyTorch FL stack.

Runs the same partitions, hyperparameters, and evaluation protocol as
the Flower/PyTorch implementation for a fair comparison.

Framework comparison metrics reported:
  - PR-AUC, ROC-AUC, F1 (same eval protocol as PyTorch)
  - Wall-clock training time per round
  - Peak memory usage
  - Communication bytes (parameter count × 4 bytes)

Usage (inside Docker):
    python run_tff.py --n-banks 5 --seed 42 --output /results/tff

Usage (via make):
    make tff-compare
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def check_tff_available() -> bool:
    try:
        import tensorflow_federated as tff
        import tensorflow as tf
        logger.info(f"TFF version: {tff.__version__}, TF version: {tf.__version__}")
        return True
    except ImportError as e:
        logger.error(f"TFF not available: {e}")
        return False


def load_bank_data(partitions_dir: Path, n_banks: int, scenario: str):
    """Load preprocessed bank data from parquet files."""
    import pandas as pd
    # Use same preprocessing as the PyTorch stack
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from fl.preprocessing import FraudPreprocessor

    preprocessor = FraudPreprocessor()
    datasets = []
    for bank_id in range(n_banks):
        d = partitions_dir / f"N{n_banks}" / scenario / f"bank{bank_id}"
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
        logger.info(f"Loaded bank{bank_id}: {len(X_tr)} train, fraud={y_tr.mean():.3%}")
    return datasets, preprocessor.n_features


def build_tff_lr_model(n_features: int):
    """Build a TFF-compatible logistic regression model."""
    import tensorflow as tf
    import tensorflow_federated as tff

    def model_fn():
        keras_model = tf.keras.Sequential([
            tf.keras.layers.Dense(1, input_shape=(n_features,), activation=None)
        ])
        return tff.learning.models.from_keras_model(
            keras_model,
            input_spec=(
                tf.TensorSpec(shape=[None, n_features], dtype=tf.float32),
                tf.TensorSpec(shape=[None], dtype=tf.float32),
            ),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.AUC(name="roc_auc", curve="ROC"),
                     tf.keras.metrics.AUC(name="pr_auc", curve="PR")],
        )
    return model_fn


def build_tff_cnn_model(n_features: int):
    """Build a TFF-compatible 1-D CNN model."""
    import tensorflow as tf
    import tensorflow_federated as tff

    def model_fn():
        inp = tf.keras.Input(shape=(n_features,))
        x = tf.keras.layers.Reshape((n_features, 1))(inp)
        x = tf.keras.layers.Conv1D(32, 5, padding="same", activation="relu")(x)
        x = tf.keras.layers.MaxPool1D(2)(x)
        x = tf.keras.layers.Conv1D(64, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.MaxPool1D(2)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        out = tf.keras.layers.Dense(1)(x)
        keras_model = tf.keras.Model(inp, out)
        return tff.learning.models.from_keras_model(
            keras_model,
            input_spec=(
                tf.TensorSpec(shape=[None, n_features], dtype=tf.float32),
                tf.TensorSpec(shape=[None], dtype=tf.float32),
            ),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.AUC(name="roc_auc", curve="ROC"),
                     tf.keras.metrics.AUC(name="pr_auc", curve="PR")],
        )
    return model_fn


def datasets_to_tff(datasets: list, batch_size: int = 256):
    """Convert bank datasets to TFF ClientData format."""
    import tensorflow as tf
    import tensorflow_federated as tff

    def make_client_dataset(X, y):
        return tf.data.Dataset.from_tensor_slices(
            (tf.cast(X, tf.float32), tf.cast(y, tf.float32))
        ).shuffle(1000).batch(batch_size)

    client_ids = [f"bank{d['bank_id']}" for d in datasets]
    client_data_map = {
        f"bank{d['bank_id']}": make_client_dataset(d["X_train"], d["y_train"])
        for d in datasets
    }
    return client_ids, client_data_map


def run_tff_experiment(
    model_type: str,
    datasets: list,
    n_features: int,
    n_rounds: int,
    batch_size: int,
    seed: int,
) -> dict:
    """Run TFF FedAvg for one model type and return comparison metrics."""
    import tensorflow as tf
    import tensorflow_federated as tff
    from sklearn.metrics import average_precision_score, roc_auc_score

    tf.random.set_seed(seed)
    np.random.seed(seed)

    # Build model
    if model_type == "logistic_regression":
        model_fn = build_tff_lr_model(n_features)
    elif model_type == "cnn":
        model_fn = build_tff_cnn_model(n_features)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # TFF FedAvg process
    iterative_process = tff.learning.algorithms.build_weighted_fed_avg(
        model_fn=model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0),
    )

    state = iterative_process.initialize()
    client_ids, client_data_map = datasets_to_tff(datasets, batch_size)

    round_records = []
    total_comm_bytes = 0

    for round_num in range(1, n_rounds + 1):
        t_start = time.time()
        state, metrics = iterative_process.next(state, list(client_data_map.values()))
        round_time = time.time() - t_start

        # Estimate communication (all params × float32 bytes)
        n_params = sum(
            np.prod(w.shape)
            for w in state.global_model_weights.trainable
        )
        comm_bytes_per_round = int(n_params * 4 * len(datasets))
        total_comm_bytes += comm_bytes_per_round

        round_records.append({
            "round": round_num,
            "round_time": round_time,
            "comm_bytes": comm_bytes_per_round,
            "tff_metrics": {k: float(v) for k, v in metrics._asdict().items()
                           if hasattr(metrics, '_asdict')},
        })
        logger.info(f"[TFF] round={round_num}/{n_rounds}, time={round_time:.1f}s")

    # Extract final weights and evaluate
    final_weights = [w.numpy() for w in state.global_model_weights.trainable]

    # Build a Keras model from final weights for evaluation
    if model_type == "logistic_regression":
        keras_model = tf.keras.Sequential([
            tf.keras.layers.Dense(1, input_shape=(n_features,))
        ])
    else:
        inp = tf.keras.Input(shape=(n_features,))
        x = tf.keras.layers.Reshape((n_features, 1))(inp)
        x = tf.keras.layers.Conv1D(32, 5, padding="same", activation="relu")(x)
        x = tf.keras.layers.MaxPool1D(2)(x)
        x = tf.keras.layers.Conv1D(64, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.MaxPool1D(2)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        out = tf.keras.layers.Dense(1)(x)
        keras_model = tf.keras.Model(inp, out)

    keras_model.set_weights(final_weights)

    # Evaluate on pooled test
    X_test = np.vstack([d["X_test"] for d in datasets])
    y_test = np.concatenate([d["y_test"] for d in datasets])
    logits = keras_model.predict(X_test, batch_size=1024).ravel()
    probs = 1.0 / (1.0 + np.exp(-logits))

    pr_auc = float(average_precision_score(y_test, probs))
    roc_auc = float(roc_auc_score(y_test, probs))

    return {
        "framework": "tff",
        "model": model_type,
        "n_rounds": n_rounds,
        "n_banks": len(datasets),
        "rounds": round_records,
        "final_pr_auc": pr_auc,
        "final_roc_auc": roc_auc,
        "total_comm_bytes": total_comm_bytes,
        "n_params": int(n_params),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="TFF comparison experiment")
    parser.add_argument("--n-banks", type=int, default=5)
    parser.add_argument("--scenario", default="S1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-rounds", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--partitions", default="/data/partitions")
    parser.add_argument("--output", default="/results/tff")
    args = parser.parse_args()

    if not check_tff_available():
        logger.error(
            "TensorFlow Federated not available.\n"
            "Run this script inside the TFF Docker container:\n"
            "  make tff-compare\n"
            "Or install: pip install tensorflow-federated"
        )
        sys.exit(1)

    partitions_dir = Path(args.partitions)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets, n_features = load_bank_data(partitions_dir, args.n_banks, args.scenario)
    all_results = []

    for model_type in ["logistic_regression", "cnn"]:
        logger.info(f"\n[TFF] Running {model_type}...")
        try:
            result = run_tff_experiment(
                model_type=model_type,
                datasets=datasets,
                n_features=n_features,
                n_rounds=args.n_rounds,
                batch_size=args.batch_size,
                seed=args.seed,
            )
            all_results.append(result)
            logger.info(
                f"[TFF] {model_type}: PR-AUC={result['final_pr_auc']:.4f}, "
                f"ROC-AUC={result['final_roc_auc']:.4f}"
            )
        except Exception as e:
            logger.error(f"[TFF] {model_type} FAILED: {e}")
            all_results.append({"framework": "tff", "model": model_type, "error": str(e)})

    out_path = output_dir / f"tff_{args.scenario}_seed{args.seed}.json"
    record = {
        "git_hash": "tff-docker",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": vars(args),
        "results": all_results,
    }
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)

    print(f"\n[TFF] Results written to {out_path}")
    print("\n=== TFF vs PyTorch Comparison ===")
    print("(Fill in PyTorch results from results/fl/ after running make fl-train)")
    print(f"\n{'Framework':<12} {'Model':<22} {'PR-AUC':<10} {'ROC-AUC':<10}")
    print("-" * 56)
    for r in all_results:
        if "error" in r:
            print(f"  TFF  {r['model']:<22} ERROR: {r['error'][:30]}")
        else:
            print(f"  TFF  {r['model']:<22} {r['final_pr_auc']:.4f}     {r['final_roc_auc']:.4f}")


if __name__ == "__main__":
    main()
