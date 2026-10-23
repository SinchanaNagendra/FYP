"""
data/scripts/compute_vocab.py
==============================
Builds the public frequency-bucket vocabulary for categorical encoding.

This script reads ONLY the training partitions (never val/test) from ALL banks
to compute a shared vocabulary. The vocabulary is public (committed to the repo),
so any bank can use it for encoding without sharing data.

The federated alternative (for production) would be to use the secure-sum
aggregation round to compute global feature statistics. This script is the
offline equivalent used to bootstrap the system.

Outputs:
  data/vocab/categorical_vocab.json   – top-K most frequent values per column
  data/vocab/numeric_stats.json       – global count/sum/sum_sq per numeric col
  data/vocab/feature_schema.json      – final ordered feature list

Usage:
    python data/scripts/compute_vocab.py --partitions data/partitions --n-banks 5 --scenario S1
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


# Categorical columns that need frequency-bucket encoding
CATEGORICAL_COLS = [
    "ProductCD", "card4", "card6",
    "P_emaildomain", "R_emaildomain",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
    "DeviceType", "DeviceInfo",
] + [f"id_{i:02d}" for i in range(12, 39)]

# Max categories to keep per column; rest → "__OTHER__"
TOP_K = 50

# Numeric columns (V*, C*, D*, amounts, etc.) — the rest
NUMERIC_PATTERNS = ["V", "C", "D", "id_0", "id_1"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partitions", default="data/partitions")
    parser.add_argument("--n-banks", type=int, default=5)
    parser.add_argument("--scenario", default="S1")
    parser.add_argument("--output", default="data/vocab")
    args = parser.parse_args()

    base = Path(args.partitions) / f"N{args.n_banks}" / args.scenario
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load TRAINING splits only from all banks
    print("[vocab] Loading training splits from all banks (vocab is computed from train only)...")
    train_frames = []
    for bank_id in range(args.n_banks):
        p = base / f"bank{bank_id}" / "train.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            train_frames.append(df)
            print(f"  bank{bank_id}: {len(df):,} rows")
        else:
            print(f"  WARNING: {p} not found, skipping bank{bank_id}")

    if not train_frames:
        print("[ERROR] No training files found.")
        return

    all_train = pd.concat(train_frames, ignore_index=True)
    print(f"[vocab] Total training rows: {len(all_train):,}")

    # ------------------------------------------------------------------
    # Categorical vocabulary
    # ------------------------------------------------------------------
    cat_vocab = {}
    for col in CATEGORICAL_COLS:
        if col not in all_train.columns:
            continue
        counts = Counter(all_train[col].fillna("__MISSING__").astype(str))
        top = [val for val, _ in counts.most_common(TOP_K)]
        cat_vocab[col] = top
        print(f"  {col}: {len(top)} buckets (from {len(counts)} unique values)")

    vocab_path = output_dir / "categorical_vocab.json"
    with open(vocab_path, "w") as f:
        json.dump(cat_vocab, f, indent=2)
    print(f"[vocab] Categorical vocab saved to {vocab_path}")

    # ------------------------------------------------------------------
    # Numeric statistics (for federated scaling)
    # ------------------------------------------------------------------
    # These would normally come from a federated secure-sum round.
    # Here we compute them once from all training data as a bootstrap.
    numeric_cols = [
        c for c in all_train.columns
        if c not in CATEGORICAL_COLS + ["TransactionID", "isFraud", "TransactionDT", "uid", "bank_id"]
        and all_train[c].dtype in [np.float64, np.float32, np.int64, np.int32]
    ]

    numeric_stats = {}
    for col in numeric_cols:
        vals = all_train[col].fillna(0).astype(float)
        numeric_stats[col] = {
            "count": int(vals.count()),
            "sum": float(vals.sum()),
            "sum_sq": float((vals ** 2).sum()),
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "min": float(vals.min()),
            "max": float(vals.max()),
        }

    stats_path = output_dir / "numeric_stats.json"
    with open(stats_path, "w") as f:
        json.dump(numeric_stats, f, indent=2)
    print(f"[vocab] Numeric stats saved to {stats_path} ({len(numeric_stats)} columns)")

    # ------------------------------------------------------------------
    # Feature schema: ordered list of final feature names
    # ------------------------------------------------------------------
    # After encoding: each categorical col becomes TOP_K+1 binary indicators
    # (top K values + "__OTHER__") + 1 missing indicator
    feature_names = []

    # Numeric features (with missing indicator)
    for col in numeric_cols:
        feature_names.append(col)
        feature_names.append(f"{col}__MISSING")

    # Categorical features (one-hot over vocab + OTHER + MISSING)
    for col in CATEGORICAL_COLS:
        if col not in cat_vocab:
            continue
        for val in cat_vocab[col]:
            feature_names.append(f"{col}__{val}")
        feature_names.append(f"{col}____OTHER__")
        feature_names.append(f"{col}____MISSING__")

    schema = {
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "numeric_cols": numeric_cols,
        "categorical_cols": list(cat_vocab.keys()),
        "label_col": "isFraud",
    }

    schema_path = output_dir / "feature_schema.json"
    with open(schema_path, "w") as f:
        json.dump(schema, f, indent=2)
    print(f"[vocab] Feature schema saved to {schema_path}")
    print(f"        Total features: {len(feature_names)}")


if __name__ == "__main__":
    main()
