"""
fl/preprocessing.py
===================
Fixed, public preprocessing schema applied identically at each bank.

Design:
  - Vocabulary and feature schema are loaded from data/vocab/ (committed to repo).
  - Scaling statistics (mean/std) come from either:
      (a) data/vocab/numeric_stats.json (bootstrap, single-machine mode), or
      (b) the output of a federated secure-sum round (production mode).
  - No per-bank statistics are used; each bank applies the GLOBAL statistics.
  - This file must produce identical outputs for identical inputs regardless of
    which bank calls it.

Privacy invariant: this module processes only a single bank's local data in memory.
It never sends data out; it only consumes the public vocabulary.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Default paths (can be overridden)
# ---------------------------------------------------------------------------
DEFAULT_VOCAB_DIR = Path(__file__).parent.parent / "data" / "vocab"


class FraudPreprocessor:
    """
    Transforms a raw bank DataFrame into a (X, y) numpy array pair.

    Steps:
      1. Fill missing values (numeric → 0.0 + missing indicator; categorical → '__MISSING__')
      2. Categorical frequency-bucket encoding using public vocabulary
      3. Numeric z-score scaling using global statistics
      4. Assemble feature matrix in fixed order from feature_schema.json

    Parameters
    ----------
    vocab_dir : Path
        Directory containing categorical_vocab.json, numeric_stats.json,
        and feature_schema.json.
    """

    def __init__(self, vocab_dir: Optional[Path] = None):
        vocab_dir = Path(vocab_dir) if vocab_dir else DEFAULT_VOCAB_DIR
        self._load_vocab(vocab_dir)

    def _load_vocab(self, vocab_dir: Path) -> None:
        cat_vocab_path = vocab_dir / "categorical_vocab.json"
        num_stats_path = vocab_dir / "numeric_stats.json"
        schema_path = vocab_dir / "feature_schema.json"

        if not cat_vocab_path.exists():
            raise FileNotFoundError(
                f"Vocabulary not found at {cat_vocab_path}. "
                "Run: python data/scripts/compute_vocab.py"
            )

        with open(cat_vocab_path) as f:
            self.cat_vocab: dict[str, list[str]] = json.load(f)

        with open(num_stats_path) as f:
            self.num_stats: dict[str, dict] = json.load(f)

        with open(schema_path) as f:
            schema = json.load(f)

        self.feature_names: list[str] = schema["feature_names"]
        self.numeric_cols: list[str] = schema["numeric_cols"]
        self.categorical_cols: list[str] = schema["categorical_cols"]
        self.n_features: int = schema["n_features"]
        self.label_col: str = schema["label_col"]

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Transform a DataFrame into (X, y) arrays.

        Parameters
        ----------
        df : pd.DataFrame
            Raw bank data (train, val, or test split).

        Returns
        -------
        X : np.ndarray of shape (n_samples, n_features), dtype float32
        y : np.ndarray of shape (n_samples,), dtype int32
        """
        df = df.copy()

        # Label
        if self.label_col not in df.columns:
            raise ValueError(f"Label column '{self.label_col}' not found in DataFrame")
        y = df[self.label_col].fillna(0).astype(np.int32).values

        # Build feature dict: {feature_name: np.ndarray}
        feature_dict: dict[str, np.ndarray] = {}

        # ------------------------------------------------------------------
        # Numeric features: z-score scaling + missing indicator
        # ------------------------------------------------------------------
        for col in self.numeric_cols:
            stats = self.num_stats.get(col, {})
            mean = stats.get("mean", 0.0)
            std = stats.get("std", 1.0)
            if std < 1e-8:
                std = 1.0

            if col in df.columns:
                missing_mask = df[col].isna().astype(np.float32).values
                vals = df[col].fillna(0.0).astype(np.float32).values
                scaled = (vals - mean) / std
            else:
                n = len(df)
                missing_mask = np.ones(n, dtype=np.float32)
                scaled = np.zeros(n, dtype=np.float32)

            feature_dict[col] = scaled
            feature_dict[f"{col}__MISSING"] = missing_mask

        # ------------------------------------------------------------------
        # Categorical features: one-hot over vocabulary + OTHER + MISSING
        # ------------------------------------------------------------------
        for col in self.categorical_cols:
            vocab = self.cat_vocab.get(col, [])
            if not vocab:
                continue

            if col in df.columns:
                str_vals = df[col].fillna("__MISSING__").astype(str).values
            else:
                str_vals = np.full(len(df), "__MISSING__")

            # For each vocab entry: 1 if this row matches, else 0
            is_other = np.ones(len(df), dtype=np.float32)
            for val in vocab:
                indicator = (str_vals == val).astype(np.float32)
                feature_dict[f"{col}__{val}"] = indicator
                is_other = is_other * (1.0 - indicator)  # OTHER only if none matched

            feature_dict[f"{col}____OTHER__"] = is_other
            feature_dict[f"{col}____MISSING__"] = (str_vals == "__MISSING__").astype(np.float32)

        # ------------------------------------------------------------------
        # Assemble in fixed schema order
        # ------------------------------------------------------------------
        n_samples = len(df)
        X = np.zeros((n_samples, self.n_features), dtype=np.float32)

        for i, feat in enumerate(self.feature_names):
            if feat in feature_dict:
                X[:, i] = feature_dict[feat]
            # else: column not in data → stays 0.0

        return X, y

    def fit_transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Alias for transform (no fitting; schema is fixed and public)."""
        return self.transform(df)

    @property
    def output_dim(self) -> int:
        return self.n_features


# ---------------------------------------------------------------------------
# Federated scaling statistics update (for use with secure-sum round)
# ---------------------------------------------------------------------------

def compute_local_sufficient_stats(df: pd.DataFrame, numeric_cols: list[str]) -> dict:
    """
    Compute per-column (count, sum, sum_sq) for one bank's training data.
    These sufficient statistics can be securely summed across banks to derive
    global mean and std without sharing raw data.

    Returns a dict suitable for JSON serialisation and secure aggregation.
    """
    stats = {}
    for col in numeric_cols:
        if col in df.columns:
            vals = df[col].dropna().astype(float)
            stats[col] = {
                "count": int(vals.count()),
                "sum": float(vals.sum()),
                "sum_sq": float((vals ** 2).sum()),
            }
        else:
            stats[col] = {"count": 0, "sum": 0.0, "sum_sq": 0.0}
    return stats


def aggregate_sufficient_stats(bank_stats: list[dict]) -> dict:
    """
    Aggregate sufficient statistics from multiple banks into global mean/std.
    This is the server-side operation after the secure sum round.

    Parameters
    ----------
    bank_stats : list of per-bank sufficient stat dicts

    Returns
    -------
    global_stats : dict mapping col → {mean, std, count}
    """
    if not bank_stats:
        return {}

    cols = bank_stats[0].keys()
    global_stats = {}

    for col in cols:
        total_count = sum(b[col]["count"] for b in bank_stats)
        total_sum = sum(b[col]["sum"] for b in bank_stats)
        total_sum_sq = sum(b[col]["sum_sq"] for b in bank_stats)

        if total_count == 0:
            mean, std = 0.0, 1.0
        else:
            mean = total_sum / total_count
            variance = total_sum_sq / total_count - mean ** 2
            std = float(np.sqrt(max(variance, 0.0)))
            if std < 1e-8:
                std = 1.0

        global_stats[col] = {
            "count": total_count,
            "sum": total_sum,
            "sum_sq": total_sum_sq,
            "mean": mean,
            "std": std,
        }

    return global_stats
