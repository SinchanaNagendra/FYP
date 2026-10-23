"""
data/scripts/partition.py
=========================
Partitions the IEEE-CIS and ULB datasets into N banks under three scenarios:

  S1 – IID:               random uid assignment
  S2 – Feature/domain skew: assign uid clusters by ProductCD / card4 / card6
  S3 – Label+quantity skew: Dirichlet(alpha) over uid→bank assignment

Privacy-preserving assignment rules:
  - uid = SHA-256(card1 || card2 || card3 || card4 || card5 || addr1)[:16]
  - A uid and ALL its transactions go to exactly ONE bank.
  - Temporal split per bank: 70% train / 15% val / 15% test (by TransactionDT).
  - Validation and test sets are NEVER resampled.

Outputs per (N, scenario):
  data/partitions/N{n}/{scenario}/bank{i}/train.parquet
  data/partitions/N{n}/{scenario}/bank{i}/val.parquet
  data/partitions/N{n}/{scenario}/bank{i}/test.parquet
  data/partitions/N{n}/{scenario}/stats.json

Usage:
    python data/scripts/partition.py --n-banks 5 --seed 42 --output data/partitions
    python data/scripts/partition.py --n-banks 3 --seed 42 --scenarios S1 S2 S3
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy


# ---------------------------------------------------------------------------
# Feature columns used for the uid hash
# ---------------------------------------------------------------------------
UID_COLS = ["card1", "card2", "card3", "card4", "card5", "addr1"]

# Columns to keep from train_transaction after join (full feature list)
# V* columns included; missing ones filled with 0 or -1
KEEP_COLS_TRANSACTION = [
    "TransactionID", "isFraud", "TransactionDT", "TransactionAmt",
    "ProductCD", "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2", "dist1", "dist2",
    "P_emaildomain", "R_emaildomain",
    "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10",
    "C11", "C12", "C13", "C14",
    "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10",
    "D11", "D12", "D13", "D14", "D15",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
] + [f"V{i}" for i in range(1, 340)]

KEEP_COLS_IDENTITY = [
    "TransactionID",
    "id_01", "id_02", "id_03", "id_04", "id_05", "id_06",
    "id_07", "id_08", "id_09", "id_10", "id_11", "id_12",
    "id_13", "id_14", "id_15", "id_16", "id_17", "id_18",
    "id_19", "id_20", "id_21", "id_22", "id_23", "id_24",
    "id_25", "id_26", "id_27", "id_28", "id_29", "id_30",
    "id_31", "id_32", "id_33", "id_34", "id_35", "id_36",
    "id_37", "id_38",
    "DeviceType", "DeviceInfo",
]


# ---------------------------------------------------------------------------
# UID construction
# ---------------------------------------------------------------------------

def make_uid(df: pd.DataFrame) -> pd.Series:
    """
    Compute uid = SHA-256(str(card1)||str(card2)||...||str(addr1))[:16]
    for every row. fillna('NA') so missing values still hash consistently.
    """
    def row_hash(row):
        key = "|".join(str(row.get(c, "NA")) for c in UID_COLS)
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    return df.apply(row_hash, axis=1)


# ---------------------------------------------------------------------------
# Load and join
# ---------------------------------------------------------------------------

def load_ieee_cis(raw_dir: Path) -> pd.DataFrame:
    ieee_dir = raw_dir / "ieee_cis"
    tx_path = ieee_dir / "train_transaction.csv"
    id_path = ieee_dir / "train_identity.csv"

    if not tx_path.exists():
        print(f"[ERROR] {tx_path} not found. Run make data-download first.")
        sys.exit(1)

    print("[load] Reading train_transaction.csv ...")
    tx = pd.read_csv(tx_path, low_memory=False)

    print("[load] Reading train_identity.csv ...")
    id_ = pd.read_csv(id_path, low_memory=False)

    print(f"  train_transaction shape: {tx.shape}")
    print(f"  train_identity shape   : {id_.shape}")

    # Keep only columns that exist
    tx_cols = [c for c in KEEP_COLS_TRANSACTION if c in tx.columns]
    id_cols = [c for c in KEEP_COLS_IDENTITY if c in id_.columns]

    tx = tx[tx_cols].copy()
    id_ = id_[id_cols].copy()

    print("[load] Joining on TransactionID (left join) ...")
    df = tx.merge(id_, on="TransactionID", how="left")
    print(f"  Joined shape: {df.shape}")
    print(f"  Fraud rate  : {df['isFraud'].mean():.4%}")

    return df


# ---------------------------------------------------------------------------
# uid assignment helpers
# ---------------------------------------------------------------------------

def assign_s1_iid(uid_list: list, n_banks: int, rng: np.random.Generator) -> dict:
    """Random uid assignment (IID)."""
    shuffled = rng.permutation(uid_list)
    return {uid: int(i % n_banks) for i, uid in enumerate(shuffled)}


def assign_s2_feature_skew(df: pd.DataFrame, uid_list: list, n_banks: int) -> dict:
    """
    Feature/domain skew: cluster uids by (ProductCD, card4, card6) modal value,
    then assign clusters round-robin to banks.
    """
    # Compute modal (ProductCD, card4, card6) per uid
    uid_features = (
        df.groupby("uid")[["ProductCD", "card4", "card6"]]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "UNK")
        .reset_index()
    )
    # Assign cluster label as string combination
    uid_features["cluster"] = (
        uid_features["ProductCD"].astype(str) + "_"
        + uid_features["card4"].astype(str) + "_"
        + uid_features["card6"].astype(str)
    )
    # Sort clusters and assign banks round-robin so dominant clusters split across banks
    uid_features = uid_features.sort_values("cluster")
    uid_features["bank"] = [i % n_banks for i in range(len(uid_features))]
    return dict(zip(uid_features["uid"], uid_features["bank"].astype(int)))


def assign_s3_dirichlet(
    df: pd.DataFrame, uid_list: list, n_banks: int, alpha: float, rng: np.random.Generator
) -> dict:
    """
    Label + quantity skew via Dirichlet(alpha).
    For each fraud label (0, 1), draw a Dirichlet distribution over N banks,
    then assign uids proportionally.
    """
    uid_fraud = df.groupby("uid")["isFraud"].mean().reset_index()
    uid_fraud["binary_fraud"] = (uid_fraud["isFraud"] > 0).astype(int)

    uid_to_bank = {}
    for label in [0, 1]:
        label_uids = uid_fraud.loc[uid_fraud["binary_fraud"] == label, "uid"].tolist()
        if not label_uids:
            continue
        # Dirichlet proportions over N banks
        proportions = rng.dirichlet(alpha=np.full(n_banks, alpha))
        counts = (proportions * len(label_uids)).astype(int)
        # Adjust rounding
        diff = len(label_uids) - counts.sum()
        counts[np.argmax(proportions)] += diff
        # Assign
        shuffled = rng.permutation(label_uids)
        idx = 0
        for bank_id, count in enumerate(counts):
            for uid in shuffled[idx: idx + count]:
                uid_to_bank[uid] = int(bank_id)
            idx += count

    # Any unassigned uid (rounding edge) → bank 0
    for uid in uid_list:
        if uid not in uid_to_bank:
            uid_to_bank[uid] = 0

    return uid_to_bank


# ---------------------------------------------------------------------------
# Temporal split
# ---------------------------------------------------------------------------

def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split by TransactionDT (ascending): 70% train, 15% val, 15% test.
    Never shuffled. Returns (train, val, test).
    """
    df_sorted = df.sort_values("TransactionDT").reset_index(drop=True)
    n = len(df_sorted)
    n_train = int(n * 0.70)
    n_val = int(n * 0.85)  # 70% + 15%
    train = df_sorted.iloc[:n_train].copy()
    val = df_sorted.iloc[n_train:n_val].copy()
    test = df_sorted.iloc[n_val:].copy()
    return train, val, test


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def js_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen–Shannon distance between two probability distributions."""
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    js = 0.5 * entropy(p, m, base=2) + 0.5 * entropy(q, m, base=2)
    return float(np.sqrt(np.clip(js, 0, None)))


def compute_stats(
    bank_splits: dict[int, dict[str, pd.DataFrame]],
    global_df: pd.DataFrame,
) -> dict:
    global_fraud_rate = float(global_df["isFraud"].mean())
    global_tx_dist = None  # computed below

    stats = {
        "global": {
            "n_transactions": int(len(global_df)),
            "fraud_rate": global_fraud_rate,
        },
        "banks": {},
    }

    # Global fraud distribution as 2-bin distribution [legit, fraud]
    gfr = global_fraud_rate
    global_dist = np.array([1.0 - gfr, gfr])

    for bank_id, splits in bank_splits.items():
        train = splits["train"]
        val = splits["val"]
        test = splits["test"]
        total = pd.concat([train, val, test], ignore_index=True)

        n_total = len(total)
        n_train = len(train)
        n_val = len(val)
        n_test = len(test)
        fraud_rate = float(total["isFraud"].mean()) if n_total > 0 else 0.0
        train_fraud_rate = float(train["isFraud"].mean()) if n_train > 0 else 0.0

        # JS distance between this bank's label distribution and global
        bank_dist = np.array([1.0 - fraud_rate, fraud_rate])
        js_dist = js_distance(bank_dist + 1e-10, global_dist + 1e-10)

        stats["banks"][str(bank_id)] = {
            "n_total": n_total,
            "n_train": n_train,
            "n_val": n_val,
            "n_test": n_test,
            "fraud_rate_total": fraud_rate,
            "fraud_rate_train": train_fraud_rate,
            "js_distance_from_global": round(js_dist, 6),
            "min_TransactionDT": int(total["TransactionDT"].min()) if n_total > 0 else None,
            "max_TransactionDT": int(total["TransactionDT"].max()) if n_total > 0 else None,
        }

    return stats


# ---------------------------------------------------------------------------
# Main partitioner
# ---------------------------------------------------------------------------

def partition(
    df: pd.DataFrame,
    n_banks: int,
    scenario: str,
    alpha: float,
    seed: int,
    output_dir: Path,
) -> dict:
    rng = np.random.default_rng(seed)

    print(f"\n[partition] scenario={scenario}, n_banks={n_banks}, seed={seed}")

    
    # Build uid
    print("  Building pseudo-cardholder uid ...")
    df = df.copy()
    df["uid"] = make_uid(df)
    uid_list = df["uid"].unique().tolist()
    print(f"  Unique uids: {len(uid_list):,}")

    # Assign uids to banks
    if scenario == "S1":
        uid_to_bank = assign_s1_iid(uid_list, n_banks, rng)
    elif scenario == "S2":
        uid_to_bank = assign_s2_feature_skew(df, uid_list, n_banks)
    elif scenario == "S3" or scenario.startswith("S3_a"):
        uid_to_bank = assign_s3_dirichlet(
            df, uid_list, n_banks, alpha, rng
        )
    else:
        print(f"[ERROR] Unknown scenario: {scenario}")
        sys.exit(1)


    # Assign bank column
    df["bank_id"] = df["uid"].map(uid_to_bank)

    # Verify: no uid in two banks
    uid_bank_counts = df.groupby("uid")["bank_id"].nunique()
    multi_bank_uids = (uid_bank_counts > 1).sum()
    assert multi_bank_uids == 0, f"BUG: {multi_bank_uids} uids appear in >1 bank"

    # Per-bank temporal split and save
    base = output_dir / f"N{n_banks}" / scenario
    bank_splits = {}

    for bank_id in range(n_banks):
        bank_df = df[df["bank_id"] == bank_id].copy()
        if len(bank_df) == 0:
            print(f"  WARNING: bank {bank_id} has 0 transactions!")
            bank_splits[bank_id] = {"train": pd.DataFrame(), "val": pd.DataFrame(), "test": pd.DataFrame()}
            continue

        train, val, test = temporal_split(bank_df)
        bank_splits[bank_id] = {"train": train, "val": val, "test": test}

        bank_dir = base / f"bank{bank_id}"
        bank_dir.mkdir(parents=True, exist_ok=True)

        train.to_parquet(bank_dir / "train.parquet", index=False)
        val.to_parquet(bank_dir / "val.parquet", index=False)
        test.to_parquet(bank_dir / "test.parquet", index=False)

        print(
            f"  bank{bank_id}: total={len(bank_df):6,}  "
            f"train={len(train):6,}  val={len(val):5,}  test={len(test):5,}  "
            f"fraud={bank_df['isFraud'].mean():.3%}"
        )

    # Compute and save stats
    stats = compute_stats(bank_splits, df)
    stats["config"] = {
        "n_banks": n_banks,
        "scenario": scenario,
        "alpha": alpha,
        "seed": seed,
    }
    stats_path = base / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Stats written to {stats_path}")

    return stats


# ---------------------------------------------------------------------------
# ULB partitioner (same logic, simpler schema)
# ---------------------------------------------------------------------------

def partition_ulb(
    raw_dir: Path,
    n_banks: int,
    scenario: str,
    alpha: float,
    seed: int,
    output_dir: Path,
) -> None:
    ulb_path = raw_dir / "ulb" / "creditcard.csv"
    if not ulb_path.exists():
        print(f"[ulb] {ulb_path} not found, skipping ULB partitioning.")
        return

    print(f"\n[ulb-partition] Loading {ulb_path} ...")
    df = pd.read_csv(ulb_path)
    print(f"  Shape: {df.shape}, Fraud rate: {df['Class'].mean():.4%}")

    # ULB has no cardholder key; use row index hash as uid proxy
    # (ULB is only used for paper comparison, IID split is sufficient)
    df = df.rename(columns={"Class": "isFraud"})
    df["TransactionDT"] = df.index  # use row order as time proxy
    df["uid"] = [hashlib.sha256(str(i).encode()).hexdigest()[:16] for i in df.index]

    rng = np.random.default_rng(seed)
    uid_list = df["uid"].tolist()
    uid_to_bank = assign_s1_iid(uid_list, n_banks, rng)
    df["bank_id"] = df["uid"].map(uid_to_bank)

    base = output_dir / "ulb" / f"N{n_banks}" / scenario
    for bank_id in range(n_banks):
        bank_df = df[df["bank_id"] == bank_id].copy()
        train, val, test = temporal_split(bank_df)
        bank_dir = base / f"bank{bank_id}"
        bank_dir.mkdir(parents=True, exist_ok=True)
        train.to_parquet(bank_dir / "train.parquet", index=False)
        val.to_parquet(bank_dir / "val.parquet", index=False)
        test.to_parquet(bank_dir / "test.parquet", index=False)

    print(f"[ulb-partition] Done. Output: {base}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Partition fraud datasets into N banks")
    parser.add_argument("--n-banks", type=int, nargs="+", default=[5],
                        help="Number of banks (can specify multiple, e.g. 3 5 8)")
    parser.add_argument("--scenarios", nargs="+", default=["S1", "S2", "S3"],
                        choices=["S1", "S2", "S3"])
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.5, 1.0],
                        help="Dirichlet alpha values for S3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--raw", default="data/raw", help="Path to raw data directory")
    parser.add_argument("--output", default="data/partitions")
    parser.add_argument("--include-ulb", action="store_true",
                        help="Also partition ULB dataset")
    args = parser.parse_args()

    raw_dir = Path(args.raw)
    output_dir = Path(args.output)

    # Load IEEE-CIS once
    df = load_ieee_cis(raw_dir)

    for n_banks in args.n_banks:
        for scenario in args.scenarios:
            if scenario == "S3":
                for alpha in args.alphas:
                    # For S3, embed alpha in scenario name
                    tag = f"S3_a{alpha}"
                    partition(df, n_banks, tag, alpha, args.seed, output_dir)
            else:
                partition(df, n_banks, scenario, 1.0, args.seed, output_dir)

        if args.include_ulb:
            partition_ulb(raw_dir, n_banks, "S1", 1.0, args.seed, output_dir)

    print("\n[partition] All partitions complete.")
    print(f"  Output: {output_dir}")
    print("  Next: make data-verify")


if __name__ == "__main__":
    main()
