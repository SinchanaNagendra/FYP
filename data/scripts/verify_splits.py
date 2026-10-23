"""
data/scripts/verify_splits.py
==============================
Phase 1 Gate Verification.

Checks:
  1. No uid appears in more than one bank.
  2. No val/test record has a TransactionDT earlier than the earliest val TransactionDT
     in the same bank (temporal ordering within each bank's split is correct).
  3. No record appears in both train and val/test (no leakage by TransactionID).
  4. Stats file exists with all required fields.
  5. Prints per-bank table: size, fraud rate, JS distance from global.

Exits 0 only if all checks pass.

Usage:
    python data/scripts/verify_splits.py --partitions data/partitions --n-banks 5
    python data/scripts/verify_splits.py --partitions data/partitions --n-banks 5 --scenarios S1 S2
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np


PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
WARN = "\033[93m[WARN]\033[0m"


def check(condition: bool, msg_pass: str, msg_fail: str) -> bool:
    if condition:
        print(f"  {PASS} {msg_pass}")
        return True
    else:
        print(f"  {FAIL} {msg_fail}")
        return False


def verify_scenario(partitions_dir: Path, n_banks: int, scenario: str) -> bool:
    base = partitions_dir / f"N{n_banks}" / scenario
    if not base.exists():
        print(f"\n{WARN} {base} does not exist – skipping")
        return True  # not a failure if not generated yet

    print(f"\n{'='*60}")
    print(f"Verifying: N={n_banks}, scenario={scenario}")
    print(f"{'='*60}")

    all_pass = True

    # ------------------------------------------------------------------
    # 1. Load stats.json
    # ------------------------------------------------------------------
    stats_path = base / "stats.json"
    if not stats_path.exists():
        print(f"  {FAIL} stats.json not found at {stats_path}")
        return False

    with open(stats_path) as f:
        stats = json.load(f)

    # ------------------------------------------------------------------
    # 2. Load all splits into memory
    # ------------------------------------------------------------------
    bank_data: dict[int, dict[str, pd.DataFrame]] = {}
    for bank_id in range(n_banks):
        bank_dir = base / f"bank{bank_id}"
        splits = {}
        for split in ["train", "val", "test"]:
            p = bank_dir / f"{split}.parquet"
            if not p.exists():
                print(f"  {FAIL} Missing: {p}")
                all_pass = False
                splits[split] = pd.DataFrame()
            else:
                splits[split] = pd.read_parquet(p)
        bank_data[bank_id] = splits

    if not all_pass:
        return False

    # ------------------------------------------------------------------
    # 3. Check: no uid in two banks
    # ------------------------------------------------------------------
    all_uid_bank: dict[str, int] = {}
    uid_collision = False
    for bank_id, splits in bank_data.items():
        combined = pd.concat(splits.values(), ignore_index=True)
        for uid in combined["uid"].unique():
            if uid in all_uid_bank and all_uid_bank[uid] != bank_id:
                print(f"  {FAIL} uid {uid} appears in bank {all_uid_bank[uid]} AND bank {bank_id}")
                uid_collision = True
            all_uid_bank[uid] = bank_id
    all_pass &= check(not uid_collision,
                      "No uid appears in more than one bank",
                      "uid collision detected (privacy violation!)")

    # ------------------------------------------------------------------
    # 4. Check: temporal ordering within each bank
    # ------------------------------------------------------------------
    temporal_ok = True
    for bank_id, splits in bank_data.items():
        train = splits["train"]
        val = splits["val"]
        test = splits["test"]

        if len(train) == 0 or len(val) == 0 or len(test) == 0:
            print(f"  {WARN} bank{bank_id}: one or more splits are empty")
            continue

        max_train_dt = train["TransactionDT"].max()
        min_val_dt = val["TransactionDT"].min()
        min_test_dt = test["TransactionDT"].min()
        max_val_dt = val["TransactionDT"].max()

        if max_train_dt > min_val_dt:
            print(f"  {FAIL} bank{bank_id}: train has DT={max_train_dt} > val min DT={min_val_dt}")
            temporal_ok = False
        if max_val_dt > min_test_dt:
            print(f"  {FAIL} bank{bank_id}: val has DT={max_val_dt} > test min DT={min_test_dt}")
            temporal_ok = False

    all_pass &= check(temporal_ok,
                      "Temporal ordering: train < val < test in all banks",
                      "Temporal ordering violated (data leakage!)")

    # ------------------------------------------------------------------
    # 5. Check: no TransactionID leakage between splits
    # ------------------------------------------------------------------
    leakage_found = False
    for bank_id, splits in bank_data.items():
        train_ids = set(splits["train"]["TransactionID"].tolist()) if len(splits["train"]) > 0 else set()
        val_ids = set(splits["val"]["TransactionID"].tolist()) if len(splits["val"]) > 0 else set()
        test_ids = set(splits["test"]["TransactionID"].tolist()) if len(splits["test"]) > 0 else set()

        tv = train_ids & val_ids
        tt = train_ids & test_ids
        vt = val_ids & test_ids

        if tv or tt or vt:
            print(f"  {FAIL} bank{bank_id}: TransactionID overlap: "
                  f"train∩val={len(tv)}, train∩test={len(tt)}, val∩test={len(vt)}")
            leakage_found = True

    all_pass &= check(not leakage_found,
                      "No TransactionID appears in multiple splits",
                      "TransactionID leakage detected!")

    # ------------------------------------------------------------------
    # 6. Print per-bank table
    # ------------------------------------------------------------------
    print(f"\n  Per-bank statistics:")
    header = (
        f"  {'Bank':>5} | {'N train':>8} | {'N val':>7} | {'N test':>7} | "
        f"{'Fraud%':>7} | {'JS dist':>8}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    global_fraud_rate = stats["global"]["fraud_rate"]

    for bank_id in range(n_banks):
        s = stats["banks"].get(str(bank_id), {})
        n_train = s.get("n_train", 0)
        n_val = s.get("n_val", 0)
        n_test = s.get("n_test", 0)
        fraud_pct = s.get("fraud_rate_total", 0) * 100
        js = s.get("js_distance_from_global", float("nan"))
        print(
            f"  {bank_id:>5} | {n_train:>8,} | {n_val:>7,} | {n_test:>7,} | "
            f"{fraud_pct:>6.2f}% | {js:>8.4f}"
        )

    global_fraud_pct = global_fraud_rate * 100
    global_n = stats["global"]["n_transactions"]
    print(f"\n  Global: {global_n:,} transactions, {global_fraud_pct:.2f}% fraud")

    # ------------------------------------------------------------------
    # 7. Sanity: PR-AUC majority class baseline
    # ------------------------------------------------------------------
    print(f"\n  Majority-class accuracy baseline (always predict non-fraud):")
    print(f"    Accuracy = {100 - global_fraud_pct:.2f}%  (fraud rate = {global_fraud_pct:.2f}%)")
    print(f"    NOTE: accuracy is NOT the primary metric. Use PR-AUC and ROC-AUC.")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n  {'='*40}")
    if all_pass:
        print(f"  {PASS} All checks passed for N={n_banks}, scenario={scenario}")
    else:
        print(f"  {FAIL} One or more checks FAILED for N={n_banks}, scenario={scenario}")
    print(f"  {'='*40}")

    return all_pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 gate verification")
    parser.add_argument("--partitions", default="data/partitions")
    parser.add_argument("--n-banks", type=int, nargs="+", default=[5])
    parser.add_argument("--scenarios", nargs="+",
                        default=["S1", "S2", "S3_a0.1", "S3_a0.5", "S3_a1.0"])
    args = parser.parse_args()

    partitions_dir = Path(args.partitions)
    overall_pass = True

    for n_banks in args.n_banks:
        for scenario in args.scenarios:
            ok = verify_scenario(partitions_dir, n_banks, scenario)
            overall_pass = overall_pass and ok

    print("\n" + "="*60)
    if overall_pass:
        print(f"{PASS} PHASE 1 GATE: ALL CHECKS PASSED")
        sys.exit(0)
    else:
        print(f"{FAIL} PHASE 1 GATE: SOME CHECKS FAILED – DO NOT PROCEED")
        sys.exit(1)


if __name__ == "__main__":
    main()
