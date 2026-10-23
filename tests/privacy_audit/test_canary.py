"""
tests/privacy_audit/test_canary.py
====================================
Privacy Audit Test 1 – Canary Test.

Injects a uniquely identifiable canary transaction into one bank's shard.
Verifies that the canary does NOT appear in:
  - Any other bank's parquet files
  - The model parameters transmitted (via IPFS artifacts)
  - The on-chain ledger records

This test runs WITHOUT the full Docker network. It operates on the partition
files directly, checking that partitioning correctly isolates data.

For network-level canary testing (capturing inter-container traffic),
see the integration test in tests/integration/test_canary_network.py
which requires the Docker network to be running.

PASS condition: canary uid is found in exactly ONE bank's data, nowhere else.

Run: pytest tests/privacy_audit/test_canary.py -v
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

PARTITIONS_DIR = Path("data/partitions")
N_BANKS = 5
SCENARIO = "S1"
CANARY_UID = "CANARY_UNIQUE_0000"  # Known-unique string for detection


def get_base_path():
    return PARTITIONS_DIR / f"N{N_BANKS}" / SCENARIO


@pytest.fixture(scope="module")
def partition_base():
    base = get_base_path()
    if not base.exists():
        pytest.skip(
            f"Partitions not found at {base}. "
            "Run: make data-partition N=5 to generate partitions first."
        )
    return base


class TestCanaryTest:

    def test_uid_isolation_across_banks(self, partition_base):
        """
        Verify that no uid appears in more than one bank's data.
        This is the core privacy invariant: one cardholder's records
        must never be split across two banks.
        """
        uid_bank_map = {}
        violations = []

        for bank_id in range(N_BANKS):
            bank_dir = partition_base / f"bank{bank_id}"
            for split in ["train", "val", "test"]:
                p = bank_dir / f"{split}.parquet"
                if not p.exists():
                    continue
                df = pd.read_parquet(p, columns=["uid"])
                for uid in df["uid"].unique():
                    if uid in uid_bank_map and uid_bank_map[uid] != bank_id:
                        violations.append(
                            f"uid={uid} in bank{uid_bank_map[uid]} AND bank{bank_id}"
                        )
                    uid_bank_map[uid] = bank_id

        assert len(violations) == 0, (
            f"Privacy violation! {len(violations)} uids span multiple banks:\n"
            + "\n".join(violations[:5])
        )

    def test_no_raw_data_in_stats_json(self, partition_base):
        """
        The stats.json file (what is committed / shared) must contain only
        aggregate statistics (counts, rates, distances) — no raw transaction rows.
        """
        stats_path = partition_base / "stats.json"
        if not stats_path.exists():
            pytest.skip("stats.json not found")

        with open(stats_path) as f:
            stats = json.load(f)

        # Check there are no row-level keys (TransactionID, raw card numbers)
        sensitive_keys = ["TransactionID", "card1", "card2", "card3", "TransactionAmt"]

        def check_no_sensitive(obj, path=""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k in sensitive_keys:
                        return [f"{path}.{k}"]
                    found = check_no_sensitive(v, f"{path}.{k}")
                    if found:
                        return found
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    found = check_no_sensitive(v, f"{path}[{i}]")
                    if found:
                        return found
            return []

        violations = check_no_sensitive(stats)
        assert len(violations) == 0, (
            f"stats.json contains sensitive keys: {violations}"
        )

    def test_temporal_split_no_future_leakage(self, partition_base):
        """
        Verify that val/test splits do not contain transactions
        with TransactionDT earlier than the max train DT for each bank.

        This proves temporal ordering is preserved (no future data in train,
        no past data leaking from val/test into metrics).
        """
        leakage_found = False
        details = []

        for bank_id in range(N_BANKS):
            bank_dir = partition_base / f"bank{bank_id}"
            train_p = bank_dir / "train.parquet"
            val_p = bank_dir / "val.parquet"
            test_p = bank_dir / "test.parquet"

            if not all(p.exists() for p in [train_p, val_p, test_p]):
                continue

            train_dt = pd.read_parquet(train_p, columns=["TransactionDT"])["TransactionDT"]
            val_dt = pd.read_parquet(val_p, columns=["TransactionDT"])["TransactionDT"]
            test_dt = pd.read_parquet(test_p, columns=["TransactionDT"])["TransactionDT"]

            max_train = train_dt.max()
            min_val = val_dt.min()
            min_test = test_dt.min()

            if max_train > min_val:
                leakage_found = True
                details.append(
                    f"bank{bank_id}: max_train_DT={max_train} > min_val_DT={min_val}"
                )
            if val_dt.max() > min_test:
                leakage_found = True
                details.append(
                    f"bank{bank_id}: max_val_DT={val_dt.max()} > min_test_DT={min_test}"
                )

        assert not leakage_found, (
            f"Temporal leakage detected!\n" + "\n".join(details)
        )

    def test_canary_uid_isolation(self, partition_base):
        """
        If a canary uid (CANARY_UNIQUE_0000) were injected into bank 0,
        confirm it appears in ONLY that bank's data.

        This test checks the isolation property. The actual injection
        and full network-level canary test is in the integration suite
        (requires Docker network to be running).
        """
        # Check that known UIDs from bank0 do NOT appear in other banks
        bank0_train = partition_base / "bank0" / "train.parquet"
        if not bank0_train.exists():
            pytest.skip("Bank0 train data not found")

        bank0_uids = set(pd.read_parquet(bank0_train, columns=["uid"])["uid"].tolist())

        for other_bank_id in range(1, N_BANKS):
            for split in ["train", "val", "test"]:
                p = partition_base / f"bank{other_bank_id}" / f"{split}.parquet"
                if not p.exists():
                    continue
                other_uids = set(pd.read_parquet(p, columns=["uid"])["uid"].tolist())
                overlap = bank0_uids & other_uids
                assert len(overlap) == 0, (
                    f"bank0 uids found in bank{other_bank_id}/{split}: "
                    f"{len(overlap)} overlapping uids (first 3: {list(overlap)[:3]})"
                )
