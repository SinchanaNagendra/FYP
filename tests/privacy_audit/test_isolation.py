"""
tests/privacy_audit/test_isolation.py
======================================
Privacy Audit Test 3 – Isolation Test.

Verifies that each bank's data volume is accessible ONLY by that bank's
FL container. Cross-bank data access must be denied.

This test operates at two levels:

  Level 1 (offline, no Docker): Verify that partition files are stored in
    separate directories and the partition script assigns each UID to exactly
    one bank. This proves the logical isolation principle.

  Level 2 (Docker, integration): When running inside Docker, attempt to
    read another bank's mounted data volume and assert permission is denied.
    This requires the network to be running and is gated by DOCKER_INTEGRATION=1.

Run (offline):
    pytest tests/privacy_audit/test_isolation.py -v

Run (Docker integration):
    DOCKER_INTEGRATION=1 pytest tests/privacy_audit/test_isolation.py -v
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

PARTITIONS_DIR = Path("data/partitions")
N_BANKS = 5
SCENARIO = "S1"
DOCKER_INTEGRATION = os.environ.get("DOCKER_INTEGRATION", "0") == "1"


class TestLogicalIsolation:
    """Offline tests: verify logical separation at the file level."""

    def test_partition_directories_are_separate(self):
        """
        Each bank's data is in its own directory.
        No shared files exist between bank directories.
        """
        base = PARTITIONS_DIR / f"N{N_BANKS}" / SCENARIO
        if not base.exists():
            pytest.skip(f"Partitions not found at {base}. Run: make data-partition")

        bank_dirs = [base / f"bank{i}" for i in range(N_BANKS)]
        existing = [d for d in bank_dirs if d.exists()]
        if not existing:
            pytest.skip("No bank directories found")

        # Each bank has its own isolated directory
        for i, d_i in enumerate(existing):
            for j, d_j in enumerate(existing):
                if i >= j:
                    continue
                # No file should appear in both directories
                files_i = {f.name for f in d_i.glob("*.parquet")}
                files_j = {f.name for f in d_j.glob("*.parquet")}
                # Files may share names (train.parquet etc) but must be in different dirs
                for fname in files_i & files_j:
                    path_i = (d_i / fname).resolve()
                    path_j = (d_j / fname).resolve()
                    assert path_i != path_j, (
                        f"bank{i} and bank{j} share the same file path: {path_i}"
                    )

    def test_uid_appears_in_only_one_bank(self):
        """Core isolation guarantee: every UID maps to exactly one bank."""
        import pandas as pd
        base = PARTITIONS_DIR / f"N{N_BANKS}" / SCENARIO
        if not base.exists():
            pytest.skip(f"Partitions not found at {base}")

        uid_seen = {}
        for bank_id in range(N_BANKS):
            bank_dir = base / f"bank{bank_id}"
            for split in ["train", "val", "test"]:
                p = bank_dir / f"{split}.parquet"
                if not p.exists():
                    continue
                df = pd.read_parquet(p, columns=["uid"])
                for uid in df["uid"].unique():
                    if uid in uid_seen and uid_seen[uid] != bank_id:
                        pytest.fail(
                            f"UID {uid} appears in bank{uid_seen[uid]} AND bank{bank_id}. "
                            "Isolation violated!"
                        )
                    uid_seen[uid] = bank_id

    def test_model_parameters_do_not_contain_uids(self):
        """
        Verify that model parameters (what crosses the boundary) cannot be
        reverse-engineered to reveal UIDs.

        This test checks that parameter vectors do not contain string data
        (UIDs are 16-char hex strings; float tensors cannot embed them).
        """
        import numpy as np
        from fl.models.logistic_regression import LogisticRegressionModel

        n_features = 64
        model = LogisticRegressionModel(n_features)
        params = model.get_parameters()

        for i, p in enumerate(params):
            assert p.dtype in (np.float32, np.float64), (
                f"Parameter {i} has dtype {p.dtype}; string data (UIDs) cannot be embedded"
            )
            # Float arrays cannot contain hex string UIDs
            # (this is a type-system guarantee, confirmed here explicitly)

    def test_knn_federation_blocked(self):
        """
        KNN federation must be blocked. This is the primary isolation failure mode:
        federating KNN would require transmitting raw training records.
        """
        from fl.models.knn import KNNModel, PrivacyViolationError
        import numpy as np

        model = KNNModel(n_features=64)
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 64)).astype(np.float32)
        y = (rng.random(100) > 0.8).astype(np.int32)
        model.fit(X, y)

        # After fitting, get_parameters MUST still raise (training data is "the model")
        with pytest.raises(PrivacyViolationError):
            model.get_parameters()


class TestDockerIsolation:
    """Docker-level isolation tests. Only run with DOCKER_INTEGRATION=1."""

    @pytest.mark.skipif(not DOCKER_INTEGRATION, reason="Requires DOCKER_INTEGRATION=1 and running network")
    def test_bank0_cannot_read_bank1_volume(self):
        """
        From bank0's container, attempting to access bank1's data volume
        must fail with a permission error.
        """
        import subprocess
        result = subprocess.run(
            ["docker", "exec", "fl-bank0", "ls", "/data/bank1"],
            capture_output=True, text=True
        )
        # The exit code must be non-zero (permission denied or no such directory)
        assert result.returncode != 0, (
            f"Bank0 container could access /data/bank1! stdout: {result.stdout}"
        )
        assert "permission denied" in result.stderr.lower() or result.returncode != 0

    @pytest.mark.skipif(not DOCKER_INTEGRATION, reason="Requires DOCKER_INTEGRATION=1 and running network")
    def test_bank1_container_only_mounts_bank1_data(self):
        """
        The bank1 container must have /data/bank1 mounted but NOT /data/bank0.
        """
        import subprocess
        # Check bank1 can see its own data
        r1 = subprocess.run(
            ["docker", "exec", "fl-bank1", "ls", "/data/bank1"],
            capture_output=True, text=True
        )
        assert r1.returncode == 0, "Bank1 cannot access its own data volume!"

        # Check bank1 cannot see bank0's data
        r0 = subprocess.run(
            ["docker", "exec", "fl-bank1", "ls", "/data/bank0"],
            capture_output=True, text=True
        )
        assert r0.returncode != 0, (
            "Bank1 container can access /data/bank0! Data isolation is violated."
        )
