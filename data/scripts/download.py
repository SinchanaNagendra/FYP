"""
data/scripts/download.py
========================
Downloads the IEEE-CIS Fraud Detection dataset and the ULB Credit Card
Fraud dataset from Kaggle.

Usage:
    python data/scripts/download.py --output data/raw

Requirements:
    - Kaggle API credentials in ~/.kaggle/kaggle.json
    - pip install kaggle==1.6.12
    - Accepted competition rules for ieee-fraud-detection on Kaggle

Datasets:
    Primary : ieee-fraud-detection (Vesta Corporation)
              https://www.kaggle.com/competitions/ieee-fraud-detection
    Secondary: mlg-ulb/creditcardfraud (ULB)
               https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
"""

import argparse
import hashlib
import os
import subprocess
import sys
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Expected file checksums (SHA-256) — filled after first download and committed.
# Set to None to skip checksum verification during initial download.
# ---------------------------------------------------------------------------
EXPECTED_SHA256 = {
    "train_transaction.csv": None,  # fill after first download
    "train_identity.csv": None,
    "test_transaction.csv": None,
    "creditcard.csv": None,
}

IEEE_CIS_COMPETITION = "ieee-fraud-detection"
ULB_DATASET = "mlg-ulb/creditcardfraud"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_file(path: Path) -> None:
    expected = EXPECTED_SHA256.get(path.name)
    if expected is None:
        print(f"  [SKIP checksum] {path.name} (not yet recorded)")
        return
    actual = sha256_file(path)
    if actual != expected:
        print(f"  [ERROR] Checksum mismatch for {path.name}")
        print(f"    expected: {expected}")
        print(f"    actual  : {actual}")
        sys.exit(1)
    print(f"  [OK] {path.name} checksum verified")


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=False)
    if result.returncode != 0:
        print(f"[ERROR] Command failed with code {result.returncode}")
        sys.exit(result.returncode)


def download_ieee_cis(output_dir: Path) -> None:
    ieee_dir = output_dir / "ieee_cis"
    ieee_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    required = ["train_transaction.csv", "train_identity.csv"]
    if all((ieee_dir / f).exists() for f in required):
        print(f"[ieee-cis] Files already present in {ieee_dir}, skipping download.")
        for f in required:
            verify_file(ieee_dir / f)
        return

    print(f"[ieee-cis] Downloading from Kaggle competition: {IEEE_CIS_COMPETITION}")
    print("  NOTE: You must have accepted the competition rules on Kaggle first.")
    print("  Credentials: ~/.kaggle/kaggle.json")

    run(
        ["kaggle", "competitions", "download", "-c", IEEE_CIS_COMPETITION, "-p", str(ieee_dir)],
    )

    # Unzip
    for zip_path in ieee_dir.glob("*.zip"):
        print(f"  [unzip] {zip_path.name}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(ieee_dir)
        zip_path.unlink()

    for f in required:
        fpath = ieee_dir / f
        if not fpath.exists():
            print(f"[ERROR] Expected file not found after extraction: {fpath}")
            sys.exit(1)
        verify_file(fpath)

    print(f"[ieee-cis] Done. Files in {ieee_dir}")


def download_ulb(output_dir: Path) -> None:
    ulb_dir = output_dir / "ulb"
    ulb_dir.mkdir(parents=True, exist_ok=True)

    target = ulb_dir / "creditcard.csv"
    if target.exists():
        print(f"[ulb] Already present in {ulb_dir}, skipping download.")
        verify_file(target)
        return

    print(f"[ulb] Downloading from Kaggle dataset: {ULB_DATASET}")
    run(
        ["kaggle", "datasets", "download", "-d", ULB_DATASET, "-p", str(ulb_dir)],
    )

    for zip_path in ulb_dir.glob("*.zip"):
        print(f"  [unzip] {zip_path.name}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(ulb_dir)
        zip_path.unlink()

    if not target.exists():
        print(f"[ERROR] creditcard.csv not found after extraction in {ulb_dir}")
        sys.exit(1)

    verify_file(target)
    print(f"[ulb] Done. File: {target}")


def record_checksums(output_dir: Path) -> None:
    """Print checksums for all downloaded files so they can be committed."""
    print("\n[checksums] Recording SHA-256 for all downloaded files:")
    for csv in sorted(output_dir.rglob("*.csv")):
        digest = sha256_file(csv)
        print(f"  {csv.relative_to(output_dir)}: {digest}")
    print("\nCopy the above into EXPECTED_SHA256 in this script and commit.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download fraud detection datasets")
    parser.add_argument("--output", default="data/raw", help="Output directory for raw data")
    parser.add_argument("--ieee-only", action="store_true", help="Download only IEEE-CIS")
    parser.add_argument("--ulb-only", action="store_true", help="Download only ULB")
    parser.add_argument("--record-checksums", action="store_true",
                        help="Print checksums of downloaded files")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check kaggle CLI available
    result = subprocess.run(["kaggle", "--version"], capture_output=True)
    if result.returncode != 0:
        print("[ERROR] kaggle CLI not found. Install with: pip install kaggle==1.6.12")
        print("        Then set up ~/.kaggle/kaggle.json with your API credentials.")
        sys.exit(1)

    if not args.ulb_only:
        download_ieee_cis(output_dir)

    if not args.ieee_only:
        download_ulb(output_dir)

    if args.record_checksums:
        record_checksums(output_dir)

    print("\n[download] All datasets ready.")
    print(f"  IEEE-CIS: {output_dir}/ieee_cis/")
    print(f"  ULB     : {output_dir}/ulb/")


if __name__ == "__main__":
    main()
