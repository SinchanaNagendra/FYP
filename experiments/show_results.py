"""Quick script to display the baseline results table."""
import csv
import sys
from pathlib import Path

results_dir = Path("results/baselines/centralized")
csv_files = sorted(results_dir.glob("*_progress.csv"), key=lambda p: p.stat().st_mtime, reverse=True)

if not csv_files:
    print("No results found yet. Run: python experiments/run_baselines.py --quick ...")
    sys.exit(0)

path = csv_files[0]
print(f"\nResults from: {path}\n")

with open(path) as f:
    rows = list(csv.DictReader(f))

rows = [r for r in rows if "error" not in r]

if not rows:
    print("No completed rows yet.")
    sys.exit(0)

# Header
print(f"  {'model':<22} {'balancing':<10} {'PR-AUC':<8} {'ROC-AUC':<8} {'F1':<8} {'Accuracy':<9} {'Maj-cls':<9} {'Time(s)'}")
print("  " + "-" * 85)

for r in rows:
    def fmt(key, decimals=4):
        v = r.get(key)
        try:
            return f"{float(v):.{decimals}f}"
        except (TypeError, ValueError):
            return str(v) if v else "N/A"

    print(
        f"  {r.get('model','?'):<22} "
        f"{r.get('balancing','?'):<10} "
        f"{fmt('pr_auc'):<8} "
        f"{fmt('roc_auc'):<8} "
        f"{fmt('f1'):<8} "
        f"{fmt('accuracy'):<9} "
        f"{fmt('majority_class_accuracy'):<9} "
        f"{fmt('train_time', 1)}"
    )

print(f"\n  Total rows: {len(rows)}")
print("  NOTE: PR-AUC is the primary metric for this imbalanced dataset (3.5% fraud).")
print("  Majority-class baseline accuracy (always predict non-fraud) is shown for comparison.")
