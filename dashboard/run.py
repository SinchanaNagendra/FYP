"""
dashboard/run.py
================
One-command dashboard launcher.

Usage:
    python dashboard/run.py

Opens: http://localhost:8000
"""
import http.server
import json
import os
import socketserver
import subprocess
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FRONTEND = ROOT / "dashboard" / "frontend" / "public"
PORT = 8000


def build_data_json():
    """Read all experiment results and write data.json for the dashboard."""
    import glob
    import statistics

    data = {
        "central_table": [],
        "fl_comparison": [],
        "round_metrics": [],
        "bank_metrics": [],
        "s3_bank_metrics": [],
    }

    # ── Centralized baselines ──────────────────────────────────────────────
    bl_path = RESULTS / "baselines" / "centralized" / "S1_seed42.json"
    if bl_path.exists():
        bl = json.load(open(bl_path))
        for r in bl["results"]["rows"]:
            if "error" not in r:
                data["central_table"].append({
                    "model":        r["model"],
                    "balancing":    r["balancing"],
                    "pr_auc":       round(r.get("pr_auc") or 0, 4),
                    "roc_auc":      round(r.get("roc_auc") or 0, 4),
                    "f1":           round(r.get("f1") or 0, 4),
                    "accuracy":     round(r.get("accuracy") or 0, 4),
                    "majority_acc": round(r.get("majority_class_accuracy") or 0, 4),
                    "train_time":   round(r.get("train_time") or 0, 1),
                })

    # ── FL results ─────────────────────────────────────────────────────────
    for f in sorted(glob.glob(str(RESULTS / "fl" / "*.json"))):
        d = json.load(open(f))
        name = os.path.basename(f).replace(".json", "")
        cfg = d.get("config", {})
        test = d["results"].get("final_test", [])
        pr_aucs = [r.get("pr_auc") for r in test if r.get("pr_auc")]
        roc_aucs = [r.get("roc_auc") for r in test if r.get("roc_auc")]
        rounds = d["results"].get("rounds", [])
        if pr_aucs:
            data["fl_comparison"].append({
                "name":         name,
                "model":        cfg.get("model", name),
                "scenario":     cfg.get("scenario", "S1"),
                "n_rounds":     len(rounds),
                "pr_auc_mean":  round(sum(pr_aucs) / len(pr_aucs), 4),
                "pr_auc_std":   round(statistics.stdev(pr_aucs) if len(pr_aucs) > 1 else 0, 4),
                "roc_auc_mean": round(sum(roc_aucs) / len(roc_aucs), 4) if roc_aucs else None,
            })

    # ── LR convergence curve ───────────────────────────────────────────────
    lr_path = RESULTS / "fl" / "logistic_regression_none_S1_seed42.json"
    if lr_path.exists():
        d = json.load(open(lr_path))
        for r in d["results"].get("rounds", []):
            val = r.get("val_metrics", [])
            pr_aucs = [m.get("pr_auc") for m in val if m.get("pr_auc")]
            data["round_metrics"].append({
                "round": r.get("round_num"),
                "mean_pr_auc": round(sum(pr_aucs) / len(pr_aucs), 4) if pr_aucs else None,
                "time": round(r.get("round_time", 0), 1),
            })

    # ── Per-bank results (RF S1) ───────────────────────────────────────────
    rf_path = RESULTS / "fl" / "random_forest_none_S1_seed42.json"
    if rf_path.exists():
        d = json.load(open(rf_path))
        for b in d["results"].get("final_test", []):
            data["bank_metrics"].append({
                "bank_id":    b.get("bank_id"),
                "pr_auc":     round(b.get("pr_auc") or 0, 4),
                "roc_auc":    round(b.get("roc_auc") or 0, 4),
                "f1":         round(b.get("f1") or 0, 4),
                "recall":     round(b.get("recall") or 0, 4),
                "fraud_rate": round(b.get("fraud_rate") or 0, 4),
                "n_samples":  b.get("n_samples"),
            })

    # ── S3 non-IID bank results ────────────────────────────────────────────
    s3_path = RESULTS / "fl" / "random_forest_none_S3_a0.1_seed42.json"
    if s3_path.exists():
        d = json.load(open(s3_path))
        for b in d["results"].get("final_test", []):
            data["s3_bank_metrics"].append({
                "bank_id":    b.get("bank_id"),
                "pr_auc":     round(b.get("pr_auc") or 0, 4),
                "fraud_rate": round(b.get("fraud_rate") or 0, 4),
                "n_samples":  b.get("n_samples"),
            })

    # Write
    out = FRONTEND / "data.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return data


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FRONTEND), **kwargs)

    def log_message(self, fmt, *args):
        pass  # suppress per-request logs


def main():
    print("=" * 55)
    print("  FL Fraud Detection Dashboard")
    print("=" * 55)

    print("\n[1/2] Reading experiment results...")
    data = build_data_json()
    n_central = len(data["central_table"])
    n_fl = len(data["fl_comparison"])
    n_banks = len(data["bank_metrics"])
    print(f"      Centralized baselines : {n_central} runs")
    print(f"      Federated runs         : {n_fl}")
    print(f"      Per-bank results       : {n_banks} banks")

    print(f"\n[2/2] Starting server on http://localhost:{PORT}")
    print(f"\n  ✅ Open this URL in your browser:")
    print(f"\n     http://localhost:{PORT}\n")
    print("  Press Ctrl+C to stop.\n")

    webbrowser.open(f"http://localhost:{PORT}")

    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        httpd.allow_reuse_address = True
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\nDashboard stopped.")


if __name__ == "__main__":
    main()
