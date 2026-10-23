"""
experiments/generate_report_tables.py
=======================================
Reads results/full_matrix/summary.csv and generates LaTeX + CSV tables
for the final report.

Tables generated:
  1. Centralized vs federated PR-AUC gap (per model, per scenario)
  2. Per-balancing method comparison (best model, S1 IID)
  3. Privacy–utility trade-off (ε vs PR-AUC)
  4. Attack results (Mode A vs Mode B)
  5. Framework comparison (PyTorch/Flower vs TFF)

Usage:
    python experiments/generate_report_tables.py --output results/tables
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_summary(results_dir: Path) -> list:
    csv_path = results_dir / "full_matrix" / "summary.csv"
    if not csv_path.exists():
        print(f"[tables] summary.csv not found at {csv_path}")
        print("  Run: make experiments-all first")
        return []
    import csv
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def table_centralized_vs_federated(rows: list, output_dir: Path) -> None:
    """Table 1: Centralized vs federated PR-AUC gap."""
    if not rows:
        return
    models = ["logistic_regression", "cnn", "gaussian_nb", "decision_tree", "random_forest"]
    scenarios = ["S1", "S2", "S3_a0.1"]

    lines = [
        r"\begin{table}[h]",
        r"\caption{Centralized vs Federated PR-AUC (N=5 banks, best balancing per model)}",
        r"\label{tab:central_vs_fed}",
        r"\begin{tabular}{lllrrrr}",
        r"\hline",
        r"Model & Scenario & Balancing & Central & Local & Federated & Gap (C–F) \\",
        r"\hline",
    ]

    def get_pr(mode, model, scenario, balancing="none"):
        for r in rows:
            if (r.get("mode") == mode and r.get("model") == model
                    and r.get("scenario") == scenario
                    and r.get("balancing") == balancing
                    and r.get("n_banks") == "5"):
                v = r.get("pr_auc_mean")
                return float(v) if v else None
        return None

    for model in models:
        for scenario in scenarios:
            c = get_pr("centralized", model, scenario)
            l = get_pr("local", model, scenario)
            f = get_pr("federated", model, scenario)
            gap = round(c - f, 4) if c and f else None
            c_s = f"{c:.4f}" if c else "—"
            l_s = f"{l:.4f}" if l else "—"
            f_s = f"{f:.4f}" if f else "—"
            g_s = f"{gap:.4f}" if gap else "—"
            short_model = model.replace("_", " ").title()
            lines.append(f"{short_model} & {scenario} & none & {c_s} & {l_s} & {f_s} & {g_s} \\\\")

    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]

    out = output_dir / "tab_central_vs_fed.tex"
    out.write_text("\n".join(lines))
    print(f"[tables] Written {out}")


def table_privacy_utility(results_dir: Path, output_dir: Path) -> None:
    """Table 3: Privacy–utility (ε vs PR-AUC)."""
    dp_path = results_dir / "dp_sweep"
    if not dp_path.exists():
        print("[tables] dp_sweep results not found, skipping privacy table")
        return

    import glob
    files = sorted(dp_path.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return

    with open(files[0]) as f:
        data = json.load(f)

    results = data.get("results", {}).get("results", [])
    lines = [
        r"\begin{table}[h]",
        r"\caption{Privacy–Utility Trade-off (LR and CNN, N=5, S1 IID)}",
        r"\label{tab:privacy_utility}",
        r"\begin{tabular}{llrrr}",
        r"\hline",
        r"Model & Setting & $\sigma$ & Cum-$\varepsilon$ & PR-AUC \\",
        r"\hline",
    ]

    for r in results:
        if "error" in r:
            continue
        ps = r.get("privacy_summary", {})
        cum_e = ps.get("cumulative_epsilon")
        eps_str = f"{cum_e:.2f}" if cum_e else r"$\infty$"
        pr = r.get("final_test_pr_auc_mean")
        noise = r.get("noise_multiplier", "—")
        model = r.get("model", "").replace("_", " ").title()
        label = r.get("noise_label", "")
        lines.append(
            f"{model} & {label} & {noise} & {eps_str} & "
            f"{pr:.4f if pr else '—'} \\\\"
        )

    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    out = output_dir / "tab_privacy_utility.tex"
    out.write_text("\n".join(lines))
    print(f"[tables] Written {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate report tables")
    parser.add_argument("--results", default="results")
    parser.add_argument("--output", default="results/tables")
    args = parser.parse_args()

    results_dir = Path(args.results)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_summary(results_dir)
    table_centralized_vs_federated(rows, output_dir)
    table_privacy_utility(results_dir, output_dir)

    print(f"\n[tables] All tables written to {output_dir}/")
    print("  Include in LaTeX report with: \\input{results/tables/tab_*.tex}")


if __name__ == "__main__":
    main()
