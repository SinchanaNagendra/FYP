"""
experiments/generate_report_figures.py
========================================
Generates all report figures as PNG files in results/figures/.

Figures:
  fig1_privacy_utility_curve.png  – ε vs PR-AUC for LR and CNN
  fig2_central_vs_fed_gap.png     – bar chart: centralized vs federated gap
  fig3_attack_results.png         – grouped bar: attack PR-AUC under Mode A vs B
  fig4_learning_curves.png        – PR-AUC per round across scenarios
  fig5_balancing_comparison.png   – box plot: PR-AUC by balancing method

Usage:
    python experiments/generate_report_figures.py --output results/figures
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def setup_style():
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    plt.style.use("dark_background")
    return plt


def fig_privacy_utility(results_dir: Path, output_dir: Path) -> None:
    plt = setup_style()
    dp_files = sorted(
        (results_dir / "dp_sweep").glob("*.json"),
        key=lambda p: p.stat().st_mtime, reverse=True
    )
    if not dp_files:
        print("[figures] No DP sweep results")
        return
    with open(dp_files[0]) as f:
        data = json.load(f)

    results = [r for r in data.get("results", {}).get(
        "results", []) if "error" not in r]
    if not results:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {"logistic_regression": "#60a5fa", "cnn": "#34d399"}

    for ax, model_type in zip(axes, ["logistic_regression", "cnn"]):
        model_results = [r for r in results if r.get("model") == model_type]
        epsilons, pr_aucs, labels = [], [], []
        for r in model_results:
            ps = r.get("privacy_summary", {})
            cum_e = ps.get("cumulative_epsilon")
            pr = r.get("final_test_pr_auc_mean")
            if pr:
                epsilons.append(cum_e if cum_e else 100)  # ∞ mapped to 100
                pr_aucs.append(pr)
                labels.append(r.get("noise_label", ""))

        if epsilons:
            ax.scatter(epsilons, pr_aucs, s=80, color=colors.get(model_type, "#f472b6"),
                       zorder=5)
            ax.plot(epsilons, pr_aucs, "--", alpha=0.5,
                    color=colors.get(model_type, "#f472b6"))
            for i, lbl in enumerate(labels):
                ax.annotate(lbl, (epsilons[i], pr_aucs[i]), textcoords="offset points",
                            xytext=(5, 5), fontsize=9, color="#94a3b8")

        ax.set_xlabel("Cumulative ε (100 = ∞)")
        ax.set_ylabel("PR-AUC (test)")
        ax.set_title(model_type.replace("_", " ").title())
        ax.grid(True, alpha=0.2)

    plt.suptitle(
        "Privacy–Utility Trade-off (DP-SGD, N=5 banks, S1 IID)", y=1.02)
    plt.tight_layout()
    out = output_dir / "fig1_privacy_utility_curve.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[figures] Written {out}")


def fig_learning_curves(results_dir: Path, output_dir: Path) -> None:
    plt = setup_style()
    fl_files = sorted(
        list((results_dir / "fl_fabric").glob("*.json"))
        + list((results_dir / "fl").glob("*.json")),
        key=lambda p: p.stat().st_mtime, reverse=True
    )
    if not fl_files:
        print("[figures] No FL results for learning curves")
        return
    with open(fl_files[0]) as f:
        data = json.load(f)

    rounds = data.get("results", {}).get("rounds", [])
    if not rounds:
        return

    round_nums = [r["round_num"] for r in rounds]
    pr_aucs = [r.get("mean_pr_auc") for r in rounds]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(round_nums, pr_aucs, "o-",
            color="#60a5fa", linewidth=2, markersize=5)
    ax.set_xlabel("Round")
    ax.set_ylabel("Mean PR-AUC across banks")
    ax.set_title("Federated Learning Convergence (PR-AUC per round)")
    ax.grid(True, alpha=0.2)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    out = output_dir / "fig4_learning_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[figures] Written {out}")


def fig_attack_results(results_dir: Path, output_dir: Path) -> None:
    plt = setup_style()
    attack_files = sorted(
        (results_dir / "attacks").glob("*.json"),
        key=lambda p: p.stat().st_mtime, reverse=True
    )
    if not attack_files:
        print("[figures] No attack results")
        return
    with open(attack_files[0]) as f:
        data = json.load(f)

    attacks = data.get("results", {}).get("results", [])
    if not attacks:
        return

    # Filter to LR, group by attack type
    lr_attacks = [a for a in attacks if a.get(
        "model") == "logistic_regression" and "error" not in a]
    labels = []
    mode_a_pr = []  # label_flip/free_rider = Mode A context
    mode_b_pr = []  # robust agg = Mode B

    mode_a = [a for a in lr_attacks if "robust_method" not in a]
    mode_b = [a for a in lr_attacks if "robust_method" in a]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if mode_a:
        def mode_a_label(a):
            pt = a.get('poison_type', '') or f"m={a.get('n_malicious', '?')}"
            return f"{a.get('attack', '?')}\n{pt}".strip()
        a_labels = [mode_a_label(a) for a in mode_a]
        a_prs = [a.get("mean_pr_auc", 0) or 0 for a in mode_a]
        axes[0].barh(a_labels, a_prs, color="#60a5fa")
        axes[0].set_xlim(0, 1)
        axes[0].set_xlabel("PR-AUC")
        axes[0].set_title("Mode A (FedAvg + norm clip + reputation)")
        axes[0].axvline(0.5, color="#f87171", linestyle="--",
                        alpha=0.5, label="0.5 baseline")

    if mode_b:
        b_labels = [
            f"{a.get('robust_method', '?')}\nm={a.get('n_malicious', '?')}" for a in mode_b]
        b_prs = [a.get("mean_pr_auc", 0) or 0 for a in mode_b]
        axes[1].barh(b_labels, b_prs, color="#34d399")
        axes[1].set_xlim(0, 1)
        axes[1].set_xlabel("PR-AUC")
        axes[1].set_title("Mode B (Robust aggregation, no secure agg)")
        axes[1].axvline(0.5, color="#f87171", linestyle="--", alpha=0.5)

    plt.suptitle("Attack & Defence Results – LR model, N=5, S1", y=1.02)
    plt.tight_layout()
    out = output_dir / "fig3_attack_results.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[figures] Written {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate report figures")
    parser.add_argument("--results", default="results")
    parser.add_argument("--output", default="results/figures")
    args = parser.parse_args()

    results_dir = Path(args.results)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")

    fig_privacy_utility(results_dir, output_dir)
    fig_learning_curves(results_dir, output_dir)
    fig_attack_results(results_dir, output_dir)

    print(f"\n[figures] All figures written to {output_dir}/")


if __name__ == "__main__":
    main()
