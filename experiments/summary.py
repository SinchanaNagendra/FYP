"""Quick summary of all results generated so far."""
import json
import glob
import os

results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
results_dir = os.path.normpath(results_dir)

print("=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)

# ── Centralized baselines ──────────────────────────────────────
bl_path = os.path.join(results_dir, 'baselines',
                       'centralized', 'S1_seed42.json')
if os.path.exists(bl_path):
    d = json.load(open(bl_path))
    rows = [r for r in d['results']['rows'] if 'error' not in r]
    print(f"\nCENTRALIZED BASELINES ({len(rows)} runs)")
    print(f"  {'Model':<22} {'Balancing':<10} {'PR-AUC':<8} {'ROC-AUC'}")
    for r in rows:
        pr = r.get('pr_auc')
        roc = r.get('roc_auc')
        pr_s = f"{pr:.4f}" if pr else "N/A"
        roc_s = f"{roc:.4f}" if roc else "N/A"
        print(f"  {r['model']:<22} {r['balancing']:<10} {pr_s:<8} {roc_s}")
else:
    print("\nCENTRALIZED BASELINES: not found")

# ── FL results ─────────────────────────────────────────────────
fl_files = glob.glob(os.path.join(results_dir, 'fl', '*.json'))
if fl_files:
    print(f"\nFEDERATED RESULTS ({len(fl_files)} runs)")
    print(f"  {'Run':<45} {'Rounds':<8} {'Test PR-AUC (mean)'}")
    for f in sorted(fl_files):
        d = json.load(open(f))
        name = os.path.basename(f).replace('.json', '')
        rounds = d['results'].get('rounds', [])
        test = d['results'].get('final_test', [])
        pr_aucs = [r.get('pr_auc') for r in test if r.get('pr_auc')]
        if pr_aucs:
            mean = sum(pr_aucs) / len(pr_aucs)
            std = (sum((x - mean)**2 for x in pr_aucs) / len(pr_aucs)) ** 0.5
            print(f"  {name:<45} {len(rounds):<8} {mean:.4f} ± {std:.4f}")
        else:
            print(f"  {name:<45} {len(rounds):<8} (no test metrics)")
else:
    print("\nFEDERATED RESULTS: none yet")

# ── Comparison table ───────────────────────────────────────────
print("\n" + "=" * 60)
print("CENTRALIZED vs FEDERATED (PR-AUC)")
print("=" * 60)

central = {}
if os.path.exists(bl_path):
    d = json.load(open(bl_path))
    for r in d['results']['rows']:
        if 'error' not in r and r.get('balancing') == 'none':
            central[r['model']] = r.get('pr_auc')

fed = {}
for f in sorted(fl_files):
    d = json.load(open(f))
    name = os.path.basename(f).replace('.json', '')
    # e.g. random_forest_none_S1_seed42
    parts = name.split('_')
    # find model name (up to balancing method)
    test = d['results'].get('final_test', [])
    pr_aucs = [r.get('pr_auc') for r in test if r.get('pr_auc')]
    if not pr_aucs:
        continue
    mean = sum(pr_aucs) / len(pr_aucs)
    # extract model from config
    cfg = d.get('config', {})
    model = cfg.get('model', name)
    scenario = cfg.get('scenario', 'S1')
    if scenario == 'S1':
        fed[model] = mean

if central or fed:
    all_models = sorted(set(list(central.keys()) + list(fed.keys())))
    print(f"  {'Model':<25} {'Centralized':<14} {'Federated':<14} {'Gap'}")
    print("  " + "-" * 60)
    for m in all_models:
        c = central.get(m)
        f = fed.get(m)
        gap = (f - c) if (c and f) else None
        c_s = f"{c:.4f}" if c else "N/A"
        f_s = f"{f:.4f}" if f else "N/A"
        g_s = f"{gap:+.4f}" if gap is not None else "N/A"
        print(f"  {m:<25} {c_s:<14} {f_s:<14} {g_s}")
    print("\n  Positive gap = federated BETTER than centralized")
    print("  (RF federated bagging grows the ensemble across banks)")

print("\n" + "=" * 60)
print("NEXT STEPS")
print("=" * 60)

remaining = []
if 'logistic_regression' not in fed:
    remaining.append(
        "Run federated LR: python experiments/run_fl.py --model logistic_regression --no-blockchain")
if 'gaussian_nb' not in fed:
    remaining.append(
        "Run federated GNB (fast): python experiments/run_fl.py --model gaussian_nb --no-blockchain")
if not glob.glob(os.path.join(results_dir, 'attacks', '*.json')):
    remaining.append(
        "Run attacks: python experiments/run_attacks.py --n-banks 5 --seed 42")
if not glob.glob(os.path.join(results_dir, 'dp_sweep', '*.json')):
    remaining.append(
        "Run DP sweep: python experiments/run_dp_sweep.py --n-banks 5 --seed 42")

if remaining:
    for i, r in enumerate(remaining, 1):
        print(f"  {i}. {r}")
else:
    print("  All major experiments complete!")
