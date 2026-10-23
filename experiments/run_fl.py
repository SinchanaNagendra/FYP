"""
experiments/run_fl.py
======================
Phase 3 / Phase 5: Federated learning runner.

Modes:
  --no-blockchain  (Phase 3): runs FL rounds entirely in-process; no Fabric/IPFS.
  --with-blockchain (Phase 5): drives rounds through Fabric chaincode + IPFS.

Usage:
    python experiments/run_fl.py --n-banks 5 --scenario S1 --seed 42 --no-blockchain
    python experiments/run_fl.py --n-banks 5 --scenario S1 --seed 42 --with-blockchain --rounds 20
"""

# --- path setup MUST come before any fl.* imports ---
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fl.secure_agg.aggregator import SecureAggregator          # noqa: E402
from fl.secure_agg.keys import KeyPair, generate_pairwise_seeds, derive_shared_seed  # noqa: E402
from fl.preprocessing import FraudPreprocessor                  # noqa: E402
from fl.bank_node.evaluator import evaluate, metrics_to_summary_str  # noqa: E402
from fl.balancing.methods import BalancingMethod                # noqa: E402
from experiments.utils import set_global_seed, write_results, mean_std  # noqa: E402


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def build_bank_nodes(
    n_banks: int,
    scenario: str,
    partitions_dir: Path,
    model_type: str,
    balancing: BalancingMethod,
    n_epochs: int,
    use_dp: bool,
    dp_noise_multiplier: float,
    use_secure_agg: bool,
    use_blockchain: bool,
    device: str,
    seed: int,
):
    """Build and return a list of BankNode instances."""
    from fl.bank_node.node import BankNode

    nodes = []
    for bank_id in range(n_banks):
        data_path = partitions_dir / f"bank{bank_id}"
        node = BankNode(
            bank_id=bank_id,
            n_banks=n_banks,
            data_path=data_path,
            model_type=model_type,
            balancing=balancing,
            n_epochs=n_epochs,
            use_dp=use_dp,
            dp_noise_multiplier=dp_noise_multiplier,
            use_secure_agg=use_secure_agg,
            use_blockchain=use_blockchain,
            device=device,
            random_state=seed + bank_id,
        )
        node.load_data()
        nodes.append(node)
        logger.info(f"[FL] Bank {bank_id} node initialised")

    return nodes


def get_strategy(model_type: str, n_banks: int, use_secure_agg: bool, mu: float = 0.0):
    from fl.strategies.fabric_fedavg import FabricFedAvg
    from fl.strategies.fabric_fedprox import FabricFedProx
    from fl.strategies.gnb_strategy import GNBStrategy
    from fl.strategies.tree_strategy import TreeStrategy

    if model_type == "gaussian_nb":
        return GNBStrategy(n_banks)
    elif model_type in ("decision_tree", "random_forest"):
        return TreeStrategy(n_banks, model_type)
    elif mu > 0.0:
        return FabricFedProx(n_banks, mu=mu, use_secure_agg=use_secure_agg)
    else:
        return FabricFedAvg(n_banks, use_secure_agg=use_secure_agg)


def _is_parametric(model_type: str) -> bool:
    """Parametric models (LR, CNN) have weight tensors from __init__.
    Non-parametric models (RF, DT, GNB) must be fitted before get_parameters()."""
    return model_type in ("logistic_regression", "cnn")


def run_fl_rounds(
    nodes: list,
    n_rounds: int,
    model_type: str,
    strategy,
    use_secure_agg: bool,
    use_blockchain: bool,
    encryption_key: bytes,
    seed: int,
) -> List[Dict[str, Any]]:
    """
    Run N federated learning rounds. Returns list of per-round result dicts.
    """
    n_banks = len(nodes)
    round_results = []

    # For parametric models (LR/CNN) we can extract random-init weights immediately.
    # For tree/GNB models, global_params starts as None — each bank trains from
    # scratch in round 1, then the aggregate becomes global_params for round 2+.
    global_params = nodes[0].model.get_parameters(
    ) if _is_parametric(model_type) else None

    for round_num in range(1, n_rounds + 1):
        round_id = f"round_{round_num:04d}"
        t_round = time.time()
        logger.info(
            f"\n{'='*50}\n[FL] Starting round {round_num}/{n_rounds}\n{'='*50}")

        # Key exchange for secure aggregation
        if use_secure_agg:
            keypairs = {i: KeyPair.generate() for i in range(n_banks)}
            all_seeds: Dict = {}
            for i in range(n_banks):
                for j in range(i + 1, n_banks):
                    seed_ij = derive_shared_seed(
                        keypairs[i], keypairs[j].public_bytes, i, j, round_id
                    )
                    all_seeds[(i, j)] = seed_ij

            # Each bank gets the public keys of all peers
            peer_public_keys = {
                i: keypairs[i].public_bytes for i in range(n_banks)}
        else:
            all_seeds = {}
            peer_public_keys = {}
            keypairs = {}

        # Each bank trains locally and produces a (masked) update
        bank_results = []
        results_for_agg = []  # (params, n_samples) for strategy.aggregate_fit

        # Assign the global params to each bank node's model.
        # Skip for tree/GNB in round 1 (global_params is None — each bank
        # trains from scratch and the aggregate becomes global_params after).
        if global_params is not None:
            for node in nodes:
                node.model.set_parameters(global_params)

        if use_secure_agg and _is_parametric(model_type):
            # Secure aggregation path for parametric models (LR, CNN).
            # In no-blockchain mode: train locally then mask in-process (no IPFS).
            # In blockchain mode: run_round handles IPFS upload + chain submission.

            # Step 1: local training on every bank
            for node in nodes:
                train_result = node._train_local_only(global_params)
                bank_results.append({"bank_id": node.bank_id, **train_result})

            # Step 2: compute secure-aggregate in-process
            flat_size = sum(p.size for p in (
                global_params or nodes[0].model.get_parameters()))
            agg = SecureAggregator(n_banks=n_banks, n_elements=flat_size)

            for node in nodes:
                bank_id = node.bank_id
                local_p = node.model.get_parameters()
                flat = np.concatenate([p.ravel() for p in local_p])
                bank_seeds = {k: v for k, v in all_seeds.items()
                              if bank_id in k}
                masked = SecureAggregator.apply_mask(
                    bank_id, flat, bank_seeds, flat_size)
                agg.add_masked_update(bank_id, masked)

            agg_flat = agg.aggregate()

            # Step 3: reshape flat aggregate back to parameter structure
            ref_params = global_params or nodes[0].model.get_parameters()
            offset = 0
            new_global_params = []
            for p in ref_params:
                new_global_params.append(
                    agg_flat[offset: offset + p.size].reshape(p.shape))
                offset += p.size
            global_params = new_global_params

        else:
            # Plain FedAvg (no secure agg — for robust aggregation mode)
            for node in nodes:
                train_result = node._train_local_only(global_params)
                bank_results.append({"bank_id": node.bank_id, **train_result})
                results_for_agg.append(
                    (node.model.get_parameters(), train_result["n_samples"]))

            global_params = strategy.aggregate_fit(results_for_agg)

        # Update all nodes with the new global model.
        # global_params is None only if tree/GNB failed to produce any results.
        if global_params is not None:
            for node in nodes:
                node.model.set_parameters(global_params)

        # Evaluate all banks on their validation set
        round_val_metrics = []
        for node in nodes:
            m = node.evaluate_val()
            round_val_metrics.append({"bank_id": node.bank_id, **m})

        round_time = time.time() - t_round
        round_summary = {
            "round_num": round_num,
            "round_id": round_id,
            "round_time": round_time,
            "bank_results": bank_results,
            "val_metrics": round_val_metrics,
            "mean_pr_auc": mean_std([m.get("pr_auc") for m in round_val_metrics])[0],
        }
        round_results.append(round_summary)

        logger.info(
            f"[FL] Round {round_num} done in {round_time:.1f}s. "
            f"Mean PR-AUC={round_summary['mean_pr_auc']}"
        )

    return round_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run federated learning experiment")
    parser.add_argument("--n-banks", type=int, default=5)
    parser.add_argument("--scenario", default="S1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--model", default="logistic_regression",
                        choices=["logistic_regression", "cnn", "gaussian_nb",
                                 "decision_tree", "random_forest"])
    parser.add_argument("--balancing", default="none",
                        choices=[b.value for b in BalancingMethod])
    parser.add_argument("--n-epochs", type=int, default=5)
    parser.add_argument("--no-blockchain", action="store_true")
    parser.add_argument("--with-blockchain", action="store_true")
    parser.add_argument("--no-secure-agg", action="store_true")
    parser.add_argument("--use-dp", action="store_true")
    parser.add_argument("--dp-noise-multiplier", type=float, default=1.1)
    parser.add_argument("--fedprox-mu", type=float, default=0.0)
    parser.add_argument("--partitions", default="data/partitions")
    parser.add_argument("--output", default="results/fl")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    use_blockchain = args.with_blockchain and not args.no_blockchain
    use_secure_agg = not args.no_secure_agg
    balancing = BalancingMethod(args.balancing)

    set_global_seed(args.seed)

    partitions_dir = Path(args.partitions) / f"N{args.n_banks}" / args.scenario
    if not partitions_dir.exists():
        logger.error(
            f"Partitions not found: {partitions_dir}. Run: make data-partition")
        sys.exit(1)

    logger.info(
        f"[FL] Config: n_banks={args.n_banks}, scenario={args.scenario}, "
        f"model={args.model}, balancing={args.balancing}, rounds={args.rounds}, "
        f"blockchain={use_blockchain}, secure_agg={use_secure_agg}, dp={args.use_dp}"
    )

    nodes = build_bank_nodes(
        n_banks=args.n_banks,
        scenario=args.scenario,
        partitions_dir=partitions_dir,
        model_type=args.model,
        balancing=balancing,
        n_epochs=args.n_epochs,
        use_dp=args.use_dp,
        dp_noise_multiplier=args.dp_noise_multiplier,
        use_secure_agg=use_secure_agg,
        use_blockchain=use_blockchain,
        device=args.device,
        seed=args.seed,
    )

    strategy = get_strategy(args.model, args.n_banks,
                            use_secure_agg, args.fedprox_mu)
    # round-specific key (in production: distributed via key exchange)
    encryption_key = os.urandom(32)

    round_results = run_fl_rounds(
        nodes=nodes,
        n_rounds=args.rounds,
        model_type=args.model,
        strategy=strategy,
        use_secure_agg=use_secure_agg,
        use_blockchain=use_blockchain,
        encryption_key=encryption_key,
        seed=args.seed,
    )

    # Final test evaluation
    test_metrics = []
    for node in nodes:
        m = node.evaluate_test()
        test_metrics.append({"bank_id": node.bank_id, **m})

    config = vars(args)
    config["use_blockchain"] = use_blockchain
    config["use_secure_agg"] = use_secure_agg

    output_tag = f"{args.model}_{args.balancing}_{args.scenario}_seed{args.seed}"
    write_results(
        {
            "rounds": round_results,
            "final_test": test_metrics,
            "n_rounds": args.rounds,
        },
        Path(args.output),
        output_tag,
        config,
    )

    # Print summary
    final_pr_aucs = [m.get("pr_auc")
                     for m in test_metrics if m.get("pr_auc") is not None]
    if final_pr_aucs:
        mu, std = mean_std(final_pr_aucs)
        print(f"\n[FL] Final PR-AUC across banks: {mu:.4f} ± {std:.4f}")

    print(f"\n[FL] Results written to {args.output}/{output_tag}.json")


if __name__ == "__main__":
    main()
