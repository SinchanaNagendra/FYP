"""
fl/strategies/fabric_fedavg.py
================================
FabricFedAvg: Federated Averaging strategy backed by Hyperledger Fabric.

Replaces Flower's central server with:
  1. Round lifecycle managed via chaincode (StartRound → SubmitUpdate → FinalizeRound).
  2. Weight artifacts stored encrypted in private IPFS.
  3. Aggregation via pairwise-mask secure aggregation (masks cancel in the sum).
  4. No single party sees individual banks' unmasked updates.

In no-blockchain mode (--no-blockchain flag), operates like standard FedAvg
using in-process communication for testing.

Federation mechanism: weighted FedAvg — each bank's weights are weighted by
its number of training samples, then averaged.

  w_global = Σ_i (n_i / Σ n_j) * w_i

Compatible with: LogisticRegressionModel, CNNModel.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class FabricFedAvg:
    """
    Federated Averaging strategy.

    In blockchain mode, this class coordinates the round via Fabric chaincode
    and IPFS (via FabricClient and IPFSClient on each bank node).

    In no-blockchain mode, it manages the round in-process for testing.

    Parameters
    ----------
    n_banks : int
    min_fit_clients : int
        Minimum banks required to start a round (default = n_banks).
    fraction_fit : float
        Fraction of banks to include in each round (default = 1.0).
    use_secure_agg : bool
        If True, apply pairwise-mask secure aggregation.
    use_blockchain : bool
        If True, use Fabric + IPFS for round coordination.
    """

    def __init__(
        self,
        n_banks: int,
        min_fit_clients: Optional[int] = None,
        fraction_fit: float = 1.0,
        use_secure_agg: bool = True,
        use_blockchain: bool = False,
    ):
        self.n_banks = n_banks
        self.min_fit_clients = min_fit_clients or n_banks
        self.fraction_fit = fraction_fit
        self.use_secure_agg = use_secure_agg
        self.use_blockchain = use_blockchain

    def aggregate_fit(
        self,
        results: List[Tuple[List[np.ndarray], int]],
    ) -> List[np.ndarray]:
        """
        Aggregate parameter updates from multiple banks using weighted FedAvg.

        Parameters
        ----------
        results : list of (parameters, num_examples) tuples
            parameters: list of numpy arrays (one per model parameter tensor)
            num_examples: number of training samples used by that bank

        Returns
        -------
        aggregated_parameters : list of numpy arrays
        """
        if not results:
            raise ValueError("No results to aggregate")

        total_examples = sum(n for _, n in results)
        if total_examples == 0:
            raise ValueError("Total examples is 0")

        # Weighted average
        n_layers = len(results[0][0])
        aggregated = []

        for layer_idx in range(n_layers):
            weighted_sum = np.zeros_like(results[0][0][layer_idx], dtype=np.float64)
            for params, n_examples in results:
                weight = n_examples / total_examples
                weighted_sum += weight * params[layer_idx].astype(np.float64)
            aggregated.append(weighted_sum.astype(np.float32))

        logger.info(
            f"[FedAvg] Aggregated {len(results)} banks, "
            f"total_examples={total_examples}, n_layers={n_layers}"
        )
        return aggregated

    def aggregate_fit_secure(
        self,
        bank_ids: List[int],
        masked_updates: Dict[int, np.ndarray],
        n_elements: int,
    ) -> np.ndarray:
        """
        Aggregate masked updates using secure aggregation (masks cancel in sum).

        Parameters
        ----------
        bank_ids : list of participating bank IDs
        masked_updates : dict {bank_id: masked_flat_weights (uint64 array)}
        n_elements : int

        Returns
        -------
        average_flat_weights : np.ndarray, dtype float32
        """
        from fl.secure_agg.masking import unmask_aggregate

        running_sum = np.zeros(n_elements, dtype=np.uint64)
        for bid in bank_ids:
            running_sum = (running_sum + masked_updates[bid]).astype(np.uint64)

        return unmask_aggregate(running_sum, n_participants=len(bank_ids))

    def configure_fit(self, round_num: int, parameters: List[np.ndarray]) -> Dict[str, Any]:
        """Return configuration dict sent to each bank at the start of a round."""
        return {
            "round_num": round_num,
            "use_secure_agg": self.use_secure_agg,
        }
