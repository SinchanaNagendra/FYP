"""
fl/strategies/tree_strategy.py
================================
Aggregation strategy for Decision Tree and Random Forest.

Decision Tree:  global model = soft-vote ensemble of all banks' local trees.
Random Forest:  global model = union of all banks' local forest trees (federated bagging).

Trees are serialised with leaf PROBABILITIES (not counts) before transmission.
"""

import logging
from typing import List

import numpy as np

from fl.models.random_forest import aggregate_forests

logger = logging.getLogger(__name__)


class TreeStrategy:
    """
    Aggregates tree models from multiple banks.

    For Random Forest: unions all individual trees into one global forest.
    For Decision Tree: collects all trees into a soft-vote ensemble.
    """

    def __init__(self, n_banks: int, model_type: str = "random_forest"):
        self.n_banks = n_banks
        assert model_type in ("random_forest", "decision_tree")
        self.model_type = model_type

    def aggregate_fit(
        self,
        results: List[tuple],  # list of (params: list[np.ndarray], n_examples: int)
    ) -> List[np.ndarray]:
        """
        Aggregate tree parameters from all banks.

        Parameters
        ----------
        results : list of ([serialised_trees_blob], n_examples)

        Returns
        -------
        aggregated_params : [serialised_all_trees_blob]
        """
        if not results:
            raise ValueError("No results to aggregate")

        bank_params = [params for params, _ in results]
        agg = aggregate_forests(bank_params)  # works for both DT and RF

        total = sum(n for _, n in results)
        logger.info(f"[TreeStrategy] Aggregated {len(results)} banks, total_examples={total}")
        return agg
