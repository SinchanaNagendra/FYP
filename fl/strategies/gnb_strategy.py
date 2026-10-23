"""
fl/strategies/gnb_strategy.py
===============================
Aggregation strategy for Gaussian Naive Bayes.

GNB uses EXACT aggregation of sufficient statistics, not FedAvg.
The per-class (count, sum, sum_sq) from all banks are summed to derive
the global GNB model parameters.

No floating-point averaging approximation is used.
"""

import logging
from typing import List

import numpy as np

from fl.models.gaussian_nb import aggregate_gnb_params

logger = logging.getLogger(__name__)


class GNBStrategy:
    """
    Aggregates GNB sufficient statistics from multiple banks.

    Aggregation is exact: the aggregate statistics equal what would be
    obtained by training on all data pooled together.
    """

    def __init__(self, n_banks: int):
        self.n_banks = n_banks

    def aggregate_fit(
        self,
        results: List[tuple],  # list of (gnb_params: list[np.ndarray], n_examples: int)
    ) -> List[np.ndarray]:
        """
        Aggregate GNB sufficient statistics from all banks.

        Parameters
        ----------
        results : list of ([counts, sum, sum_sq], n_examples)

        Returns
        -------
        aggregated_params : [agg_counts, agg_sum, agg_sum_sq]
        """
        if not results:
            raise ValueError("No results to aggregate")

        bank_params = [params for params, _ in results]
        agg = aggregate_gnb_params(bank_params)

        total = sum(n for _, n in results)
        logger.info(f"[GNBStrategy] Aggregated {len(results)} banks, total_examples={total}")
        return agg
