"""
fl/secure_agg/aggregator.py
============================
Coordinates pairwise-mask secure aggregation across N banks.

In the real system, this logic runs DISTRIBUTED — each bank holds only its own
masked update; no single party sees all unmasked weights. The "aggregation"
is just a sum of masked values that any party can compute.

This class represents the coordinator role (could run on any bank node, or
on the chaincode-triggered aggregation step).

The key security property:
  - The aggregate sum equals the sum of plain weights (masks cancel exactly)
  - No individual masked update reveals anything about the underlying weights
    (assuming the PRG is secure and the shared seed is known only to the pair)

See tests/unit/test_secure_agg.py for verification.
"""

import logging
from typing import Optional

import numpy as np

from .masking import mask_weights, unmask_aggregate, generate_mask

logger = logging.getLogger(__name__)


class SecureAggregator:
    """
    Manages the secure aggregation of model updates from N banks.

    Usage (coordinator perspective):
        agg = SecureAggregator(n_banks=5)
        for bank_id, masked_update in bank_submissions:
            agg.add_masked_update(bank_id, masked_update)
        average_weights = agg.aggregate()

    Usage (bank perspective):
        masked = SecureAggregator.apply_mask(
            bank_id=my_id,
            weights=my_weights,
            pairwise_seeds=my_seeds,
            n_elements=len(my_weights),
        )
        # transmit masked to coordinator
    """

    def __init__(self, n_banks: int, n_elements: int):
        self.n_banks = n_banks
        self.n_elements = n_elements
        self._masked_updates: dict[int, np.ndarray] = {}

    def add_masked_update(self, bank_id: int, masked_update: np.ndarray) -> None:
        """
        Accept one bank's masked update.

        Parameters
        ----------
        bank_id : int
        masked_update : np.ndarray, dtype uint64, shape (n_elements,)
        """
        assert masked_update.shape == (self.n_elements,), (
            f"Expected shape ({self.n_elements},), got {masked_update.shape}"
        )
        assert masked_update.dtype == np.uint64
        if bank_id in self._masked_updates:
            raise ValueError(f"Duplicate submission from bank {bank_id}")
        self._masked_updates[bank_id] = masked_update
        logger.info(f"[SecureAgg] Received masked update from bank {bank_id}")

    def is_quorum_reached(self, quorum: Optional[int] = None) -> bool:
        quorum = quorum or self.n_banks
        return len(self._masked_updates) >= quorum

    def aggregate(self, participating_banks: Optional[list[int]] = None) -> np.ndarray:
        """
        Compute the aggregate (average) of all submitted masked updates.

        The masks cancel in the sum, yielding the sum of plain encoded weights.
        Decoding divides by (scale × n_participants) to get the average.

        Parameters
        ----------
        participating_banks : list of int, optional
            If provided, use only these banks' updates (for dropout handling).

        Returns
        -------
        average_weights : np.ndarray, dtype float32
        """
        if participating_banks is None:
            participating_banks = list(self._masked_updates.keys())

        if not participating_banks:
            raise RuntimeError("No updates to aggregate.")

        n = len(participating_banks)
        running_sum = np.zeros(self.n_elements, dtype=np.uint64)

        for bank_id in participating_banks:
            if bank_id not in self._masked_updates:
                raise KeyError(f"No masked update for bank {bank_id}")
            running_sum = (running_sum + self._masked_updates[bank_id]).astype(np.uint64)

        result = unmask_aggregate(running_sum, n_participants=n)
        logger.info(f"[SecureAgg] Aggregated {n} updates. L2 norm: {np.linalg.norm(result):.4f}")
        return result

    def get_submitted_bank_ids(self) -> list[int]:
        return list(self._masked_updates.keys())

    def reset(self) -> None:
        self._masked_updates.clear()

    @staticmethod
    def apply_mask(
        bank_id: int,
        weights: np.ndarray,
        pairwise_seeds: dict[tuple[int, int], bytes],
        n_elements: int,
    ) -> np.ndarray:
        """
        Convenience method: generate mask + apply to weights.

        Parameters
        ----------
        bank_id : int
        weights : np.ndarray, dtype float32, shape (n_elements,)
        pairwise_seeds : dict {(i,j): seed_bytes}
        n_elements : int

        Returns
        -------
        masked_update : np.ndarray, dtype uint64
        """
        mask = generate_mask(bank_id, pairwise_seeds, n_elements)
        return mask_weights(weights, mask)

    @staticmethod
    def verify_masks_cancel(
        bank_ids: list[int],
        pairwise_seeds: dict[tuple[int, int], bytes],
        n_elements: int = 100,
    ) -> bool:
        """
        Verification test: assert that the sum of all banks' masks is zero.
        This is the fundamental correctness property of the protocol.

        Returns True if verified, raises AssertionError if not.
        """
        from .masking import generate_mask
        total = np.zeros(n_elements, dtype=np.uint64)
        for bank_id in bank_ids:
            m = generate_mask(bank_id, pairwise_seeds, n_elements)
            total = (total + m).astype(np.uint64)

        all_zero = np.all(total == 0)
        assert all_zero, (
            f"Mask cancellation FAILED: {n_elements} elements, "
            f"non-zero count={np.count_nonzero(total)}"
        )
        return True
