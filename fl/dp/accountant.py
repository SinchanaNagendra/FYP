"""
fl/dp/accountant.py
====================
Privacy accountant for cumulative epsilon tracking across FL rounds.

Opacus uses Rényi Differential Privacy (RDP) internally. This module
provides a wrapper that:
  1. Tracks per-round ε.
  2. Computes cumulative ε using basic composition (sum of per-round ε).
  3. Produces a summary dict suitable for on-chain reporting.

Note: Basic composition (sum of ε) is a valid but conservative bound.
Advanced composition (Kairouz et al.) would give tighter bounds; this is
a future-work item noted in the report.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RoundPrivacyRecord:
    """Record of privacy cost for one FL round."""
    round_id: str
    epsilon: float
    delta: float
    noise_multiplier: float
    max_grad_norm: float
    n_steps: int
    n_samples: int


class PrivacyAccountant:
    """
    Tracks cumulative differential privacy budget across FL rounds.

    Parameters
    ----------
    delta : float
        Target δ. Should be < 1/n_training_samples.
    """

    def __init__(self, delta: float = 1e-5):
        self.delta = delta
        self._records: List[RoundPrivacyRecord] = []

    def record_round(
        self,
        round_id: str,
        epsilon: float,
        noise_multiplier: float,
        max_grad_norm: float,
        n_steps: int,
        n_samples: int,
    ) -> None:
        """Record the privacy cost of one FL round."""
        record = RoundPrivacyRecord(
            round_id=round_id,
            epsilon=epsilon,
            delta=self.delta,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            n_steps=n_steps,
            n_samples=n_samples,
        )
        self._records.append(record)
        logger.info(
            f"[PrivacyAccountant] Round {round_id}: ε={epsilon:.4f}, "
            f"cumulative_ε={self.cumulative_epsilon:.4f}"
        )

    @property
    def cumulative_epsilon(self) -> float:
        """
        Total ε spent across all recorded rounds (basic composition).
        Returns 0.0 if no rounds recorded.
        """
        return sum(r.epsilon for r in self._records)

    @property
    def n_rounds(self) -> int:
        return len(self._records)

    def get_summary(self) -> dict:
        """Return a summary dict for on-chain reporting or the dashboard."""
        return {
            "n_rounds": self.n_rounds,
            "delta": self.delta,
            "cumulative_epsilon": self.cumulative_epsilon,
            "per_round": [
                {
                    "round_id": r.round_id,
                    "epsilon": r.epsilon,
                    "noise_multiplier": r.noise_multiplier,
                    "max_grad_norm": r.max_grad_norm,
                }
                for r in self._records
            ],
        }

    def reset(self) -> None:
        """Reset the accountant (e.g., start of a new experiment)."""
        self._records.clear()

    @staticmethod
    def compute_epsilon_opacus(
        n_samples: int,
        batch_size: int,
        noise_multiplier: float,
        n_steps: int,
        delta: float,
    ) -> float:
        """
        Compute ε for given parameters using Opacus's RDP accountant.

        Parameters
        ----------
        n_samples : int
        batch_size : int
        noise_multiplier : float
        n_steps : int
            Total number of gradient steps (epochs × batches_per_epoch).
        delta : float

        Returns
        -------
        epsilon : float
        """
        try:
            from opacus.accountants import RDPAccountant
            from opacus.accountants.utils import get_noise_multiplier
        except ImportError:
            logger.warning("Opacus not available; returning ε=inf")
            return float("inf")

        accountant = RDPAccountant()
        sample_rate = batch_size / n_samples

        for _ in range(n_steps):
            accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)

        eps, _ = accountant.get_privacy_spent(delta=delta)
        return float(eps)
