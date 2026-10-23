"""
attacks/poisoning/free_rider.py
=================================
Free-rider attack: a bank submits random or stale updates
without performing real local training.
"""

import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class FreeRiderBank:
    """
    Simulates a free-rider bank that submits low-quality updates.

    Parameters
    ----------
    strategy : str
        'random' | 'stale' | 'zeros'
    random_state : int
    """

    def __init__(self, strategy: str = "random", random_state: int = 42):
        assert strategy in ("random", "stale", "zeros")
        self.strategy = strategy
        self.rng = np.random.default_rng(random_state)
        self._last_params: Optional[List[np.ndarray]] = None

    def get_update(
        self,
        global_params: List[np.ndarray],
        n_samples: int = 1000,
    ) -> tuple:
        """
        Return a (fake_params, n_samples) tuple as if training was done.

        Parameters
        ----------
        global_params : list of np.ndarray
        n_samples : int (claimed, not real)

        Returns
        -------
        (fake_params, n_samples)
        """
        if self.strategy == "random":
            fake = [self.rng.standard_normal(p.shape).astype(p.dtype) * 0.01
                    for p in global_params]
        elif self.strategy == "stale":
            if self._last_params is None:
                fake = [p.copy() for p in global_params]
            else:
                fake = self._last_params
        elif self.strategy == "zeros":
            fake = [np.zeros_like(p) for p in global_params]

        self._last_params = [p.copy() for p in global_params]
        logger.info(f"[FreeRider] Submitting {self.strategy} update (n_samples={n_samples})")
        return fake, n_samples
