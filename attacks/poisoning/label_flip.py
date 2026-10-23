"""
attacks/poisoning/label_flip.py
================================
Label-flipping poisoning attack.

A malicious bank flips a fraction of its fraud labels (1→0) in its training data.
This degrades the global model's recall on the poisoned class.

Attack modes:
  - random_flip: randomly select fraud samples and flip to non-fraud
  - targeted_flip: flip all fraud samples (strongest attack)

Used in Phase 7 experiments with m ∈ {1, 2} malicious banks out of N.
"""

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


def label_flip_attack(
    X: np.ndarray,
    y: np.ndarray,
    flip_rate: float = 1.0,
    target_label: int = 1,
    flip_to: int = 0,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Flip a fraction of target_label samples to flip_to.

    Parameters
    ----------
    X : np.ndarray
    y : np.ndarray
    flip_rate : float
        Fraction of target_label samples to flip. 1.0 = flip all.
    target_label : int
        The label to flip FROM (default: 1 = fraud).
    flip_to : int
        The label to flip TO (default: 0 = non-fraud).
    random_state : int

    Returns
    -------
    X_poisoned, y_poisoned : np.ndarray
    """
    rng = np.random.default_rng(random_state)
    y_poisoned = y.copy()

    target_indices = np.where(y == target_label)[0]
    n_flip = int(len(target_indices) * flip_rate)

    if n_flip == 0:
        logger.warning("[LabelFlip] No samples to flip; attack has no effect.")
        return X, y_poisoned

    flip_indices = rng.choice(target_indices, size=n_flip, replace=False)
    y_poisoned[flip_indices] = flip_to

    logger.info(
        f"[LabelFlip] Flipped {n_flip}/{len(target_indices)} "
        f"label={target_label} → {flip_to} samples"
    )
    return X, y_poisoned
