"""
attacks/poisoning/model_poison.py
===================================
Model poisoning attacks on the federated update.

Attacks implemented:
  1. Scaled update: multiply the update by a large scalar.
  2. Sign-flipped update: negate all model parameters.
  3. Backdoor-style update: inject a targeted perturbation pattern.

These are applied AFTER local training, replacing the legitimate update
with a poisoned one before submission to the aggregator.
"""

import logging
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


def scaled_update_attack(
    params: List[np.ndarray],
    scale: float = 10.0,
) -> List[np.ndarray]:
    """
    Scale the model update by a large factor.

    A high scale factor causes the malicious bank's contribution to
    dominate the FedAvg aggregate.
    """
    poisoned = [p * scale for p in params]
    logger.info(f"[ModelPoison] Scaled update by {scale}x")
    return poisoned


def sign_flip_attack(params: List[np.ndarray]) -> List[np.ndarray]:
    """
    Negate all model parameters.

    Pushes the global model away from the legitimate optimum.
    """
    poisoned = [-p for p in params]
    logger.info("[ModelPoison] Sign-flipped all parameters")
    return poisoned


def backdoor_attack(
    params: List[np.ndarray],
    trigger_layer_idx: int = 0,
    trigger_scale: float = 5.0,
    pattern_seed: int = 42,
) -> List[np.ndarray]:
    """
    Backdoor-style update: add a fixed pattern to a specific layer.

    In a real backdoor attack, this pattern encodes a trigger that causes
    the model to misclassify inputs containing the trigger.
    For the evaluation, this injects a structured perturbation.

    Parameters
    ----------
    params : list of np.ndarray
    trigger_layer_idx : int
        Which parameter tensor to perturb.
    trigger_scale : float
    pattern_seed : int
    """
    rng = np.random.default_rng(pattern_seed)
    poisoned = [p.copy() for p in params]

    if trigger_layer_idx < len(poisoned):
        p = poisoned[trigger_layer_idx]
        pattern = rng.standard_normal(p.shape).astype(p.dtype)
        pattern = pattern / (np.linalg.norm(pattern) + 1e-8) * trigger_scale
        poisoned[trigger_layer_idx] = p + pattern
        logger.info(
            f"[ModelPoison] Backdoor pattern injected into layer {trigger_layer_idx}, "
            f"scale={trigger_scale}"
        )
    return poisoned


def random_update_attack(
    params: List[np.ndarray],
    random_state: int = 42,
) -> List[np.ndarray]:
    """
    Free-rider / random update: replace parameters with Gaussian noise.
    Simulates a bank that does no real training.
    """
    rng = np.random.default_rng(random_state)
    poisoned = [rng.standard_normal(p.shape).astype(p.dtype) * 0.01 for p in params]
    logger.info("[ModelPoison] Random update (free-rider attack)")
    return poisoned


def stale_update_attack(
    params: List[np.ndarray],
    stale_params: List[np.ndarray],
) -> List[np.ndarray]:
    """
    Free-rider: submit an old (stale) update from a previous round.
    """
    logger.info("[ModelPoison] Stale update attack (free-rider)")
    return [p.copy() for p in stale_params]
