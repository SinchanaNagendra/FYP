"""
attacks/defences/robust_agg.py
================================
Robust aggregation methods for Mode B (no secure aggregation).

Defences implemented:
  1. Coordinate-wise median
  2. Trimmed mean (trim top and bottom α fraction)
  3. Krum (select the update closest to its k nearest neighbours)

MODE B: These defences require seeing individual (unmasked) updates.
They CANNOT be combined with secure aggregation (Mode A).
This is an inherent security/privacy trade-off:
  - Mode A (secure agg + DP + norm clip): privacy but no robust aggregation
  - Mode B (robust aggregation): Byzantine resilience but no update privacy

This trade-off is measured and reported in Phase 7.
See docs/ARCHITECTURE.md section 4 (Security Model) for full discussion.
"""

import logging
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


def coordinate_median(
    params_list: List[List[np.ndarray]],
) -> List[np.ndarray]:
    """
    Coordinate-wise median aggregation.

    For each parameter coordinate, take the median across all banks.
    Resistant to Byzantine attacks as long as < n/2 banks are malicious.

    Parameters
    ----------
    params_list : list of (list of np.ndarray)
        Each element is one bank's parameter list.

    Returns
    -------
    aggregated : list of np.ndarray
    """
    n_layers = len(params_list[0])
    aggregated = []

    for layer_idx in range(n_layers):
        layer_stack = np.stack(
            [p[layer_idx].astype(np.float64) for p in params_list], axis=0
        )  # shape: (n_banks, *param_shape)
        median = np.median(layer_stack, axis=0).astype(np.float32)
        aggregated.append(median)

    logger.info(f"[RobustAgg] Coordinate median over {len(params_list)} banks")
    return aggregated


def trimmed_mean(
    params_list: List[List[np.ndarray]],
    trim_fraction: float = 0.1,
) -> List[np.ndarray]:
    """
    Trimmed mean aggregation.

    For each coordinate, remove the top and bottom trim_fraction fraction
    of values, then average the remainder.

    Parameters
    ----------
    params_list : list of parameter lists
    trim_fraction : float
        Fraction to trim from each end. Total trimmed = 2 * trim_fraction.
        Must be < 0.5.

    Returns
    -------
    aggregated : list of np.ndarray
    """
    assert 0 < trim_fraction < 0.5
    n_banks = len(params_list)
    n_trim = int(trim_fraction * n_banks)

    if n_trim == 0:
        logger.warning("[RobustAgg] trim_fraction too small; no trimming applied")
        # Fall back to mean
        return _fedavg_mean(params_list)

    n_layers = len(params_list[0])
    aggregated = []

    for layer_idx in range(n_layers):
        layer_stack = np.stack(
            [p[layer_idx].astype(np.float64) for p in params_list], axis=0
        )
        sorted_stack = np.sort(layer_stack, axis=0)
        trimmed = sorted_stack[n_trim: n_banks - n_trim]
        mean = trimmed.mean(axis=0).astype(np.float32)
        aggregated.append(mean)

    logger.info(
        f"[RobustAgg] Trimmed mean: n_banks={n_banks}, trim={n_trim} each side"
    )
    return aggregated


def krum(
    params_list: List[List[np.ndarray]],
    n_malicious: int = 1,
    multi_krum: bool = False,
) -> List[np.ndarray]:
    """
    Krum aggregation rule.

    Selects the update whose sum of squared distances to its
    (n_banks - n_malicious - 2) nearest neighbours is minimised.
    Multi-Krum selects the top-k updates and averages them.

    Parameters
    ----------
    params_list : list of parameter lists
    n_malicious : int
        Assumed maximum number of malicious banks.
    multi_krum : bool
        If True, select and average multiple "good" updates.

    Returns
    -------
    aggregated : list of np.ndarray
    """
    n_banks = len(params_list)
    assert n_banks > 2 * n_malicious + 2, (
        f"Krum requires n_banks > 2*n_malicious+2; got {n_banks} banks, {n_malicious} malicious"
    )

    # Flatten all parameters into vectors
    flat = np.array([
        np.concatenate([p.ravel().astype(np.float64) for p in params])
        for params in params_list
    ])  # shape: (n_banks, total_params)

    # Pairwise squared distances
    dists = np.sum((flat[:, None, :] - flat[None, :, :]) ** 2, axis=2)  # (n_banks, n_banks)

    n_closest = n_banks - n_malicious - 2
    scores = np.zeros(n_banks)
    for i in range(n_banks):
        row = np.sort(dists[i])
        scores[i] = row[1: n_closest + 1].sum()  # skip distance to self (=0)

    if multi_krum:
        # Select the n_banks - n_malicious best
        n_select = n_banks - n_malicious
        selected_indices = np.argsort(scores)[:n_select]
        selected = [params_list[i] for i in selected_indices]
        result = _fedavg_mean(selected)
        logger.info(f"[RobustAgg] Multi-Krum: selected {n_select}/{n_banks} banks")
    else:
        best_idx = int(np.argmin(scores))
        result = params_list[best_idx]
        logger.info(f"[RobustAgg] Krum: selected bank {best_idx}/{n_banks}")

    return result


def norm_clipping(
    params_list: List[List[np.ndarray]],
    clip_norm: float = 10.0,
) -> List[List[np.ndarray]]:
    """
    Clip each bank's update to have L2 norm ≤ clip_norm.
    Applied before aggregation in both Mode A and Mode B.

    Parameters
    ----------
    params_list : list of parameter lists
    clip_norm : float

    Returns
    -------
    clipped_params_list
    """
    clipped = []
    for params in params_list:
        flat = np.concatenate([p.ravel().astype(np.float64) for p in params])
        norm = np.linalg.norm(flat)
        if norm > clip_norm:
            scale = clip_norm / norm
            clipped.append([p * scale for p in params])
            logger.debug(f"[RobustAgg] Clipped norm {norm:.2f} → {clip_norm}")
        else:
            clipped.append(params)
    return clipped


def _fedavg_mean(params_list: List[List[np.ndarray]]) -> List[np.ndarray]:
    n = len(params_list)
    n_layers = len(params_list[0])
    return [
        np.mean(
            np.stack([p[i].astype(np.float64) for p in params_list], axis=0),
            axis=0
        ).astype(np.float32)
        for i in range(n_layers)
    ]
