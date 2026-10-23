"""
fl/balancing/methods.py
========================
Resampling and balancing methods applied LOCALLY within a bank's training split.

CRITICAL RULES (from master prompt):
  1. These methods are applied ONLY to each bank's TRAINING split.
  2. NEVER applied to validation or test data.
  3. NEVER used as a substitute for real data (SMOTE etc. augments train only).
  4. Class-weighted training is handled separately via loss weights.

Methods:
  NONE         – no resampling (use raw class distribution)
  WEIGHTED     – class-weighted loss (no data resampling)
  ROS          – RandomOverSampler
  RUS          – RandomUnderSampler
  SMOTE        – Synthetic Minority Oversampling Technique
  ADASYN       – Adaptive Synthetic Sampling
  HYBRID       – SMOTEENN (SMOTE + Edited Nearest Neighbours cleaning)
               NOTE: we use SMOTEENN, not SMOTE+Tomek, because SMOTEENN gives
               more aggressive cleaning. Both are valid; this choice is
               documented in DATASET.md.
"""

from enum import Enum
from typing import Optional, Tuple

import numpy as np


class BalancingMethod(str, Enum):
    NONE = "none"
    WEIGHTED = "weighted"
    ROS = "ros"
    RUS = "rus"
    SMOTE = "smote"
    ADASYN = "adasyn"
    HYBRID = "hybrid"  # SMOTEENN


def get_balancer(method: BalancingMethod, random_state: int = 42):
    """
    Return an imbalanced-learn sampler for the given method,
    or None for NONE/WEIGHTED (which require no resampling).

    Parameters
    ----------
    method : BalancingMethod
    random_state : int

    Returns
    -------
    sampler or None
    """
    if method in (BalancingMethod.NONE, BalancingMethod.WEIGHTED):
        return None

    # Import here to avoid hard dependency if imbalanced-learn not installed
    try:
        from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
        from imblearn.under_sampling import RandomUnderSampler
        from imblearn.combine import SMOTEENN
    except ImportError as e:
        raise ImportError(
            "imbalanced-learn is required for resampling. "
            "Install with: pip install imbalanced-learn==0.12.2"
        ) from e

    if method == BalancingMethod.ROS:
        return RandomOverSampler(random_state=random_state)

    elif method == BalancingMethod.RUS:
        return RandomUnderSampler(random_state=random_state)

    elif method == BalancingMethod.SMOTE:
        # k_neighbors default=5; reduce if bank has very few minority samples
        return SMOTE(random_state=random_state)

    elif method == BalancingMethod.ADASYN:
        return ADASYN(random_state=random_state)

    elif method == BalancingMethod.HYBRID:
        # SMOTEENN: first oversample minority with SMOTE, then clean with ENN
        return SMOTEENN(random_state=random_state)

    else:
        raise ValueError(f"Unknown balancing method: {method}")


def apply_balancing(
    X: np.ndarray,
    y: np.ndarray,
    method: BalancingMethod,
    random_state: int = 42,
    min_minority_samples: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply the chosen balancing method to training data (X, y).

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,)
    method : BalancingMethod
    random_state : int
    min_minority_samples : int
        If minority class has fewer than this many samples, fall back to
        WEIGHTED (no resampling) and warn. SMOTE/ADASYN fail on tiny classes.

    Returns
    -------
    X_resampled, y_resampled
    """
    if method in (BalancingMethod.NONE, BalancingMethod.WEIGHTED):
        return X, y

    # Check minority class size
    unique, counts = np.unique(y, return_counts=True)
    min_count = counts.min()

    if min_count < min_minority_samples:
        print(
            f"[balancing] WARNING: minority class has only {min_count} samples. "
            f"Falling back to NONE for method={method.value}."
        )
        return X, y

    # SMOTE requires k_neighbors < min_minority_samples
    if method in (BalancingMethod.SMOTE, BalancingMethod.ADASYN, BalancingMethod.HYBRID):
        if min_count < 6:
            print(
                f"[balancing] WARNING: too few minority samples for {method.value}. "
                "Falling back to ROS."
            )
            method = BalancingMethod.ROS

    sampler = get_balancer(method, random_state)
    try:
        X_res, y_res = sampler.fit_resample(X, y)
        return X_res, y_res
    except Exception as e:
        print(f"[balancing] WARNING: {method.value} failed ({e}). Returning original data.")
        return X, y


def compute_class_weights(y: np.ndarray) -> dict:
    """
    Compute inverse-frequency class weights for the WEIGHTED method.

    Returns dict {class_label: weight} suitable for sklearn's class_weight
    parameter or torch loss weighting.
    """
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    n_classes = len(unique)
    weights = total / (n_classes * counts)
    return {int(cls): float(w) for cls, w in zip(unique, weights)}
