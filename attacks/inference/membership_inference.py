"""
attacks/inference/membership_inference.py
==========================================
Membership Inference Attacks (MIA) against the global model.

Two attack types:
  1. Loss-threshold attack: classify samples with loss < threshold as members.
  2. Shadow model attack: train a meta-classifier on shadow model outputs.

Evaluated with and without DP (DP should reduce MIA success).

Metric: MIA advantage = 2 * (MIA_AUC - 0.5)
  0 = no advantage (random guessing = good privacy)
  1 = perfect attack (bad privacy)
"""

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


def loss_threshold_attack(
    model,
    X_members: np.ndarray,
    y_members: np.ndarray,
    X_nonmembers: np.ndarray,
    y_nonmembers: np.ndarray,
    model_type: str = "torch",
    device: str = "cpu",
    batch_size: int = 1024,
) -> dict:
    """
    Loss-threshold MIA: predict membership based on per-sample loss.
    Samples with lower loss are more likely members (they were trained on).

    Parameters
    ----------
    model : trained model
    X_members, y_members : training samples (known members)
    X_nonmembers, y_nonmembers : held-out samples (known non-members)

    Returns
    -------
    result : dict with 'auc', 'advantage', 'threshold'
    """
    from sklearn.metrics import roc_auc_score

    losses_members = _compute_losses(model, X_members, y_members, model_type, device, batch_size)
    losses_nonmembers = _compute_losses(model, X_nonmembers, y_nonmembers, model_type, device, batch_size)

    # Lower loss → more likely a member → negate for ROC AUC convention
    scores = np.concatenate([-losses_members, -losses_nonmembers])
    labels = np.concatenate([np.ones(len(X_members)), np.zeros(len(X_nonmembers))])

    auc = float(roc_auc_score(labels, scores))
    advantage = 2.0 * (auc - 0.5)

    # Optimal threshold (Youden's J)
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(labels, scores)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    best_threshold = float(thresholds[best_idx])

    logger.info(f"[MIA/loss-threshold] AUC={auc:.4f}, advantage={advantage:.4f}")
    return {
        "attack_type": "loss_threshold",
        "auc": auc,
        "advantage": advantage,
        "threshold": best_threshold,
        "n_members": len(X_members),
        "n_nonmembers": len(X_nonmembers),
    }


def shadow_model_attack(
    target_model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_shadow: int = 4,
    model_type: str = "torch",
    device: str = "cpu",
    random_state: int = 42,
) -> dict:
    """
    Shadow model MIA: train N shadow models on disjoint subsets of the training data.
    Use shadow model outputs + membership labels to train a meta-classifier.
    Apply the meta-classifier to the target model's outputs.

    This is a simplified version (single-dataset shadow models) sufficient for
    measuring DP's effect on MIA success.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(random_state)
    n_total = len(X_train)
    shadow_size = n_total // (n_shadow * 2)

    # Collect meta-features: (model_output, is_member) for shadow models
    meta_X, meta_y = [], []

    for shadow_idx in range(n_shadow):
        # Sample shadow train/test subsets
        indices = rng.permutation(n_total)
        shadow_train_idx = indices[:shadow_size]
        shadow_test_idx = indices[shadow_size: shadow_size * 2]

        X_sh_train = X_train[shadow_train_idx]
        y_sh_train = y_train[shadow_train_idx]
        X_sh_test = X_train[shadow_test_idx]
        y_sh_test = y_train[shadow_test_idx]

        # Train a shadow model (simple LR for speed)
        shadow_lr = _train_shadow_lr(X_sh_train, y_sh_train)

        # Collect outputs
        proba_train = shadow_lr.predict_proba(X_sh_train)[:, 1]
        proba_test = shadow_lr.predict_proba(X_sh_test)[:, 1]

        meta_X.append(proba_train.reshape(-1, 1))
        meta_y.append(np.ones(len(proba_train)))
        meta_X.append(proba_test.reshape(-1, 1))
        meta_y.append(np.zeros(len(proba_test)))

    meta_X = np.vstack(meta_X)
    meta_y = np.concatenate(meta_y)

    # Train meta-classifier
    meta_clf = LogisticRegression(random_state=random_state)
    meta_clf.fit(meta_X, meta_y)

    # Apply to target model
    target_proba_train = _get_proba(target_model, X_train[:shadow_size], model_type, device)
    target_proba_test = _get_proba(target_model, X_test[:shadow_size], model_type, device)

    attack_scores_train = meta_clf.predict_proba(target_proba_train.reshape(-1, 1))[:, 1]
    attack_scores_test = meta_clf.predict_proba(target_proba_test.reshape(-1, 1))[:, 1]

    attack_scores = np.concatenate([attack_scores_train, attack_scores_test])
    attack_labels = np.concatenate([
        np.ones(len(attack_scores_train)),
        np.zeros(len(attack_scores_test)),
    ])

    auc = float(roc_auc_score(attack_labels, attack_scores))
    advantage = 2.0 * (auc - 0.5)

    logger.info(f"[MIA/shadow] AUC={auc:.4f}, advantage={advantage:.4f}")
    return {
        "attack_type": "shadow_model",
        "auc": auc,
        "advantage": advantage,
        "n_shadow_models": n_shadow,
    }


def _compute_losses(model, X, y, model_type, device, batch_size):
    """Compute per-sample BCE loss for the model on (X, y)."""
    import torch
    import torch.nn as nn

    if model_type == "torch":
        model.eval()
        losses = []
        n = len(X)
        for start in range(0, n, batch_size):
            xb = torch.tensor(X[start: start + batch_size], dtype=torch.float32).to(device)
            yb = torch.tensor(y[start: start + batch_size], dtype=torch.float32).to(device)
            with torch.no_grad():
                logits = model(xb)
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logits, yb, reduction="none"
                )
            losses.append(loss.cpu().numpy())
        return np.concatenate(losses)
    else:
        proba = model.predict_proba(X)
        if proba.ndim == 2:
            proba = proba[:, 1]
        eps = 1e-10
        proba = np.clip(proba, eps, 1 - eps)
        return -(y * np.log(proba) + (1 - y) * np.log(1 - proba))


def _get_proba(model, X, model_type, device):
    if model_type == "torch":
        import torch
        model.eval()
        with torch.no_grad():
            xb = torch.tensor(X, dtype=torch.float32).to(device)
            proba = model.predict_proba(xb).cpu().numpy()
        return proba
    else:
        proba = model.predict_proba(X)
        return proba[:, 1] if proba.ndim == 2 else proba


def _train_shadow_lr(X, y):
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(max_iter=100, random_state=42)
    lr.fit(X, y)
    return lr
