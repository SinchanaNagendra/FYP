"""
fl/bank_node/evaluator.py
==========================
Evaluates a trained model on a local bank's validation or test set.

Metrics computed:
  - Accuracy (reported with majority-class baseline beside it)
  - Precision, Recall, F1 (binary, threshold=0.5)
  - PR-AUC (primary metric for imbalanced data)
  - ROC-AUC
  - Recall at 1% FPR (false positive rate)
  - Log-loss
  - Majority-class baseline accuracy

All metric computations are based only on the local bank's data.
Metrics (not raw predictions) are what get reported on-chain.
"""

import logging
from typing import Any, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


def evaluate(
    model,
    X: np.ndarray,
    y: np.ndarray,
    split_name: str = "val",
    model_type: str = "torch",
    device: str = "cpu",
    batch_size: int = 1024,
) -> dict[str, Any]:
    """
    Compute all evaluation metrics.

    Parameters
    ----------
    model : model object (LogisticRegressionModel, CNNModel, GaussianNBModel, etc.)
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,)
    split_name : str
        'val' or 'test' (for logging only)
    model_type : str
        'torch' | 'sklearn' | 'gnb' | 'tree'
    device : str
    batch_size : int

    Returns
    -------
    metrics : dict
        All computed metrics. Ready for on-chain reporting.
    """
    y_true = y.astype(np.int32)
    n = len(y_true)

    # Get predicted probabilities
    y_prob = _predict_proba(model, X, model_type, device, batch_size)

    # Binary predictions at 0.5 threshold
    y_pred = (y_prob >= 0.5).astype(np.int32)

    # Fraud rate in this split
    fraud_rate = float(y_true.mean())
    majority_class = int(np.argmax(np.bincount(y_true.astype(np.int64))))
    majority_acc = float((y_true == majority_class).mean())

    # Core metrics
    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    # PR-AUC (primary metric)
    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except Exception:
        pr_auc = float("nan")

    # ROC-AUC
    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc_auc = float("nan")

    # Recall at 1% FPR
    recall_at_1_fpr = _recall_at_fpr(y_true, y_prob, target_fpr=0.01)

    # Log-loss
    try:
        logloss = float(log_loss(y_true, y_prob))
    except Exception:
        logloss = float("nan")

    metrics = {
        "split": split_name,
        "n_samples": n,
        "fraud_rate": round(fraud_rate, 6),
        "majority_class_accuracy": round(majority_acc, 6),
        # Accuracy reported WITH baseline for context
        "accuracy": round(accuracy, 6),
        "accuracy_vs_majority": round(accuracy - majority_acc, 6),
        # Primary metrics
        "pr_auc": round(pr_auc, 6) if not np.isnan(pr_auc) else None,
        "roc_auc": round(roc_auc, 6) if not np.isnan(roc_auc) else None,
        # Secondary metrics
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "recall_at_1pct_fpr": round(recall_at_1_fpr, 6) if recall_at_1_fpr is not None else None,
        "log_loss": round(logloss, 6) if not np.isnan(logloss) else None,
    }

    logger.info(
        f"[eval] {split_name}: n={n}, fraud={fraud_rate:.3%}, "
        f"PR-AUC={pr_auc:.4f}, ROC-AUC={roc_auc:.4f}, F1={f1:.4f}"
    )

    return metrics


def _predict_proba(
    model,
    X: np.ndarray,
    model_type: str,
    device: str,
    batch_size: int,
) -> np.ndarray:
    """
    Get predicted fraud probabilities from any model type.
    Returns 1-D array of fraud (class=1) probabilities.
    """
    if model_type in ("torch",):
        import torch
        model.eval()
        probs = []
        n = len(X)
        for start in range(0, n, batch_size):
            xb = torch.tensor(X[start: start + batch_size], dtype=torch.float32).to(device)
            with torch.no_grad():
                pb = model.predict_proba(xb).cpu().numpy()
            probs.append(pb)
        return np.concatenate(probs)

    elif model_type in ("sklearn", "gnb", "tree", "forest"):
        proba = model.predict_proba(X)
        if proba.ndim == 2:
            # (n_samples, n_classes) → take class-1 column
            return proba[:, 1]
        return proba

    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def _recall_at_fpr(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_fpr: float = 0.01,
) -> Optional[float]:
    """
    Compute recall (TPR) at a fixed false positive rate (FPR).

    Parameters
    ----------
    y_true : binary labels
    y_prob : fraud probability scores
    target_fpr : float (default 0.01 = 1%)

    Returns
    -------
    recall at that FPR, or None if not computable.
    """
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        # Find the threshold closest to target_fpr
        idx = np.searchsorted(fpr, target_fpr, side="right") - 1
        idx = max(0, min(idx, len(tpr) - 1))
        return float(tpr[idx])
    except Exception:
        return None


def metrics_to_summary_str(metrics: dict) -> str:
    """Pretty-print a metrics dict for logging/display."""
    lines = [
        f"  PR-AUC          : {metrics.get('pr_auc', 'N/A')}",
        f"  ROC-AUC         : {metrics.get('roc_auc', 'N/A')}",
        f"  F1              : {metrics.get('f1', 'N/A')}",
        f"  Precision       : {metrics.get('precision', 'N/A')}",
        f"  Recall          : {metrics.get('recall', 'N/A')}",
        f"  Recall@1%FPR    : {metrics.get('recall_at_1pct_fpr', 'N/A')}",
        f"  Log-loss        : {metrics.get('log_loss', 'N/A')}",
        f"  Accuracy        : {metrics.get('accuracy', 'N/A')} "
        f"(majority={metrics.get('majority_class_accuracy', 'N/A')})",
        f"  N samples       : {metrics.get('n_samples', 'N/A')}",
        f"  Fraud rate      : {metrics.get('fraud_rate', 'N/A'):.2%}"
        if isinstance(metrics.get('fraud_rate'), float) else
        f"  Fraud rate      : {metrics.get('fraud_rate', 'N/A')}",
    ]
    return "\n".join(lines)
