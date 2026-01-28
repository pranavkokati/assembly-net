"""
Evaluation metrics for Assembly-Net.

Provides metrics for both classification and regression tasks.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        num_classes: Number of classes.

    Returns:
        Dictionary of metrics.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if num_classes is None:
        num_classes = max(len(np.unique(y_true)), len(np.unique(y_pred)))

    metrics = {}

    # Overall accuracy
    metrics["accuracy"] = 100.0 * np.mean(y_true == y_pred)

    # Per-class metrics
    for c in range(num_classes):
        # True positives, false positives, false negatives
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))

        # Precision
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        metrics[f"precision_class_{c}"] = 100.0 * precision

        # Recall
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics[f"recall_class_{c}"] = 100.0 * recall

        # F1
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        metrics[f"f1_class_{c}"] = 100.0 * f1

        # Class accuracy
        class_total = np.sum(y_true == c)
        if class_total > 0:
            metrics[f"accuracy_class_{c}"] = 100.0 * tp / class_total

    # Macro averages
    precisions = [metrics[f"precision_class_{c}"] for c in range(num_classes)]
    recalls = [metrics[f"recall_class_{c}"] for c in range(num_classes)]
    f1s = [metrics[f"f1_class_{c}"] for c in range(num_classes)]

    metrics["macro_precision"] = np.mean(precisions)
    metrics["macro_recall"] = np.mean(recalls)
    metrics["macro_f1"] = np.mean(f1s)

    # Confusion matrix
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        confusion[t, p] += 1
    metrics["confusion_matrix"] = confusion

    return metrics


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute regression metrics.

    Args:
        y_true: True values.
        y_pred: Predicted values.

    Returns:
        Dictionary of metrics.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    metrics = {}

    # Mean Squared Error
    metrics["mse"] = float(np.mean((y_true - y_pred) ** 2))

    # Root Mean Squared Error
    metrics["rmse"] = float(np.sqrt(metrics["mse"]))

    # Mean Absolute Error
    metrics["mae"] = float(np.mean(np.abs(y_true - y_pred)))

    # R-squared
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    metrics["r2"] = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Mean Absolute Percentage Error (if no zeros)
    if np.all(y_true != 0):
        metrics["mape"] = float(100.0 * np.mean(np.abs((y_true - y_pred) / y_true)))

    # Pearson correlation
    if len(y_true) > 1:
        correlation = np.corrcoef(y_true, y_pred)[0, 1]
        metrics["pearson_r"] = float(correlation) if not np.isnan(correlation) else 0.0

    return metrics


def compute_topological_metrics(
    true_betti: List[np.ndarray],
    pred_betti: List[np.ndarray],
) -> Dict[str, float]:
    """
    Compute metrics for topological feature prediction.

    Args:
        true_betti: True Betti numbers for each sample.
        pred_betti: Predicted Betti numbers.

    Returns:
        Dictionary of metrics.
    """
    true_betti = np.array(true_betti)
    pred_betti = np.array(pred_betti)

    metrics = {}

    # Per-dimension metrics
    for dim in range(min(true_betti.shape[1], pred_betti.shape[1])):
        true_dim = true_betti[:, dim]
        pred_dim = pred_betti[:, dim]

        metrics[f"betti_{dim}_mae"] = float(np.mean(np.abs(true_dim - pred_dim)))
        metrics[f"betti_{dim}_rmse"] = float(np.sqrt(np.mean((true_dim - pred_dim) ** 2)))

        # Exact match rate
        metrics[f"betti_{dim}_exact_match"] = float(100.0 * np.mean(true_dim == pred_dim))

    return metrics


def confusion_matrix_plot(
    confusion_matrix: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = False,
) -> Tuple[Any, Any]:
    """
    Create confusion matrix plot data.

    Args:
        confusion_matrix: Confusion matrix array.
        class_names: Names for classes.
        normalize: Whether to normalize by row.

    Returns:
        Tuple of (figure, axes) for plotting.
    """
    import matplotlib.pyplot as plt

    if normalize:
        row_sums = confusion_matrix.sum(axis=1, keepdims=True)
        confusion_matrix = confusion_matrix / np.maximum(row_sums, 1)

    fig, ax = plt.subplots(figsize=(8, 8))

    im = ax.imshow(confusion_matrix, cmap="Blues")

    n_classes = confusion_matrix.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    # Add text annotations
    thresh = confusion_matrix.max() / 2
    fmt = ".2f" if normalize else "d"
    for i in range(n_classes):
        for j in range(n_classes):
            color = "white" if confusion_matrix[i, j] > thresh else "black"
            text = format(confusion_matrix[i, j], fmt)
            ax.text(j, i, text, ha="center", va="center", color=color)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix" + (" (Normalized)" if normalize else ""))

    fig.colorbar(im, ax=ax)
    plt.tight_layout()

    return fig, ax


def balanced_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Compute balanced accuracy (average of per-class accuracies).

    Args:
        y_true: True labels.
        y_pred: Predicted labels.

    Returns:
        Balanced accuracy percentage.
    """
    classes = np.unique(y_true)
    per_class_acc = []

    for c in classes:
        mask = y_true == c
        if mask.sum() > 0:
            acc = (y_pred[mask] == c).mean()
            per_class_acc.append(acc)

    return 100.0 * np.mean(per_class_acc)


def top_k_accuracy(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    k: int = 3,
) -> float:
    """
    Compute top-k accuracy.

    Args:
        y_true: True labels.
        y_probs: Predicted class probabilities.
        k: Number of top classes to consider.

    Returns:
        Top-k accuracy percentage.
    """
    top_k_preds = np.argsort(y_probs, axis=1)[:, -k:]
    correct = np.array([y in preds for y, preds in zip(y_true, top_k_preds)])
    return 100.0 * correct.mean()
