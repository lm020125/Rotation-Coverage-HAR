from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def classification_metrics(
    targets: Sequence[int], predictions: Sequence[int], num_classes: int
) -> dict[str, object]:
    targets_array = np.asarray(targets, dtype=int)
    predictions_array = np.asarray(predictions, dtype=int)
    labels = list(range(num_classes))
    return {
        "accuracy": float(accuracy_score(targets_array, predictions_array)),
        "macro_f1": float(
            f1_score(
                targets_array, predictions_array, labels=labels, average="macro", zero_division=0
            )
        ),
        "weighted_f1": float(
            f1_score(
                targets_array, predictions_array, labels=labels, average="weighted", zero_division=0
            )
        ),
        "confusion_matrix": confusion_matrix(
            targets_array, predictions_array, labels=labels
        ).tolist(),
        "per_class": classification_report(
            targets_array,
            predictions_array,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
    }


def classification_metrics_from_confusion(matrix: Sequence[Sequence[int]]) -> dict[str, object]:
    """Compute global metrics after summing out-of-fold confusion matrices."""
    confusion = np.asarray(matrix, dtype=np.int64)
    if confusion.ndim != 2 or confusion.shape[0] != confusion.shape[1]:
        raise ValueError("confusion matrix must be square")
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    true_positive = np.diag(confusion)
    precision = np.divide(
        true_positive, predicted, out=np.zeros_like(true_positive, dtype=float), where=predicted > 0
    )
    recall = np.divide(
        true_positive, support, out=np.zeros_like(true_positive, dtype=float), where=support > 0
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    total = int(confusion.sum())
    return {
        "accuracy": float(true_positive.sum() / total) if total else 0.0,
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.average(f1, weights=support)) if total else 0.0,
        "confusion_matrix": confusion.tolist(),
        "per_class_f1": f1.tolist(),
        "support": support.tolist(),
    }
