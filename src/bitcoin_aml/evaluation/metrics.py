from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


@dataclass(frozen=True)
class BinaryMetrics:
    """
    defines the result contract
    """

    precision: float
    recall: float
    f1: float
    average_precision: float

    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int

    threshold: float

    @property
    def num_illicit(self) -> int:
        return self.true_positives + self.false_negatives

    @property
    def num_licit(self) -> int:
        return self.true_negatives + self.false_positives


def compute_binary_metrics(
    y_true: NDArray[np.integer],
    scores: NDArray[np.floating],
    *,
    threshold: float,
) -> BinaryMetrics:
    """
    computes BinaryMetrics from illicit-class (label 1) scores

    average_precision is nan when y_true contains only one class, since
    ranking quality is undefined in that case
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    if y_true.shape != scores.shape:
        raise ValueError("y_true and scores must have the same shape")

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie in [0, 1]")

    y_pred = (scores >= threshold).astype(int)

    true_negatives, false_positives, false_negatives, true_positives = (
        confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    )

    if np.unique(y_true).size < 2:
        # ranking quality is undefined with only one class present; sklearn
        # would otherwise silently return a degenerate 0.0 or 1.0 with a warning
        average_precision = float("nan")
    else:
        average_precision = float(average_precision_score(y_true, scores))

    return BinaryMetrics(
        precision=float(
            precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        recall=float(
            recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        f1=float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        average_precision=average_precision,
        true_positives=int(true_positives),
        false_positives=int(false_positives),
        true_negatives=int(true_negatives),
        false_negatives=int(false_negatives),
        threshold=threshold,
    )