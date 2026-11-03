# probability-calibration diagnostics, built on raw scores (no recalibration)

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class CalibrationDiagnostics:
    """
    diagnostic-only calibration summary: how well `scores` already behave
    like probabilities, without fitting any recalibrator
    """

    brier_score: float
    expected_calibration_error: float
    bin_edges: tuple[float, ...]
    bin_confidence: tuple[float, ...]
    bin_accuracy: tuple[float, ...]
    bin_counts: tuple[int, ...]


def compute_calibration_diagnostics(
    y_true: NDArray[np.integer],
    scores: NDArray[np.floating],
    *,
    n_bins: int = 10,
) -> CalibrationDiagnostics:
    """
    computes Brier score and Expected Calibration Error (ECE) over
    `n_bins` equal-width bins spanning [0, 1]

    unlike compute_binary_metrics's average_precision, neither statistic is
    NaN-guarded for single-class y_true: both remain well-defined (if less
    informative) when only one class is present
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)

    if y_true.shape != scores.shape:
        raise ValueError("y_true and scores must have the same shape")

    if n_bins < 1:
        raise ValueError("n_bins must be a positive integer")

    brier_score = float(np.mean((scores - y_true) ** 2))

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    # right-inclusive bins, except the first, so a score of exactly 0.0
    # still lands in the first bin
    bin_indices = np.clip(np.digitize(scores, bin_edges[1:-1], right=True), 0, n_bins - 1)

    bin_confidence: list[float] = []
    bin_accuracy: list[float] = []
    bin_counts: list[int] = []
    ece = 0.0
    n = len(scores)

    for bin_index in range(n_bins):
        mask = bin_indices == bin_index
        count = int(np.sum(mask))
        bin_counts.append(count)

        if count == 0:
            bin_confidence.append(float("nan"))
            bin_accuracy.append(float("nan"))
            continue

        confidence = float(np.mean(scores[mask]))
        accuracy = float(np.mean(y_true[mask]))
        bin_confidence.append(confidence)
        bin_accuracy.append(accuracy)

        ece += (count / n) * abs(accuracy - confidence)

    return CalibrationDiagnostics(
        brier_score=brier_score,
        expected_calibration_error=float(ece),
        bin_edges=tuple(float(edge) for edge in bin_edges),
        bin_confidence=tuple(bin_confidence),
        bin_accuracy=tuple(bin_accuracy),
        bin_counts=tuple(bin_counts),
    )
