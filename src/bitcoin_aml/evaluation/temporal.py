# per-timestep evaluation, built on evaluation/metrics.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from bitcoin_aml.data.dataset import EllipticDataset
from bitcoin_aml.evaluation.metrics import BinaryMetrics, compute_binary_metrics


@dataclass(frozen=True, slots=True)
class TemporalMetrics:
    overall: BinaryMetrics
    per_timestep: Mapping[int, BinaryMetrics]


def evaluate_per_timestep(
    dataset: EllipticDataset,
    predict_idx: NDArray[np.integer],
    y_true: NDArray[np.integer],
    scores: NDArray[np.floating],
    *,
    threshold: float,
) -> TemporalMetrics:
    """
    y_true and scores must be aligned to predict_idx (same order, same length)
    """
    if len(predict_idx) != len(y_true) or len(predict_idx) != len(scores):
        raise ValueError("predict_idx, y_true, and scores must have equal length")

    overall = compute_binary_metrics(y_true, scores, threshold=threshold)

    timesteps = dataset.timestep[predict_idx]

    per_timestep: dict[int, BinaryMetrics] = {}

    for observed_timestep in sorted({int(value) for value in timesteps}):
        mask = timesteps == observed_timestep

        per_timestep[observed_timestep] = compute_binary_metrics(
            y_true[mask],
            scores[mask],
            threshold=threshold,
        )

    return TemporalMetrics(overall=overall, per_timestep=per_timestep)