import math

import numpy as np
import pytest

from bitcoin_aml.evaluation.metrics import compute_binary_metrics


def test_perfect_predictions() -> None:
    y_true = np.array([0, 0, 1, 1])
    scores = np.array([0.0, 0.1, 0.9, 1.0])

    metrics = compute_binary_metrics(y_true, scores, threshold=0.5)

    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0
    assert metrics.average_precision == 1.0
    assert metrics.true_positives == 2
    assert metrics.true_negatives == 2
    assert metrics.false_positives == 0
    assert metrics.false_negatives == 0


def test_all_wrong_predictions() -> None:
    y_true = np.array([0, 0, 1, 1])
    scores = np.array([1.0, 1.0, 0.0, 0.0])

    metrics = compute_binary_metrics(y_true, scores, threshold=0.5)

    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.true_positives == 0
    assert metrics.false_negatives == 2
    assert metrics.false_positives == 2


def test_degenerate_single_class_true_labels_yields_nan_average_precision() -> None:
    y_true = np.array([0, 0, 0, 0])
    scores = np.array([0.1, 0.4, 0.6, 0.9])

    metrics = compute_binary_metrics(y_true, scores, threshold=0.5)

    assert math.isnan(metrics.average_precision)
    # precision/recall/f1 remain well-defined (zero_division=0)
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.num_illicit == 0
    assert metrics.num_licit == 4


def test_threshold_boundary_is_inclusive() -> None:
    y_true = np.array([1])
    scores = np.array([0.5])

    metrics = compute_binary_metrics(y_true, scores, threshold=0.5)

    assert metrics.true_positives == 1


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        compute_binary_metrics(np.array([0, 1]), np.array([0.5]), threshold=0.5)


def test_invalid_threshold_raises() -> None:
    with pytest.raises(ValueError):
        compute_binary_metrics(np.array([0, 1]), np.array([0.1, 0.9]), threshold=1.5)