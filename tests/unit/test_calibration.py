import math

import numpy as np
import pytest

from bitcoin_aml.evaluation.calibration import compute_calibration_diagnostics


def test_perfect_predictions_have_zero_brier_score() -> None:
    y_true = np.array([0, 0, 1, 1])
    scores = np.array([0.0, 0.0, 1.0, 1.0])

    diagnostics = compute_calibration_diagnostics(y_true, scores, n_bins=2)

    assert diagnostics.brier_score == pytest.approx(0.0)
    assert diagnostics.expected_calibration_error == pytest.approx(0.0)


def test_ece_matches_hand_computed_value() -> None:
    # 2 bins: [0, 0.5] and (0.5, 1.0]
    # bin 0: scores 0.2, 0.3 -> confidence 0.25, y_true [0, 0] -> accuracy 0.0
    # bin 1: scores 0.7, 0.9 -> confidence 0.8,  y_true [1, 0] -> accuracy 0.5
    y_true = np.array([0, 0, 1, 0])
    scores = np.array([0.2, 0.3, 0.7, 0.9])

    diagnostics = compute_calibration_diagnostics(y_true, scores, n_bins=2)

    expected_ece = 0.5 * abs(0.0 - 0.25) + 0.5 * abs(0.5 - 0.8)
    assert diagnostics.expected_calibration_error == pytest.approx(expected_ece)
    assert diagnostics.bin_counts == (2, 2)
    assert diagnostics.bin_confidence[0] == pytest.approx(0.25)
    assert diagnostics.bin_accuracy[1] == pytest.approx(0.5)


def test_empty_bins_reported_as_nan_and_excluded_from_ece() -> None:
    y_true = np.array([0, 1])
    scores = np.array([0.05, 0.1])

    diagnostics = compute_calibration_diagnostics(y_true, scores, n_bins=10)

    assert diagnostics.bin_counts[-1] == 0
    assert math.isnan(diagnostics.bin_confidence[-1])
    assert math.isnan(diagnostics.bin_accuracy[-1])
    assert not math.isnan(diagnostics.expected_calibration_error)


def test_single_class_y_true_does_not_raise() -> None:
    y_true = np.array([0, 0, 0, 0])
    scores = np.array([0.1, 0.4, 0.6, 0.9])

    diagnostics = compute_calibration_diagnostics(y_true, scores, n_bins=4)

    assert not math.isnan(diagnostics.brier_score)
    assert not math.isnan(diagnostics.expected_calibration_error)


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        compute_calibration_diagnostics(np.array([0, 1]), np.array([0.5]))


def test_non_positive_n_bins_raises() -> None:
    with pytest.raises(ValueError):
        compute_calibration_diagnostics(np.array([0, 1]), np.array([0.1, 0.9]), n_bins=0)
