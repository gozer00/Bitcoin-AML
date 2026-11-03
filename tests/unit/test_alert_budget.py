import math

import numpy as np
import pytest

from bitcoin_aml.evaluation.alert_budget import compute_alert_budget_metrics


def test_precision_recall_at_known_budgets() -> None:
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    y_true = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])

    results = compute_alert_budget_metrics(y_true, scores, (0.1, 0.3, 0.5))
    by_budget = {result.budget_fraction: result for result in results}

    top_1 = by_budget[0.1]
    assert top_1.num_alerts == 1
    assert top_1.precision == pytest.approx(1.0)
    assert top_1.recall == pytest.approx(0.5)
    assert top_1.alerts_per_true_positive == pytest.approx(1.0)

    top_3 = by_budget[0.3]
    assert top_3.num_alerts == 3
    assert top_3.precision == pytest.approx(2 / 3)
    assert top_3.recall == pytest.approx(1.0)

    top_5 = by_budget[0.5]
    assert top_5.num_alerts == 5
    assert top_5.precision == pytest.approx(0.4)
    assert top_5.recall == pytest.approx(1.0)


def test_ties_break_by_original_index_order() -> None:
    # three tied scores (indices 0, 1, 2) followed by a clearly lower one;
    # the sole positive sits at the tied index that a stable sort should
    # NOT select when only the top 2 alerts fit the budget
    scores = np.array([0.5, 0.5, 0.5, 0.1])
    y_true = np.array([0, 0, 1, 0])

    (result,) = compute_alert_budget_metrics(y_true, scores, (0.5,))

    assert result.num_alerts == 2
    assert result.precision == pytest.approx(0.0)


def test_zero_precision_yields_infinite_cost() -> None:
    scores = np.array([0.9, 0.1])
    y_true = np.array([0, 1])

    (result,) = compute_alert_budget_metrics(y_true, scores, (0.5,))

    assert result.precision == pytest.approx(0.0)
    assert math.isinf(result.alerts_per_true_positive)


def test_no_illicit_labels_yields_nan_recall() -> None:
    scores = np.array([0.9, 0.1])
    y_true = np.array([0, 0])

    (result,) = compute_alert_budget_metrics(y_true, scores, (0.5,))

    assert math.isnan(result.recall)


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        compute_alert_budget_metrics(np.array([0, 1]), np.array([0.5]), (0.1,))


def test_invalid_budget_fraction_raises() -> None:
    with pytest.raises(ValueError):
        compute_alert_budget_metrics(np.array([0, 1]), np.array([0.1, 0.9]), (0.0,))

    with pytest.raises(ValueError):
        compute_alert_budget_metrics(np.array([0, 1]), np.array([0.1, 0.9]), (1.5,))


def test_empty_input_raises() -> None:
    with pytest.raises(ValueError):
        compute_alert_budget_metrics(np.array([]), np.array([]), (0.1,))
