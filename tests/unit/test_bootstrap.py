import functools
import math

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, f1_score

from bitcoin_aml.evaluation.bootstrap import (
    average_precision_metric,
    bootstrap_confidence_interval,
    f1_metric,
    paired_bootstrap_difference,
)


def _mean_score(y_true: np.ndarray, scores: np.ndarray) -> float:
    del y_true
    return float(np.mean(scores))


def _synthetic_data(n: int = 200, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    scores = rng.uniform(0.0, 1.0, size=n)
    y_true = (scores > 0.5).astype(int)
    return y_true, scores


def test_point_estimate_matches_full_data_metric() -> None:
    y_true, scores = _synthetic_data()

    result = bootstrap_confidence_interval(
        y_true, scores, metric_fn=_mean_score, n_resamples=100, seed=1
    )

    assert result.point_estimate == pytest.approx(float(np.mean(scores)))


def test_ci_brackets_point_estimate() -> None:
    y_true, scores = _synthetic_data()

    result = bootstrap_confidence_interval(
        y_true, scores, metric_fn=_mean_score, n_resamples=2000, seed=1
    )

    assert result.lower <= result.point_estimate <= result.upper


def test_same_seed_is_deterministic() -> None:
    y_true, scores = _synthetic_data()

    first = bootstrap_confidence_interval(
        y_true, scores, metric_fn=_mean_score, n_resamples=500, seed=7
    )
    second = bootstrap_confidence_interval(
        y_true, scores, metric_fn=_mean_score, n_resamples=500, seed=7
    )

    assert first == second


def test_varying_n_resamples_yields_valid_ordered_bounds() -> None:
    # a strict "more resamples -> strictly narrower CI" claim is not
    # statistically guaranteed (percentile estimation noise can go either
    # way), so this only checks the bounds stay well-formed across a range
    # of resample counts, not that width monotonically shrinks
    y_true, scores = _synthetic_data()

    for n_resamples in (5, 50, 500, 2000):
        result = bootstrap_confidence_interval(
            y_true, scores, metric_fn=_mean_score, n_resamples=n_resamples, seed=3
        )
        assert result.lower <= result.upper
        assert result.n_resamples == n_resamples


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        bootstrap_confidence_interval(
            np.array([0, 1]), np.array([0.5]), metric_fn=_mean_score, seed=0
        )


def test_empty_input_raises() -> None:
    with pytest.raises(ValueError):
        bootstrap_confidence_interval(
            np.array([]), np.array([]), metric_fn=_mean_score, seed=0
        )


def test_invalid_n_resamples_raises() -> None:
    with pytest.raises(ValueError):
        bootstrap_confidence_interval(
            np.array([0, 1]),
            np.array([0.1, 0.9]),
            metric_fn=_mean_score,
            n_resamples=0,
            seed=0,
        )


def test_invalid_confidence_raises() -> None:
    with pytest.raises(ValueError):
        bootstrap_confidence_interval(
            np.array([0, 1]),
            np.array([0.1, 0.9]),
            metric_fn=_mean_score,
            confidence=1.0,
            seed=0,
        )


def test_average_precision_metric_matches_sklearn_on_untied_random_data() -> None:
    rng = np.random.default_rng(0)

    for _ in range(500):
        n = rng.integers(5, 60)
        y_true = (rng.random(n) < 0.3).astype(int)
        scores = rng.random(n)  # continuous -> ties are ~never present

        if y_true.sum() == 0:
            continue

        expected = average_precision_score(y_true, scores)
        actual = average_precision_metric(y_true, scores)
        assert actual == pytest.approx(expected)


def test_average_precision_metric_is_nan_with_no_positives() -> None:
    assert math.isnan(average_precision_metric(np.array([0, 0, 0]), np.array([0.1, 0.5, 0.9])))


def test_f1_metric_matches_sklearn_on_random_data() -> None:
    rng = np.random.default_rng(0)

    for _ in range(500):
        n = rng.integers(5, 60)
        y_true = (rng.random(n) < 0.3).astype(int)
        scores = rng.random(n)
        threshold = 0.5

        expected = f1_score(y_true, scores >= threshold, pos_label=1, zero_division=0)
        actual = f1_metric(y_true, scores, threshold=threshold)
        assert actual == pytest.approx(expected)


def test_f1_metric_partial_application_works_as_metric_fn() -> None:
    y_true, scores = _synthetic_data()

    bound_f1 = functools.partial(f1_metric, threshold=0.5)
    result = bootstrap_confidence_interval(
        y_true, scores, metric_fn=bound_f1, n_resamples=50, seed=0
    )

    assert 0.0 <= result.point_estimate <= 1.0
    assert 0.0 <= result.lower <= result.upper <= 1.0



def _informative_scores(
    y_true: np.ndarray, noise: float, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return y_true + rng.normal(0.0, noise, size=len(y_true))


def test_paired_difference_of_identical_models_is_zero() -> None:
    rng = np.random.default_rng(0)
    y_true = (rng.uniform(size=300) < 0.2).astype(int)
    scores = _informative_scores(y_true, noise=1.0, seed=1)

    result = paired_bootstrap_difference(
        y_true,
        scores,
        scores,
        metric_fn=average_precision_metric,
        n_resamples=200,
        seed=2,
    )

    assert result.point_estimate == 0.0
    assert result.lower == 0.0
    assert result.upper == 0.0
    assert result.fraction_non_positive == 1.0


def test_paired_difference_point_estimate_is_seed_mean_difference() -> None:
    rng = np.random.default_rng(3)
    y_true = (rng.uniform(size=300) < 0.2).astype(int)
    scores_a = np.stack(
        [_informative_scores(y_true, noise=0.5, seed=s) for s in range(3)]
    )
    scores_b = np.stack(
        [_informative_scores(y_true, noise=2.0, seed=s) for s in range(10, 12)]
    )

    result = paired_bootstrap_difference(
        y_true,
        scores_a,
        scores_b,
        metric_fn=average_precision_metric,
        n_resamples=200,
        seed=4,
    )

    expected = np.mean(
        [average_precision_score(y_true, row) for row in scores_a]
    ) - np.mean([average_precision_score(y_true, row) for row in scores_b])

    assert result.point_estimate == pytest.approx(expected)
    assert result.lower <= result.point_estimate <= result.upper
    # a is far less noisy than b, so a should win in essentially every resample
    assert result.lower > 0.0
    assert result.fraction_non_positive < 0.01


def test_paired_difference_stratified_resampling_runs() -> None:
    rng = np.random.default_rng(5)
    y_true = (rng.uniform(size=300) < 0.3).astype(int)
    strata = np.repeat(np.arange(3), 100)
    scores_a = _informative_scores(y_true, noise=0.8, seed=6)
    scores_b = _informative_scores(y_true, noise=0.9, seed=7)

    result = paired_bootstrap_difference(
        y_true,
        scores_a,
        scores_b,
        metric_fn=average_precision_metric,
        strata=strata,
        n_resamples=100,
        seed=8,
    )

    assert result.lower <= result.upper
    assert 0.0 <= result.fraction_non_positive <= 1.0


def test_paired_difference_rejects_misaligned_scores() -> None:
    y_true = np.array([0, 1, 0, 1])

    with pytest.raises(ValueError):
        paired_bootstrap_difference(
            y_true,
            np.zeros(4),
            np.zeros(3),
            metric_fn=average_precision_metric,
            seed=0,
        )
