# bootstrap confidence intervals over a metric, by resampling test nodes

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

MetricFn = Callable[[NDArray[np.integer], NDArray[np.floating]], float]


@dataclass(frozen=True, slots=True)
class BootstrapCI:
    point_estimate: float
    lower: float
    upper: float
    n_resamples: int
    confidence: float


def bootstrap_confidence_interval(
    y_true: NDArray[np.integer],
    scores: NDArray[np.floating],
    *,
    metric_fn: MetricFn,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int,
) -> BootstrapCI:
    """
    resamples (y_true, scores) pairs with replacement `n_resamples` times,
    recomputes `metric_fn` on each resample, and returns a percentile
    confidence interval

    the point estimate is computed once on the full, un-resampled data --
    only the interval bounds come from resampling
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    if y_true.shape != scores.shape:
        raise ValueError("y_true and scores must have the same shape")

    n = len(scores)
    if n == 0:
        raise ValueError("y_true/scores must be non-empty")

    if n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer")

    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")

    point_estimate = float(metric_fn(y_true, scores))

    rng = np.random.default_rng(seed)
    resampled_values = np.empty(n_resamples, dtype=np.float64)

    for i in range(n_resamples):
        resample_idx = rng.integers(0, n, size=n)
        resampled_values[i] = metric_fn(y_true[resample_idx], scores[resample_idx])

    alpha = 1.0 - confidence
    lower, upper = np.nanpercentile(
        resampled_values, [100 * alpha / 2, 100 * (1 - alpha / 2)]
    )

    return BootstrapCI(
        point_estimate=point_estimate,
        lower=float(lower),
        upper=float(upper),
        n_resamples=n_resamples,
        confidence=confidence,
    )


def average_precision_metric(
    y_true: NDArray[np.integer], scores: NDArray[np.floating]
) -> float:
    """
    fast, numpy-only average precision, for use as a bootstrap `metric_fn`

    `evaluation/metrics.py::compute_binary_metrics` wraps sklearn's
    `average_precision_score`, which is too slow to call once per resample
    at bootstrap scale (thousands of resamples x hundreds of
    config/timestep groups): its per-call input-validation overhead alone
    dominates runtime. This computes the equivalent quantity directly --
    the mean of precision-at-rank over positive-labeled ranks, ranked by
    score descending -- verified against sklearn on randomized data
    (see tests/unit/test_bootstrap.py). It can diverge from sklearn's
    result only under exact score ties (sklearn averages across a tied
    group; this breaks ties by original index via a stable sort), which
    essentially never occurs for continuous model scores.

    NaN when no positive label is present, matching compute_binary_metrics.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    num_positive = int(np.sum(y_true))
    if num_positive == 0:
        return float("nan")

    order = np.argsort(-scores, kind="stable")
    y_sorted = y_true[order]
    cumulative_true_positives = np.cumsum(y_sorted)
    ranks = np.arange(1, len(y_sorted) + 1)
    precision_at_rank = cumulative_true_positives / ranks

    return float(np.sum(precision_at_rank * y_sorted) / num_positive)


def f1_metric(
    y_true: NDArray[np.integer],
    scores: NDArray[np.floating],
    *,
    threshold: float,
) -> float:
    """
    fast, numpy-only F1 at a fixed threshold, for use as a bootstrap
    `metric_fn` (via functools.partial(f1_metric, threshold=...)) -- avoids
    the same sklearn per-call overhead as average_precision_metric

    0.0 when precision + recall is 0, matching sklearn's zero_division=0
    """
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(scores) >= threshold

    true_positives = int(np.sum(y_pred & y_true))
    false_positives = int(np.sum(y_pred & ~y_true))
    false_negatives = int(np.sum(~y_pred & y_true))

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )

    if precision + recall == 0.0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


@dataclass(frozen=True, slots=True)
class PairedBootstrapResult:
    """
    difference metric(a) - metric(b) on the same test nodes

    `fraction_non_positive` is the share of resamples in which a does not
    beat b -- a one-sided bootstrap p-value for "a > b"
    """

    point_estimate: float
    lower: float
    upper: float
    fraction_non_positive: float
    n_resamples: int
    confidence: float


def _seed_mean_metric(
    y_true: NDArray[np.integer],
    scores: NDArray[np.floating],
    metric_fn: MetricFn,
) -> float:
    return float(np.mean([metric_fn(y_true, row) for row in scores]))


def paired_bootstrap_difference(
    y_true: NDArray[np.integer],
    scores_a: NDArray[np.floating],
    scores_b: NDArray[np.floating],
    *,
    metric_fn: MetricFn,
    strata: NDArray[np.integer] | None = None,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int,
) -> PairedBootstrapResult:
    """
    paired bootstrap of the seed-averaged metric difference between two
    models scored on the same, identically ordered test nodes

    `scores_a`/`scores_b` are (n_seeds, n_nodes) arrays (a 1-D array is
    treated as a single seed); each resample draws ONE set of node indices
    and applies it to every seed of both models, so the interval reflects
    test-set sampling uncertainty with the model pairing preserved, and the
    point estimate is mean_seeds metric(a) - mean_seeds metric(b) on the
    full test set

    with `strata` (e.g. per-node timestep), nodes are resampled within each
    stratum, preserving the per-timestep composition of the test window
    """
    y_true = np.asarray(y_true)
    scores_a = np.atleast_2d(np.asarray(scores_a))
    scores_b = np.atleast_2d(np.asarray(scores_b))

    n = len(y_true)
    if n == 0:
        raise ValueError("y_true must be non-empty")

    if scores_a.shape[1] != n or scores_b.shape[1] != n:
        raise ValueError("scores_a/scores_b must have one column per node")

    if n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer")

    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")

    if strata is None:
        groups = [np.arange(n)]
    else:
        strata = np.asarray(strata)
        if strata.shape != y_true.shape:
            raise ValueError("strata must align with y_true")
        groups = [np.flatnonzero(strata == value) for value in np.unique(strata)]

    point_estimate = _seed_mean_metric(
        y_true, scores_a, metric_fn
    ) - _seed_mean_metric(y_true, scores_b, metric_fn)

    rng = np.random.default_rng(seed)
    differences = np.empty(n_resamples, dtype=np.float64)

    for i in range(n_resamples):
        resample_idx = np.concatenate(
            [group[rng.integers(0, len(group), size=len(group))] for group in groups]
        )
        y_resampled = y_true[resample_idx]
        differences[i] = _seed_mean_metric(
            y_resampled, scores_a[:, resample_idx], metric_fn
        ) - _seed_mean_metric(y_resampled, scores_b[:, resample_idx], metric_fn)

    alpha = 1.0 - confidence
    lower, upper = np.nanpercentile(
        differences, [100 * alpha / 2, 100 * (1 - alpha / 2)]
    )

    return PairedBootstrapResult(
        point_estimate=float(point_estimate),
        lower=float(lower),
        upper=float(upper),
        fraction_non_positive=float(np.mean(differences <= 0.0)),
        n_resamples=n_resamples,
        confidence=confidence,
    )
