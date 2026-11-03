# alert-budget evaluation: precision-at-k under a fixed alert budget

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class AlertBudgetResult:
    """
    metrics at one fixed alert budget -- the top `budget_fraction` of
    scored transactions, ranked by score descending
    """

    budget_fraction: float
    num_alerts: int
    precision: float
    recall: float
    alerts_per_true_positive: float


def compute_alert_budget_metrics(
    y_true: NDArray[np.integer],
    scores: NDArray[np.floating],
    budget_fractions: tuple[float, ...],
) -> tuple[AlertBudgetResult, ...]:
    """
    for each budget fraction, flags the top ceil(budget_fraction * N)
    scored transactions (stable sort by score descending, so tied scores
    break deterministically in original-index order) and computes
    precision/recall of that alert set against y_true

    `alerts_per_true_positive` is 1/precision (an "investigator cost per
    confirmed illicit transaction" reading), or inf when precision is 0
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    if y_true.shape != scores.shape:
        raise ValueError("y_true and scores must have the same shape")

    n = len(scores)
    if n == 0:
        raise ValueError("y_true/scores must be non-empty")

    num_illicit = int(np.sum(y_true))

    order = np.argsort(-scores, kind="stable")

    results = []

    for budget_fraction in budget_fractions:
        if not 0.0 < budget_fraction <= 1.0:
            raise ValueError("budget_fraction must lie in (0, 1]")

        num_alerts = max(1, math.ceil(budget_fraction * n))
        alert_idx = order[:num_alerts]

        true_positives = int(np.sum(y_true[alert_idx]))
        precision = true_positives / num_alerts
        recall = true_positives / num_illicit if num_illicit > 0 else float("nan")
        alerts_per_true_positive = (
            1.0 / precision if precision > 0 else float("inf")
        )

        results.append(
            AlertBudgetResult(
                budget_fraction=budget_fraction,
                num_alerts=num_alerts,
                precision=precision,
                recall=recall,
                alerts_per_true_positive=alerts_per_true_positive,
            )
        )

    return tuple(results)
