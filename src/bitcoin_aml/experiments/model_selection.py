# model-agnostic hyperparameter selection

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Mapping

from bitcoin_aml.evaluation.metrics import compute_binary_metrics
from bitcoin_aml.experiments.specification import SelectionDirection
from bitcoin_aml.models.base import ModelInput
from bitcoin_aml.models.registry import create_model
from bitcoin_aml.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class CandidateResult:
    parameters: Mapping[str, Any]
    metric_value: float


@dataclass(frozen=True, slots=True)
class ModelSelectionResult:
    best_parameters: Mapping[str, Any]
    best_metric_value: float
    candidates: tuple[CandidateResult, ...]


def generate_candidates(
    search_space: Mapping[str, tuple[Any, ...]],
) -> tuple[dict[str, Any], ...]:
    """
    cartesian product of the search space, in deterministic order

    parameter names are sorted so that iteration order (and therefore any
    tie-breaking during selection) does not depend on dict insertion order
    """
    if not search_space:
        return ({},)

    names = sorted(search_space)
    value_lists = [search_space[name] for name in names]

    return tuple(
        dict(zip(names, combination))
        for combination in itertools.product(*value_lists)
    )


def select_best_hyperparameters(
    *,
    model_name: str,
    fixed_parameters: Mapping[str, Any],
    search_space: Mapping[str, tuple[Any, ...]],
    train_data: ModelInput,
    metric: str,
    direction: SelectionDirection,
    threshold: float,
    seed: int,
) -> ModelSelectionResult:
    """
    fits one candidate model per point in `search_space`, evaluates each on
    `train_data.predict_idx`, and selects the best according to `metric`
    and `direction`

    ties are broken deterministically: `generate_candidates` produces a
    fixed iteration order, and the first candidate reaching the best value
    wins
    """
    y_validation = train_data.y[train_data.predict_idx]

    all_candidates = generate_candidates(search_space)
    logger.info(
        "model_selection start model=%s candidates=%d metric=%s direction=%s",
        model_name,
        len(all_candidates),
        metric,
        direction.value,
    )

    candidates: list[CandidateResult] = []

    for index, candidate_params in enumerate(all_candidates, start=1):
        parameters = {**fixed_parameters, **candidate_params}

        model = create_model(model_name, parameters)
        model.fit(train_data, seed=seed)
        scores = model.predict_scores(train_data)

        metrics = compute_binary_metrics(y_validation, scores, threshold=threshold)
        metric_value = getattr(metrics, metric)

        logger.info(
            "model_selection candidate %d/%d parameters=%s %s=%.4f",
            index,
            len(all_candidates),
            candidate_params,
            metric,
            metric_value,
        )

        candidates.append(
            CandidateResult(parameters=candidate_params, metric_value=metric_value)
        )

    best = _pick_best(candidates, direction)
    logger.info(
        "model_selection complete model=%s best_parameters=%s best_%s=%.4f",
        model_name,
        best.parameters,
        metric,
        best.metric_value,
    )

    return ModelSelectionResult(
        best_parameters=best.parameters,
        best_metric_value=best.metric_value,
        candidates=tuple(candidates),
    )


def _pick_best(
    candidates: list[CandidateResult],
    direction: SelectionDirection,
) -> CandidateResult:
    if direction == SelectionDirection.MAXIMIZE:
        return max(candidates, key=lambda candidate: candidate.metric_value)

    if direction == SelectionDirection.MINIMIZE:
        return min(candidates, key=lambda candidate: candidate.metric_value)

    raise ValueError(f"unsupported selection direction: {direction}")