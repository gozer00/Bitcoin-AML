from __future__ import annotations

import numpy as np
import pytest

from bitcoin_aml.experiments.model_selection import (
    generate_candidates,
    select_best_hyperparameters,
)
from bitcoin_aml.experiments.specification import SelectionDirection
from bitcoin_aml.models.base import ModelAdapter, ModelCapabilities, ModelInput
from bitcoin_aml.models.registry import MODEL_REGISTRY, register_model

_FAKE_MODEL_NAME = "test_only_fake_model"


def test_generate_candidates_empty_search_space() -> None:
    assert generate_candidates({}) == ({},)


def test_generate_candidates_cartesian_product_deterministic_order() -> None:
    search_space = {
        "b": (1, 2),
        "a": ("x", "y"),
    }

    candidates = generate_candidates(search_space)

    assert candidates == (
        {"a": "x", "b": 1},
        {"a": "x", "b": 2},
        {"a": "y", "b": 1},
        {"a": "y", "b": 2},
    )


if _FAKE_MODEL_NAME not in MODEL_REGISTRY:

    @register_model(_FAKE_MODEL_NAME)
    class _FakeAdapter(ModelAdapter):
        """
        deterministic test double: scores are an affine blend between the
        true labels (quality=1 -> perfect ranking) and their complement
        (quality=0 -> perfectly inverted ranking)
        """

        capabilities = ModelCapabilities(
            requires_graph=False,
            requires_directed_graph=False,
            support_inductive_inference=True,
        )

        def __init__(self, *, quality: float = 1.0) -> None:
            self._quality = quality

        def fit(self, data: ModelInput, *, seed: int) -> None:
            pass

        def predict_scores(self, data: ModelInput) -> np.ndarray:
            y = data.y[data.predict_idx].astype(np.float64)
            return self._quality * y + (1 - self._quality) * (1 - y)


def _model_input() -> ModelInput:
    x = np.zeros((4, 1), dtype=np.float32)
    y = np.array([0, 1, 0, 1], dtype=np.int64)

    fit_idx = np.array([0, 1], dtype=np.int64)
    predict_idx = np.array([2, 3], dtype=np.int64)

    return ModelInput(x=x, y=y, fit_idx=fit_idx, predict_idx=predict_idx)


def test_select_best_hyperparameters_picks_maximum() -> None:
    result = select_best_hyperparameters(
        model_name=_FAKE_MODEL_NAME,
        fixed_parameters={},
        search_space={"quality": (0.0, 1.0)},
        train_data=_model_input(),
        metric="average_precision",
        direction=SelectionDirection.MAXIMIZE,
        threshold=0.5,
        seed=0,
    )

    assert result.best_parameters == {"quality": 1.0}
    assert result.best_metric_value == pytest.approx(1.0)
    assert len(result.candidates) == 2


def test_select_best_hyperparameters_picks_minimum() -> None:
    result = select_best_hyperparameters(
        model_name=_FAKE_MODEL_NAME,
        fixed_parameters={},
        search_space={"quality": (0.0, 1.0)},
        train_data=_model_input(),
        metric="average_precision",
        direction=SelectionDirection.MINIMIZE,
        threshold=0.5,
        seed=0,
    )

    assert result.best_parameters == {"quality": 0.0}