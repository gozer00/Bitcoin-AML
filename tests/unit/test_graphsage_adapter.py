import numpy as np
import pytest

from bitcoin_aml.models.base import ModelInput
from bitcoin_aml.models.graph.graphsage import (
    GraphSAGEFullBatchAdapter,
    GraphSAGESampledAdapter,
)
from bitcoin_aml.utils.seeding import set_global_seed

_ADAPTERS = pytest.mark.parametrize(
    "adapter_cls, extra_hyperparameters",
    [
        (GraphSAGEFullBatchAdapter, {}),
        (GraphSAGESampledAdapter, {"fanout": [3, 3]}),
    ],
    ids=["full_batch", "sampled"],
)


def _separable_model_input() -> ModelInput:
    x = np.array(
        [
            [0.0],
            [0.1],
            [0.2],
            [5.0],
            [5.1],
            [5.2],
        ],
        dtype=np.float32,
    )
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.float32)

    fit_idx = np.array([0, 1, 3, 4], dtype=np.int64)
    predict_idx = np.array([2, 5], dtype=np.int64)

    # licit neighborhood: 0 -> 2, 1 -> 2 ; illicit neighborhood: 3 -> 5, 4 -> 5
    edge_index = np.array(
        [
            [0, 1, 3, 4],
            [2, 2, 5, 5],
        ],
        dtype=np.int64,
    )

    return ModelInput(
        x=x,
        y=y,
        fit_idx=fit_idx,
        predict_idx=predict_idx,
        edge_index=edge_index,
    )


@_ADAPTERS
def test_capabilities_require_graph(adapter_cls, extra_hyperparameters) -> None:
    capabilities = adapter_cls.capabilities

    assert capabilities.requires_graph is True
    assert capabilities.requires_directed_graph is False
    assert capabilities.support_inductive_inference is True


@_ADAPTERS
def test_fit_and_predict_scores_shape_and_range(adapter_cls, extra_hyperparameters) -> None:
    set_global_seed(0)
    data = _separable_model_input()
    model = adapter_cls(
        num_layers=2,
        hidden_dim=4,
        dropout=0.0,
        epochs=100,
        learning_rate=0.05,
        weight_decay=0.0,
        **extra_hyperparameters,
    )
    model.fit(data, seed=0)

    scores = model.predict_scores(data)

    assert scores.shape == (len(data.predict_idx),)
    assert np.all((scores >= 0.0) & (scores <= 1.0))


@_ADAPTERS
def test_predict_scores_ranks_illicit_class_higher(adapter_cls, extra_hyperparameters) -> None:
    set_global_seed(0)
    data = _separable_model_input()
    model = adapter_cls(
        num_layers=2,
        hidden_dim=4,
        dropout=0.0,
        epochs=100,
        learning_rate=0.05,
        weight_decay=0.0,
        **extra_hyperparameters,
    )
    model.fit(data, seed=0)

    scores = model.predict_scores(data)

    # predict_idx = [2 (licit, label 0), 5 (illicit, label 1)]
    assert scores[1] > scores[0]


@_ADAPTERS
def test_predict_scores_before_fit_raises(adapter_cls, extra_hyperparameters) -> None:
    model = adapter_cls(**extra_hyperparameters)
    data = _separable_model_input()

    with pytest.raises(RuntimeError):
        model.predict_scores(data)


def test_sampled_requires_fanout() -> None:
    model = GraphSAGESampledAdapter(num_layers=2, hidden_dim=4, epochs=1)
    data = _separable_model_input()

    with pytest.raises(ValueError, match="fanout"):
        model.fit(data, seed=0)


def test_sampled_fanout_length_must_match_num_layers() -> None:
    model = GraphSAGESampledAdapter(num_layers=2, hidden_dim=4, epochs=1, fanout=[3, 3, 3])
    data = _separable_model_input()

    with pytest.raises(ValueError, match="num_layers"):
        model.fit(data, seed=0)


def test_full_batch_rejects_fanout() -> None:
    model = GraphSAGEFullBatchAdapter(num_layers=2, hidden_dim=4, epochs=1, fanout=[3, 3])
    data = _separable_model_input()

    with pytest.raises(ValueError, match="fanout"):
        model.fit(data, seed=0)
