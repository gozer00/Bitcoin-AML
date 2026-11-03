import numpy as np
import pytest

from bitcoin_aml.models.base import ModelInput
from bitcoin_aml.models.graph.gcn import GCNAdapter
from bitcoin_aml.utils.seeding import set_global_seed


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


def test_capabilities_require_graph() -> None:
    capabilities = GCNAdapter.capabilities

    assert capabilities.requires_graph is True
    assert capabilities.requires_directed_graph is False
    assert capabilities.support_inductive_inference is True


def test_fit_and_predict_scores_shape_and_range() -> None:
    set_global_seed(0)
    data = _separable_model_input()
    model = GCNAdapter(
        num_layers=2, hidden_dim=4, dropout=0.0, epochs=100, learning_rate=0.05, weight_decay=0.0
    )
    model.fit(data, seed=0)

    scores = model.predict_scores(data)

    assert scores.shape == (len(data.predict_idx),)
    assert np.all((scores >= 0.0) & (scores <= 1.0))


def test_predict_scores_ranks_illicit_class_higher() -> None:
    set_global_seed(0)
    data = _separable_model_input()
    model = GCNAdapter(
        num_layers=2, hidden_dim=4, dropout=0.0, epochs=100, learning_rate=0.05, weight_decay=0.0
    )
    model.fit(data, seed=0)

    scores = model.predict_scores(data)

    # predict_idx = [2 (licit, label 0), 5 (illicit, label 1)]
    assert scores[1] > scores[0]


def test_predict_scores_before_fit_raises() -> None:
    model = GCNAdapter()
    data = _separable_model_input()

    with pytest.raises(RuntimeError):
        model.predict_scores(data)
