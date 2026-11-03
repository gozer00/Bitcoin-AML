import numpy as np
import pytest

from bitcoin_aml.models.base import ModelInput
from bitcoin_aml.models.graph.evolvegcn_o import EvolveGCNOAdapter
from bitcoin_aml.utils.seeding import set_global_seed


def _separable_model_input() -> ModelInput:
    """
    two timesteps: nodes 0-2 (timestep 1, licit neighborhood 0,1 -> 2) and
    nodes 3-5 (timestep 2, illicit neighborhood 3,4 -> 5) -- same shape as
    gcn/graphsage's toy fixture, but split across two timesteps so the
    recurrence actually has more than one step to evolve through.
    """
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
    timestep = np.array([1, 1, 1, 2, 2, 2], dtype=np.int64)

    fit_idx = np.array([0, 1, 3, 4], dtype=np.int64)
    predict_idx = np.array([2, 5], dtype=np.int64)

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
        timestep=timestep,
    )


_HYPERPARAMETERS = dict(
    num_layers=2, hidden_dim=4, dropout=0.0, epochs=30, learning_rate=0.05, weight_decay=0.0
)


def test_capabilities_require_graph() -> None:
    capabilities = EvolveGCNOAdapter.capabilities

    assert capabilities.requires_graph is True
    assert capabilities.requires_directed_graph is False
    assert capabilities.support_inductive_inference is True


def test_fit_and_predict_scores_shape_and_range() -> None:
    set_global_seed(0)
    data = _separable_model_input()
    model = EvolveGCNOAdapter(**_HYPERPARAMETERS)
    model.fit(data, seed=0)

    scores = model.predict_scores(data)

    assert scores.shape == (len(data.predict_idx),)
    assert np.all((scores >= 0.0) & (scores <= 1.0))


def test_predict_scores_ranks_illicit_class_higher() -> None:
    set_global_seed(0)
    data = _separable_model_input()
    model = EvolveGCNOAdapter(**_HYPERPARAMETERS)
    model.fit(data, seed=0)

    scores = model.predict_scores(data)

    # predict_idx = [2 (licit, label 0, timestep 1), 5 (illicit, label 1, timestep 2)]
    assert scores[1] > scores[0]


def test_predict_scores_before_fit_raises() -> None:
    model = EvolveGCNOAdapter()
    data = _separable_model_input()

    with pytest.raises(RuntimeError):
        model.predict_scores(data)


def test_fit_without_timestep_raises() -> None:
    data = _separable_model_input()
    data_no_timestep = ModelInput(
        x=data.x,
        y=data.y,
        fit_idx=data.fit_idx,
        predict_idx=data.predict_idx,
        edge_index=data.edge_index,
    )
    model = EvolveGCNOAdapter(**_HYPERPARAMETERS)

    with pytest.raises(ValueError):
        model.fit(data_no_timestep, seed=0)


def test_fit_without_edge_index_raises() -> None:
    data = _separable_model_input()
    data_no_graph = ModelInput(
        x=data.x,
        y=data.y,
        fit_idx=data.fit_idx,
        predict_idx=data.predict_idx,
        timestep=data.timestep,
    )
    model = EvolveGCNOAdapter(**_HYPERPARAMETERS)

    with pytest.raises(ValueError):
        model.fit(data_no_graph, seed=0)


def test_unsupported_hyperparameter_raises() -> None:
    data = _separable_model_input()
    model = EvolveGCNOAdapter(**_HYPERPARAMETERS, not_a_real_hyperparameter=1)

    with pytest.raises(ValueError):
        model.fit(data, seed=0)
