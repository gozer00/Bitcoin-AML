import numpy as np
import pytest

from bitcoin_aml.models.base import ModelInput
from bitcoin_aml.models.graph.predecessor_gnn import (
    PredecessorGNNOneHopAdapter,
    PredecessorGNNTwoHopAdapter,
)
from bitcoin_aml.utils.seeding import set_global_seed

_ADAPTERS = pytest.mark.parametrize(
    "adapter_cls",
    [PredecessorGNNOneHopAdapter, PredecessorGNNTwoHopAdapter],
    ids=["one_hop", "two_hop"],
)

_HYPERPARAMETERS = dict(
    hidden_dim=4,
    dropout=0.0,
    epochs=100,
    learning_rate=0.05,
    weight_decay=0.0,
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
def test_capabilities_require_directed_graph(adapter_cls) -> None:
    capabilities = adapter_cls.capabilities

    assert capabilities.requires_graph is True
    assert capabilities.requires_directed_graph is True
    assert capabilities.support_inductive_inference is True


@_ADAPTERS
def test_fit_and_predict_scores_shape_and_range(adapter_cls) -> None:
    set_global_seed(0)
    data = _separable_model_input()
    model = adapter_cls(**_HYPERPARAMETERS)
    model.fit(data, seed=0)

    scores = model.predict_scores(data)

    assert scores.shape == (len(data.predict_idx),)
    assert np.all((scores >= 0.0) & (scores <= 1.0))


@_ADAPTERS
def test_predict_scores_ranks_illicit_class_higher(adapter_cls) -> None:
    set_global_seed(0)
    data = _separable_model_input()
    model = adapter_cls(**_HYPERPARAMETERS)
    model.fit(data, seed=0)

    scores = model.predict_scores(data)

    # predict_idx = [2 (licit, label 0), 5 (illicit, label 1)]
    assert scores[1] > scores[0]


@_ADAPTERS
def test_predict_scores_before_fit_raises(adapter_cls) -> None:
    model = adapter_cls(**_HYPERPARAMETERS)
    data = _separable_model_input()

    with pytest.raises(RuntimeError):
        model.predict_scores(data)


@_ADAPTERS
def test_fit_without_edge_index_raises(adapter_cls) -> None:
    model = adapter_cls(**_HYPERPARAMETERS)
    data = _separable_model_input()
    data_without_graph = ModelInput(
        x=data.x,
        y=data.y,
        fit_idx=data.fit_idx,
        predict_idx=data.predict_idx,
        edge_index=None,
    )

    with pytest.raises(ValueError, match="requires a graph"):
        model.fit(data_without_graph, seed=0)


@_ADAPTERS
def test_predict_scores_without_edge_index_raises(adapter_cls) -> None:
    set_global_seed(0)
    data = _separable_model_input()
    model = adapter_cls(**_HYPERPARAMETERS)
    model.fit(data, seed=0)

    data_without_graph = ModelInput(
        x=data.x,
        y=data.y,
        fit_idx=data.fit_idx,
        predict_idx=data.predict_idx,
        edge_index=None,
    )

    with pytest.raises(ValueError, match="requires a graph"):
        model.predict_scores(data_without_graph)


@_ADAPTERS
def test_unsupported_hyperparameter_raises(adapter_cls) -> None:
    model = adapter_cls(epochs=1, bogus_key=1)
    data = _separable_model_input()

    with pytest.raises(ValueError, match="unsupported"):
        model.fit(data, seed=0)


@_ADAPTERS
@pytest.mark.parametrize("key", ["hops", "num_layers"])
def test_hops_is_not_overridable_via_hyperparameters(adapter_cls, key) -> None:
    model = adapter_cls(epochs=1, **{key: 5})
    data = _separable_model_input()

    with pytest.raises(ValueError, match="unsupported"):
        model.fit(data, seed=0)


def _hop_depth_model_input(*, p2_feature: float) -> ModelInput:
    """
    0 = T (target, predict-only)
    1 = P1 (1-hop predecessor of T)
    2 = P2 (2-hop predecessor of T, via P1 -- the perturbation target)
    3 = A (fit node, label 0)
    4 = A's 1-hop predecessor
    5 = B (fit node, label 1)
    6 = B's 1-hop predecessor

    edges (predecessor direction u -> v): 1->0, 2->1, 4->3, 6->5.

    Training only ever touches the {3, 4, 5, 6} component (neither fit_idx
    entry has T/P1/P2 within its 2-hop predecessor union), so the fit-time
    weights never depend on p2_feature -- isolating the structural claim
    (whether P2 is in T's receptive field at all) from anything learned.
    """
    x = np.array(
        [
            [0.0],
            [0.0],
            [p2_feature],
            [0.0],
            [0.1],
            [5.0],
            [5.1],
        ],
        dtype=np.float32,
    )
    y = np.array([0, 0, 0, 0, 0, 1, 1], dtype=np.float32)

    fit_idx = np.array([3, 5], dtype=np.int64)
    predict_idx = np.array([0], dtype=np.int64)

    edge_index = np.array(
        [
            [1, 2, 4, 6],
            [0, 1, 3, 5],
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


def test_hop_depth_controls_receptive_field() -> None:
    hyperparameters = dict(
        hidden_dim=4,
        dropout=0.0,
        epochs=100,
        learning_rate=0.05,
        weight_decay=0.0,
    )

    set_global_seed(3)
    two_hop_model = PredecessorGNNTwoHopAdapter(**hyperparameters)
    two_hop_model.fit(_hop_depth_model_input(p2_feature=-9.0), seed=3)
    scores_baseline_two_hop = two_hop_model.predict_scores(
        _hop_depth_model_input(p2_feature=-9.0)
    )
    scores_perturbed_two_hop = two_hop_model.predict_scores(
        _hop_depth_model_input(p2_feature=9.0)
    )

    assert abs(scores_perturbed_two_hop[0] - scores_baseline_two_hop[0]) > 1e-4

    set_global_seed(3)
    one_hop_model = PredecessorGNNOneHopAdapter(**hyperparameters)
    one_hop_model.fit(_hop_depth_model_input(p2_feature=-9.0), seed=3)
    scores_baseline_one_hop = one_hop_model.predict_scores(
        _hop_depth_model_input(p2_feature=-9.0)
    )
    scores_perturbed_one_hop = one_hop_model.predict_scores(
        _hop_depth_model_input(p2_feature=9.0)
    )

    np.testing.assert_array_equal(scores_baseline_one_hop, scores_perturbed_one_hop)
