import numpy as np
import pytest

from bitcoin_aml.graphs.views import make_bidirectional
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

_DIRECTIONS = pytest.mark.parametrize(
    "direction",
    ["directed", "reversed", "bidirectional"],
)

_HYPERPARAMETERS = dict(
    hidden_dim=4,
    dropout=0.0,
    epochs=20,
    learning_rate=0.05,
    weight_decay=0.0,
)


def _apply_direction(
    edge_index: np.ndarray, *, num_nodes: int, direction: str
) -> np.ndarray:
    """
    applies the same edge-index transform each ablation's graph.view
    applies (see graphs/views.py::edge_index_for_view) -- exercises the
    no-future-leakage guarantee under all three views this adapter can now
    be run against, not just the default directed/incoming one.
    """
    if direction == "directed":
        return edge_index

    if direction == "reversed":
        return np.stack([edge_index[1], edge_index[0]])

    if direction == "bidirectional":
        return make_bidirectional(edge_index, num_nodes=num_nodes)

    raise ValueError(f"unknown direction {direction!r}")


def _base_model_input(direction: str = "directed") -> ModelInput:
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

    edge_index = np.array(
        [
            [0, 1, 3, 4],
            [2, 2, 5, 5],
        ],
        dtype=np.int64,
    )

    edge_index = _apply_direction(
        edge_index, num_nodes=len(x), direction=direction
    )

    return ModelInput(
        x=x,
        y=y,
        fit_idx=fit_idx,
        predict_idx=predict_idx,
        edge_index=edge_index,
    )


def _model_input_with_disconnected_future_nodes(
    direction: str = "directed",
) -> ModelInput:
    """
    same shared nodes (0-5) and edges as `_base_model_input`, plus two extra
    "future timestep" nodes (6, 7) that are structurally disconnected from
    every shared node -- mirroring the Elliptic dataset's no-cross-timestep-
    edges invariant (docs/data_contract.md)
    """
    base_x = np.array(
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
    base_y = np.array([0, 0, 0, 1, 1, 1], dtype=np.float32)
    base_edge_index = np.array(
        [
            [0, 1, 3, 4],
            [2, 2, 5, 5],
        ],
        dtype=np.int64,
    )

    fit_idx = np.array([0, 1, 3, 4], dtype=np.int64)
    predict_idx = np.array([2, 5], dtype=np.int64)

    future_x = np.array([[-9.0], [9.0]], dtype=np.float32)
    future_y = np.array([0, 1], dtype=np.float32)

    x = np.concatenate([base_x, future_x], axis=0)
    y = np.concatenate([base_y, future_y], axis=0)

    # future nodes only connect to each other, never to the shared 0-5 nodes
    future_edges = np.array([[6], [7]], dtype=np.int64)
    edge_index = np.concatenate([base_edge_index, future_edges], axis=1)

    edge_index = _apply_direction(
        edge_index, num_nodes=len(x), direction=direction
    )

    return ModelInput(
        x=x,
        y=y,
        fit_idx=fit_idx,
        predict_idx=predict_idx,
        edge_index=edge_index,
    )


@_ADAPTERS
@_DIRECTIONS
def test_predictions_for_shared_nodes_are_unaffected_by_future_nodes(
    adapter_cls, direction
) -> None:
    set_global_seed(7)
    model_a = adapter_cls(**_HYPERPARAMETERS)
    model_a.fit(_base_model_input(direction), seed=7)
    scores_a = model_a.predict_scores(_base_model_input(direction))

    set_global_seed(7)
    model_b = adapter_cls(**_HYPERPARAMETERS)
    model_b.fit(_model_input_with_disconnected_future_nodes(direction), seed=7)
    scores_b = model_b.predict_scores(
        _model_input_with_disconnected_future_nodes(direction)
    )

    np.testing.assert_allclose(scores_a, scores_b, rtol=1e-5, atol=1e-6)
