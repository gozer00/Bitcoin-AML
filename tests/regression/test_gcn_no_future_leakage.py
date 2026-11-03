import numpy as np

from bitcoin_aml.models.base import ModelInput
from bitcoin_aml.models.graph.gcn import GCNAdapter
from bitcoin_aml.utils.seeding import set_global_seed

_HYPERPARAMETERS = dict(
    num_layers=2,
    hidden_dim=4,
    dropout=0.0,
    epochs=20,
    learning_rate=0.05,
    weight_decay=0.0,
)


def _base_model_input() -> ModelInput:
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

    return ModelInput(
        x=x,
        y=y,
        fit_idx=fit_idx,
        predict_idx=predict_idx,
        edge_index=edge_index,
    )


def _model_input_with_disconnected_future_nodes() -> ModelInput:
    """
    same shared nodes (0-5) and edges as `_base_model_input`, plus two extra
    "future timestep" nodes (6, 7) that are structurally disconnected from
    every shared node -- mirroring the Elliptic dataset's no-cross-timestep-
    edges invariant (docs/data_contract.md)
    """
    base = _base_model_input()

    future_x = np.array([[-9.0], [9.0]], dtype=np.float32)
    future_y = np.array([0, 1], dtype=np.float32)

    x = np.concatenate([base.x, future_x], axis=0)
    y = np.concatenate([base.y, future_y], axis=0)

    # future nodes only connect to each other, never to the shared 0-5 nodes
    future_edges = np.array([[6], [7]], dtype=np.int64)
    edge_index = np.concatenate([base.edge_index, future_edges], axis=1)

    return ModelInput(
        x=x,
        y=y,
        fit_idx=base.fit_idx,
        predict_idx=base.predict_idx,
        edge_index=edge_index,
    )


def test_predictions_for_shared_nodes_are_unaffected_by_future_nodes() -> None:
    set_global_seed(7)
    model_a = GCNAdapter(**_HYPERPARAMETERS)
    model_a.fit(_base_model_input(), seed=7)
    scores_a = model_a.predict_scores(_base_model_input())

    set_global_seed(7)
    model_b = GCNAdapter(**_HYPERPARAMETERS)
    model_b.fit(_model_input_with_disconnected_future_nodes(), seed=7)
    scores_b = model_b.predict_scores(_model_input_with_disconnected_future_nodes())

    np.testing.assert_allclose(scores_a, scores_b, rtol=1e-5, atol=1e-6)
