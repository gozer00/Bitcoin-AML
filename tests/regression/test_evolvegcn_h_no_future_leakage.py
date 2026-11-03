import numpy as np

from bitcoin_aml.models.base import ModelInput
from bitcoin_aml.models.graph.evolvegcn_h import EvolveGCNHAdapter
from bitcoin_aml.utils.seeding import set_global_seed

_HYPERPARAMETERS = dict(
    num_layers=2,
    hidden_dim=4,
    dropout=0.0,
    epochs=20,
    learning_rate=0.05,
    weight_decay=0.0,
)

_FIT_IDX = np.array([0, 1, 3, 4], dtype=np.int64)
_PREDICT_IDX = np.array([2, 5, 8], dtype=np.int64)


def _shared_x() -> np.ndarray:
    return np.array(
        [
            [0.0],
            [0.1],
            [0.2],  # timestep 1: licit neighborhood 0, 1 -> 2
            [5.0],
            [5.1],
            [5.2],  # timestep 2: illicit neighborhood 3, 4 -> 5
        ],
        dtype=np.float32,
    )


def _model_input_with_future_timestep(*, future_x: np.ndarray) -> ModelInput:
    """
    shared nodes 0-5 (timesteps 1-2) plus a "future" timestep-3 neighborhood
    (6, 7 -> 8) that is part of predict_idx (so predict_scores actually
    processes it) but never part of fit_idx (so training never touches it).
    Mirrors Elliptic's zero-cross-timestep-edges invariant: nodes 6-8 only
    connect to each other, never to the shared nodes.
    """
    x = np.concatenate([_shared_x(), future_x], axis=0)
    y = np.concatenate(
        [np.array([0, 0, 0, 1, 1, 1], dtype=np.float32), np.array([0, 0, 0], dtype=np.float32)]
    )
    timestep = np.concatenate(
        [np.array([1, 1, 1, 2, 2, 2], dtype=np.int64), np.array([3, 3, 3], dtype=np.int64)]
    )
    future_edges = np.array([[6, 7], [8, 8]], dtype=np.int64)
    edge_index = np.concatenate(
        [np.array([[0, 1, 3, 4], [2, 2, 5, 5]], dtype=np.int64), future_edges], axis=1
    )

    return ModelInput(
        x=x,
        y=y,
        fit_idx=_FIT_IDX,
        predict_idx=_PREDICT_IDX,
        edge_index=edge_index,
        timestep=timestep,
    )


def test_predictions_at_earlier_timesteps_are_unaffected_by_future_timestep_perturbation() -> None:
    """
    the future timestep-3 neighborhood (nodes 6-8) is genuinely processed by
    predict_scores -- it's part of predict_idx, so t_end reaches timestep 3
    -- but is never part of fit_idx, so training (and therefore the learned
    parameters) never touches it. Perturbing its feature values between
    models A and B must not change the predict-time scores at the shared,
    earlier-timestep nodes 2 (timestep 1) and 5 (timestep 2): the recurrence
    only ever evolves forward, and each timestep's node embeddings (the
    GRU's input) are themselves computed purely from that timestep's own
    local subgraph, so a later timestep's data cannot feed back into an
    already-computed earlier timestep's weights or logits (see
    evolvegcn_h.py's `_run_sequence`, which computes and stores each
    timestep's logits before moving on to the next).
    """
    future_x_a = np.array([[1.0], [1.1], [1.2]], dtype=np.float32)
    future_x_b = np.array([[-40.0], [83.0], [-6.0]], dtype=np.float32)

    set_global_seed(11)
    model_a = EvolveGCNHAdapter(**_HYPERPARAMETERS)
    data_a = _model_input_with_future_timestep(future_x=future_x_a)
    model_a.fit(data_a, seed=11)
    scores_a = model_a.predict_scores(data_a)

    set_global_seed(11)
    model_b = EvolveGCNHAdapter(**_HYPERPARAMETERS)
    data_b = _model_input_with_future_timestep(future_x=future_x_b)
    model_b.fit(data_b, seed=11)
    scores_b = model_b.predict_scores(data_b)

    # predict_idx = [2 (t=1, shared), 5 (t=2, shared), 8 (t=3, perturbed)]
    np.testing.assert_allclose(scores_a[:2], scores_b[:2], rtol=1e-5, atol=1e-6)

    # sanity check: the perturbation is not vacuous -- the perturbed
    # timestep's own score is allowed, and expected, to differ
    assert not np.allclose(scores_a[2], scores_b[2])
