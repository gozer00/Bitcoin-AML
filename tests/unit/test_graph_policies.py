import numpy as np
import pytest

from bitcoin_aml.data.dataset import EllipticDataset
from bitcoin_aml.graphs.policies import (
    build_graph_snapshot,
    resolve_phase_edge_index,
    validate_graph_snapshot,
)
from bitcoin_aml.graphs.specification import (
    GraphAccessPolicy,
    GraphSpec,
    GraphViewName,
)
from bitcoin_aml.graphs.views import (
    edge_index_for_view,
)


def test_strict_snapshot_excludes_future_nodes(
    synthetic_elliptic_dataset: EllipticDataset,
) -> None:
    dataset = synthetic_elliptic_dataset

    edge_index = edge_index_for_view(
        dataset,
        GraphSpec(
            view=GraphViewName.DIRECTED,
        ),
    )

    snapshot = build_graph_snapshot(
        dataset,
        edge_index,
        access_policy=GraphAccessPolicy.STRICT_ROLLING,
        max_timestep=34,
    )

    assert snapshot.num_nodes == 34 * 2
    assert snapshot.num_edges == 34

    assert np.all(
        dataset.timestep[
            snapshot.node_indices
        ] <= 34
    )

    assert np.all(
        dataset.timestep[
            snapshot.edge_index[0]
        ] <= 34
    )

    assert np.all(
        dataset.timestep[
            snapshot.edge_index[1]
        ] <= 34
    )


def test_strict_snapshot_is_validated(
    synthetic_elliptic_dataset: EllipticDataset,
) -> None:
    dataset = synthetic_elliptic_dataset

    snapshot = build_graph_snapshot(
        dataset,
        dataset.edge_index,
        access_policy=GraphAccessPolicy.STRICT_ROLLING,
        max_timestep=29,
    )

    validate_graph_snapshot(
        dataset,
        snapshot,
    )


def test_strict_snapshot_requires_timestep(
    synthetic_elliptic_dataset: EllipticDataset,
) -> None:
    with pytest.raises(
        ValueError,
        match="requires max_timestep",
    ):
        build_graph_snapshot(
            synthetic_elliptic_dataset,
            synthetic_elliptic_dataset.edge_index,
            access_policy=(
                GraphAccessPolicy.STRICT_ROLLING
            ),
        )


def test_full_transductive_contains_all_nodes(
    synthetic_elliptic_dataset: EllipticDataset,
) -> None:
    dataset = synthetic_elliptic_dataset

    snapshot = build_graph_snapshot(
        dataset,
        dataset.edge_index,
        access_policy=(
            GraphAccessPolicy.FULL_TRANSDUCTIVE
        ),
    )

    assert snapshot.num_nodes == dataset.num_nodes
    assert snapshot.num_edges == dataset.num_edges

    assert np.all(snapshot.node_mask)


def test_full_transductive_rejects_max_timestep(
    synthetic_elliptic_dataset: EllipticDataset,
) -> None:
    with pytest.raises(
        ValueError,
        match="does not accept max_timestep",
    ):
        build_graph_snapshot(
            synthetic_elliptic_dataset,
            synthetic_elliptic_dataset.edge_index,
            access_policy=(
                GraphAccessPolicy.FULL_TRANSDUCTIVE
            ),
            max_timestep=34,
        )


def test_resolve_phase_edge_index_strict_fit_masks_future_edges(
    synthetic_elliptic_dataset: EllipticDataset,
) -> None:
    dataset = synthetic_elliptic_dataset

    result = resolve_phase_edge_index(
        dataset,
        dataset.edge_index,
        access_policy=GraphAccessPolicy.STRICT_ROLLING,
        phase="fit",
        fit_cutoff_timestep=34,
    )

    # one edge per timestep in this fixture -- 34 timesteps admissible
    assert result.shape[1] == 34
    assert np.all(dataset.timestep[result[0]] <= 34)
    assert np.all(dataset.timestep[result[1]] <= 34)


def test_resolve_phase_edge_index_strict_eval_sees_full_graph(
    synthetic_elliptic_dataset: EllipticDataset,
) -> None:
    dataset = synthetic_elliptic_dataset

    result = resolve_phase_edge_index(
        dataset,
        dataset.edge_index,
        access_policy=GraphAccessPolicy.STRICT_ROLLING,
        phase="eval",
        fit_cutoff_timestep=34,
    )

    np.testing.assert_array_equal(result, dataset.edge_index)


def test_resolve_phase_edge_index_full_transductive_ignores_cutoff(
    synthetic_elliptic_dataset: EllipticDataset,
) -> None:
    dataset = synthetic_elliptic_dataset

    for phase in ("fit", "eval"):
        result = resolve_phase_edge_index(
            dataset,
            dataset.edge_index,
            access_policy=GraphAccessPolicy.FULL_TRANSDUCTIVE,
            phase=phase,
            fit_cutoff_timestep=1,
        )

        np.testing.assert_array_equal(result, dataset.edge_index)