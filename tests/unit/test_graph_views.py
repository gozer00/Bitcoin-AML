import numpy as np

from bitcoin_aml.data.dataset import EllipticDataset
from bitcoin_aml.graphs.specification import (
    GraphSpec,
    GraphViewName,
)
from bitcoin_aml.graphs.views import (
    edge_index_for_view,
    make_bidirectional,
)


def test_directed_view_preserves_canonical_edges(
    synthetic_elliptic_dataset: EllipticDataset,
) -> None:
    dataset = synthetic_elliptic_dataset

    spec = GraphSpec(
        view=GraphViewName.DIRECTED,
    )

    edge_index = edge_index_for_view(
        dataset,
        spec,
    )

    assert edge_index is dataset.edge_index


def test_none_view_contains_no_edges(
    synthetic_elliptic_dataset: EllipticDataset,
) -> None:
    dataset = synthetic_elliptic_dataset

    spec = GraphSpec(
        view=GraphViewName.NONE,
    )

    edge_index = edge_index_for_view(
        dataset,
        spec,
    )

    assert edge_index.shape == (2, 0)


def test_incoming_base_view_preserves_direction(
    synthetic_elliptic_dataset: EllipticDataset,
) -> None:
    dataset = synthetic_elliptic_dataset

    spec = GraphSpec(
        view=GraphViewName.INCOMING,
        hops=2,
    )

    edge_index = edge_index_for_view(
        dataset,
        spec,
    )

    assert edge_index is dataset.edge_index


def test_reversed_view_swaps_source_and_target(
    synthetic_elliptic_dataset: EllipticDataset,
) -> None:
    dataset = synthetic_elliptic_dataset

    spec = GraphSpec(
        view=GraphViewName.REVERSED,
    )

    edge_index = edge_index_for_view(
        dataset,
        spec,
    )

    np.testing.assert_array_equal(
        edge_index[0],
        dataset.edge_index[1],
    )

    np.testing.assert_array_equal(
        edge_index[1],
        dataset.edge_index[0],
    )


def test_bidirectional_view_adds_reverse_edges(
    synthetic_elliptic_dataset: EllipticDataset,
) -> None:
    dataset = synthetic_elliptic_dataset

    bidirectional = make_bidirectional(
        dataset.edge_index,
        num_nodes=dataset.num_nodes,
    )

    canonical_pairs = {
        tuple(pair)
        for pair in dataset.edge_index.T
    }

    bidirectional_pairs = {
        tuple(pair)
        for pair in bidirectional.T
    }

    for source, target in canonical_pairs:
        assert (
            source,
            target,
        ) in bidirectional_pairs

        assert (
            target,
            source,
        ) in bidirectional_pairs


def test_bidirectional_view_has_no_duplicate_edges() -> None:
    edge_index = np.array(
        [
            [0, 1],
            [1, 0],
        ],
        dtype=np.int64,
    )

    bidirectional = make_bidirectional(
        edge_index,
        num_nodes=2,
    )

    pairs = bidirectional.T

    assert len(pairs) == 2
    assert len(np.unique(
        pairs,
        axis=0,
    )) == 2