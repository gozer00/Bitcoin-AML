import numpy as np
import pytest

from bitcoin_aml.data.dataset import EllipticDataset
from bitcoin_aml.graphs.predecessors import (
    PredecessorBatch,
    build_incoming_csr,
    build_predecessor_batch,
    extract_predecessor_neighborhood,
)


def test_incoming_csr(
    predecessor_elliptic_dataset: EllipticDataset,
) -> None:
    csr = build_incoming_csr(
        predecessor_elliptic_dataset
    )

    np.testing.assert_array_equal(
        csr.predecessors(3),
        np.array(
            [0, 1],
            dtype=np.int64,
        ),
    )

    np.testing.assert_array_equal(
        csr.predecessors(4),
        np.array(
            [2, 3],
            dtype=np.int64,
        ),
    )

    np.testing.assert_array_equal(
        csr.predecessors(5),
        np.array(
            [4],
            dtype=np.int64,
        ),
    )


def test_one_hop_predecessors(
    predecessor_elliptic_dataset: EllipticDataset,
) -> None:
    csr = build_incoming_csr(
        predecessor_elliptic_dataset
    )

    neighborhood = (
        extract_predecessor_neighborhood(
            csr,
            target_node=4,
            hops=1,
        )
    )

    np.testing.assert_array_equal(
        neighborhood.nodes_by_hop[0],
        np.array(
            [2, 3],
            dtype=np.int64,
        ),
    )


def test_two_hop_predecessors(
    predecessor_elliptic_dataset: EllipticDataset,
) -> None:
    csr = build_incoming_csr(
        predecessor_elliptic_dataset
    )

    neighborhood = (
        extract_predecessor_neighborhood(
            csr,
            target_node=4,
            hops=2,
        )
    )

    np.testing.assert_array_equal(
        neighborhood.nodes_by_hop[0],
        np.array(
            [2, 3],
            dtype=np.int64,
        ),
    )

    np.testing.assert_array_equal(
        neighborhood.nodes_by_hop[1],
        np.array(
            [0, 1],
            dtype=np.int64,
        ),
    )

    np.testing.assert_array_equal(
        neighborhood.all_predecessors,
        np.array(
            [2, 3, 0, 1],
            dtype=np.int64,
        ),
    )


def test_predecessors_use_shortest_hop_only(
    predecessor_elliptic_dataset: EllipticDataset,
) -> None:
    csr = build_incoming_csr(
        predecessor_elliptic_dataset
    )

    neighborhood = (
        extract_predecessor_neighborhood(
            csr,
            target_node=5,
            hops=3,
        )
    )

    assert len(
        np.unique(
            neighborhood.all_predecessors
        )
    ) == len(
        neighborhood.all_predecessors
    )


def test_allowed_mask_is_enforced(
    predecessor_elliptic_dataset: EllipticDataset,
) -> None:
    dataset = predecessor_elliptic_dataset

    csr = build_incoming_csr(
        dataset
    )

    allowed_mask = np.ones(
        dataset.num_nodes,
        dtype=bool,
    )

    # Disallow predecessor node 3.
    allowed_mask[3] = False

    neighborhood = (
        extract_predecessor_neighborhood(
            csr,
            target_node=4,
            hops=2,
            allowed_mask=allowed_mask,
        )
    )

    np.testing.assert_array_equal(
        neighborhood.nodes_by_hop[0],
        np.array(
            [2],
            dtype=np.int64,
        ),
    )

    # Since traversal cannot pass through inaccessible node 3,
    # its predecessors 0 and 1 cannot become reachable either.
    assert len(
        neighborhood.nodes_by_hop[1]
    ) == 0


def test_inaccessible_target_is_rejected(
    predecessor_elliptic_dataset: EllipticDataset,
) -> None:
    dataset = predecessor_elliptic_dataset

    csr = build_incoming_csr(
        dataset
    )

    allowed_mask = np.ones(
        dataset.num_nodes,
        dtype=bool,
    )

    allowed_mask[4] = False

    with pytest.raises(
        ValueError,
        match="Target node is not permitted",
    ):
        extract_predecessor_neighborhood(
            csr,
            target_node=4,
            hops=2,
            allowed_mask=allowed_mask,
        )


def test_batch_single_target_matches_single_extraction(
    predecessor_elliptic_dataset: EllipticDataset,
) -> None:
    csr = build_incoming_csr(
        predecessor_elliptic_dataset
    )

    neighborhood = (
        extract_predecessor_neighborhood(
            csr,
            target_node=4,
            hops=2,
        )
    )

    batch = build_predecessor_batch(
        csr,
        target_nodes=np.array(
            [4],
            dtype=np.int64,
        ),
        hops=2,
    )

    expected_global_node_ids = np.unique(
        np.concatenate(
            [
                np.array(
                    [4],
                    dtype=np.int64,
                ),
                neighborhood.all_predecessors,
            ]
        )
    )

    np.testing.assert_array_equal(
        batch.global_node_ids,
        expected_global_node_ids,
    )

    np.testing.assert_array_equal(
        batch.target_nodes,
        np.array(
            [4],
            dtype=np.int64,
        ),
    )

    assert (
        batch.global_node_ids[batch.target_local_ids[0]]
        == 4
    )


def test_batch_overlapping_neighborhoods_dedupes_union(
    multi_target_elliptic_dataset: EllipticDataset,
) -> None:
    csr = build_incoming_csr(
        multi_target_elliptic_dataset
    )

    batch = build_predecessor_batch(
        csr,
        target_nodes=np.array(
            [4, 7],
            dtype=np.int64,
        ),
        hops=2,
    )

    # Node 2 is a shared 1-hop predecessor of both targets.
    assert (
        np.count_nonzero(batch.global_node_ids == 2)
        == 1
    )

    local_2 = int(
        np.searchsorted(
            batch.global_node_ids,
            2,
        )
    )

    local_4 = int(
        batch.target_local_ids[
            list(batch.target_nodes).index(4)
        ]
    )

    local_7 = int(
        batch.target_local_ids[
            list(batch.target_nodes).index(7)
        ]
    )

    edges = set(
        zip(
            batch.local_edge_index[0].tolist(),
            batch.local_edge_index[1].tolist(),
        )
    )

    # Both branches out of the shared ancestor are present.
    assert (local_2, local_4) in edges
    assert (local_2, local_7) in edges


def test_batch_disjoint_targets_produces_two_components(
    multi_target_elliptic_dataset: EllipticDataset,
) -> None:
    csr = build_incoming_csr(
        multi_target_elliptic_dataset
    )

    batch = build_predecessor_batch(
        csr,
        target_nodes=np.array(
            [7, 5],
            dtype=np.int64,
        ),
        hops=2,
    )

    # Target 7's component reaches nodes {0, 1, 2, 7};
    # target 5's component reaches nodes {3, 5}. Neither
    # component contains node 4 or 6, which only connect to target 4.
    component_a = {0, 1, 2, 7}
    component_b = {3, 5}

    assert set(batch.global_node_ids.tolist()) == (
        component_a | component_b
    )

    def component_of(global_id: int) -> set[int]:
        if global_id in component_a:
            return component_a
        return component_b

    for source_local, target_local in zip(
        batch.local_edge_index[0].tolist(),
        batch.local_edge_index[1].tolist(),
    ):
        source_global = int(
            batch.global_node_ids[source_local]
        )

        target_global = int(
            batch.global_node_ids[target_local]
        )

        assert component_of(source_global) == component_of(
            target_global
        )


def test_batch_determinism(
    multi_target_elliptic_dataset: EllipticDataset,
) -> None:
    csr = build_incoming_csr(
        multi_target_elliptic_dataset
    )

    batch_a = build_predecessor_batch(
        csr,
        target_nodes=np.array(
            [4, 7],
            dtype=np.int64,
        ),
        hops=2,
    )

    batch_b = build_predecessor_batch(
        csr,
        target_nodes=np.array(
            [4, 7],
            dtype=np.int64,
        ),
        hops=2,
    )

    np.testing.assert_array_equal(
        batch_a.target_nodes,
        batch_b.target_nodes,
    )

    np.testing.assert_array_equal(
        batch_a.global_node_ids,
        batch_b.global_node_ids,
    )

    np.testing.assert_array_equal(
        batch_a.local_edge_index,
        batch_b.local_edge_index,
    )

    np.testing.assert_array_equal(
        batch_a.target_local_ids,
        batch_b.target_local_ids,
    )

    # Shuffled, duplicated input targets must collapse to the same
    # sorted-unique result.
    batch_shuffled = build_predecessor_batch(
        csr,
        target_nodes=np.array(
            [7, 4, 7],
            dtype=np.int64,
        ),
        hops=2,
    )

    np.testing.assert_array_equal(
        batch_shuffled.target_nodes,
        batch_a.target_nodes,
    )

    np.testing.assert_array_equal(
        batch_shuffled.global_node_ids,
        batch_a.global_node_ids,
    )

    np.testing.assert_array_equal(
        batch_shuffled.local_edge_index,
        batch_a.local_edge_index,
    )

    np.testing.assert_array_equal(
        batch_shuffled.target_local_ids,
        batch_a.target_local_ids,
    )


def test_batch_allowed_mask_enforced_across_whole_batch(
    predecessor_elliptic_dataset: EllipticDataset,
) -> None:
    dataset = predecessor_elliptic_dataset

    csr = build_incoming_csr(
        dataset
    )

    allowed_mask = np.ones(
        dataset.num_nodes,
        dtype=bool,
    )

    # Disallow node 3, which is the only path from targets 4/5 to
    # nodes 0 and 1.
    allowed_mask[3] = False

    batch = build_predecessor_batch(
        csr,
        target_nodes=np.array(
            [4, 5],
            dtype=np.int64,
        ),
        hops=2,
        allowed_mask=allowed_mask,
    )

    # Node 3 itself, and nodes 0/1 which were only reachable through
    # it, are excluded from the whole batch -- not just from one
    # target's contribution.
    np.testing.assert_array_equal(
        batch.global_node_ids,
        np.array(
            [2, 4, 5],
            dtype=np.int64,
        ),
    )


def test_batch_rejects_inaccessible_target(
    predecessor_elliptic_dataset: EllipticDataset,
) -> None:
    dataset = predecessor_elliptic_dataset

    csr = build_incoming_csr(
        dataset
    )

    allowed_mask = np.ones(
        dataset.num_nodes,
        dtype=bool,
    )

    allowed_mask[4] = False

    with pytest.raises(
        ValueError,
        match="Target node is not permitted",
    ):
        build_predecessor_batch(
            csr,
            target_nodes=np.array(
                [2, 4],
                dtype=np.int64,
            ),
            hops=2,
            allowed_mask=allowed_mask,
        )


def test_batch_rejects_empty_target_nodes(
    predecessor_elliptic_dataset: EllipticDataset,
) -> None:
    csr = build_incoming_csr(
        predecessor_elliptic_dataset
    )

    with pytest.raises(
        ValueError,
        match="must contain at least one target",
    ):
        build_predecessor_batch(
            csr,
            target_nodes=np.array(
                [],
                dtype=np.int64,
            ),
            hops=1,
        )


def test_batch_rejects_out_of_bounds_target(
    predecessor_elliptic_dataset: EllipticDataset,
) -> None:
    csr = build_incoming_csr(
        predecessor_elliptic_dataset
    )

    with pytest.raises(IndexError):
        build_predecessor_batch(
            csr,
            target_nodes=np.array(
                [csr.num_nodes],
                dtype=np.int64,
            ),
            hops=1,
        )


def test_predecessor_batch_post_init_rejects_unsorted_global_node_ids() -> None:
    with pytest.raises(
        ValueError,
        match="sorted in ascending order",
    ):
        PredecessorBatch(
            target_nodes=np.array(
                [0],
                dtype=np.int64,
            ),
            global_node_ids=np.array(
                [1, 0],
                dtype=np.int64,
            ),
            local_edge_index=np.empty(
                (2, 0),
                dtype=np.int64,
            ),
            target_local_ids=np.array(
                [1],
                dtype=np.int64,
            ),
        )


def test_predecessor_batch_post_init_rejects_duplicate_global_node_ids() -> None:
    with pytest.raises(
        ValueError,
        match="duplicate nodes",
    ):
        PredecessorBatch(
            target_nodes=np.array(
                [0],
                dtype=np.int64,
            ),
            global_node_ids=np.array(
                [0, 0],
                dtype=np.int64,
            ),
            local_edge_index=np.empty(
                (2, 0),
                dtype=np.int64,
            ),
            target_local_ids=np.array(
                [0],
                dtype=np.int64,
            ),
        )


def test_predecessor_batch_post_init_rejects_bad_dtype() -> None:
    with pytest.raises(TypeError):
        PredecessorBatch(
            target_nodes=np.array(
                [0],
                dtype=np.int32,
            ),
            global_node_ids=np.array(
                [0],
                dtype=np.int64,
            ),
            local_edge_index=np.empty(
                (2, 0),
                dtype=np.int64,
            ),
            target_local_ids=np.array(
                [0],
                dtype=np.int64,
            ),
        )


def test_predecessor_batch_post_init_rejects_mismatched_target_local_ids() -> None:
    with pytest.raises(
        ValueError,
        match="does not match target_nodes",
    ):
        PredecessorBatch(
            target_nodes=np.array(
                [0],
                dtype=np.int64,
            ),
            global_node_ids=np.array(
                [0, 1],
                dtype=np.int64,
            ),
            local_edge_index=np.empty(
                (2, 0),
                dtype=np.int64,
            ),
            target_local_ids=np.array(
                [1],
                dtype=np.int64,
            ),
        )