from pathlib import Path

import numpy as np
import pytest

from bitcoin_aml.data.repository import EllipticRepository
from bitcoin_aml.data.schema import (
    ELLIPTIC_NUM_FEATURES,
    ELLIPTIC_TIMESTEPS,
)
from bitcoin_aml.graphs.policies import (
    build_graph_snapshot,
)
from bitcoin_aml.graphs.predecessors import (
    build_incoming_csr,
    build_predecessor_batch,
)
from bitcoin_aml.graphs.specification import (
    GraphAccessPolicy,
)


DATASET_DIR = Path("data/processed/elliptic_v1")


@pytest.mark.skipif(
    not DATASET_DIR.exists(),
    reason="Local processed Elliptic dataset is unavailable.",
)
def test_real_elliptic_dataset_contract() -> None:
    dataset = EllipticRepository.load(
        DATASET_DIR,
        verify_hashes=True,
    )

    assert dataset.num_nodes > 0
    assert dataset.num_edges > 0
    assert dataset.num_features == ELLIPTIC_NUM_FEATURES

    assert tuple(
        int(t) for t in np.unique(dataset.timestep)
    ) == ELLIPTIC_TIMESTEPS

    source = dataset.edge_index[0]
    target = dataset.edge_index[1]

    assert np.all(
        dataset.timestep[source]
        ==
        dataset.timestep[target]
    )

    assert set(
        int(value) for value in np.unique(dataset.labels)
    ).issubset({-1, 0, 1})


@pytest.mark.skipif(
    not DATASET_DIR.exists(),
    reason="Local processed Elliptic dataset is unavailable.",
)
def test_real_strict_graph_snapshot() -> None:
    dataset = EllipticRepository.load(
        DATASET_DIR,
        verify_hashes=True,
    )

    snapshot = build_graph_snapshot(
        dataset,
        dataset.edge_index,
        access_policy=(
            GraphAccessPolicy.STRICT_ROLLING
        ),
        max_timestep=34,
    )

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


@pytest.mark.skipif(
    not DATASET_DIR.exists(),
    reason="Local processed Elliptic dataset is unavailable.",
)
def test_real_incoming_csr_edge_count() -> None:
    dataset = EllipticRepository.load(
        DATASET_DIR,
        verify_hashes=True,
    )

    csr = build_incoming_csr(dataset)

    assert csr.num_nodes == dataset.num_nodes
    assert csr.num_edges == dataset.num_edges


@pytest.mark.skipif(
    not DATASET_DIR.exists(),
    reason="Local processed Elliptic dataset is unavailable.",
)
def test_real_predecessor_batch_shapes() -> None:
    dataset = EllipticRepository.load(
        DATASET_DIR,
        verify_hashes=True,
    )

    csr = build_incoming_csr(dataset)

    target_nodes = np.arange(
        200,
        dtype=np.int64,
    )

    batch = build_predecessor_batch(
        csr,
        target_nodes=target_nodes,
        hops=2,
    )

    assert 0 < batch.num_nodes <= dataset.num_nodes
    assert batch.num_edges >= 0

    assert np.all(
        np.isin(
            batch.target_nodes,
            batch.global_node_ids,
        )
    )

@pytest.mark.skipif(
    not DATASET_DIR.exists(),
    reason="Local processed Elliptic dataset is unavailable.",
)
def test_real_local21_feature_view() -> None:
    from bitcoin_aml.features.selectors import select_feature_view
    from bitcoin_aml.features.specification import (
        FeatureSetName,
        FeatureSpec,
        ScalingStrategy,
    )

    dataset = EllipticRepository.load(DATASET_DIR)

    features = select_feature_view(
        dataset,
        FeatureSpec(
            feature_set=FeatureSetName.LOCAL_21,
            scaling=ScalingStrategy.NONE,
        ),
    )

    assert features.shape == (dataset.num_nodes, 21)
    assert np.all(np.isfinite(features))
