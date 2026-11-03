import numpy as np
import pytest

from bitcoin_aml.data.dataset import EllipticDataset


@pytest.fixture
def synthetic_elliptic_dataset() -> EllipticDataset:
    """
    represents a synthetic full 49-timestep Elliptic-compatible dataset

    - two nodes per timestep
    - one directed edge within every timestep
    """
    num_timesteps = 49
    nodes_per_timestep = 2

    num_nodes = (
        num_timesteps
        * nodes_per_timestep
    )

    tx_id = np.arange(
        10_000,
        10_000 + num_nodes,
        dtype=np.int64,
    )

    timestep = np.repeat(
        np.arange(
            1,
            num_timesteps + 1,
            dtype=np.int16,
        ),
        nodes_per_timestep,
    )

    features = np.zeros(
        (num_nodes, 165),
        dtype=np.float32,
    )

    # Deterministic non-constant values.
    features[:, 0] = np.arange(
        num_nodes,
        dtype=np.float32,
    )

    features[:, 92] = 92.0
    features[:, 93] = 93.0
    features[:, 164] = 164.0

    labels = np.resize(
        np.array(
            [1, 0, -1],
            dtype=np.int8,
        ),
        num_nodes,
    )

    source = np.arange(
        0,
        num_nodes,
        2,
        dtype=np.int64,
    )

    target = source + 1

    edge_index = np.vstack(
        [source, target]
    )

    return EllipticDataset(
        tx_id=tx_id,
        timestep=timestep,
        features=features,
        labels=labels,
        edge_index=edge_index,
    )

@pytest.fixture
def predecessor_elliptic_dataset() -> EllipticDataset:
    """
    First timestep contains:

        0 -> 3
        1 -> 3
        2 -> 4
        3 -> 4
        4 -> 5

    Timesteps 2-49 each contain one isolated node.
    """
    first_slice_nodes = 6
    remaining_timesteps = 48

    num_nodes = (
        first_slice_nodes
        + remaining_timesteps
    )

    tx_id = np.arange(
        20_000,
        20_000 + num_nodes,
        dtype=np.int64,
    )

    timestep = np.concatenate(
        [
            np.ones(
                first_slice_nodes,
                dtype=np.int16,
            ),
            np.arange(
                2,
                50,
                dtype=np.int16,
            ),
        ]
    )

    features = np.zeros(
        (num_nodes, 165),
        dtype=np.float32,
    )

    labels = np.zeros(
        num_nodes,
        dtype=np.int8,
    )

    edge_index = np.array(
        [
            [0, 1, 2, 3, 4],
            [3, 3, 4, 4, 5],
        ],
        dtype=np.int64,
    )

    return EllipticDataset(
        tx_id=tx_id,
        timestep=timestep,
        features=features,
        labels=labels,
        edge_index=edge_index,
    )

@pytest.fixture
def multi_target_elliptic_dataset() -> EllipticDataset:
    """
    First timestep contains two components:

        0 -> 2
        1 -> 2
        2 -> 4
        2 -> 7
        6 -> 4
        3 -> 5

    Node 2 is a shared ancestor of targets 4 and 7 (both 1-hop, with
    0 and 1 shared as 2-hop predecessors through it), so targets 4 and
    7 exercise overlapping neighborhoods. Node 6 is a 1-hop predecessor
    of target 4 only. Node 3/5 form a component disjoint from targets
    4 and 7.

    Timesteps 2-49 each contain one isolated node.
    """
    first_slice_nodes = 8
    remaining_timesteps = 48

    num_nodes = (
        first_slice_nodes
        + remaining_timesteps
    )

    tx_id = np.arange(
        30_000,
        30_000 + num_nodes,
        dtype=np.int64,
    )

    timestep = np.concatenate(
        [
            np.ones(
                first_slice_nodes,
                dtype=np.int16,
            ),
            np.arange(
                2,
                50,
                dtype=np.int16,
            ),
        ]
    )

    features = np.zeros(
        (num_nodes, 165),
        dtype=np.float32,
    )

    labels = np.zeros(
        num_nodes,
        dtype=np.int8,
    )

    edge_index = np.array(
        [
            [0, 1, 2, 2, 6, 3],
            [2, 2, 4, 7, 4, 5],
        ],
        dtype=np.int64,
    )

    return EllipticDataset(
        tx_id=tx_id,
        timestep=timestep,
        features=features,
        labels=labels,
        edge_index=edge_index,
    )