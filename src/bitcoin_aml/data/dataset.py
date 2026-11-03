from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from bitcoin_aml.data.schema import (
    ELLIPTIC_NUM_FEATURES,
    ELLIPTIC_TIMESTEPS,
    TransactionLabel,
)


@dataclass(frozen=True, slots=True)
class EllipticDataset:
    """
    Canonical in-memory representation of the Elliptic data.
    We use same node ordering in all arrays.
    """

    tx_id: NDArray[np.int64]  # shape: [N], original transaction ids
    timestep: NDArray[np.integer]    # shape: [N], slices from 1 to 49
    features: NDArray[np.floating]    # shape: [N, 165], type: float, all features
    labels: NDArray[np.integer]  # shape: [N], type: int, -1=unknown, 0=licit, 1=illicit
    edge_index: NDArray[np.integer]  # shape: [2, E] PyTorch Geometric convention, type: int, source -> target

    def __post_init__(self) -> None:
        self._validate_dimensions()
        self._validate_dtypes()
        self._validate_node_data()
        self._validate_edges()

    def _validate_dimensions(self) -> None:
        if self.tx_id.ndim != 1:
            raise ValueError("tx_id must be one-dimensional")

        if self.timestep.ndim != 1:
            raise ValueError("timestep must be one-dimensional")

        if self.labels.ndim != 1:
            raise ValueError("labels must be one-dimensional")

        if self.features.ndim != 2:
            raise ValueError("features must have shape (N, F)")

        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape (2, E)")

        n_nodes = len(self.tx_id)

        if len(self.timestep) != n_nodes:
            raise ValueError("tx_id and timestep are not aligned")

        if len(self.labels) != n_nodes:
            raise ValueError("tx_id and labels are not aligned")

        if self.features.shape[0] != n_nodes:
            raise ValueError("features must contain one row per node")

        if self.features.shape[1] != ELLIPTIC_NUM_FEATURES:
            raise ValueError(
                f"Expected {ELLIPTIC_NUM_FEATURES} features, "
                f"received {self.features.shape[1]}"
            )

    def _validate_dtypes(self) -> None:
        if not np.issubdtype(self.tx_id.dtype, np.integer):
            raise TypeError("tx_id must contain integers")

        if not np.issubdtype(self.timestep.dtype, np.integer):
            raise TypeError("timestep must contain integers")

        if not np.issubdtype(self.features.dtype, np.floating):
            raise TypeError("features must contain floats")

        if not np.issubdtype(self.labels.dtype, np.integer):
            raise TypeError("labels must contain integers")

        if not np.issubdtype(self.edge_index.dtype, np.integer):
            raise TypeError("edge_index must contain integers")

    def _validate_node_data(self) -> None:
        if len(np.unique(self.tx_id)) != self.num_nodes:
            raise ValueError("tx_id values must be unique")

        actual_timesteps = tuple(
            int(value) for value in np.unique(self.timestep)
        )

        if actual_timesteps != ELLIPTIC_TIMESTEPS:
            raise ValueError(
                "canonical EllipticDataset must contain all timesteps 1-49"
            )

        valid_labels = {
            int(TransactionLabel.UNKNOWN),
            int(TransactionLabel.LICIT),
            int(TransactionLabel.ILLICIT),
        }

        actual_labels = {
            int(value) for value in np.unique(self.labels)
        }

        if not actual_labels.issubset(valid_labels):
            raise ValueError(
                f"Unsupported canonical labels "
                f"{sorted(actual_labels - valid_labels)}"
            )

    def _validate_edges(self) -> None:
        if self.edge_index.size == 0:
            return

        min_node = int(self.edge_index.min())
        max_node = int(self.edge_index.max())

        if min_node < 0:
            raise ValueError("edge_index contains a negative node index")

        if max_node >= self.num_nodes:
            raise ValueError(
                "edge_index contains a node index outside the dataset"
            )

        source = self.edge_index[0]
        target = self.edge_index[1]

        if np.any(source == target):
            raise ValueError("Self-edges are not permitted")

        if np.any(self.timestep[source] != self.timestep[target]):
            raise ValueError(
                "Elliptic edge connects nodes from different timesteps"
            )

    @property
    def num_nodes(self) -> int:
        return len(self.tx_id)

    @property
    def num_features(self) -> int:
        return self.features.shape[1]

    @property
    def num_edges(self) -> int:
        return self.edge_index.shape[1]
