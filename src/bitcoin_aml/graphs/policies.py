# contains logic for accessing nodes
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from bitcoin_aml.data.dataset import EllipticDataset
from bitcoin_aml.data.schema import (
    ELLIPTIC_NUM_TIMESTEPS,
)
from bitcoin_aml.graphs.specification import (
    GraphAccessPolicy,
)


@dataclass(frozen=True, slots=True)
class GraphSnapshot:
    """
    manages logic for admissible graph information

    node_indices: Canonical global node indices that are available.
    node_mask: Boolean membership mask over the full canonical dataset.
    edge_index: Permitted edges using canonical global node indices.
    """

    node_indices: NDArray[np.int64]
    node_mask: NDArray[np.bool_]
    edge_index: NDArray[np.int64]

    access_policy: GraphAccessPolicy
    max_timestep: int | None

    def __post_init__(self) -> None:
        if self.node_indices.ndim != 1:
            raise ValueError(
                "node_indices must be one-dimensional."
            )

        if self.node_mask.ndim != 1:
            raise ValueError(
                "node_mask must be one-dimensional."
            )

        if self.node_mask.dtype != np.bool_:
            raise TypeError(
                "node_mask must have boolean dtype."
            )

        if self.edge_index.ndim != 2:
            raise ValueError(
                "edge_index must be two-dimensional."
            )

        if self.edge_index.shape[0] != 2:
            raise ValueError(
                "edge_index must have shape (2, E)."
            )

        if not np.issubdtype(
            self.edge_index.dtype,
            np.integer,
        ):
            raise TypeError(
                "edge_index must contain integers."
            )

        expected_nodes = np.flatnonzero(
            self.node_mask
        )

        if not np.array_equal(
            self.node_indices,
            expected_nodes,
        ):
            raise ValueError(
                "node_indices and node_mask describe different node sets."
            )

        if self.edge_index.size > 0:
            if int(self.edge_index.min()) < 0:
                raise ValueError(
                    "edge_index contains negative nodes."
                )

            if int(self.edge_index.max()) >= len(
                self.node_mask
            ):
                raise ValueError(
                    "edge_index references a node outside the canonical dataset."
                )

            if not np.all(
                self.node_mask[
                    self.edge_index[0]
                ]
            ):
                raise ValueError(
                    "Graph contains an unavailable source node."
                )

            if not np.all(
                self.node_mask[
                    self.edge_index[1]
                ]
            ):
                raise ValueError(
                    "Graph contains an unavailable target node."
                )

    @property
    def num_nodes(self) -> int:
        return len(self.node_indices)

    @property
    def num_edges(self) -> int:
        return self.edge_index.shape[1]


def build_graph_snapshot(
    dataset: EllipticDataset,
    edge_index: NDArray[np.integer],
    *,
    access_policy: GraphAccessPolicy,
    max_timestep: int | None = None,
) -> GraphSnapshot:
    """
    Apply a temporal information-access policy to a graph view.

    Returned edges retain canonical global node indices.
    """
    _validate_edge_index_against_dataset(
        dataset,
        edge_index,
    )

    if access_policy == GraphAccessPolicy.FULL_TRANSDUCTIVE:
        if max_timestep is not None:
            raise ValueError(
                "FULL_TRANSDUCTIVE does not accept max_timestep."
            )

        node_mask = np.ones(
            dataset.num_nodes,
            dtype=bool,
        )

        node_indices = np.arange(
            dataset.num_nodes,
            dtype=np.int64,
        )

        snapshot_edges = np.asarray(
            edge_index,
            dtype=np.int64,
        )

        return GraphSnapshot(
            node_indices=node_indices,
            node_mask=node_mask,
            edge_index=snapshot_edges,
            access_policy=access_policy,
            max_timestep=None,
        )

    if access_policy == GraphAccessPolicy.STRICT_ROLLING:
        if max_timestep is None:
            raise ValueError(
                "STRICT_ROLLING requires max_timestep."
            )

        if not (
            1
            <= max_timestep
            <= ELLIPTIC_NUM_TIMESTEPS
        ):
            raise ValueError(
                "max_timestep must lie between 1 and 49."
            )

        node_mask = (
            dataset.timestep <= max_timestep
        )

        node_indices = np.flatnonzero(
            node_mask
        ).astype(
            np.int64,
            copy=False,
        )

        source = edge_index[0]
        target = edge_index[1]

        edge_mask = (
            node_mask[source]
            & node_mask[target]
        )

        snapshot_edges = np.ascontiguousarray(
            edge_index[:, edge_mask],
            dtype=np.int64,
        )

        snapshot = GraphSnapshot(
            node_indices=node_indices,
            node_mask=np.asarray(
                node_mask,
                dtype=np.bool_,
            ),
            edge_index=snapshot_edges,
            access_policy=access_policy,
            max_timestep=max_timestep,
        )

        validate_graph_snapshot(
            dataset,
            snapshot,
        )

        return snapshot

    raise ValueError(
        f"Unsupported graph access policy: "
        f"{access_policy}"
    )


def validate_graph_snapshot(
    dataset: EllipticDataset,
    snapshot: GraphSnapshot,
) -> None:
    """
    Runtime leakage validation for a materialized graph snapshot.
    """
    if len(snapshot.node_mask) != dataset.num_nodes:
        raise ValueError(
            "Graph snapshot node mask does not match "
            "the canonical dataset."
        )

    if (
        snapshot.access_policy
        == GraphAccessPolicy.STRICT_ROLLING
    ):
        if snapshot.max_timestep is None:
            raise ValueError(
                "Strict rolling snapshot has no max_timestep."
            )

        if snapshot.num_nodes > 0:
            observed_node_max = int(
                dataset.timestep[
                    snapshot.node_indices
                ].max()
            )

            if (
                observed_node_max
                > snapshot.max_timestep
            ):
                raise ValueError(
                    "Strict graph snapshot contains "
                    "future nodes."
                )

        if snapshot.num_edges > 0:
            source = snapshot.edge_index[0]
            target = snapshot.edge_index[1]

            observed_edge_max = int(
                max(
                    dataset.timestep[source].max(),
                    dataset.timestep[target].max(),
                )
            )

            if (
                observed_edge_max
                > snapshot.max_timestep
            ):
                raise ValueError(
                    "Strict graph snapshot contains "
                    "future edges."
                )

    elif (
        snapshot.access_policy
        == GraphAccessPolicy.FULL_TRANSDUCTIVE
    ):
        if snapshot.num_nodes != dataset.num_nodes:
            raise ValueError(
                "Full transductive snapshot must "
                "contain all canonical nodes."
            )

        if snapshot.max_timestep is not None:
            raise ValueError(
                "Full transductive snapshot must not "
                "have max_timestep."
            )

    else:
        raise ValueError(
            "Unknown graph access policy."
        )


def resolve_phase_edge_index(
    dataset: EllipticDataset,
    edge_index: NDArray[np.integer],
    *,
    access_policy: GraphAccessPolicy,
    phase: Literal["fit", "eval"],
    fit_cutoff_timestep: int,
) -> NDArray[np.int64]:
    """
    Return the edge_index visible to a given experiment phase under an
    access policy.

    Evaluation always sees the full, unrestricted graph regardless of
    policy: predicting on a node requires its real neighbors, and Elliptic's
    zero-cross-timestep-edges invariant means a node's only possible
    neighbors share its own timestep -- masking the eval-time graph would
    not test leakage, it would just delete those nodes' only neighbors.
    Under strict_rolling, only the fit phase (model selection and refit) is
    masked to fit_cutoff_timestep, so the model is trained as if
    later-timestep nodes did not yet exist.
    """
    if access_policy == GraphAccessPolicy.FULL_TRANSDUCTIVE or phase == "eval":
        return build_graph_snapshot(
            dataset,
            edge_index,
            access_policy=GraphAccessPolicy.FULL_TRANSDUCTIVE,
        ).edge_index

    return build_graph_snapshot(
        dataset,
        edge_index,
        access_policy=GraphAccessPolicy.STRICT_ROLLING,
        max_timestep=fit_cutoff_timestep,
    ).edge_index


def _validate_edge_index_against_dataset(
    dataset: EllipticDataset,
    edge_index: NDArray[np.integer],
) -> None:
    if edge_index.ndim != 2:
        raise ValueError(
            "edge_index must be two-dimensional."
        )

    if edge_index.shape[0] != 2:
        raise ValueError(
            "edge_index must have shape (2, E)."
        )

    if not np.issubdtype(
        edge_index.dtype,
        np.integer,
    ):
        raise TypeError(
            "edge_index must contain integers."
        )

    if edge_index.size == 0:
        return

    if int(edge_index.min()) < 0:
        raise ValueError(
            "edge_index contains a negative node."
        )

    if int(edge_index.max()) >= dataset.num_nodes:
        raise ValueError(
            "edge_index references a node outside "
            "the canonical dataset."
        )