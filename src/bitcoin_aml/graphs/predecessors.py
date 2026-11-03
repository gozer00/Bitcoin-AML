# builder for incoming neighbor representation

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from bitcoin_aml.data.dataset import EllipticDataset


@dataclass(frozen=True, slots=True)
class IncomingCSR:
    """
    Compressed sparse representation of canonical incoming neighbors.

    For node v:

        indices[indptr[v]:indptr[v + 1]]

    contains all u such that:

        u -> v
    """

    indptr: NDArray[np.int64]
    indices: NDArray[np.int64]

    def __post_init__(self) -> None:
        if self.indptr.ndim != 1:
            raise ValueError(
                "indptr must be one-dimensional."
            )

        if self.indices.ndim != 1:
            raise ValueError(
                "indices must be one-dimensional."
            )

        if self.indptr.dtype != np.int64:
            raise TypeError(
                "indptr must have dtype int64."
            )

        if self.indices.dtype != np.int64:
            raise TypeError(
                "indices must have dtype int64."
            )

        if len(self.indptr) == 0:
            raise ValueError(
                "indptr must contain at least one value."
            )

        if self.indptr[0] != 0:
            raise ValueError(
                "indptr must begin with zero."
            )

        if np.any(
            self.indptr[1:] < self.indptr[:-1]
        ):
            raise ValueError(
                "indptr must be monotonically non-decreasing."
            )

        if self.indptr[-1] != len(self.indices):
            raise ValueError(
                "Final indptr value must equal "
                "the number of stored neighbors."
            )

        if len(self.indices) > 0:
            if int(self.indices.min()) < 0:
                raise ValueError(
                    "indices contains a negative node."
                )

            if int(self.indices.max()) >= self.num_nodes:
                raise ValueError(
                    "indices references a node outside "
                    "the graph."
                )

    @property
    def num_nodes(self) -> int:
        return len(self.indptr) - 1

    @property
    def num_edges(self) -> int:
        return len(self.indices)

    def predecessors(
        self,
        node: int,
    ) -> NDArray[np.int64]:
        if not 0 <= node < self.num_nodes:
            raise IndexError(
                f"Node {node} lies outside the graph."
            )

        start = int(self.indptr[node])
        end = int(self.indptr[node + 1])

        return self.indices[start:end]


@dataclass(frozen=True, slots=True)
class PredecessorNeighborhood:
    """
    Target-centered predecessor neighborhood.

    nodes_by_hop[0] contains exact 1-hop predecessors,
    nodes_by_hop[1] exact 2-hop predecessors, etc.

    Each node appears at its shortest predecessor distance only.
    The target itself is excluded.
    """

    target_node: int
    nodes_by_hop: tuple[NDArray[np.int64], ...]

    @property
    def hops(self) -> int:
        return len(self.nodes_by_hop)

    @property
    def all_predecessors(self) -> NDArray[np.int64]:
        non_empty = [
            nodes
            for nodes in self.nodes_by_hop
            if len(nodes) > 0
        ]

        if not non_empty:
            return np.empty(
                0,
                dtype=np.int64,
            )

        return np.concatenate(
            non_empty
        ).astype(
            np.int64,
            copy=False,
        )


@dataclass(frozen=True, slots=True)
class PredecessorBatch:
    """
    Union-of-neighborhoods minibatch over multiple target nodes.

    global_node_ids[i] is the canonical dataset index of local node i;
    it is the row to use when gathering features/labels for local id i.

    local_edge_index uses local ids 0..len(global_node_ids) - 1 and
    contains exactly the canonical predecessor-direction edges (u -> v)
    for which both u and v are included in global_node_ids.

    target_local_ids gives, for each entry of target_nodes, that
    target's position in global_node_ids.

    target_nodes and global_node_ids are both sorted ascending and
    deduplicated, so repeated calls with the same inputs are
    byte-identical: target_nodes is the caller's targets after
    np.unique, regardless of the order or duplicates the caller passed
    in.
    """

    target_nodes: NDArray[np.int64]
    global_node_ids: NDArray[np.int64]
    local_edge_index: NDArray[np.int64]
    target_local_ids: NDArray[np.int64]

    def __post_init__(self) -> None:
        if self.target_nodes.ndim != 1:
            raise ValueError(
                "target_nodes must be one-dimensional."
            )

        if self.global_node_ids.ndim != 1:
            raise ValueError(
                "global_node_ids must be one-dimensional."
            )

        if self.local_edge_index.ndim != 2:
            raise ValueError(
                "local_edge_index must be two-dimensional."
            )

        if self.target_local_ids.ndim != 1:
            raise ValueError(
                "target_local_ids must be one-dimensional."
            )

        if self.target_nodes.dtype != np.int64:
            raise TypeError(
                "target_nodes must have dtype int64."
            )

        if self.global_node_ids.dtype != np.int64:
            raise TypeError(
                "global_node_ids must have dtype int64."
            )

        if self.local_edge_index.dtype != np.int64:
            raise TypeError(
                "local_edge_index must have dtype int64."
            )

        if self.target_local_ids.dtype != np.int64:
            raise TypeError(
                "target_local_ids must have dtype int64."
            )

        if len(self.target_nodes) == 0:
            raise ValueError(
                "target_nodes must contain at least one target."
            )

        if self.local_edge_index.shape[0] != 2:
            raise ValueError(
                "local_edge_index must have shape (2, E)."
            )

        if len(self.target_local_ids) != len(self.target_nodes):
            raise ValueError(
                "target_local_ids must have one entry per target."
            )

        if len(self.global_node_ids) != len(
            np.unique(self.global_node_ids)
        ):
            raise ValueError(
                "global_node_ids must not contain duplicate nodes."
            )

        if np.any(
            self.global_node_ids[:-1] >= self.global_node_ids[1:]
        ):
            raise ValueError(
                "global_node_ids must be sorted in ascending order."
            )

        if self.local_edge_index.size > 0:
            if int(self.local_edge_index.min()) < 0 or int(
                self.local_edge_index.max()
            ) >= self.num_nodes:
                raise ValueError(
                    "local_edge_index references a node outside "
                    "the batch."
                )

        if len(self.target_local_ids) > 0:
            if int(self.target_local_ids.min()) < 0 or int(
                self.target_local_ids.max()
            ) >= self.num_nodes:
                raise ValueError(
                    "target_local_ids references a node outside "
                    "the batch."
                )

        if not np.array_equal(
            self.global_node_ids[self.target_local_ids],
            self.target_nodes,
        ):
            raise ValueError(
                "target_local_ids does not match target_nodes "
                "under global_node_ids."
            )

    @property
    def num_nodes(self) -> int:
        return len(self.global_node_ids)

    @property
    def num_edges(self) -> int:
        return self.local_edge_index.shape[1]

    @property
    def num_targets(self) -> int:
        return len(self.target_nodes)


def build_incoming_csr_from_edges(
    edge_index: NDArray[np.integer],
    *,
    num_nodes: int,
) -> IncomingCSR:
    """
    Build incoming-neighbor CSR from an arbitrary directed edge_index over
    num_nodes canonical node indices.

    This is the shared implementation behind build_incoming_csr. Callers
    that only have a phase-masked edge_index (e.g. a model adapter given a
    ModelInput.edge_index that graphs.policies.resolve_phase_edge_index
    already masked under GraphAccessPolicy.STRICT_ROLLING's fit phase, not
    the full EllipticDataset object) must call this directly instead of
    build_incoming_csr.
    """
    source = np.asarray(
        edge_index[0],
        dtype=np.int64,
    )

    target = np.asarray(
        edge_index[1],
        dtype=np.int64,
    )

    if len(source) == 0:
        return IncomingCSR(
            indptr=np.zeros(
                num_nodes + 1,
                dtype=np.int64,
            ),
            indices=np.empty(
                0,
                dtype=np.int64,
            ),
        )

    # Deterministic order:
    # target first, predecessor/source second.
    order = np.lexsort(
        (
            source,
            target,
        )
    )

    sorted_target = target[order]
    sorted_source = source[order]

    counts = np.bincount(
        sorted_target,
        minlength=num_nodes,
    )

    indptr = np.empty(
        num_nodes + 1,
        dtype=np.int64,
    )

    indptr[0] = 0

    np.cumsum(
        counts,
        out=indptr[1:],
    )

    indices = np.ascontiguousarray(
        sorted_source,
        dtype=np.int64,
    )

    return IncomingCSR(
        indptr=indptr,
        indices=indices,
    )


def build_incoming_csr(
    dataset: EllipticDataset,
) -> IncomingCSR:
    """
    Build incoming-neighbor CSR directly from the canonical directed
    Elliptic graph.

    The bidirectional experimental graph must never be supplied here.
    """
    return build_incoming_csr_from_edges(
        dataset.edge_index,
        num_nodes=dataset.num_nodes,
    )


def extract_predecessor_neighborhood(
    csr: IncomingCSR,
    *,
    target_node: int,
    hops: int,
    allowed_mask: NDArray[np.bool_] | None = None,
) -> PredecessorNeighborhood:
    """
    Extract exact predecessor layers using breadth-first traversal.

    If allowed_mask is supplied, inaccessible nodes are removed before
    traversal can continue through them.
    """
    if not 0 <= target_node < csr.num_nodes:
        raise IndexError(
            "target_node lies outside the graph."
        )

    if hops < 1:
        raise ValueError(
            "hops must be at least 1."
        )

    if allowed_mask is not None:
        if allowed_mask.ndim != 1:
            raise ValueError(
                "allowed_mask must be one-dimensional."
            )

        if allowed_mask.dtype != np.bool_:
            raise TypeError(
                "allowed_mask must have boolean dtype."
            )

        if len(allowed_mask) != csr.num_nodes:
            raise ValueError(
                "allowed_mask does not match graph size."
            )

        if not allowed_mask[target_node]:
            raise ValueError(
                "Target node is not permitted by the graph "
                "information policy."
            )

    visited = np.zeros(
        csr.num_nodes,
        dtype=bool,
    )

    visited[target_node] = True

    frontier = np.array(
        [target_node],
        dtype=np.int64,
    )

    layers: list[NDArray[np.int64]] = []

    for _ in range(hops):
        predecessor_parts = [
            csr.predecessors(int(node))
            for node in frontier
        ]

        predecessor_parts = [
            values
            for values in predecessor_parts
            if len(values) > 0
        ]

        if not predecessor_parts:
            current = np.empty(
                0,
                dtype=np.int64,
            )

            layers.append(current)
            frontier = current
            continue

        candidates = np.unique(
            np.concatenate(
                predecessor_parts
            )
        ).astype(
            np.int64,
            copy=False,
        )

        if allowed_mask is not None:
            candidates = candidates[
                allowed_mask[candidates]
            ]

        candidates = candidates[
            ~visited[candidates]
        ]

        visited[candidates] = True

        current = np.ascontiguousarray(
            candidates,
            dtype=np.int64,
        )

        layers.append(current)
        frontier = current

    return PredecessorNeighborhood(
        target_node=target_node,
        nodes_by_hop=tuple(layers),
    )


def build_predecessor_batch(
    csr: IncomingCSR,
    *,
    target_nodes: NDArray[np.integer],
    hops: int,
    allowed_mask: NDArray[np.bool_] | None = None,
) -> PredecessorBatch:
    """
    Build one renumbered local subgraph covering the union of every
    target's exact k-hop predecessor neighborhood.

    Elliptic's zero-cross-timestep-edges invariant means every edge
    returned here already lies within a single timestep -- this
    function performs no timestep filtering of its own; any temporal
    restriction must be encoded into allowed_mask by the caller (e.g.
    via GraphSnapshot's node_mask) before calling this function.

    The returned target_nodes is the sorted, deduplicated set of input
    targets, not the caller's original array: use target_nodes together
    with target_local_ids (not the caller's own array/order) to recover
    each target's local id.
    """
    if target_nodes.ndim != 1:
        raise ValueError(
            "target_nodes must be one-dimensional."
        )

    if len(target_nodes) == 0:
        raise ValueError(
            "target_nodes must contain at least one target."
        )

    target_nodes_i64 = target_nodes.astype(
        np.int64,
        copy=False,
    )

    if int(target_nodes_i64.min()) < 0 or int(
        target_nodes_i64.max()
    ) >= csr.num_nodes:
        raise IndexError(
            "target_nodes contains a node outside the graph."
        )

    if hops < 1:
        raise ValueError(
            "hops must be at least 1."
        )

    unique_targets = np.unique(
        target_nodes_i64
    )

    node_parts = [unique_targets]

    for target in unique_targets:
        neighborhood = extract_predecessor_neighborhood(
            csr,
            target_node=int(target),
            hops=hops,
            allowed_mask=allowed_mask,
        )

        node_parts.append(
            neighborhood.all_predecessors
        )

    global_node_ids = np.unique(
        np.concatenate(node_parts)
    ).astype(
        np.int64,
        copy=False,
    )

    target_local_ids = np.searchsorted(
        global_node_ids,
        unique_targets,
    ).astype(
        np.int64,
        copy=False,
    )

    # global_node_ids already reflects allowed_mask, since it is built
    # only from extract_predecessor_neighborhood's (already-masked)
    # output -- an excluded node never appears here, so edges touching
    # it are excluded automatically without a second mask check below.
    in_batch = np.zeros(
        csr.num_nodes,
        dtype=bool,
    )

    in_batch[global_node_ids] = True

    local_sources: list[NDArray[np.int64]] = []
    local_targets: list[NDArray[np.int64]] = []

    for local_target_id, node in enumerate(global_node_ids):
        predecessors = csr.predecessors(int(node))

        if len(predecessors) == 0:
            continue

        kept = predecessors[in_batch[predecessors]]

        if len(kept) == 0:
            continue

        local_sources.append(
            np.searchsorted(
                global_node_ids,
                kept,
            )
        )

        local_targets.append(
            np.full(
                len(kept),
                local_target_id,
                dtype=np.int64,
            )
        )

    if not local_sources:
        local_edge_index = np.empty(
            (2, 0),
            dtype=np.int64,
        )
    else:
        local_source = np.concatenate(
            local_sources
        ).astype(
            np.int64,
            copy=False,
        )

        local_target = np.concatenate(
            local_targets
        ).astype(
            np.int64,
            copy=False,
        )

        # Deterministic order:
        # target first, predecessor/source second.
        order = np.lexsort(
            (
                local_source,
                local_target,
            )
        )

        local_edge_index = np.vstack(
            [
                local_source[order],
                local_target[order],
            ]
        ).astype(
            np.int64,
            copy=False,
        )

    return PredecessorBatch(
        target_nodes=unique_targets,
        global_node_ids=global_node_ids,
        local_edge_index=local_edge_index,
        target_local_ids=target_local_ids,
    )