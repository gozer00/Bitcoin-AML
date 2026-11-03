# transforms the graph data into the required view

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from bitcoin_aml.data.dataset import EllipticDataset
from bitcoin_aml.graphs.specification import GraphSpec, GraphViewName


def edge_index_for_view(
        dataset: EllipticDataset,
        spec: GraphSpec,
) -> NDArray[np.integer]:
    """
    returns global edge repr. according to GraphSpec
    node indices remain untouched
    """

    if spec.view == GraphViewName.NONE:
        return np.empty(
            (2, 0),
            dtype=np.uint64
        )

    if spec.view == GraphViewName.DIRECTED or spec.view == GraphViewName.INCOMING:
        return dataset.edge_index

    if spec.view == GraphViewName.BIDIRECTIONAL:
        return make_bidirectional(
            dataset.edge_index,
            num_nodes=dataset.num_nodes,
        )

    if spec.view == GraphViewName.REVERSED:
        return np.stack(
            [
                dataset.edge_index[1],
                dataset.edge_index[0],
            ]
        )

    raise ValueError(f"Unknown view {spec.view}")
def make_bidirectional(
        edge_index: NDArray[np.integer],
        *,
        num_nodes: int
) -> NDArray[np.int64]:
    """
    adds the reverse of every edge
    removes duplicates first
    """

    _validate_edge_index(
        edge_index,
        num_nodes=num_nodes,
    )

    if edge_index.shape[1] == 0:
        return np.empty(
            (2, 0),
            dtype=np.int64,
        )

    original = np.asarray(
        edge_index,
        dtype=np.int64,
    )

    reverse = original[::-1]

    pairs = np.concatenate([original.T, reverse.T,], axis=0)

    # Primary sort key: source.
    # Secondary sort key: target.
    order = np.lexsort(
        (
            pairs[:, 1],
            pairs[:, 0],
        )
    )

    pairs = pairs[order]

    keep = np.ones(len(pairs), dtype=bool,)

    if len(pairs) > 1:
        keep[1:] = np.any(
            pairs[1:] != pairs[:-1],
            axis=1,
        )

    unique_pairs = pairs[keep]

    return np.ascontiguousarray(
        unique_pairs.T,
        dtype=np.int64,
    )

def _validate_edge_index(
        edge_index: NDArray[np.integer],
        *,
        num_nodes: int,
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

    if int(edge_index.max()) >= num_nodes:
        raise ValueError(
            "edge_index references a node outside the graph."
        )

    if np.any(
            edge_index[0] == edge_index[1]
    ):
        raise ValueError(
            "Self-edges are not supported."
        )