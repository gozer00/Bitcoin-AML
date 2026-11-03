from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from bitcoin_aml.data.dataset import EllipticDataset
from bitcoin_aml.data.schema import TransactionLabel
from bitcoin_aml.splits.specification import (
    SplitIndices,
    SplitSpec,
    TimestepRange,
)


def build_chronological_split(
    dataset: EllipticDataset,
    spec: SplitSpec,
) -> SplitIndices:
    """
    converts timestep ranges to global node-index arrays
    """
    train = _indices_for_range(dataset.timestep, spec.train,)

    validation = (
        None
        if spec.validation is None
        else _indices_for_range(
            dataset.timestep,
            spec.validation,
        )
    )

    test = _indices_for_range(dataset.timestep, spec.test,)

    return SplitIndices(
        train=train,
        validation=validation,
        test=test,
    )


def labeled_indices(
    dataset: EllipticDataset,
    candidate_indices: NDArray[np.integer],
) -> NDArray[np.int64]:
    """
    returns only nodes with known labels
    """
    _validate_indices(candidate_indices, dataset.num_nodes,)

    candidate_labels = dataset.labels[candidate_indices]

    known_mask = (candidate_labels != int(TransactionLabel.UNKNOWN))

    return np.asarray(
        candidate_indices[known_mask],
        dtype=np.int64,
    )


def _indices_for_range(
    timestep: NDArray[np.integer],
    timestep_range: TimestepRange,
) -> NDArray[np.int64]:
    mask = (
        (timestep >= timestep_range.start)
        & (timestep <= timestep_range.end)
    )

    indices = np.flatnonzero(mask).astype(np.int64, copy=False,)

    if len(indices) == 0:
        raise ValueError(
            "Timestep range "
            f"{timestep_range.start}-{timestep_range.end} "
            "does not contain any nodes."
        )

    return indices


def _validate_indices(
    indices: NDArray[np.integer],
    n_nodes: int,
) -> None:
    if indices.ndim != 1:
        raise ValueError("Indices must be one-dimensional.")

    if not np.issubdtype(indices.dtype, np.integer):
        raise TypeError("Indices must contain integers.")

    if len(indices) == 0:
        return

    if int(indices.min()) < 0:
        raise ValueError("Indices contain a negative node index.")

    if int(indices.max()) >= n_nodes:
        raise ValueError("Indices contain a node outside the dataset.")