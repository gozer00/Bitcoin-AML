# runtime safety checks for:
# train ∩ validation = ∅
# train ∩ test = ∅
# validation ∩ test = ∅
# max(train timestep) < min(validation timestep)
# max(validation timestep) < min(test timestep)

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


def validate_temporal_split(
    dataset: EllipticDataset,
    spec: SplitSpec,
    split: SplitIndices,
) -> None:
    """
    verifies that concrete split indices correspond exactly to SplitSpec
    """
    _assert_matches_range(
        dataset,
        split.train,
        spec.train,
        name="train",
    )

    if spec.validation is None:
        if split.validation is not None:
            raise ValueError(
                "Split contains validation indices although the "
                "specification has no validation range."
            )
    else:
        if split.validation is None:
            raise ValueError(
                "Split specification requires validation indices."
            )

        _assert_matches_range(
            dataset,
            split.validation,
            spec.validation,
            name="validation",
        )

    _assert_matches_range(
        dataset,
        split.test,
        spec.test,
        name="test",
    )


def assert_indices_subset(
    candidate: NDArray[np.integer],
    allowed: NDArray[np.integer],
    *,
    candidate_name: str,
    allowed_name: str,
) -> None:
    """
    ensure that every candidate node belongs to the permitted set
    """
    if len(candidate) == 0:
        return

    if not np.all(np.isin(candidate, allowed)):
        illegal = candidate[
            ~np.isin(candidate, allowed)
        ]

        raise ValueError(
            f"{candidate_name} contains nodes outside "
            f"{allowed_name}. "
            f"Examples: {illegal[:10].tolist()}"
        )


def assert_supervised_indices_labeled(
    dataset: EllipticDataset,
    indices: NDArray[np.integer],
    *,
    name: str,
) -> None:
    """
    ensure that every supervised target has a known class
    """
    labels = dataset.labels[indices]

    unknown_mask = (labels == int(TransactionLabel.UNKNOWN))

    if np.any(unknown_mask):
        unknown_indices = indices[unknown_mask]

        raise ValueError(
            f"{name} contains unknown-label nodes. "
            f"Examples: {unknown_indices[:10].tolist()}"
        )


def assert_max_timestep(
    dataset: EllipticDataset,
    indices: NDArray[np.integer],
    *,
    max_timestep: int,
    name: str,
) -> None:
    """
    ensure that an information set does not contain future nodes
    """
    if len(indices) == 0:
        return

    observed_max = int(dataset.timestep[indices].max())

    if observed_max > max_timestep:
        raise ValueError(
            f"{name} contains future information. "
            f"Allowed maximum timestep: {max_timestep}; "
            f"observed: {observed_max}."
        )


def _assert_matches_range(
    dataset: EllipticDataset,
    actual_indices: NDArray[np.integer],
    expected_range: TimestepRange,
    *,
    name: str,
) -> None:
    expected_indices = np.flatnonzero(
        (
            dataset.timestep >= expected_range.start
        )
        & (
            dataset.timestep <= expected_range.end
        )
    )

    if not np.array_equal(
        actual_indices,
        expected_indices,
    ):
        raise ValueError(
            f"{name} indices do not exactly correspond to "
            f"timesteps {expected_range.start}-"
            f"{expected_range.end}."
        )