import numpy as np
import pytest

from bitcoin_aml.data.dataset import EllipticDataset
from bitcoin_aml.splits.chronological import (
    build_chronological_split,
    labeled_indices,
)
from bitcoin_aml.splits.leakage import (
    assert_indices_subset,
    assert_max_timestep,
    assert_supervised_indices_labeled,
    validate_temporal_split,
)
from bitcoin_aml.splits.specification import (
    SplitSpec,
    TimestepRange,
)


@pytest.fixture
def thesis_split_spec() -> SplitSpec:
    return SplitSpec(
        train=TimestepRange(1, 29),
        validation=TimestepRange(30, 34),
        test=TimestepRange(35, 49),
    )


def test_build_thesis_split(
    synthetic_elliptic_dataset: EllipticDataset,
    thesis_split_spec: SplitSpec,
) -> None:
    dataset = synthetic_elliptic_dataset

    split = build_chronological_split(
        dataset,
        thesis_split_spec,
    )

    # Two nodes per timestep.
    assert len(split.train) == 29 * 2
    assert split.validation is not None
    assert len(split.validation) == 5 * 2
    assert len(split.test) == 15 * 2

    assert np.all(
        dataset.timestep[split.train] <= 29
    )

    assert np.all(
        (
            dataset.timestep[split.validation] >= 30
        )
        & (
            dataset.timestep[split.validation] <= 34
        )
    )

    assert np.all(
        dataset.timestep[split.test] >= 35
    )


def test_temporal_split_validation(
    synthetic_elliptic_dataset: EllipticDataset,
    thesis_split_spec: SplitSpec,
) -> None:
    split = build_chronological_split(
        synthetic_elliptic_dataset,
        thesis_split_spec,
    )

    validate_temporal_split(
        synthetic_elliptic_dataset,
        thesis_split_spec,
        split,
    )


def test_labeled_indices_remove_unknown_nodes(
    synthetic_elliptic_dataset: EllipticDataset,
    thesis_split_spec: SplitSpec,
) -> None:
    dataset = synthetic_elliptic_dataset

    split = build_chronological_split(
        dataset,
        thesis_split_spec,
    )

    train_labeled = labeled_indices(
        dataset,
        split.train,
    )

    assert len(train_labeled) < len(split.train)

    assert np.all(
        dataset.labels[train_labeled] != -1
    )


def test_supervised_unknown_labels_are_rejected(
    synthetic_elliptic_dataset: EllipticDataset,
) -> None:
    dataset = synthetic_elliptic_dataset

    unknown_index = np.flatnonzero(
        dataset.labels == -1
    )[:1]

    with pytest.raises(
        ValueError,
        match="unknown-label",
    ):
        assert_supervised_indices_labeled(
            dataset,
            unknown_index,
            name="training targets",
        )


def test_validation_indices_are_not_training_subset(
    synthetic_elliptic_dataset: EllipticDataset,
    thesis_split_spec: SplitSpec,
) -> None:
    split = build_chronological_split(
        synthetic_elliptic_dataset,
        thesis_split_spec,
    )

    assert split.validation is not None

    with pytest.raises(
        ValueError,
        match="outside",
    ):
        assert_indices_subset(
            split.validation[:1],
            split.train,
            candidate_name="preprocessing fit indices",
            allowed_name="training split",
        )


def test_future_information_is_rejected(
    synthetic_elliptic_dataset: EllipticDataset,
    thesis_split_spec: SplitSpec,
) -> None:
    dataset = synthetic_elliptic_dataset

    split = build_chronological_split(
        dataset,
        thesis_split_spec,
    )

    with pytest.raises(
        ValueError,
        match="future information",
    ):
        assert_max_timestep(
            dataset,
            split.test,
            max_timestep=34,
            name="final-fit information",
        )