import numpy as np
import pytest

from bitcoin_aml.data.dataset import EllipticDataset
from bitcoin_aml.features.selectors import (
    expected_feature_count,
    select_feature_view,
)
from bitcoin_aml.features.specification import (
    FeatureSetName,
    FeatureSpec,
    ScalingStrategy,
)


def test_local_feature_view(
    synthetic_elliptic_dataset: EllipticDataset,
) -> None:
    dataset = synthetic_elliptic_dataset

    spec = FeatureSpec(
        feature_set=FeatureSetName.LOCAL_93,
        scaling=ScalingStrategy.STANDARD,
    )

    features = select_feature_view(
        dataset,
        spec,
    )

    assert features.shape == (
        dataset.num_nodes,
        93,
    )

    assert np.shares_memory(
        features,
        dataset.features,
    )


def test_local21_feature_view(
    synthetic_elliptic_dataset: EllipticDataset,
) -> None:
    """
    local_21 is columns 0-20: the transaction-local features that
    Safar et al. (2026) identify before the address-level block
    (timestep is not part of the canonical feature matrix)
    """
    dataset = synthetic_elliptic_dataset

    spec = FeatureSpec(
        feature_set=FeatureSetName.LOCAL_21,
        scaling=ScalingStrategy.STANDARD,
    )

    features = select_feature_view(
        dataset,
        spec,
    )

    assert features.shape == (
        dataset.num_nodes,
        21,
    )

    assert np.shares_memory(
        features,
        dataset.features,
    )

    np.testing.assert_array_equal(
        features,
        dataset.features[:, :21],
    )


def test_local21_is_prefix_of_local93(
    synthetic_elliptic_dataset: EllipticDataset,
) -> None:
    dataset = synthetic_elliptic_dataset

    local21 = select_feature_view(
        dataset,
        FeatureSpec(
            feature_set=FeatureSetName.LOCAL_21,
            scaling=ScalingStrategy.NONE,
        ),
    )

    local93 = select_feature_view(
        dataset,
        FeatureSpec(
            feature_set=FeatureSetName.LOCAL_93,
            scaling=ScalingStrategy.NONE,
        ),
    )

    np.testing.assert_array_equal(
        local21,
        local93[:, :21],
    )


def test_aggregate_feature_view(
    synthetic_elliptic_dataset: EllipticDataset,
) -> None:
    dataset = synthetic_elliptic_dataset

    spec = FeatureSpec(
        feature_set=FeatureSetName.AGGREGATE_72,
        scaling=ScalingStrategy.NONE,
    )

    features = select_feature_view(
        dataset,
        spec,
    )

    assert features.shape == (
        dataset.num_nodes,
        72,
    )

    assert np.shares_memory(
        features,
        dataset.features,
    )


def test_all_feature_view(
    synthetic_elliptic_dataset: EllipticDataset,
) -> None:
    dataset = synthetic_elliptic_dataset

    spec = FeatureSpec(
        feature_set=FeatureSetName.ALL_165,
        scaling=ScalingStrategy.STANDARD,
    )

    features = select_feature_view(
        dataset,
        spec,
    )

    assert features.shape == (
        dataset.num_nodes,
        165,
    )

    assert features is dataset.features


def test_expected_feature_counts() -> None:
    assert expected_feature_count(
        FeatureSetName.LOCAL_21
    ) == 21

    assert expected_feature_count(
        FeatureSetName.LOCAL_93
    ) == 93

    assert expected_feature_count(
        FeatureSetName.AGGREGATE_72
    ) == 72

    assert expected_feature_count(
        FeatureSetName.ALL_165
    ) == 165


@pytest.mark.parametrize(
    "feature_set",
    list(FeatureSetName),
)
def test_every_feature_set_has_count_and_view(
    synthetic_elliptic_dataset: EllipticDataset,
    feature_set: FeatureSetName,
) -> None:
    features = select_feature_view(
        synthetic_elliptic_dataset,
        FeatureSpec(
            feature_set=feature_set,
            scaling=ScalingStrategy.NONE,
        ),
    )

    assert features.shape[1] == expected_feature_count(
        feature_set
    )
