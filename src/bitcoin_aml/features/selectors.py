# selects the columns for the experiment

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from bitcoin_aml.data.dataset import EllipticDataset
from bitcoin_aml.data.schema import (
    ELLIPTIC_NUM_FEATURES,
    ELLIPTIC_NUM_LOCAL_FEATURES,
    ELLIPTIC_NUM_LOCAL21_FEATURES,
)
from bitcoin_aml.features.specification import (
    FeatureSetName,
    FeatureSpec,
)


def select_feature_view(
    dataset: EllipticDataset,
    spec: FeatureSpec,
) -> NDArray[np.floating]:
    """
    returns a column view of the canonical feature matrix
    no preprocessing or row filtering occurs here
    """
    if spec.feature_set == FeatureSetName.LOCAL_21:
        return dataset.features[
            :,
            :ELLIPTIC_NUM_LOCAL21_FEATURES,
        ]

    if spec.feature_set == FeatureSetName.LOCAL_93:
        return dataset.features[
            :,
            :ELLIPTIC_NUM_LOCAL_FEATURES,
        ]

    if spec.feature_set == FeatureSetName.AGGREGATE_72:
        return dataset.features[
            :,
            ELLIPTIC_NUM_LOCAL_FEATURES:,
        ]

    if spec.feature_set == FeatureSetName.ALL_165:
        return dataset.features

    raise ValueError(f"unsupported feature set: {spec.feature_set}")


def expected_feature_count(
    feature_set: FeatureSetName,
) -> int:
    if feature_set == FeatureSetName.LOCAL_21:
        return ELLIPTIC_NUM_LOCAL21_FEATURES

    if feature_set == FeatureSetName.LOCAL_93:
        return ELLIPTIC_NUM_LOCAL_FEATURES

    if feature_set == FeatureSetName.AGGREGATE_72:
        return (
            ELLIPTIC_NUM_FEATURES
            - ELLIPTIC_NUM_LOCAL_FEATURES
        )

    if feature_set == FeatureSetName.ALL_165:
        return ELLIPTIC_NUM_FEATURES

    raise ValueError(f"Unsupported feature set: {feature_set}")