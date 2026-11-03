import numpy as np
import pytest

from bitcoin_aml.features.preprocessing import (
    fit_feature_preprocessor,
)
from bitcoin_aml.features.specification import (
    ScalingStrategy,
)


def test_standard_scaler_uses_only_fit_indices() -> None:
    features = np.array(
        [
            [0.0],
            [2.0],
            [1000.0],
        ],
        dtype=np.float32,
    )

    train_idx = np.array(
        [0, 1],
        dtype=np.int64,
    )

    validation_idx = np.array(
        [2],
        dtype=np.int64,
    )

    preprocessor = fit_feature_preprocessor(
        features,
        train_idx,
        strategy=ScalingStrategy.STANDARD,
    )

    transformed_train = (
        preprocessor.transform_rows(
            features,
            train_idx,
        )
    )

    transformed_validation = (
        preprocessor.transform_rows(
            features,
            validation_idx,
        )
    )

    # Mean and variance come only from [0, 2].
    assert transformed_train[0, 0] == pytest.approx(-1.0)
    assert transformed_train[1, 0] == pytest.approx(1.0)

    # If validation=1000 had influenced fitting, this value
    # would be different.
    assert transformed_validation[0, 0] > 100.0


def test_final_refit_produces_new_preprocessing_parameters() -> None:
    features = np.array(
        [
            [0.0],
            [2.0],
            [4.0],
            [1000.0],
        ],
        dtype=np.float32,
    )

    train_idx = np.array(
        [0, 1],
        dtype=np.int64,
    )

    validation_idx = np.array(
        [2],
        dtype=np.int64,
    )

    train_validation_idx = np.concatenate(
        [train_idx, validation_idx]
    )

    test_idx = np.array(
        [3],
        dtype=np.int64,
    )

    selection_preprocessor = (
        fit_feature_preprocessor(
            features,
            train_idx,
            strategy=ScalingStrategy.STANDARD,
        )
    )

    final_preprocessor = (
        fit_feature_preprocessor(
            features,
            train_validation_idx,
            strategy=ScalingStrategy.STANDARD,
        )
    )

    selection_validation_value = (
        selection_preprocessor.transform_rows(
            features,
            validation_idx,
        )[0, 0]
    )

    final_validation_value = (
        final_preprocessor.transform_rows(
            features,
            validation_idx,
        )[0, 0]
    )

    assert (
        selection_validation_value
        != pytest.approx(final_validation_value)
    )

    # Test observation was not part of either fit.
    final_test_value = (
        final_preprocessor.transform_rows(
            features,
            test_idx,
        )[0, 0]
    )

    assert final_test_value > 100.0


def test_none_strategy_preserves_values() -> None:
    features = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=np.float32,
    )

    indices = np.array(
        [0, 1],
        dtype=np.int64,
    )

    preprocessor = fit_feature_preprocessor(
        features,
        indices,
        strategy=ScalingStrategy.NONE,
    )

    transformed = (
        preprocessor.transform_rows(
            features,
            indices,
        )
    )

    np.testing.assert_array_equal(
        transformed,
        features,
    )

    assert transformed.dtype == np.float32


def test_yeo_johnson_skewed_transforms_only_skewed_columns() -> None:
    rng = np.random.default_rng(0)

    # column 0: heavily right-skewed (exponential-like), |skew| >> 1.
    # column 1: roughly symmetric, |skew| well under 1.
    skewed_column = rng.exponential(scale=1.0, size=200).astype(np.float32)
    symmetric_column = rng.normal(loc=5.0, scale=1.0, size=200).astype(np.float32)

    features = np.column_stack([skewed_column, symmetric_column]).astype(np.float32)
    fit_idx = np.arange(200, dtype=np.int64)

    preprocessor = fit_feature_preprocessor(
        features,
        fit_idx,
        strategy=ScalingStrategy.YEO_JOHNSON_SKEWED,
    )

    transformed = preprocessor.transform_rows(features, fit_idx)

    # skewed column must actually change...
    assert not np.allclose(transformed[:, 0], features[:, 0])
    # ...and the resulting distribution must be far less skewed than raw.
    raw_skew = _skew(features[:, 0].astype(np.float64))
    transformed_skew = _skew(transformed[:, 0].astype(np.float64))
    assert abs(transformed_skew) < abs(raw_skew)

    # symmetric column must stay byte-identical to the raw input.
    np.testing.assert_array_equal(transformed[:, 1], features[:, 1])


def test_yeo_johnson_skewed_constant_column_is_left_untouched() -> None:
    constant_column = np.full(50, 7.0, dtype=np.float32)
    other_column = np.linspace(0.0, 1.0, 50, dtype=np.float32)

    features = np.column_stack([constant_column, other_column]).astype(np.float32)
    fit_idx = np.arange(50, dtype=np.int64)

    preprocessor = fit_feature_preprocessor(
        features,
        fit_idx,
        strategy=ScalingStrategy.YEO_JOHNSON_SKEWED,
    )

    transformed = preprocessor.transform_rows(features, fit_idx)

    np.testing.assert_array_equal(transformed[:, 0], features[:, 0])


def test_yeo_johnson_skewed_uses_only_fit_indices() -> None:
    rng = np.random.default_rng(1)

    fit_values = rng.exponential(scale=1.0, size=100).astype(np.float32)
    outlier = np.array([1_000_000.0], dtype=np.float32)

    features = np.concatenate([fit_values, outlier]).reshape(-1, 1)
    fit_idx = np.arange(100, dtype=np.int64)
    outlier_idx = np.array([100], dtype=np.int64)

    preprocessor_with_outlier_excluded = fit_feature_preprocessor(
        features,
        fit_idx,
        strategy=ScalingStrategy.YEO_JOHNSON_SKEWED,
    )

    preprocessor_with_outlier_included = fit_feature_preprocessor(
        features,
        np.concatenate([fit_idx, outlier_idx]),
        strategy=ScalingStrategy.YEO_JOHNSON_SKEWED,
    )

    value_excluded = preprocessor_with_outlier_excluded.transform_rows(
        features, fit_idx
    )[0, 0]
    value_included = preprocessor_with_outlier_included.transform_rows(
        features, fit_idx
    )[0, 0]

    assert value_excluded != pytest.approx(value_included)


def test_yeo_johnson_skewed_refit_produces_new_preprocessing_parameters() -> None:
    rng = np.random.default_rng(2)

    train_values = rng.exponential(scale=1.0, size=100).astype(np.float32)
    validation_values = rng.exponential(scale=1.0, size=20).astype(np.float32)

    features = np.concatenate([train_values, validation_values]).reshape(-1, 1)
    train_idx = np.arange(100, dtype=np.int64)
    validation_idx = np.arange(100, 120, dtype=np.int64)
    train_validation_idx = np.concatenate([train_idx, validation_idx])

    selection_preprocessor = fit_feature_preprocessor(
        features,
        train_idx,
        strategy=ScalingStrategy.YEO_JOHNSON_SKEWED,
    )

    final_preprocessor = fit_feature_preprocessor(
        features,
        train_validation_idx,
        strategy=ScalingStrategy.YEO_JOHNSON_SKEWED,
    )

    selection_value = selection_preprocessor.transform_rows(
        features, validation_idx
    )[0, 0]
    final_value = final_preprocessor.transform_rows(features, validation_idx)[0, 0]

    assert selection_value != pytest.approx(final_value)


def test_yeo_johnson_skewed_preserves_float32_dtype() -> None:
    rng = np.random.default_rng(3)
    features = rng.exponential(scale=1.0, size=(50, 2)).astype(np.float32)
    indices = np.arange(50, dtype=np.int64)

    preprocessor = fit_feature_preprocessor(
        features,
        indices,
        strategy=ScalingStrategy.YEO_JOHNSON_SKEWED,
    )

    transformed = preprocessor.transform_rows(features, indices)

    assert transformed.dtype == np.float32


def test_yeo_johnson_skewed_empty_fit_indices_are_rejected() -> None:
    features = np.zeros((10, 3), dtype=np.float32)
    empty = np.array([], dtype=np.int64)

    with pytest.raises(ValueError, match="At least one"):
        fit_feature_preprocessor(
            features,
            empty,
            strategy=ScalingStrategy.YEO_JOHNSON_SKEWED,
        )


def _skew(values: np.ndarray) -> float:
    centered = values - values.mean()
    std = values.std()
    return float((centered**3).mean() / std**3)


def test_empty_fit_indices_are_rejected() -> None:
    features = np.zeros(
        (10, 3),
        dtype=np.float32,
    )

    empty = np.array(
        [],
        dtype=np.int64,
    )

    with pytest.raises(
        ValueError,
        match="At least one",
    ):
        fit_feature_preprocessor(
            features,
            empty,
            strategy=ScalingStrategy.STANDARD,
        )