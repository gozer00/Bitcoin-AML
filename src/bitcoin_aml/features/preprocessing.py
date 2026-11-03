# handler for model independent transformations (e.g. scaling)
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import PowerTransformer, StandardScaler

from bitcoin_aml.features.specification import (
    ScalingStrategy,
)

# threshold on |Fisher-Pearson skewness|, computed on admissible fit rows,
# above which a feature is treated as "extremely skewed" and receives a
# Yeo-Johnson power transform under ScalingStrategy.YEO_JOHNSON_SKEWED.
# 1.0 is the standard stats-textbook cutoff for "substantial/extreme" skew
# (0.5-1 is usually called "moderate"). Features at or below this stay
# untouched, matching ScalingStrategy.NONE's passthrough for those columns.
_SKEW_THRESHOLD = 1.0


@dataclass(frozen=True, slots=True)
class FittedFeaturePreprocessor:
    """
    Immutable handle to preprocessing parameters fitted on one
    explicitly supplied set of observations.
    """

    strategy: ScalingStrategy
    num_features: int

    _scaler: StandardScaler | None = field(
        default=None,
        repr=False,
    )

    _skewed_mask: NDArray[np.bool_] | None = field(
        default=None,
        repr=False,
    )

    _power_transformer: PowerTransformer | None = field(
        default=None,
        repr=False,
    )

    def transform_rows(
        self,
        features: NDArray[np.floating],
        indices: NDArray[np.integer],
    ) -> NDArray[np.float32]:
        """
        Materialize and transform only the requested rows.
        """
        _validate_feature_matrix(
            features,
            expected_features=self.num_features,
        )

        _validate_indices(
            indices,
            n_rows=features.shape[0],
            allow_empty=True,
        )

        selected = np.ascontiguousarray(
            features[indices],
            dtype=np.float32,
        )

        if not np.isfinite(selected).all():
            raise ValueError(
                "Selected feature rows contain NaN or infinite values."
            )

        if self.strategy == ScalingStrategy.NONE:
            return selected

        if self.strategy == ScalingStrategy.STANDARD:
            if self._scaler is None:
                raise RuntimeError(
                    "Standard preprocessing has no fitted scaler."
                )

            transformed = self._scaler.transform(
                selected
            )

            return np.ascontiguousarray(
                transformed,
                dtype=np.float32,
            )

        if self.strategy == ScalingStrategy.YEO_JOHNSON_SKEWED:
            if self._skewed_mask is None:
                raise RuntimeError(
                    "Selective Yeo-Johnson preprocessing has no fitted mask."
                )

            transformed = selected.copy()

            if self._power_transformer is not None:
                transformed[:, self._skewed_mask] = self._power_transformer.transform(
                    selected[:, self._skewed_mask]
                )

            return np.ascontiguousarray(
                transformed,
                dtype=np.float32,
            )

        raise RuntimeError(
            f"Unsupported fitted preprocessing strategy: "
            f"{self.strategy}"
        )


def fit_feature_preprocessor(
    features: NDArray[np.floating],
    fit_indices: NDArray[np.integer],
    *,
    strategy: ScalingStrategy,
) -> FittedFeaturePreprocessor:
    """
    Fit preprocessing statistics using exactly fit_indices.

    This function deliberately has no fit_transform shortcut.
    """
    _validate_feature_matrix(features)

    _validate_indices(
        fit_indices,
        n_rows=features.shape[0],
        allow_empty=False,
    )

    fit_rows = np.ascontiguousarray(
        features[fit_indices],
        dtype=np.float32,
    )

    if not np.isfinite(fit_rows).all():
        raise ValueError(
            "Preprocessing fit rows contain NaN or infinite values."
        )

    if strategy == ScalingStrategy.NONE:
        return FittedFeaturePreprocessor(
            strategy=strategy,
            num_features=features.shape[1],
        )

    if strategy == ScalingStrategy.STANDARD:
        scaler = StandardScaler(
            copy=True,
            with_mean=True,
            with_std=True,
        )

        scaler.fit(fit_rows)

        return FittedFeaturePreprocessor(
            strategy=strategy,
            num_features=features.shape[1],
            _scaler=scaler,
        )

    if strategy == ScalingStrategy.YEO_JOHNSON_SKEWED:
        skewness = _feature_skewness(fit_rows)
        skewed_mask = np.abs(skewness) > _SKEW_THRESHOLD

        power_transformer = None

        if skewed_mask.any():
            power_transformer = PowerTransformer(
                method="yeo-johnson",
                standardize=True,
            )
            power_transformer.fit(fit_rows[:, skewed_mask])

        return FittedFeaturePreprocessor(
            strategy=strategy,
            num_features=features.shape[1],
            _skewed_mask=skewed_mask,
            _power_transformer=power_transformer,
        )

    raise ValueError(
        f"Unsupported preprocessing strategy: {strategy}"
    )


def _feature_skewness(rows: NDArray[np.floating]) -> NDArray[np.float64]:
    """
    Per-column Fisher-Pearson skewness (population moment coefficient),
    computed directly in numpy rather than via scipy (not a project
    dependency). Zero-variance columns are reported as skewness 0.0
    (never flagged as skewed) instead of raising on the 0/0 division.
    """
    values = rows.astype(np.float64)
    mean = values.mean(axis=0)
    centered = values - mean
    std = values.std(axis=0)

    skewness = np.zeros(values.shape[1], dtype=np.float64)
    nonzero_std = std > 0

    skewness[nonzero_std] = (
        (centered[:, nonzero_std] ** 3).mean(axis=0) / std[nonzero_std] ** 3
    )

    return skewness


def _validate_feature_matrix(
    features: NDArray[np.floating],
    *,
    expected_features: int | None = None,
) -> None:
    if features.ndim != 2:
        raise ValueError(
            "Feature matrix must have shape (N, F)."
        )

    if not np.issubdtype(
        features.dtype,
        np.floating,
    ):
        raise TypeError(
            "Feature matrix must contain floating-point values."
        )

    if expected_features is not None:
        if features.shape[1] != expected_features:
            raise ValueError(
                "Feature count differs from the fitted preprocessor. "
                f"Expected {expected_features}, "
                f"received {features.shape[1]}."
            )


def _validate_indices(
    indices: NDArray[np.integer],
    *,
    n_rows: int,
    allow_empty: bool,
) -> None:
    if indices.ndim != 1:
        raise ValueError(
            "Indices must be one-dimensional."
        )

    if not np.issubdtype(
        indices.dtype,
        np.integer,
    ):
        raise TypeError(
            "Indices must contain integers."
        )

    if len(indices) == 0:
        if allow_empty:
            return

        raise ValueError(
            "At least one preprocessing fit row is required."
        )

    if int(indices.min()) < 0:
        raise ValueError(
            "Indices contain a negative row index."
        )

    if int(indices.max()) >= n_rows:
        raise ValueError(
            "Indices contain a row outside the feature matrix."
        )