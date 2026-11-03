from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class TimestepRange:
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 1:
            raise ValueError("timestep range must start at >= 1")

        if self.end < self.start:
            raise ValueError("timestep range end must be >= start ")

    def values(self) -> tuple[int, ...]:
        return tuple(range(self.start, self.end +1))


@dataclass(frozen=True, slots=True)
class SplitSpec:
    train: TimestepRange
    validation: TimestepRange | None
    test: TimestepRange

    def __post_init__(self) -> None:
        if self.validation is None:
            if self.train.end >= self.test.start:
                raise ValueError("train and test timesteps must be disjoint")
            return

        if self.train.end >= self.validation.start:
            raise ValueError("train and validation timesteps must be disjoint")

        if self.validation.end >= self.test.start:
            raise ValueError("validation and test timesteps must be disjoint")


@dataclass(frozen=True, slots=True)
class SplitIndices:
    """
    splits the canonical dataset temporally
    all nodes are considered
    """

    train: NDArray[np.integer]
    validation: NDArray[np.integer] | None
    test: NDArray[np.integer]

    def __post_init__(self) -> None:
        self._validate_index_array("train", self.train)
        self._validate_index_array("test", self.test)

        if self.validation is not None:
            self._validate_index_array(
                "validation",
                self.validation,
            )

        self._validate_disjointness()

    @staticmethod
    def _validate_index_array(
            name: str,
            indices: NDArray[np.integer],
    ) -> None:
        if indices.ndim != 1:
            raise ValueError(f"{name} indices must be one-dimensional.")

        if not np.issubdtype(indices.dtype, np.integer):
            raise TypeError(f"{name} indices must contain integers.")

        if len(indices) != len(np.unique(indices)):
            raise ValueError(f"{name} indices contain duplicates.")

    def _validate_disjointness(self) -> None:
        if np.intersect1d(self.train, self.test).size > 0:
            raise ValueError("Train and test indices overlap.")

        if self.validation is None:
            return

        if np.intersect1d(
                self.train,
                self.validation,
        ).size > 0:
            raise ValueError("Train and validation indices overlap.")

        if np.intersect1d(
                self.validation,
                self.test,
        ).size > 0:
            raise ValueError("Validation and test indices overlap.")