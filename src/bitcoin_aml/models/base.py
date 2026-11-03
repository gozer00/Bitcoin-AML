# contains three fundamental structures
# 1. ModelInput -> dataclass: defines everything the model needs for training/prediction
# 2. ModelCapabilities -> dataclass: requirements and supported settings
# 3. ModelAdapter -> abstract super class: includes methods for prediction and fitting
#       workflow for new model:
#       1. implement adapter that inherits from ModelAdapter
#       2. register adapter in MODEL_REGISTRY
#       3. add configuration

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import torch


ArrayLike = np.ndarray | torch.Tensor


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    requires_graph: bool
    requires_directed_graph: bool
    support_inductive_inference: bool


@dataclass(frozen=True, slots=True)
class ModelInput:
    """
    phase-dependent representation of input to the model

    fit_idx -> nodes whose labels are admissible for fitting
    validation_idx -> nodes whose labels are admissible for hyperparam-tuning/early-stopping
    predict_idx -> nodes whose labels are admissible for prediction
    timestep -> optional per-node timestep, aligned with x/y; ignored by
        adapters with no temporal structure, consumed by temporal adapters
        (e.g. evolvegcn_o) that self-segment into per-timestep snapshots
    """

    x: ArrayLike
    y: ArrayLike

    fit_idx: np.ndarray[np.integer]
    predict_idx: np.ndarray[np.integer]

    validation_idx: np.ndarray[np.integer] | None = None
    edge_index: ArrayLike | None = None
    timestep: ArrayLike | None = None

    def __post_init__(self) -> None:
        n_nodes = len(self.y)

        if len(self.x) != n_nodes:
            raise ValueError("x and y have same length")

        self._validate_index("fit_idx", self.fit_idx, n_nodes)
        self._validate_index("predict_idx", self.predict_idx, n_nodes)

        if self.validation_idx is not None:
            self._validate_index(
                "validation_idx",
                self.validation_idx,
                n_nodes,
            )

        if self.edge_index is not None:
            if (
                self.edge_index.ndim != 2
                or self.edge_index.shape[0] != 2
            ):
                raise ValueError("edge_index must have shape (2, E)")

        if self.timestep is not None and len(self.timestep) != n_nodes:
            raise ValueError("timestep and y have same length")

    @staticmethod
    def _validate_index(
            name:str,
            index: np.ndarray[np.integer],
            n_nodes: int,
    ) -> None:
        if index.ndim != 1:
            raise ValueError(f"{name} must be one dimensional")

        if len(index) == 0:
            return

        if int(index.min()) < 0:
            raise ValueError(f"{name} contains negative index")

        if int(index.max()) >= n_nodes:
            raise ValueError(f"{name} contains index outside of model input")


class ModelAdapter(ABC):
    """
    interface to be implemented by each prediction model
    """

    capabilities: ClassVar[ModelCapabilities]

    @abstractmethod
    def fit(self, data: ModelInput, *, seed: int) -> None:
        """
        fits model using data.fit_idx | data.val_idx
        """
        raise NotImplementedError

    @abstractmethod
    def predict_scores(self, data: ModelInput) -> np.ndarray[np.floating]:
        """
        returns prediction scores for illicit-labeled indices from data.predict_idx
        shape: (len(data.predict_idx), )
        """
        raise NotImplementedError
