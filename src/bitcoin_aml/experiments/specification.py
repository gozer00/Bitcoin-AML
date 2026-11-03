from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping

from bitcoin_aml.features.specification import FeatureSpec
from bitcoin_aml.graphs.specification import (
    GraphAccessPolicy,
    GraphSpec,
)
from bitcoin_aml.splits.specification import SplitSpec

class SelectionDirection(StrEnum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class DecisionRuleType(StrEnum):
    FIXED_THRESHOLD = "fixed_threshold"


class PreprocessingFitScope(StrEnum):
    LABELED_FIT_NODES = "labeled_fit_nodes"
    ALL_AVAILABLE_FIT_NODES = "all_available_fit_nodes"


@dataclass(frozen=True, slots=True)
class ModelSelectionSpec:
    enabled: bool
    metric: str | None = None
    direction: SelectionDirection = SelectionDirection.MAXIMIZE

    def __post_init__(self) -> None:
        if self.enabled and not self.metric:
            raise ValueError("selection metric required if model selection enabled")


@dataclass(frozen=True, slots=True)
class RefitSpec:
    enabled: bool
    include_validation: bool
    refit_processing: bool


@dataclass(frozen=True, slots=True)
class DecisionRuleSpec:
    type: DecisionRuleType
    threshold: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("classification threshold must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class EvaluationSpec:
    metrics: tuple[str, ...] = (
        "precision",
        "recall",
        "f1",
        "average_precision"
    )
    per_timestep: bool = True


@dataclass(frozen=True, slots=True)
class ProtocolSpec:
    name: str

    split: SplitSpec
    graph_access: GraphAccessPolicy

    model_selection: ModelSelectionSpec
    refit: RefitSpec
    decision_rules: DecisionRuleSpec
    evaluation: EvaluationSpec

    preprocessing_fit_scope: PreprocessingFitScope = (
        PreprocessingFitScope.LABELED_FIT_NODES
    )


@dataclass(frozen=True, slots=True)
class ModelSpec:
    name:str

    fixed_parameters: Mapping[str, Any] = field(
        default_factory=dict
    )

    search_space: Mapping[str, tuple[Any, ...]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("model name cannot be empty")

        for parameter, values in self.search_space.items():
            if len(values) == 0:
                raise ValueError(f"search space for parameter {parameter} cannot be empty")


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    name: str
    dataset_version: str

    protocol: ProtocolSpec
    features: FeatureSpec
    graph: GraphSpec
    model: ModelSpec

    seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("experiment name cannot be empty")

        if not self.dataset_version:
            raise ValueError("dataset version cannot be empty")

        if len(self.seeds) == 0:
            raise ValueError("at least one seed must be specified")

        if len(self.seeds) != len(set(self.seeds)):
            raise ValueError("random seeds must be unique")