# YAML resolution, validation, and deterministic hashing for experiment configs

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from bitcoin_aml.experiments.specification import (
    DecisionRuleSpec,
    DecisionRuleType,
    EvaluationSpec,
    ExperimentSpec,
    ModelSelectionSpec,
    ModelSpec,
    ProtocolSpec,
    RefitSpec,
    SelectionDirection,
)
from bitcoin_aml.features.specification import (
    FeatureSetName,
    FeatureSpec,
    ScalingStrategy,
)
from bitcoin_aml.graphs.specification import (
    GraphAccessPolicy,
    GraphSpec,
    GraphViewName,
)
from bitcoin_aml.splits.specification import SplitSpec, TimestepRange
from bitcoin_aml.utils.hashing import sha256_json

_SUPPORTED_REFIT_DATA = "train_validation"


class ConfigError(ValueError):
    """raised for invalid or malformed experiment configuration"""


def resolve_dataset_root(
    dataset_version: str,
    *,
    data_root: Path = Path("data/processed"),
) -> Path:
    """
    processed-data root for a dataset version, by convention
    """
    return data_root / dataset_version


def load_experiment_config(
    path: str | Path,
    *,
    configs_root: Path = Path("configs"),
) -> ExperimentSpec:
    """
    resolves an experiment YAML (plus its referenced protocol and model
    YAML files) into a validated, canonical ExperimentSpec
    """
    raw = _load_yaml(Path(path))

    _check_keys(
        raw,
        allowed={"name", "dataset", "protocol", "features", "graph", "model", "seeds"},
        context="experiment",
    )

    return ExperimentSpec(
        name=_require_str(raw, "name", context="experiment"),
        dataset_version=_resolve_dataset_version(raw, context="experiment"),
        protocol=_load_protocol(raw, configs_root),
        features=_build_feature_spec(
            _require_mapping(raw, "features", context="experiment")
        ),
        graph=_build_graph_spec(
            _require_mapping(raw, "graph", context="experiment")
        ),
        model=_load_model(raw, configs_root),
        seeds=_build_seeds(raw),
    )


def experiment_spec_to_dict(spec: ExperimentSpec) -> dict[str, Any]:
    """
    JSON-safe representation of a resolved ExperimentSpec, for hashing and
    for recording resolved configuration in run manifests
    """
    return {
        "name": spec.name,
        "dataset_version": spec.dataset_version,
        "protocol": _protocol_to_dict(spec.protocol),
        "features": {
            "feature_set": spec.features.feature_set.value,
            "scaling": spec.features.scaling.value,
        },
        "graph": {
            "view": spec.graph.view.value,
            "hops": spec.graph.hops,
        },
        "model": {
            "name": spec.model.name,
            "fixed_parameters": dict(spec.model.fixed_parameters),
            "search_space": {
                key: list(values)
                for key, values in spec.model.search_space.items()
            },
        },
        "seeds": list(spec.seeds),
    }


def hash_experiment_spec(spec: ExperimentSpec) -> str:
    """
    deterministic hash of the resolved configuration

    seeds are deliberately excluded: they identify individual runs of the
    same configuration, not the configuration itself
    """
    payload = experiment_spec_to_dict(spec)
    del payload["seeds"]

    return sha256_json(payload)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"config file does not exist: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ConfigError(f"config file must contain a mapping: {path}")

    return data


def _check_keys(raw: Mapping[str, Any], *, allowed: set[str], context: str) -> None:
    unknown = set(raw) - allowed

    if unknown:
        raise ConfigError(f"unknown keys in {context} config: {sorted(unknown)}")


def _require_str(raw: Mapping[str, Any], key: str, *, context: str) -> str:
    value = raw.get(key)

    if not isinstance(value, str) or not value:
        raise ConfigError(f"{context} config requires a non-empty string '{key}'")

    return value


def _require_mapping(
    raw: Mapping[str, Any], key: str, *, context: str
) -> dict[str, Any]:
    value = raw.get(key)

    if not isinstance(value, dict):
        raise ConfigError(f"{context} config requires a mapping '{key}'")

    return value


def _resolve_dataset_version(raw: Mapping[str, Any], *, context: str) -> str:
    dataset = _require_mapping(raw, "dataset", context=context)
    _check_keys(dataset, allowed={"version"}, context=f"{context}.dataset")
    return _require_str(dataset, "version", context=f"{context}.dataset")


def _load_protocol(raw: Mapping[str, Any], configs_root: Path) -> ProtocolSpec:
    protocol_name = _require_str(raw, "protocol", context="experiment")
    protocol_path = configs_root / "protocols" / f"{protocol_name}.yaml"
    protocol_raw = _load_yaml(protocol_path)

    return _build_protocol_spec(protocol_name, protocol_raw)


def _build_protocol_spec(
    expected_name: str, raw: Mapping[str, Any]
) -> ProtocolSpec:
    _check_keys(
        raw,
        allowed={
            "name",
            "split",
            "graph_access",
            "model_selection",
            "refit",
            "decision_rule",
            "evaluation",
        },
        context="protocol",
    )

    name = _require_str(raw, "name", context="protocol")

    if name != expected_name:
        raise ConfigError(
            f"protocol file name {name!r} does not match reference {expected_name!r}"
        )

    return ProtocolSpec(
        name=name,
        split=_build_split_spec(
            _require_mapping(raw, "split", context="protocol")
        ),
        graph_access=_build_graph_access_policy(
            _require_mapping(raw, "graph_access", context="protocol")
        ),
        model_selection=_build_model_selection_spec(raw.get("model_selection")),
        refit=_build_refit_spec(
            _require_mapping(raw, "refit", context="protocol")
        ),
        decision_rules=_build_decision_rule_spec(
            _require_mapping(raw, "decision_rule", context="protocol")
        ),
        evaluation=_build_evaluation_spec(raw.get("evaluation")),
    )


def _build_split_spec(raw: Mapping[str, Any]) -> SplitSpec:
    _check_keys(raw, allowed={"train", "validation", "test"}, context="protocol.split")

    validation = (
        None
        if raw.get("validation") is None
        else _build_timestep_range(raw, "validation")
    )

    return SplitSpec(
        train=_build_timestep_range(raw, "train"),
        validation=validation,
        test=_build_timestep_range(raw, "test"),
    )


def _build_timestep_range(raw: Mapping[str, Any], key: str) -> TimestepRange:
    value = raw.get(key)

    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or not all(isinstance(v, int) for v in value)
    ):
        raise ConfigError(
            f"protocol.split.{key} must be a two-integer [start, end] list"
        )

    start, end = value
    return TimestepRange(start=start, end=end)


def _build_graph_access_policy(raw: Mapping[str, Any]) -> GraphAccessPolicy:
    _check_keys(raw, allowed={"policy"}, context="protocol.graph_access")

    policy = _require_str(raw, "policy", context="protocol.graph_access")

    try:
        return GraphAccessPolicy(policy)
    except ValueError as exc:
        raise ConfigError(f"unknown graph access policy: {policy!r}") from exc


def _build_model_selection_spec(
    raw: Mapping[str, Any] | None,
) -> ModelSelectionSpec:
    if raw is None:
        return ModelSelectionSpec(enabled=False)

    if not isinstance(raw, dict):
        raise ConfigError("protocol.model_selection must be a mapping")

    _check_keys(raw, allowed={"metric", "direction"}, context="protocol.model_selection")

    metric = _require_str(raw, "metric", context="protocol.model_selection")

    direction_raw = raw.get("direction", SelectionDirection.MAXIMIZE.value)

    try:
        direction = SelectionDirection(direction_raw)
    except ValueError as exc:
        raise ConfigError(
            f"unknown model selection direction: {direction_raw!r}"
        ) from exc

    return ModelSelectionSpec(enabled=True, metric=metric, direction=direction)


def _build_refit_spec(raw: Mapping[str, Any]) -> RefitSpec:
    _check_keys(
        raw,
        allowed={"enabled", "data", "refit_preprocessing"},
        context="protocol.refit",
    )

    enabled = raw.get("enabled")

    if not isinstance(enabled, bool):
        raise ConfigError("protocol.refit.enabled must be a boolean")

    data = raw.get("data")

    if data != _SUPPORTED_REFIT_DATA:
        raise ConfigError(
            f"protocol.refit.data only supports {_SUPPORTED_REFIT_DATA!r}, "
            f"received {data!r}"
        )

    refit_preprocessing = raw.get("refit_preprocessing")

    if not isinstance(refit_preprocessing, bool):
        raise ConfigError(
            "protocol.refit.refit_preprocessing must be a boolean"
        )

    return RefitSpec(
        enabled=enabled,
        include_validation=True,
        refit_processing=refit_preprocessing,
    )


def _build_decision_rule_spec(raw: Mapping[str, Any]) -> DecisionRuleSpec:
    _check_keys(raw, allowed={"type", "threshold"}, context="protocol.decision_rule")

    type_raw = _require_str(raw, "type", context="protocol.decision_rule")

    try:
        rule_type = DecisionRuleType(type_raw)
    except ValueError as exc:
        raise ConfigError(f"unknown decision rule type: {type_raw!r}") from exc

    threshold = raw.get("threshold")

    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        raise ConfigError("protocol.decision_rule.threshold must be numeric")

    return DecisionRuleSpec(type=rule_type, threshold=float(threshold))


def _build_evaluation_spec(raw: Mapping[str, Any] | None) -> EvaluationSpec:
    if raw is None:
        return EvaluationSpec()

    if not isinstance(raw, dict):
        raise ConfigError("protocol.evaluation must be a mapping")

    _check_keys(raw, allowed={"metrics", "per_timestep"}, context="protocol.evaluation")

    metrics_raw = raw.get("metrics")

    if metrics_raw is None:
        metrics = EvaluationSpec().metrics
    else:
        if not isinstance(metrics_raw, list) or not all(
            isinstance(m, str) for m in metrics_raw
        ):
            raise ConfigError(
                "protocol.evaluation.metrics must be a list of strings"
            )
        metrics = tuple(metrics_raw)

    per_timestep = raw.get("per_timestep", True)

    if not isinstance(per_timestep, bool):
        raise ConfigError("protocol.evaluation.per_timestep must be a boolean")

    return EvaluationSpec(metrics=metrics, per_timestep=per_timestep)


def _load_model(raw: Mapping[str, Any], configs_root: Path) -> ModelSpec:
    model_block = _require_mapping(raw, "model", context="experiment")
    _check_keys(model_block, allowed={"name"}, context="experiment.model")

    model_name = _require_str(model_block, "name", context="experiment.model")
    model_path = configs_root / "models" / f"{model_name}.yaml"
    model_raw = _load_yaml(model_path)

    return _build_model_spec(model_name, model_raw)


def _build_model_spec(expected_name: str, raw: Mapping[str, Any]) -> ModelSpec:
    _check_keys(
        raw,
        allowed={"name", "fixed_parameters", "search_space"},
        context="model",
    )

    name = _require_str(raw, "name", context="model")

    if name != expected_name:
        raise ConfigError(
            f"model file name {name!r} does not match reference {expected_name!r}"
        )

    fixed_parameters = raw.get("fixed_parameters") or {}

    if not isinstance(fixed_parameters, dict):
        raise ConfigError("model.fixed_parameters must be a mapping")

    search_space_raw = raw.get("search_space") or {}

    if not isinstance(search_space_raw, dict):
        raise ConfigError("model.search_space must be a mapping")

    search_space: dict[str, tuple[Any, ...]] = {}

    for key, values in search_space_raw.items():
        if not isinstance(values, list):
            raise ConfigError(f"model.search_space.{key} must be a list")

        search_space[key] = tuple(values)

    return ModelSpec(
        name=name,
        fixed_parameters=fixed_parameters,
        search_space=search_space,
    )


def _build_feature_spec(raw: Mapping[str, Any]) -> FeatureSpec:
    _check_keys(raw, allowed={"set", "preprocessing"}, context="experiment.features")

    feature_set_raw = _require_str(raw, "set", context="experiment.features")

    try:
        feature_set = FeatureSetName(feature_set_raw)
    except ValueError as exc:
        raise ConfigError(f"unknown feature set: {feature_set_raw!r}") from exc

    preprocessing = _require_mapping(
        raw, "preprocessing", context="experiment.features"
    )
    _check_keys(
        preprocessing, allowed={"scaler"}, context="experiment.features.preprocessing"
    )

    scaler_raw = _require_str(
        preprocessing, "scaler", context="experiment.features.preprocessing"
    )

    try:
        scaling = ScalingStrategy(scaler_raw)
    except ValueError as exc:
        raise ConfigError(f"unknown scaling strategy: {scaler_raw!r}") from exc

    return FeatureSpec(feature_set=feature_set, scaling=scaling)


def _build_graph_spec(raw: Mapping[str, Any]) -> GraphSpec:
    _check_keys(raw, allowed={"view", "hops"}, context="experiment.graph")

    view_raw = _require_str(raw, "view", context="experiment.graph")

    try:
        view = GraphViewName(view_raw)
    except ValueError as exc:
        raise ConfigError(f"unknown graph view: {view_raw!r}") from exc

    hops = raw.get("hops")

    if hops is not None and not isinstance(hops, int):
        raise ConfigError("experiment.graph.hops must be an integer")

    return GraphSpec(view=view, hops=hops)


def _build_seeds(raw: Mapping[str, Any]) -> tuple[int, ...]:
    seeds_raw = raw.get("seeds")

    if (
        not isinstance(seeds_raw, list)
        or not seeds_raw
        or not all(isinstance(s, int) and not isinstance(s, bool) for s in seeds_raw)
    ):
        raise ConfigError("experiment.seeds must be a non-empty list of integers")

    return tuple(seeds_raw)


def _protocol_to_dict(protocol: ProtocolSpec) -> dict[str, Any]:
    return {
        "name": protocol.name,
        "split": {
            "train": [protocol.split.train.start, protocol.split.train.end],
            "validation": (
                None
                if protocol.split.validation is None
                else [
                    protocol.split.validation.start,
                    protocol.split.validation.end,
                ]
            ),
            "test": [protocol.split.test.start, protocol.split.test.end],
        },
        "graph_access": protocol.graph_access.value,
        "model_selection": {
            "enabled": protocol.model_selection.enabled,
            "metric": protocol.model_selection.metric,
            "direction": protocol.model_selection.direction.value,
        },
        "refit": {
            "enabled": protocol.refit.enabled,
            "include_validation": protocol.refit.include_validation,
            "refit_processing": protocol.refit.refit_processing,
        },
        "decision_rules": {
            "type": protocol.decision_rules.type.value,
            "threshold": protocol.decision_rules.threshold,
        },
        "evaluation": {
            "metrics": list(protocol.evaluation.metrics),
            "per_timestep": protocol.evaluation.per_timestep,
        },
        "preprocessing_fit_scope": protocol.preprocessing_fit_scope.value,
    }