from pathlib import Path

import pytest

from bitcoin_aml.experiments.config import (
    ConfigError,
    hash_experiment_spec,
    load_experiment_config,
    resolve_dataset_root,
)
from bitcoin_aml.features.specification import FeatureSetName, ScalingStrategy

REAL_EXPERIMENT_CONFIG = Path("configs/experiments/reference/lr_local93_strict.yaml")
LOCAL21_EXPERIMENT_CONFIG_DIR = Path("configs/experiments/local21")


def _protocol_yaml(*, refit_data: str = "train_validation") -> str:
    return f"""\
name: test_protocol
split:
  train: [1, 5]
  validation: [6, 8]
  test: [9, 10]
graph_access:
  policy: strict_rolling
model_selection:
  metric: average_precision
refit:
  enabled: true
  data: {refit_data}
  refit_preprocessing: true
decision_rule:
  type: fixed_threshold
  threshold: 0.5
evaluation:
  metrics: [precision, recall, f1, average_precision]
  per_timestep: true
"""


def _model_yaml(*, max_iter: int = 100) -> str:
    return f"""\
name: test_model
fixed_parameters:
  max_iter: {max_iter}
search_space:
  C: [1.0]
"""


def _experiment_yaml(
    *,
    seeds: str = "[1]",
    extra: str = "",
    scaler: str = "standard",
    feature_set: str = "local_93",
) -> str:
    return f"""\
name: test_experiment
dataset:
  version: test_v1
protocol: test_protocol
features:
  set: {feature_set}
  preprocessing:
    scaler: {scaler}
graph:
  view: none
model:
  name: test_model
seeds: {seeds}
{extra}"""


def _write_valid_configs(
    tmp_path: Path,
    *,
    refit_data: str = "train_validation",
    seeds: str = "[1]",
    extra_experiment_yaml: str = "",
    scaler: str = "standard",
    feature_set: str = "local_93",
) -> Path:
    configs_root = tmp_path / "configs"
    (configs_root / "protocols").mkdir(parents=True, exist_ok=True)
    (configs_root / "models").mkdir(parents=True, exist_ok=True)
    (configs_root / "experiments").mkdir(parents=True, exist_ok=True)

    (configs_root / "protocols" / "test_protocol.yaml").write_text(
        _protocol_yaml(refit_data=refit_data)
    )
    (configs_root / "models" / "test_model.yaml").write_text(_model_yaml())

    experiment_path = configs_root / "experiments" / "exp.yaml"
    experiment_path.write_text(
        _experiment_yaml(
            seeds=seeds,
            extra=extra_experiment_yaml,
            scaler=scaler,
            feature_set=feature_set,
        )
    )

    return experiment_path


def test_load_real_experiment_config() -> None:
    spec = load_experiment_config(REAL_EXPERIMENT_CONFIG)

    assert spec.name == "lr_local93_strict"
    assert spec.dataset_version == "elliptic_v1"
    assert spec.protocol.name == "thesis_strict_v1"
    assert spec.model.name == "logistic_regression"
    assert spec.seeds == (42, 43, 44, 45, 46)
    assert spec.protocol.split.train.values() == tuple(range(1, 30))
    assert spec.protocol.split.validation.values() == tuple(range(30, 35))
    assert spec.protocol.split.test.values() == tuple(range(35, 50))


def test_resolve_dataset_root_uses_convention() -> None:
    assert resolve_dataset_root("elliptic_v1") == Path("data/processed/elliptic_v1")


def test_hash_is_stable_across_repeated_loads() -> None:
    first = hash_experiment_spec(load_experiment_config(REAL_EXPERIMENT_CONFIG))
    second = hash_experiment_spec(load_experiment_config(REAL_EXPERIMENT_CONFIG))

    assert first == second


def test_hash_ignores_seeds(tmp_path: Path) -> None:
    configs_root = tmp_path / "configs"
    path_a = _write_valid_configs(tmp_path, seeds="[1]")
    spec_a = load_experiment_config(path_a, configs_root=configs_root)

    path_b = _write_valid_configs(tmp_path, seeds="[1, 2, 3]")
    spec_b = load_experiment_config(path_b, configs_root=configs_root)

    assert spec_a.seeds != spec_b.seeds
    assert hash_experiment_spec(spec_a) == hash_experiment_spec(spec_b)


def test_hash_changes_when_a_resolved_field_changes(tmp_path: Path) -> None:
    configs_root = tmp_path / "configs"
    experiment_path = _write_valid_configs(tmp_path)
    spec_a = load_experiment_config(experiment_path, configs_root=configs_root)

    (configs_root / "models" / "test_model.yaml").write_text(_model_yaml(max_iter=999))
    spec_b = load_experiment_config(experiment_path, configs_root=configs_root)

    assert hash_experiment_spec(spec_a) != hash_experiment_spec(spec_b)


def test_unknown_top_level_key_rejected(tmp_path: Path) -> None:
    configs_root = tmp_path / "configs"
    experiment_path = _write_valid_configs(
        tmp_path, extra_experiment_yaml="bogus: true\n"
    )

    with pytest.raises(ConfigError):
        load_experiment_config(experiment_path, configs_root=configs_root)


def test_invalid_refit_data_value_rejected(tmp_path: Path) -> None:
    configs_root = tmp_path / "configs"
    experiment_path = _write_valid_configs(tmp_path, refit_data="test_only")

    with pytest.raises(ConfigError):
        load_experiment_config(experiment_path, configs_root=configs_root)


def test_yeo_johnson_skewed_scaler_round_trips(tmp_path: Path) -> None:
    configs_root = tmp_path / "configs"
    experiment_path = _write_valid_configs(tmp_path, scaler="yeo_johnson_skewed")

    spec = load_experiment_config(experiment_path, configs_root=configs_root)

    assert spec.features.scaling == ScalingStrategy.YEO_JOHNSON_SKEWED


def test_unknown_scaler_rejected(tmp_path: Path) -> None:
    configs_root = tmp_path / "configs"
    experiment_path = _write_valid_configs(tmp_path, scaler="min_max")

    with pytest.raises(ConfigError):
        load_experiment_config(experiment_path, configs_root=configs_root)


def test_mismatched_protocol_name_rejected(tmp_path: Path) -> None:
    configs_root = tmp_path / "configs"
    experiment_path = _write_valid_configs(tmp_path)

    (configs_root / "protocols" / "test_protocol.yaml").write_text(
        _protocol_yaml().replace("name: test_protocol", "name: something_else")
    )

    with pytest.raises(ConfigError):
        load_experiment_config(experiment_path, configs_root=configs_root)


def test_local21_feature_set_round_trips(tmp_path: Path) -> None:
    configs_root = tmp_path / "configs"
    experiment_path = _write_valid_configs(tmp_path, feature_set="local_21")

    spec = load_experiment_config(experiment_path, configs_root=configs_root)

    assert spec.features.feature_set == FeatureSetName.LOCAL_21


def test_local21_hash_differs_from_local93(tmp_path: Path) -> None:
    configs_root = tmp_path / "configs"
    path_93 = _write_valid_configs(tmp_path, feature_set="local_93")
    spec_93 = load_experiment_config(path_93, configs_root=configs_root)

    path_21 = _write_valid_configs(tmp_path, feature_set="local_21")
    spec_21 = load_experiment_config(path_21, configs_root=configs_root)

    assert hash_experiment_spec(spec_93) != hash_experiment_spec(spec_21)


@pytest.mark.parametrize(
    "config_path",
    sorted(LOCAL21_EXPERIMENT_CONFIG_DIR.glob("*.yaml")),
    ids=lambda path: path.stem,
)
def test_all_local21_configs_load(config_path: Path) -> None:
    spec = load_experiment_config(config_path)

    assert spec.name == config_path.stem
    assert spec.features.feature_set == FeatureSetName.LOCAL_21
    assert spec.seeds == (42, 43, 44, 45, 46)
