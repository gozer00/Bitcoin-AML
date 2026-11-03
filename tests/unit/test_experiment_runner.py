import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pytest

from bitcoin_aml.data.dataset import EllipticDataset
from bitcoin_aml.data.repository import ARRAY_FILES, DATASET_MANIFEST_FILENAME
from bitcoin_aml.experiments.runner import ExperimentRunner
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
from bitcoin_aml.graphs.specification import GraphAccessPolicy, GraphSpec, GraphViewName
from bitcoin_aml.splits.specification import SplitSpec, TimestepRange
from bitcoin_aml.utils.hashing import sha256_file


def _write_dataset_to_disk(root: Path, dataset: EllipticDataset) -> None:
    root.mkdir(parents=True, exist_ok=True)

    arrays = {
        "tx_id": dataset.tx_id,
        "timestep": dataset.timestep,
        "features": dataset.features,
        "labels": dataset.labels,
        "edge_index": dataset.edge_index,
    }

    array_metadata = {}

    for logical_name, array in arrays.items():
        filename = ARRAY_FILES[logical_name]
        path = root / filename
        np.save(path, np.ascontiguousarray(array), allow_pickle=False)

        array_metadata[logical_name] = {
            "file": filename,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "sha256": sha256_file(path),
        }

    manifest = {"schema_version": 1, "arrays": array_metadata}

    with (root / DATASET_MANIFEST_FILENAME).open("w", encoding="utf-8") as file:
        json.dump(manifest, file)


def _thesis_strict_spec(
    dataset_version: str,
    *,
    model_name: str,
    fixed_parameters: Mapping[str, Any],
    search_space: Mapping[str, tuple[Any, ...]],
    feature_set: FeatureSetName = FeatureSetName.LOCAL_93,
    graph: GraphSpec | None = None,
) -> ExperimentSpec:
    protocol = ProtocolSpec(
        name="thesis_strict_v1",
        split=SplitSpec(
            train=TimestepRange(1, 29),
            validation=TimestepRange(30, 34),
            test=TimestepRange(35, 49),
        ),
        graph_access=GraphAccessPolicy.STRICT_ROLLING,
        model_selection=ModelSelectionSpec(
            enabled=True,
            metric="average_precision",
            direction=SelectionDirection.MAXIMIZE,
        ),
        refit=RefitSpec(enabled=True, include_validation=True, refit_processing=True),
        decision_rules=DecisionRuleSpec(
            type=DecisionRuleType.FIXED_THRESHOLD, threshold=0.5
        ),
        evaluation=EvaluationSpec(),
    )

    model = ModelSpec(
        name=model_name,
        fixed_parameters=fixed_parameters,
        search_space=search_space,
    )

    return ExperimentSpec(
        name="synthetic_smoke",
        dataset_version=dataset_version,
        protocol=protocol,
        features=FeatureSpec(
            feature_set=feature_set, scaling=ScalingStrategy.STANDARD
        ),
        graph=graph if graph is not None else GraphSpec(view=GraphViewName.NONE),
        model=model,
        seeds=(1, 2),
    )


# every registered feature-only baseline must run end to end through the
# same model-agnostic runner, unmodified -- this is the guarantee that lets
# new tabular adapters be added without touching orchestration code
_FEATURE_ONLY_BASELINE_CASES = [
    pytest.param(
        "logistic_regression",
        {"max_iter": 200, "solver": "liblinear"},
        {"C": (1.0,)},
        id="logistic_regression",
    ),
    pytest.param(
        "random_forest",
        {"n_estimators": 10},
        {"max_depth": (3,)},
        id="random_forest",
    ),
    pytest.param(
        "xgboost",
        {"n_estimators": 10, "max_depth": 2},
        {"learning_rate": (0.1,)},
        id="xgboost",
    ),
    pytest.param(
        "mlp",
        {"hidden_dim": 4, "epochs": 20, "learning_rate": 0.05, "weight_decay": 0.0},
        {"dropout": (0.0,)},
        id="mlp",
    ),
]


@pytest.mark.parametrize(
    ("model_name", "fixed_parameters", "search_space"),
    _FEATURE_ONLY_BASELINE_CASES,
)
def test_experiment_runner_runs_end_to_end(
    tmp_path: Path,
    synthetic_elliptic_dataset: EllipticDataset,
    model_name: str,
    fixed_parameters: Mapping[str, Any],
    search_space: Mapping[str, tuple[Any, ...]],
) -> None:
    dataset_version = "synthetic_v1"
    data_root = tmp_path / "data"
    _write_dataset_to_disk(data_root / dataset_version, synthetic_elliptic_dataset)

    output_root = tmp_path / "outputs"
    runner = ExperimentRunner(output_root=output_root, data_root=data_root)

    spec = _thesis_strict_spec(
        dataset_version,
        model_name=model_name,
        fixed_parameters=fixed_parameters,
        search_space=search_space,
    )
    results = runner.run(spec)

    assert len(results) == len(spec.seeds)

    seen_seeds = {result.seed for result in results}
    assert seen_seeds == set(spec.seeds)

    for result in results:
        assert result.manifest_path.is_file()
        assert result.predictions_path.is_file()
        assert result.predictions_path.name == "predictions.npz"
        assert result.predictions_path.parent == result.manifest_path.parent

        with result.manifest_path.open(encoding="utf-8") as file:
            manifest = json.load(file)

        assert manifest["config_hash"] == result.config_hash
        assert manifest["seed"] == result.seed

        overall = result.test_metrics.overall
        assert 0.0 <= overall.precision <= 1.0
        assert 0.0 <= overall.recall <= 1.0
        assert 0.0 <= overall.f1 <= 1.0
        assert overall.num_illicit + overall.num_licit == (
            overall.true_positives
            + overall.false_positives
            + overall.true_negatives
            + overall.false_negatives
        )

    # all seeded runs share the same config hash
    assert len({result.config_hash for result in results}) == 1


# local_21 (the leakage-free transaction-local columns) must flow through
# the unchanged runner for both a feature-only and a predecessor-graph model,
# and every adapter must receive exactly 21 input columns
_LOCAL21_CASES = [
    pytest.param(
        "logistic_regression",
        {"max_iter": 200, "solver": "liblinear"},
        {"C": (1.0,)},
        GraphSpec(view=GraphViewName.NONE),
        id="logistic_regression",
    ),
    pytest.param(
        "predecessor_gnn_two_hop",
        {"epochs": 5, "learning_rate": 0.01, "weight_decay": 0.0},
        {"hidden_dim": (4,), "dropout": (0.0,)},
        GraphSpec(view=GraphViewName.INCOMING, hops=2),
        id="predecessor_gnn_two_hop",
    ),
]


@pytest.mark.parametrize(
    ("model_name", "fixed_parameters", "search_space", "graph"),
    _LOCAL21_CASES,
)
def test_experiment_runner_runs_local21(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    synthetic_elliptic_dataset: EllipticDataset,
    model_name: str,
    fixed_parameters: Mapping[str, Any],
    search_space: Mapping[str, tuple[Any, ...]],
    graph: GraphSpec,
) -> None:
    import bitcoin_aml.models  # noqa: F401  (registers every adapter)
    from bitcoin_aml.models.registry import MODEL_REGISTRY

    dataset_version = "synthetic_v1"
    data_root = tmp_path / "data"
    _write_dataset_to_disk(data_root / dataset_version, synthetic_elliptic_dataset)

    seen_widths: list[int] = []
    adapter_class = MODEL_REGISTRY[model_name]
    original_fit = adapter_class.fit

    def recording_fit(self, data, *, seed):  # type: ignore[no-untyped-def]
        seen_widths.append(np.asarray(data.x).shape[1])
        return original_fit(self, data, seed=seed)

    monkeypatch.setattr(adapter_class, "fit", recording_fit)

    runner = ExperimentRunner(output_root=tmp_path / "outputs", data_root=data_root)
    spec = _thesis_strict_spec(
        dataset_version,
        model_name=model_name,
        fixed_parameters=fixed_parameters,
        search_space=search_space,
        feature_set=FeatureSetName.LOCAL_21,
        graph=graph,
    )
    results = runner.run(spec)

    assert len(results) == len(spec.seeds)
    assert seen_widths and set(seen_widths) == {21}

    for result in results:
        with result.manifest_path.open(encoding="utf-8") as file:
            manifest = json.load(file)

        assert manifest["resolved_config"]["features"]["feature_set"] == "local_21"
