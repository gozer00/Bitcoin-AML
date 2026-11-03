import json
from pathlib import Path

import pytest

from bitcoin_aml.artifacts.manifest import build_run_manifest
from bitcoin_aml.artifacts.run_store import save_run
from bitcoin_aml.evaluation.metrics import BinaryMetrics
from bitcoin_aml.evaluation.temporal import TemporalMetrics


def _sample_metrics() -> TemporalMetrics:
    overall = BinaryMetrics(
        precision=0.5,
        recall=0.6,
        f1=0.55,
        average_precision=0.4,
        true_positives=3,
        false_positives=2,
        true_negatives=10,
        false_negatives=1,
        threshold=0.5,
    )
    per_timestep_35 = BinaryMetrics(
        precision=1.0,
        recall=1.0,
        f1=1.0,
        average_precision=1.0,
        true_positives=1,
        false_positives=0,
        true_negatives=2,
        false_negatives=0,
        threshold=0.5,
    )

    return TemporalMetrics(overall=overall, per_timestep={35: per_timestep_35})


def _sample_manifest(run_id: str = "run_1") -> dict:
    return build_run_manifest(
        run_id=run_id,
        experiment_name="test_experiment",
        config_hash="abc123",
        resolved_config={"name": "test_experiment"},
        dataset_version="elliptic_v1",
        dataset_manifest={"schema_version": 1},
        seed=42,
        selected_hyperparameters={"C": 1.0},
        model_selection_metric="average_precision",
        model_selection_metric_value=0.9,
        test_metrics=_sample_metrics(),
        created_at="2026-08-14T00:00:00+00:00",
    )


def test_build_run_manifest_shape() -> None:
    manifest = _sample_manifest()

    assert manifest["schema_version"] == 1
    assert manifest["run_id"] == "run_1"
    assert manifest["config_hash"] == "abc123"
    assert manifest["seed"] == 42
    assert manifest["model_selection"]["selected_hyperparameters"] == {"C": 1.0}
    assert manifest["test_metrics"]["overall"]["precision"] == 0.5
    assert manifest["test_metrics"]["per_timestep"]["35"]["true_positives"] == 1


def test_save_run_writes_valid_json(tmp_path: Path) -> None:
    manifest = _sample_manifest()

    manifest_path = save_run(tmp_path, manifest)

    assert manifest_path == tmp_path / "run_1" / "manifest.json"

    with manifest_path.open(encoding="utf-8") as file:
        loaded = json.load(file)

    assert loaded["run_id"] == "run_1"


def test_save_run_raises_on_existing_run_dir(tmp_path: Path) -> None:
    manifest = _sample_manifest()
    save_run(tmp_path, manifest)

    with pytest.raises(FileExistsError):
        save_run(tmp_path, manifest)