from pathlib import Path

import numpy as np
import pytest

from bitcoin_aml.artifacts.manifest import build_run_manifest
from bitcoin_aml.artifacts.predictions_store import load_predictions, save_predictions
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

    return TemporalMetrics(overall=overall, per_timestep={})


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


def test_save_predictions_round_trips_arrays(tmp_path: Path) -> None:
    save_run(tmp_path, _sample_manifest())

    tx_id = np.array([10_001, 10_002, 10_003, 10_004], dtype=np.int64)
    timestep = np.array([35, 35, 49, 49], dtype=np.int16)
    y_true = np.array([0, 1, 0, 1], dtype=np.int8)
    scores = np.array([0.1, 0.9, 0.2, 0.8], dtype=np.float64)

    predictions_path = save_predictions(
        tmp_path,
        "run_1",
        tx_id=tx_id,
        timestep=timestep,
        y_true=y_true,
        scores=scores,
    )

    assert predictions_path == tmp_path / "run_1" / "predictions.npz"

    loaded = np.load(predictions_path)

    np.testing.assert_array_equal(loaded["tx_id"], tx_id)
    np.testing.assert_array_equal(loaded["timestep"], timestep)
    np.testing.assert_array_equal(loaded["y_true"], y_true)
    np.testing.assert_array_equal(loaded["scores"], scores)

    assert loaded["tx_id"].dtype == tx_id.dtype
    assert loaded["timestep"].dtype == timestep.dtype
    assert loaded["y_true"].dtype == y_true.dtype
    assert loaded["scores"].dtype == scores.dtype


def test_load_predictions_round_trips_arrays(tmp_path: Path) -> None:
    save_run(tmp_path, _sample_manifest())

    tx_id = np.array([10_001, 10_002, 10_003, 10_004], dtype=np.int64)
    timestep = np.array([35, 35, 49, 49], dtype=np.int16)
    y_true = np.array([0, 1, 0, 1], dtype=np.int8)
    scores = np.array([0.1, 0.9, 0.2, 0.8], dtype=np.float64)

    save_predictions(
        tmp_path,
        "run_1",
        tx_id=tx_id,
        timestep=timestep,
        y_true=y_true,
        scores=scores,
    )

    loaded = load_predictions(tmp_path / "run_1")

    np.testing.assert_array_equal(loaded["tx_id"], tx_id)
    np.testing.assert_array_equal(loaded["timestep"], timestep)
    np.testing.assert_array_equal(loaded["y_true"], y_true)
    np.testing.assert_array_equal(loaded["scores"], scores)


def test_save_predictions_raises_if_run_dir_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        save_predictions(
            tmp_path,
            "run_1",
            tx_id=np.array([1], dtype=np.int64),
            timestep=np.array([35], dtype=np.int16),
            y_true=np.array([0], dtype=np.int8),
            scores=np.array([0.5], dtype=np.float64),
        )
