import json
import math
from pathlib import Path

import pytest

from bitcoin_aml.artifacts.aggregation import (
    aggregate_by_config_hash,
    build_per_timestep_rows,
    build_summary_rows,
    load_run_manifests,
)
from bitcoin_aml.artifacts.manifest import build_run_manifest
from bitcoin_aml.artifacts.run_store import save_run
from bitcoin_aml.evaluation.metrics import BinaryMetrics
from bitcoin_aml.evaluation.temporal import TemporalMetrics


def _metrics(
    *, precision: float, recall: float, f1: float, average_precision: float
) -> BinaryMetrics:
    return BinaryMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        average_precision=average_precision,
        true_positives=1,
        false_positives=0,
        true_negatives=1,
        false_negatives=0,
        threshold=0.5,
    )


def _manifest(
    *,
    run_id: str,
    experiment_name: str,
    config_hash: str,
    seed: int,
    overall_precision: float,
    timestep_35_precision: float,
) -> dict:
    test_metrics = TemporalMetrics(
        overall=_metrics(
            precision=overall_precision,
            recall=0.6,
            f1=0.55,
            average_precision=0.4,
        ),
        per_timestep={
            35: _metrics(
                precision=timestep_35_precision,
                recall=0.6,
                f1=0.55,
                average_precision=0.4,
            )
        },
    )

    return build_run_manifest(
        run_id=run_id,
        experiment_name=experiment_name,
        config_hash=config_hash,
        resolved_config={"name": experiment_name},
        dataset_version="elliptic_v1",
        dataset_manifest={"schema_version": 1},
        seed=seed,
        selected_hyperparameters={"C": 1.0},
        model_selection_metric="average_precision",
        model_selection_metric_value=0.9,
        test_metrics=test_metrics,
        created_at="2026-08-14T00:00:00+00:00",
    )


def test_aggregate_by_config_hash_computes_mean_and_std_across_seeds() -> None:
    manifests = [
        _manifest(
            run_id="run_1",
            experiment_name="lr_local93_strict",
            config_hash="hash_a",
            seed=42,
            overall_precision=0.4,
            timestep_35_precision=0.2,
        ),
        _manifest(
            run_id="run_2",
            experiment_name="lr_local93_strict",
            config_hash="hash_a",
            seed=43,
            overall_precision=0.6,
            timestep_35_precision=0.8,
        ),
    ]

    aggregated = aggregate_by_config_hash(manifests)

    assert set(aggregated) == {"hash_a"}
    entry = aggregated["hash_a"]

    assert entry["experiment_name"] == "lr_local93_strict"
    assert entry["seeds"] == [42, 43]
    assert entry["n_seeds"] == 2
    assert entry["overall"]["precision_mean"] == pytest.approx(0.5)
    assert entry["overall"]["precision_std"] == pytest.approx(0.1)
    assert entry["overall"]["precision_n"] == 2
    assert entry["per_timestep"][35]["precision_mean"] == pytest.approx(0.5)
    assert entry["per_timestep"][35]["precision_std"] == pytest.approx(0.3)


def test_aggregate_by_config_hash_keeps_distinct_hashes_separate() -> None:
    manifests = [
        _manifest(
            run_id="run_1",
            experiment_name="lr_local93_strict",
            config_hash="hash_a",
            seed=42,
            overall_precision=0.4,
            timestep_35_precision=0.2,
        ),
        _manifest(
            run_id="run_2",
            experiment_name="rf_local93_strict",
            config_hash="hash_b",
            seed=42,
            overall_precision=0.9,
            timestep_35_precision=0.9,
        ),
    ]

    aggregated = aggregate_by_config_hash(manifests)

    assert set(aggregated) == {"hash_a", "hash_b"}
    assert aggregated["hash_a"]["n_seeds"] == 1
    assert aggregated["hash_b"]["n_seeds"] == 1
    assert aggregated["hash_a"]["overall"]["precision_std"] == 0.0


def test_aggregate_by_config_hash_rejects_mismatched_experiment_names() -> None:
    manifests = [
        _manifest(
            run_id="run_1",
            experiment_name="lr_local93_strict",
            config_hash="hash_a",
            seed=42,
            overall_precision=0.4,
            timestep_35_precision=0.2,
        ),
        _manifest(
            run_id="run_2",
            experiment_name="rf_local93_strict",
            config_hash="hash_a",
            seed=43,
            overall_precision=0.5,
            timestep_35_precision=0.3,
        ),
    ]

    with pytest.raises(ValueError, match="hash_a"):
        aggregate_by_config_hash(manifests)


def test_aggregate_by_config_hash_excludes_nan_average_precision_from_mean() -> None:
    nan_metrics = TemporalMetrics(
        overall=_metrics(
            precision=0.5, recall=0.5, f1=0.5, average_precision=float("nan")
        ),
        per_timestep={},
    )
    finite_metrics = TemporalMetrics(
        overall=_metrics(precision=0.5, recall=0.5, f1=0.5, average_precision=0.8),
        per_timestep={},
    )

    manifests = [
        build_run_manifest(
            run_id="run_1",
            experiment_name="lr_local93_strict",
            config_hash="hash_a",
            resolved_config={},
            dataset_version="elliptic_v1",
            dataset_manifest={},
            seed=42,
            selected_hyperparameters={},
            model_selection_metric="average_precision",
            model_selection_metric_value=0.9,
            test_metrics=nan_metrics,
            created_at="2026-08-14T00:00:00+00:00",
        ),
        build_run_manifest(
            run_id="run_2",
            experiment_name="lr_local93_strict",
            config_hash="hash_a",
            resolved_config={},
            dataset_version="elliptic_v1",
            dataset_manifest={},
            seed=43,
            selected_hyperparameters={},
            model_selection_metric="average_precision",
            model_selection_metric_value=0.9,
            test_metrics=finite_metrics,
            created_at="2026-08-14T00:00:00+00:00",
        ),
    ]

    aggregated = aggregate_by_config_hash(manifests)

    assert aggregated["hash_a"]["overall"]["average_precision_mean"] == pytest.approx(0.8)
    assert aggregated["hash_a"]["overall"]["average_precision_n"] == 1


def test_build_summary_rows_and_per_timestep_rows_flatten_aggregated() -> None:
    manifests = [
        _manifest(
            run_id="run_1",
            experiment_name="lr_local93_strict",
            config_hash="hash_a",
            seed=42,
            overall_precision=0.4,
            timestep_35_precision=0.2,
        ),
    ]

    aggregated = aggregate_by_config_hash(manifests)

    summary_rows = build_summary_rows(aggregated)
    assert summary_rows == [
        {
            "experiment_name": "lr_local93_strict",
            "config_hash": "hash_a",
            "n_seeds": 1,
            "seeds": "42",
            **aggregated["hash_a"]["overall"],
        }
    ]

    per_timestep_rows = build_per_timestep_rows(aggregated)
    assert per_timestep_rows == [
        {
            "experiment_name": "lr_local93_strict",
            "config_hash": "hash_a",
            "timestep": 35,
            "n_seeds": 1,
            **aggregated["hash_a"]["per_timestep"][35],
        }
    ]


def test_load_run_manifests_reads_every_manifest_under_runs_root(
    tmp_path: Path,
) -> None:
    manifest_a = _manifest(
        run_id="run_1",
        experiment_name="lr_local93_strict",
        config_hash="hash_a",
        seed=42,
        overall_precision=0.4,
        timestep_35_precision=0.2,
    )
    manifest_b = _manifest(
        run_id="run_2",
        experiment_name="lr_local93_strict",
        config_hash="hash_a",
        seed=43,
        overall_precision=0.6,
        timestep_35_precision=0.8,
    )

    save_run(tmp_path, manifest_a)
    save_run(tmp_path, manifest_b)

    loaded = load_run_manifests(tmp_path)

    assert {manifest["run_id"] for manifest in loaded} == {"run_1", "run_2"}
