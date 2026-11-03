# run manifest construction

from __future__ import annotations

from typing import Any, Mapping

from bitcoin_aml.evaluation.metrics import BinaryMetrics
from bitcoin_aml.evaluation.temporal import TemporalMetrics

RUN_MANIFEST_SCHEMA_VERSION = 1


def build_run_manifest(
    *,
    run_id: str,
    experiment_name: str,
    config_hash: str,
    resolved_config: Mapping[str, Any],
    dataset_version: str,
    dataset_manifest: Mapping[str, Any],
    seed: int,
    selected_hyperparameters: Mapping[str, Any],
    model_selection_metric: str,
    model_selection_metric_value: float,
    test_metrics: TemporalMetrics,
    created_at: str,
    peak_gpu_memory_gb: float | None = None,
    peak_host_memory_gb: float | None = None,
) -> dict[str, Any]:
    """
    builds a JSON-safe run manifest recording everything required to
    reproduce and audit one seeded run of a resolved experiment

    `peak_gpu_memory_gb`/`peak_host_memory_gb` are nullable: only adapters
    that opt into tracking them (e.g. gcn for GPU, evolvegcn_o for both)
    report a value, so other adapters leave these as `None`
    """
    return {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": created_at,
        "experiment_name": experiment_name,
        "config_hash": config_hash,
        "resolved_config": resolved_config,
        "seed": seed,
        "dataset": {
            "version": dataset_version,
            "manifest": dataset_manifest,
        },
        "model_selection": {
            "metric": model_selection_metric,
            "metric_value": model_selection_metric_value,
            "selected_hyperparameters": selected_hyperparameters,
        },
        "peak_gpu_memory_gb": peak_gpu_memory_gb,
        "peak_host_memory_gb": peak_host_memory_gb,
        "test_metrics": {
            "overall": _binary_metrics_to_dict(test_metrics.overall),
            "per_timestep": {
                str(timestep): _binary_metrics_to_dict(metrics)
                for timestep, metrics in test_metrics.per_timestep.items()
            },
        },
    }


def _binary_metrics_to_dict(metrics: BinaryMetrics) -> dict[str, Any]:
    return {
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "average_precision": metrics.average_precision,
        "true_positives": metrics.true_positives,
        "false_positives": metrics.false_positives,
        "true_negatives": metrics.true_negatives,
        "false_negatives": metrics.false_negatives,
        "threshold": metrics.threshold,
    }