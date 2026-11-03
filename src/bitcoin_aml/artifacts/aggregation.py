# cross-seed aggregation of persisted run manifests

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Iterable

METRIC_NAMES = ("precision", "recall", "f1", "average_precision")


def load_run_manifests(runs_root: str | Path) -> list[dict[str, Any]]:
    """
    reads every {runs_root}/{run_id}/manifest.json, sorted by run_id
    """
    manifests = [
        json.loads(manifest_path.read_text(encoding="utf-8"))
        for manifest_path in sorted(Path(runs_root).glob("*/manifest.json"))
    ]

    return manifests


def aggregate_by_config_hash(
    manifests: Iterable[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """
    groups manifests by config_hash -- i.e. one group per resolved experiment
    across its configured seeds -- and computes mean/std/n of precision,
    recall, f1, and average_precision, both overall and per test timestep
    """
    grouped: dict[str, list[dict[str, Any]]] = {}

    for manifest in manifests:
        grouped.setdefault(manifest["config_hash"], []).append(manifest)

    aggregated: dict[str, dict[str, Any]] = {}

    for config_hash, group in grouped.items():
        experiment_names = {manifest["experiment_name"] for manifest in group}

        if len(experiment_names) != 1:
            raise ValueError(
                f"config_hash {config_hash!r} spans multiple experiment names: "
                f"{sorted(experiment_names)}"
            )

        seeds = sorted(manifest["seed"] for manifest in group)

        timesteps = sorted(
            {
                int(timestep)
                for manifest in group
                for timestep in manifest["test_metrics"]["per_timestep"]
            }
        )

        aggregated[config_hash] = {
            "experiment_name": next(iter(experiment_names)),
            "config_hash": config_hash,
            "seeds": seeds,
            "n_seeds": len(seeds),
            "overall": _aggregate_metric_group(
                manifest["test_metrics"]["overall"] for manifest in group
            ),
            "per_timestep": {
                timestep: _aggregate_metric_group(
                    manifest["test_metrics"]["per_timestep"][str(timestep)]
                    for manifest in group
                    if str(timestep) in manifest["test_metrics"]["per_timestep"]
                )
                for timestep in timesteps
            },
        }

    return aggregated


def build_summary_rows(aggregated: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """
    flattens aggregated overall metrics into one row per experiment
    """
    rows = []

    for config_hash in sorted(aggregated):
        entry = aggregated[config_hash]
        row = {
            "experiment_name": entry["experiment_name"],
            "config_hash": config_hash,
            "n_seeds": entry["n_seeds"],
            "seeds": ",".join(str(seed) for seed in entry["seeds"]),
        }
        row.update(entry["overall"])
        rows.append(row)

    return rows


def build_per_timestep_rows(
    aggregated: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    flattens aggregated per-timestep metrics into one row per experiment x timestep
    """
    rows = []

    for config_hash in sorted(aggregated):
        entry = aggregated[config_hash]

        for timestep in sorted(entry["per_timestep"]):
            row = {
                "experiment_name": entry["experiment_name"],
                "config_hash": config_hash,
                "timestep": timestep,
                "n_seeds": entry["n_seeds"],
            }
            row.update(entry["per_timestep"][timestep])
            rows.append(row)

    return rows


def _aggregate_metric_group(
    metric_dicts: Iterable[dict[str, Any]],
) -> dict[str, float]:
    metric_dicts = list(metric_dicts)
    aggregate: dict[str, float] = {}

    for metric_name in METRIC_NAMES:
        values = [
            metric_dict[metric_name]
            for metric_dict in metric_dicts
            if not _is_nan(metric_dict[metric_name])
        ]

        aggregate[f"{metric_name}_mean"] = (
            statistics.fmean(values) if values else float("nan")
        )
        aggregate[f"{metric_name}_std"] = (
            statistics.pstdev(values) if values else float("nan")
        )
        aggregate[f"{metric_name}_n"] = len(values)

    return aggregate


def _is_nan(value: float) -> bool:
    return value != value
