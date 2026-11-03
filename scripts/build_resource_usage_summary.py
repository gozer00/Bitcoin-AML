from __future__ import annotations

import argparse
import csv
from pathlib import Path

from bitcoin_aml.artifacts.aggregation import load_run_manifests
from bitcoin_aml.utils.logging import configure_logging, get_logger

# Aggregates the peak_gpu_memory_gb/peak_host_memory_gb values that every
# manifest.json records into one row per experiment.

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate peak GPU/host memory across the final five-seed benchmark."
    )

    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("outputs/runs/local21"),
    )

    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/tables/local21/resource_usage_summary.csv"),
    )

    return parser.parse_args()


def _max_or_none(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return max(present) if present else None


def _build_rows(manifests: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}

    for manifest in manifests:
        grouped.setdefault(manifest["config_hash"], []).append(manifest)

    rows = []

    for config_hash in sorted(grouped):
        group = grouped[config_hash]
        experiment_name = group[0]["experiment_name"]

        gpu_values = [manifest.get("peak_gpu_memory_gb") for manifest in group]
        host_values = [manifest.get("peak_host_memory_gb") for manifest in group]

        rows.append(
            {
                "experiment_name": experiment_name,
                "config_hash": config_hash,
                "n_seeds": len(group),
                "peak_gpu_memory_gb_max": _max_or_none(gpu_values),
                "peak_gpu_memory_gb_n_reported": sum(
                    1 for value in gpu_values if value is not None
                ),
                "peak_host_memory_gb_max": _max_or_none(host_values),
                "peak_host_memory_gb_n_reported": sum(
                    1 for value in host_values if value is not None
                ),
            }
        )

    return rows


def main() -> None:
    configure_logging()

    args = parse_args()

    manifests = load_run_manifests(args.runs_root)
    logger.info("loaded %d run manifests from %s", len(manifests), args.runs_root)

    rows = _build_rows(manifests)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    with args.output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        logger.info(
            "experiment=%s peak_gpu_gb=%s (n=%d) peak_host_gb=%s (n=%d)",
            row["experiment_name"],
            row["peak_gpu_memory_gb_max"],
            row["peak_gpu_memory_gb_n_reported"],
            row["peak_host_memory_gb_max"],
            row["peak_host_memory_gb_n_reported"],
        )

    logger.info("wrote %d rows to %s", len(rows), args.output_csv)


if __name__ == "__main__":
    main()
