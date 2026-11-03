from __future__ import annotations

import argparse
import csv
from pathlib import Path

from bitcoin_aml.artifacts.aggregation import (
    aggregate_by_config_hash,
    build_per_timestep_rows,
    build_summary_rows,
    load_run_manifests,
)
from bitcoin_aml.utils.logging import configure_logging, get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate persisted run manifests into cross-seed result tables."
    )

    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("outputs/runs/local21"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/tables/local21"),
    )

    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    configure_logging()
    logger = get_logger(__name__)

    args = parse_args()

    manifests = load_run_manifests(args.runs_root)
    logger.info("loaded %d run manifests from %s", len(manifests), args.runs_root)

    aggregated = aggregate_by_config_hash(manifests)
    logger.info("aggregated into %d resolved experiments", len(aggregated))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = args.output_dir / "results_summary.csv"
    per_timestep_path = args.output_dir / "results_per_timestep.csv"

    _write_csv(summary_path, build_summary_rows(aggregated))
    _write_csv(per_timestep_path, build_per_timestep_rows(aggregated))

    for config_hash, entry in aggregated.items():
        logger.info(
            "experiment=%s config_hash=%s n_seeds=%d average_precision_mean=%.4f "
            "average_precision_std=%.4f f1_mean=%.4f",
            entry["experiment_name"],
            config_hash,
            entry["n_seeds"],
            entry["overall"]["average_precision_mean"],
            entry["overall"]["average_precision_std"],
            entry["overall"]["f1_mean"],
        )

    logger.info("wrote %s and %s", summary_path, per_timestep_path)


if __name__ == "__main__":
    main()
