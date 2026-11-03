from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

from bitcoin_aml.artifacts.aggregation import load_run_manifests
from bitcoin_aml.artifacts.predictions_store import PREDICTIONS_FILENAME
from bitcoin_aml.experiments.config import hash_experiment_spec, load_experiment_config
from bitcoin_aml.experiments.runner import ExperimentRunner
from bitcoin_aml.utils.logging import configure_logging, get_logger

# -------------------------------------------------------
# runs a benchmark group end to end
#
# Each directory under configs/experiments is one group, and each group writes
# to its own directory under --runs-root:
#   * local21: main benchmark on the 21 transaction-local features
#   * reference: local_93/all_165 runs that measure feature-leakage inflation
#   * evolvegcn: appendix comparison of EvolveGCN-H and EvolveGCN-O
#
# Arguments for calling the script:
#   * --group: benchmark group, or "all"
#   * --smoke: runs two fast configs at one seed into --runs-root/smoke
#   * --skip-existing: resumes an interrupted run
#   * --runs-root: parent directory of the group output directories
# -------------------------------------------------------

_CONFIGS_ROOT = Path("configs/experiments")
_GROUPS = ("local21", "reference", "evolvegcn")
_SMOKE_CONFIGS = (
    _CONFIGS_ROOT / "local21" / "lr_local21_strict_yeojohnson.yaml",
    _CONFIGS_ROOT / "local21" / "predecessor_gnn_two_hop_local21_incoming_strict.yaml",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a benchmark group of configured experiments."
    )

    parser.add_argument("--group", choices=(*_GROUPS, "all"), default="all")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run two fast configs at seed 42 only, ignoring --group",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip config/seed pairs that already have a finished run",
    )
    parser.add_argument("--runs-root", type=Path, default=Path("outputs/runs"))

    return parser.parse_args()


def _finished_runs(output_root: Path) -> set[tuple[str, int]]:
    """
    (config_hash, seed) of every run under output_root

    a manifest without predictions marks a run interrupted while persisting,
    which would be counted twice if it were run again next to the partial run
    """
    finished = set()

    for manifest in load_run_manifests(output_root):
        run_dir = output_root / manifest["run_id"]

        if not (run_dir / PREDICTIONS_FILENAME).exists():
            raise SystemExit(f"incomplete run, delete it and resume: {run_dir}")

        finished.add((manifest["config_hash"], manifest["seed"]))

    return finished


def _run_configs(
    config_paths: list[Path],
    *,
    output_root: Path,
    skip_existing: bool,
    seeds: tuple[int, ...] | None = None,
) -> None:
    logger = get_logger(__name__)
    runner = ExperimentRunner(output_root=output_root)
    finished = _finished_runs(output_root) if skip_existing else set()

    for config_path in config_paths:
        spec = load_experiment_config(config_path)

        if seeds is not None:
            spec = dataclasses.replace(spec, seeds=seeds)

        config_hash = hash_experiment_spec(spec)
        missing_seeds = tuple(
            seed for seed in spec.seeds if (config_hash, seed) not in finished
        )

        if not missing_seeds:
            logger.info("=== skipping finished config: %s ===", config_path.name)
            continue

        logger.info("=== running experiment with config: %s ===", config_path.name)

        for result in runner.run(dataclasses.replace(spec, seeds=missing_seeds)):
            logger.info(
                "seed=%s run_id=%s test_average_precision=%.4f test_f1=%.4f manifest=%s",
                result.seed,
                result.run_id,
                result.test_metrics.overall.average_precision,
                result.test_metrics.overall.f1,
                result.manifest_path,
            )


def main() -> None:
    configure_logging()
    args = parse_args()

    if args.smoke:
        _run_configs(
            list(_SMOKE_CONFIGS),
            output_root=args.runs_root / "smoke",
            skip_existing=args.skip_existing,
            seeds=(42,),
        )
        return

    groups = _GROUPS if args.group == "all" else (args.group,)

    for group in groups:
        _run_configs(
            sorted((_CONFIGS_ROOT / group).glob("*.yaml")),
            output_root=args.runs_root / group,
            skip_existing=args.skip_existing,
        )


if __name__ == "__main__":
    main()
