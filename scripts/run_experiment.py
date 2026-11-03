from __future__ import annotations

import argparse
from pathlib import Path

from bitcoin_aml.experiments.config import load_experiment_config
from bitcoin_aml.experiments.runner import ExperimentRunner
from bitcoin_aml.utils.logging import configure_logging, get_logger

# -------------------------------------------------------
# main executable script for experiments
#
# Arguments for calling the script:
#   * --config: filepath to experiment configuration
#   * --output-root: directory the program writes result files to
# -------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a configured experiment end to end."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/local21/lr_local21_strict.yaml"),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/runs/single"),
    )

    return parser.parse_args()


def main() -> None:
    configure_logging()
    logger = get_logger(__name__)

    args = parse_args()

    spec = load_experiment_config(args.config)
    runner = ExperimentRunner(output_root=args.output_root)

    results = runner.run(spec)

    for result in results:
        logger.info(
            "seed=%s run_id=%s test_average_precision=%.4f test_f1=%.4f manifest=%s",
            result.seed,
            result.run_id,
            result.test_metrics.overall.average_precision,
            result.test_metrics.overall.f1,
            result.manifest_path,
        )


if __name__ == "__main__":
    main()