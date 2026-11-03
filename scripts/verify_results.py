from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

from bitcoin_aml.utils.logging import configure_logging, get_logger

# -------------------------------------------------------
# compares rebuilt tables under outputs/tables with the committed results/tables
#
#   * --mode archive: tables rebuilt from the release archive must match exactly
#   * --mode rerun: tables rebuilt from a fresh benchmark run. Tabular models
#     must match. GNNs are not bit-reproducible on GPU (see docs/reproducing.md),
#     so their mean AP is only flagged when it moves beyond the published
#     cross-seed variation
# -------------------------------------------------------

_GROUPS = ("local21", "reference", "evolvegcn")
_DETERMINISTIC_PREFIXES = ("lr_", "rf_", "xgboost_", "mlp_")
_NUMERIC_TOLERANCE = 1e-9
_RERUN_EXACT_TOLERANCE = 1e-6
_GNN_STD_FACTOR = 2.0
_GNN_MIN_TOLERANCE = 0.01

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare rebuilt tables with the committed results."
    )
    parser.add_argument("--mode", choices=("archive", "rerun"), default="archive")
    parser.add_argument("--rebuilt", type=Path, default=Path("outputs/tables"))
    parser.add_argument("--published", type=Path, default=Path("results/tables"))
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _cells_match(left: str, right: str) -> bool:
    if left == right:
        return True

    try:
        return math.isclose(
            float(left), float(right), rel_tol=0.0, abs_tol=_NUMERIC_TOLERANCE
        )
    except ValueError:
        return False


def compare_exact(rebuilt_root: Path, published_root: Path) -> list[str]:
    """
    differences across every published csv, one message per differing file
    """
    problems = []

    for published in sorted(published_root.rglob("*.csv")):
        relative = published.relative_to(published_root)
        rebuilt = rebuilt_root / relative

        if not rebuilt.exists():
            problems.append(f"{relative}: missing")
            continue

        left, right = _read_csv(rebuilt), _read_csv(published)

        if len(left) != len(right):
            problems.append(f"{relative}: {len(left)} rows, expected {len(right)}")
            continue

        n_cells = sum(
            not _cells_match(row_a.get(key, ""), value)
            for row_a, row_b in zip(left, right)
            for key, value in row_b.items()
        )

        if n_cells:
            problems.append(f"{relative}: {n_cells} differing cells")

    return problems


def compare_rerun(
    rebuilt_root: Path, published_root: Path
) -> tuple[list[str], list[str]]:
    """
    (failures for tabular models, flags for GNNs), on mean average precision
    """
    failures, flags = [], []

    for group in _GROUPS:
        rebuilt_path = rebuilt_root / group / "results_summary.csv"

        if not rebuilt_path.exists():
            failures.append(f"{group}: missing {rebuilt_path}")
            continue

        rebuilt = {row["experiment_name"]: row for row in _read_csv(rebuilt_path)}

        for row in _read_csv(published_root / group / "results_summary.csv"):
            name = row["experiment_name"]

            if name not in rebuilt:
                failures.append(f"{name}: missing")
                continue

            published_ap = float(row["average_precision_mean"])
            delta = float(rebuilt[name]["average_precision_mean"]) - published_ap
            std = float(row["average_precision_std"] or 0.0)

            if name.startswith(_DETERMINISTIC_PREFIXES):
                if abs(delta) > _RERUN_EXACT_TOLERANCE:
                    failures.append(f"{name}: mean AP differs by {delta:+.6f}")
            else:
                tolerance = max(_GNN_STD_FACTOR * std, _GNN_MIN_TOLERANCE)

                if abs(delta) > tolerance:
                    flags.append(f"{name}: mean AP differs by {delta:+.4f}")

    return failures, flags


def main() -> None:
    configure_logging()
    args = parse_args()

    if args.mode == "archive":
        failures, flags = compare_exact(args.rebuilt, args.published), []
    else:
        failures, flags = compare_rerun(args.rebuilt, args.published)

    for message in flags:
        logger.warning("beyond cross-seed variation: %s", message)

    for message in failures:
        logger.error("mismatch: %s", message)

    if failures:
        sys.exit(1)

    logger.info("verified, %d GNN results flagged", len(flags))


if __name__ == "__main__":
    main()
