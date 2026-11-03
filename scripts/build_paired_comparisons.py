from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from bitcoin_aml.artifacts.predictions_store import load_predictions
from bitcoin_aml.evaluation.bootstrap import (
    average_precision_metric,
    paired_bootstrap_difference,
)
from bitcoin_aml.utils.logging import configure_logging, get_logger

# Paired, timestep-stratified bootstrap of seed-averaged AP differences for
# the thesis's predefined contrasts. Pure post-hoc analysis of persisted
# predictions.npz files -- no model is refit.
#
#   leak  (SQ1): same model, leaky feature set vs local_21
#   graph (SQ2): predecessor GNN vs feature-only/static-graph models, local_21
#   dir   (SQ3): predecessor GNN incoming vs reversed/bidirectional, local_21

_CONTRASTS: tuple[tuple[str, str, str], ...] = (
    # SQ2 -- does predecessor message passing add information on local_21?
    ("graph", "predecessor_gnn_two_hop_local21_incoming_strict", "lr_local21_strict_yeojohnson"),
    ("graph", "predecessor_gnn_two_hop_local21_incoming_strict", "lr_local21_strict"),
    ("graph", "predecessor_gnn_two_hop_local21_incoming_strict", "rf_local21_strict"),
    ("graph", "predecessor_gnn_two_hop_local21_incoming_strict", "xgboost_local21_strict"),
    ("graph", "predecessor_gnn_two_hop_local21_incoming_strict", "mlp_local21_strict"),
    ("graph", "predecessor_gnn_two_hop_local21_incoming_strict", "gcn_local21_bidirectional_strict"),
    ("graph", "predecessor_gnn_two_hop_local21_incoming_strict", "graphsage_full_batch_local21_bidirectional_strict"),
    ("graph", "predecessor_gnn_two_hop_local21_incoming_strict", "graphsage_sampled_local21_bidirectional_strict"),
    ("graph", "predecessor_gnn_three_hop_local21_bidirectional_strict_yeojohnson", "lr_local21_strict_yeojohnson"),
    # SQ3 -- direction at fixed hop depth and scaling
    ("dir", "predecessor_gnn_one_hop_local21_incoming_strict", "predecessor_gnn_one_hop_local21_reversed_strict"),
    ("dir", "predecessor_gnn_one_hop_local21_incoming_strict", "predecessor_gnn_one_hop_local21_bidirectional_strict"),
    ("dir", "predecessor_gnn_two_hop_local21_incoming_strict", "predecessor_gnn_two_hop_local21_reversed_strict"),
    ("dir", "predecessor_gnn_two_hop_local21_incoming_strict", "predecessor_gnn_two_hop_local21_bidirectional_strict"),
    ("dir", "predecessor_gnn_two_hop_local21_incoming_strict_yeojohnson", "predecessor_gnn_two_hop_local21_reversed_strict_yeojohnson"),
    ("dir", "predecessor_gnn_two_hop_local21_incoming_strict_yeojohnson", "predecessor_gnn_two_hop_local21_bidirectional_strict_yeojohnson"),
    ("dir", "predecessor_gnn_three_hop_local21_incoming_strict", "predecessor_gnn_three_hop_local21_reversed_strict"),
    ("dir", "predecessor_gnn_three_hop_local21_incoming_strict", "predecessor_gnn_three_hop_local21_bidirectional_strict"),
    # SQ1 -- inflation from address-level (local_93) and all features
    ("leak", "lr_local93_strict", "lr_local21_strict"),
    ("leak", "lr_all165_strict", "lr_local21_strict"),
    ("leak", "rf_local93_strict", "rf_local21_strict"),
    ("leak", "rf_all165_strict", "rf_local21_strict"),
    ("leak", "xgboost_local93_strict", "xgboost_local21_strict"),
    ("leak", "xgboost_all165_strict", "xgboost_local21_strict"),
    ("leak", "mlp_local93_strict", "mlp_local21_strict"),
    ("leak", "mlp_all165_strict", "mlp_local21_strict"),
    ("leak", "gcn_local93_bidirectional_strict", "gcn_local21_bidirectional_strict"),
    ("leak", "gcn_all165_bidirectional_strict", "gcn_local21_bidirectional_strict"),
    ("leak", "graphsage_full_batch_local93_bidirectional_strict", "graphsage_full_batch_local21_bidirectional_strict"),
    ("leak", "graphsage_full_batch_all165_bidirectional_strict", "graphsage_full_batch_local21_bidirectional_strict"),
    ("leak", "predecessor_gnn_two_hop_local93_incoming_strict", "predecessor_gnn_two_hop_local21_incoming_strict"),
    ("leak", "predecessor_gnn_two_hop_all165_incoming_strict", "predecessor_gnn_two_hop_local21_incoming_strict"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired bootstrap AP differences for the thesis contrasts."
    )
    parser.add_argument(
        "--runs-roots",
        type=Path,
        nargs="+",
        default=[
            Path("outputs/runs/local21"),
            Path("outputs/runs/reference"),
        ],
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/tables/paired_comparisons/paired_ap_differences.csv"),
    )
    parser.add_argument("--n-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _index_runs(runs_roots: list[Path]) -> dict[str, list[Path]]:
    """
    experiment name -> run directories, first root wins on duplicates
    """
    index: dict[str, dict[int, Path]] = {}

    for root in runs_roots:
        for manifest_path in sorted(root.glob("*/manifest.json")):
            with manifest_path.open(encoding="utf-8") as file:
                manifest = json.load(file)
            seeds = index.setdefault(manifest["experiment_name"], {})
            seeds.setdefault(int(manifest["seed"]), manifest_path.parent)

    return {
        name: [seeds[seed] for seed in sorted(seeds)] for name, seeds in index.items()
    }


def _load_stacked(run_dirs: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    (tx_id, timestep, y_true, scores[n_seeds, n_nodes]), sorted by tx_id
    """
    reference = None
    rows = []

    for run_dir in run_dirs:
        predictions = load_predictions(run_dir)
        order = np.argsort(predictions["tx_id"], kind="stable")
        tx_id = predictions["tx_id"][order]

        if reference is None:
            reference = (
                tx_id,
                predictions["timestep"][order],
                predictions["y_true"][order],
            )
        elif not np.array_equal(reference[0], tx_id):
            raise ValueError(f"test nodes differ across runs: {run_dir}")

        rows.append(predictions["scores"][order])

    assert reference is not None
    return reference[0], reference[1], reference[2], np.stack(rows)


def main() -> None:
    configure_logging()
    logger = get_logger(__name__)
    args = parse_args()

    runs = _index_runs(args.runs_roots)
    cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    def load(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if name not in cache:
            if name not in runs:
                raise KeyError(f"no persisted runs for experiment {name!r}")
            cache[name] = _load_stacked(runs[name])
        return cache[name]

    rows = []
    for family, name_a, name_b in _CONTRASTS:
        tx_a, timestep, y_true, scores_a = load(name_a)
        tx_b, _, y_true_b, scores_b = load(name_b)

        if not (np.array_equal(tx_a, tx_b) and np.array_equal(y_true, y_true_b)):
            raise ValueError(f"{name_a} and {name_b} were scored on different nodes")

        result = paired_bootstrap_difference(
            y_true,
            scores_a,
            scores_b,
            metric_fn=average_precision_metric,
            strata=timestep,
            n_resamples=args.n_resamples,
            seed=args.seed,
        )

        rows.append(
            {
                "family": family,
                "model_a": name_a,
                "model_b": name_b,
                "n_seeds_a": scores_a.shape[0],
                "n_seeds_b": scores_b.shape[0],
                "ap_a": float(np.mean([average_precision_metric(y_true, s) for s in scores_a])),
                "ap_b": float(np.mean([average_precision_metric(y_true, s) for s in scores_b])),
                "delta_ap": result.point_estimate,
                "ci_lower": result.lower,
                "ci_upper": result.upper,
                "p_one_sided": result.fraction_non_positive,
                "n_resamples": result.n_resamples,
            }
        )
        logger.info(
            "%s: %s - %s = %.4f [%.4f, %.4f] p=%.4f",
            family,
            name_a,
            name_b,
            result.point_estimate,
            result.lower,
            result.upper,
            result.fraction_non_positive,
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    logger.info("wrote %d rows to %s", len(rows), args.output_csv)


if __name__ == "__main__":
    main()
