from __future__ import annotations

import argparse
import csv
import functools
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bitcoin_aml.artifacts.aggregation import load_run_manifests
from bitcoin_aml.artifacts.predictions_store import load_predictions
from bitcoin_aml.evaluation.alert_budget import compute_alert_budget_metrics
from bitcoin_aml.evaluation.bootstrap import (
    average_precision_metric,
    bootstrap_confidence_interval,
    f1_metric,
)
from bitcoin_aml.evaluation.calibration import compute_calibration_diagnostics
from bitcoin_aml.utils.logging import configure_logging, get_logger

# Post-hoc analysis of persisted predictions.npz files: calibration
# diagnostics, precision at alert budgets, and per-timestep bootstrap
# confidence intervals. No model is fit. See docs/experiment_protocols.md,
# "Post-hoc analyses".

_N_CALIBRATION_BINS = 10
_BUDGET_FRACTIONS = (0.005, 0.01, 0.02, 0.05, 0.10)
_BOOTSTRAP_SEED = 42
_N_RESAMPLES = 2000
_CONFIDENCE = 0.95

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build calibration, alert-budget, and temporal-bootstrap tables "
            "from persisted five-seed predictions."
        )
    )

    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("outputs/runs/local21"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/tables/post_hoc"),
    )

    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("outputs/figures/post_hoc"),
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


def _mean_std(values: list[float]) -> tuple[float, float, int]:
    clean = [value for value in values if value == value]  # drop NaN

    if not clean:
        return float("nan"), float("nan"), 0

    mean = statistics.fmean(clean)
    std = statistics.pstdev(clean) if len(clean) > 1 else 0.0

    return mean, std, len(clean)


def _group_manifests_by_config_hash(
    manifests: list[dict],
) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}

    for manifest in manifests:
        grouped.setdefault(manifest["config_hash"], []).append(manifest)

    return grouped


def _load_group_predictions(
    runs_root: Path, group: list[dict]
) -> list[dict[str, np.ndarray]]:
    return [load_predictions(runs_root / manifest["run_id"]) for manifest in group]


def _build_calibration_row(
    experiment_name: str, config_hash: str, predictions: list[dict]
) -> dict:
    briers = []
    eces = []

    for prediction in predictions:
        diagnostics = compute_calibration_diagnostics(
            prediction["y_true"], prediction["scores"], n_bins=_N_CALIBRATION_BINS
        )
        briers.append(diagnostics.brier_score)
        eces.append(diagnostics.expected_calibration_error)

    brier_mean, brier_std, brier_n = _mean_std(briers)
    ece_mean, ece_std, ece_n = _mean_std(eces)

    return {
        "experiment_name": experiment_name,
        "config_hash": config_hash,
        "n_seeds": len(predictions),
        "brier_score_mean": brier_mean,
        "brier_score_std": brier_std,
        "brier_score_n": brier_n,
        "expected_calibration_error_mean": ece_mean,
        "expected_calibration_error_std": ece_std,
        "expected_calibration_error_n": ece_n,
    }


def _plot_reliability_diagram(
    experiment_name: str, predictions: list[dict], figures_dir: Path
) -> None:
    pooled_y_true = np.concatenate([p["y_true"] for p in predictions])
    pooled_scores = np.concatenate([p["scores"] for p in predictions])

    diagnostics = compute_calibration_diagnostics(
        pooled_y_true, pooled_scores, n_bins=_N_CALIBRATION_BINS
    )

    confidences = np.array(diagnostics.bin_confidence)
    accuracies = np.array(diagnostics.bin_accuracy)
    valid = ~np.isnan(confidences)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfectly calibrated")
    ax.plot(confidences[valid], accuracies[valid], marker="o", label="observed")
    ax.set_xlabel("mean predicted score")
    ax.set_ylabel("empirical fraction illicit")
    ax.set_title(experiment_name, fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7)
    fig.tight_layout()

    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / f"{experiment_name}_reliability.png", dpi=150)
    plt.close(fig)


def _build_alert_budget_rows(
    experiment_name: str, config_hash: str, predictions: list[dict]
) -> list[dict]:
    per_seed_results = [
        compute_alert_budget_metrics(
            prediction["y_true"], prediction["scores"], _BUDGET_FRACTIONS
        )
        for prediction in predictions
    ]

    rows = []

    for budget_index, budget_fraction in enumerate(_BUDGET_FRACTIONS):
        precisions = [results[budget_index].precision for results in per_seed_results]
        recalls = [results[budget_index].recall for results in per_seed_results]
        costs = [
            results[budget_index].alerts_per_true_positive
            for results in per_seed_results
            if results[budget_index].alerts_per_true_positive != float("inf")
        ]

        precision_mean, precision_std, precision_n = _mean_std(precisions)
        recall_mean, recall_std, recall_n = _mean_std(recalls)
        cost_mean, cost_std, cost_n = _mean_std(costs)

        rows.append(
            {
                "experiment_name": experiment_name,
                "config_hash": config_hash,
                "n_seeds": len(predictions),
                "budget_fraction": budget_fraction,
                "num_alerts_mean": statistics.fmean(
                    results[budget_index].num_alerts for results in per_seed_results
                ),
                "precision_mean": precision_mean,
                "precision_std": precision_std,
                "precision_n": precision_n,
                "recall_mean": recall_mean,
                "recall_std": recall_std,
                "recall_n": recall_n,
                "alerts_per_true_positive_mean": cost_mean,
                "alerts_per_true_positive_std": cost_std,
                "alerts_per_true_positive_n": cost_n,
            }
        )

    return rows


def _build_bootstrap_rows(
    experiment_name: str,
    config_hash: str,
    group: list[dict],
    predictions: list[dict],
) -> list[dict]:
    threshold = group[0]["test_metrics"]["overall"]["threshold"]

    all_timesteps = sorted(
        {
            int(value)
            for prediction in predictions
            for value in np.unique(prediction["timestep"])
        }
    )

    # uses the fast numpy-only metric_fns (evaluation/bootstrap.py), not
    # evaluation/metrics.py::compute_binary_metrics -- sklearn's per-call
    # overhead is too slow at bootstrap scale (2000 resamples x 810
    # config/timestep groups); see average_precision_metric's docstring
    _f1_at_threshold = functools.partial(f1_metric, threshold=threshold)

    rows = []

    for timestep in all_timesteps:
        pooled_y_true = np.concatenate(
            [
                prediction["y_true"][prediction["timestep"] == timestep]
                for prediction in predictions
            ]
        )
        pooled_scores = np.concatenate(
            [
                prediction["scores"][prediction["timestep"] == timestep]
                for prediction in predictions
            ]
        )

        ap_ci = bootstrap_confidence_interval(
            pooled_y_true,
            pooled_scores,
            metric_fn=average_precision_metric,
            n_resamples=_N_RESAMPLES,
            confidence=_CONFIDENCE,
            seed=_BOOTSTRAP_SEED,
        )
        f1_ci = bootstrap_confidence_interval(
            pooled_y_true,
            pooled_scores,
            metric_fn=_f1_at_threshold,
            n_resamples=_N_RESAMPLES,
            confidence=_CONFIDENCE,
            seed=_BOOTSTRAP_SEED,
        )

        rows.append(
            {
                "experiment_name": experiment_name,
                "config_hash": config_hash,
                "timestep": timestep,
                "n_pooled_nodes": len(pooled_scores),
                "average_precision_point": ap_ci.point_estimate,
                "average_precision_ci_lower": ap_ci.lower,
                "average_precision_ci_upper": ap_ci.upper,
                "f1_point": f1_ci.point_estimate,
                "f1_ci_lower": f1_ci.lower,
                "f1_ci_upper": f1_ci.upper,
                "n_resamples": _N_RESAMPLES,
                "confidence": _CONFIDENCE,
            }
        )

    return rows


def main() -> None:
    configure_logging()

    args = parse_args()

    manifests = load_run_manifests(args.runs_root)
    logger.info("loaded %d run manifests from %s", len(manifests), args.runs_root)

    grouped = _group_manifests_by_config_hash(manifests)
    logger.info("grouped into %d resolved experiments", len(grouped))

    calibration_rows = []
    alert_budget_rows = []
    bootstrap_rows = []

    for config_hash in sorted(grouped):
        group = grouped[config_hash]
        experiment_name = group[0]["experiment_name"]

        predictions = _load_group_predictions(args.runs_root, group)

        calibration_row = _build_calibration_row(
            experiment_name, config_hash, predictions
        )
        calibration_rows.append(calibration_row)
        _plot_reliability_diagram(experiment_name, predictions, args.figures_dir)

        alert_budget_rows.extend(
            _build_alert_budget_rows(experiment_name, config_hash, predictions)
        )

        bootstrap_rows.extend(
            _build_bootstrap_rows(experiment_name, config_hash, group, predictions)
        )

        logger.info(
            "experiment=%s config_hash=%s n_seeds=%d brier_mean=%.4f ece_mean=%.4f",
            experiment_name,
            config_hash,
            len(predictions),
            calibration_row["brier_score_mean"],
            calibration_row["expected_calibration_error_mean"],
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(args.output_dir / "calibration_summary.csv", calibration_rows)
    _write_csv(args.output_dir / "alert_budget_summary.csv", alert_budget_rows)
    _write_csv(args.output_dir / "temporal_bootstrap_summary.csv", bootstrap_rows)

    logger.info(
        "wrote %d calibration rows, %d alert-budget rows, %d bootstrap rows to %s",
        len(calibration_rows),
        len(alert_budget_rows),
        len(bootstrap_rows),
        args.output_dir,
    )
    logger.info("wrote reliability diagrams to %s", args.figures_dir)


if __name__ == "__main__":
    main()
