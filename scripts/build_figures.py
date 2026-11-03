from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from bitcoin_aml.artifacts.predictions_store import load_predictions  # noqa: E402
from bitcoin_aml.evaluation.calibration import compute_calibration_diagnostics  # noqa: E402
from bitcoin_aml.utils.logging import configure_logging, get_logger  # noqa: E402

# Results figures of the thesis, built from persisted five-seed outputs only.
# Every figure shows the same five local_21 models with the same colour and
# marker per model (colour follows the entity, markers as secondary encoding,
# since the thesis is printed).

# (label, experiment name, colour, marker)
_SERIES: tuple[tuple[str, str, str, str], ...] = (
    ("Dir-GraphSAGE (2-hop, incoming)", "predecessor_gnn_two_hop_local21_incoming_strict", "#2a78d6", "o"),
    ("GraphSAGE (full batch)", "graphsage_full_batch_local21_bidirectional_strict", "#eb6834", "s"),
    ("RF", "rf_local21_strict", "#1baf7a", "^"),
    ("LR (Yeo-Johnson)", "lr_local21_strict_yeojohnson", "#eda100", "D"),
    ("XGBoost", "xgboost_local21_strict", "#e87ba4", "v"),
)

_GRID_COLOR = "#e4e3df"
_N_CALIBRATION_BINS = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build thesis figures.")
    parser.add_argument(
        "--per-timestep",
        type=Path,
        default=Path("outputs/tables/local21/results_per_timestep.csv"),
    )
    parser.add_argument(
        "--alert-budget",
        type=Path,
        default=Path("outputs/tables/post_hoc/alert_budget_summary.csv"),
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("outputs/runs/local21"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/figures"),
    )
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color=_GRID_COLOR, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)


def plot_per_timestep_ap(rows: list[dict[str, str]], output: Path) -> None:
    logger = get_logger(__name__)
    fig, ax = plt.subplots(figsize=(6.3, 3.4))

    # the per-timestep figure keeps its original four series
    for label, name, color, marker in _SERIES[:4]:
        points = sorted(
            (int(row["timestep"]), float(row["average_precision_mean"]))
            for row in rows
            if row["experiment_name"] == name and row["average_precision_mean"]
        )
        if not points:
            logger.warning("no per-timestep rows for %s", name)
            continue
        timesteps, values = zip(*points)
        ax.plot(timesteps, values, color=color, linewidth=1.5, marker=marker, markersize=4, label=label)

    ax.set_xlabel("Test timestep")
    ax.set_ylabel("Average precision")
    ax.set_xticks(range(35, 50))
    ax.set_ylim(0.0, 1.0)
    _style_axis(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    _save(fig, output)


def plot_precision_at_budget(rows: list[dict[str, str]], output: Path) -> None:
    logger = get_logger(__name__)
    fig, ax = plt.subplots(figsize=(6.3, 3.4))

    for label, name, color, marker in _SERIES:
        points = sorted(
            (100 * float(row["budget_fraction"]), float(row["precision_mean"]))
            for row in rows
            if row["experiment_name"] == name
        )
        if not points:
            logger.warning("no alert-budget rows for %s", name)
            continue
        budgets, values = zip(*points)
        ax.plot(budgets, values, color=color, linewidth=1.5, marker=marker, markersize=4, label=label)

    budgets = sorted({100 * float(row["budget_fraction"]) for row in rows})
    ax.set_xscale("log")
    ax.set_xticks(budgets)
    ax.set_xticklabels([f"{b:g}" for b in budgets])
    ax.minorticks_off()
    ax.set_xlabel("Alert budget (% of test transactions reviewed)")
    ax.set_ylabel("Precision")
    ax.set_ylim(0.0, 1.0)
    _style_axis(ax)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    _save(fig, output)


def _pooled_predictions(runs_root: Path, experiment_name: str) -> tuple[np.ndarray, np.ndarray]:
    """
    pools the test-set predictions of all seeds of one experiment
    """
    y_true: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    for manifest_path in sorted(runs_root.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["experiment_name"] != experiment_name:
            continue
        predictions = load_predictions(manifest_path.parent)
        y_true.append(predictions["y_true"])
        scores.append(predictions["scores"])
    if not y_true:
        raise FileNotFoundError(f"no runs for {experiment_name} in {runs_root}")
    return np.concatenate(y_true), np.concatenate(scores)


def plot_reliability(runs_root: Path, output: Path) -> None:
    fig, axes = plt.subplots(1, len(_SERIES), figsize=(6.3, 2.0), sharey=True, layout="constrained")

    for ax, (label, name, color, marker) in zip(axes, _SERIES):
        y_true, scores = _pooled_predictions(runs_root, name)
        diagnostics = compute_calibration_diagnostics(y_true, scores, n_bins=_N_CALIBRATION_BINS)
        confidence = np.array(diagnostics.bin_confidence)
        accuracy = np.array(diagnostics.bin_accuracy)
        valid = ~np.isnan(confidence)

        ax.plot([0, 1], [0, 1], linestyle="--", color="#8a8984", linewidth=0.8)
        ax.plot(confidence[valid], accuracy[valid], color=color, linewidth=1.5, marker=marker, markersize=3.5)
        ax.set_title(label.replace(" (", "\n("), fontsize=7)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.tick_params(labelsize=7)
        ax.set_aspect("equal")
        _style_axis(ax)

    axes[0].set_ylabel("Share illicit", fontsize=8)
    axes[len(axes) // 2].set_xlabel("Mean predicted score", fontsize=8)
    _save(fig, output)


def main() -> None:
    configure_logging()
    logger = get_logger(__name__)
    args = parse_args()

    outputs = {
        "fig_per_timestep_ap_local21.pdf": lambda path: plot_per_timestep_ap(_read_csv(args.per_timestep), path),
        "fig_precision_budget_local21.pdf": lambda path: plot_precision_at_budget(_read_csv(args.alert_budget), path),
        "fig_reliability_local21.pdf": lambda path: plot_reliability(args.runs_root, path),
    }
    for filename, build in outputs.items():
        build(args.output_dir / filename)
        logger.info("wrote %s", args.output_dir / filename)


if __name__ == "__main__":
    main()
