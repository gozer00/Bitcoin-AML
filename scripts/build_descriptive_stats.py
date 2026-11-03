from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bitcoin_aml.data.dataset import EllipticDataset
from bitcoin_aml.data.repository import EllipticRepository
from bitcoin_aml.data.schema import TransactionLabel
from bitcoin_aml.features.preprocessing import _SKEW_THRESHOLD, _feature_skewness
from bitcoin_aml.features.selectors import select_feature_view
from bitcoin_aml.features.specification import (
    FeatureSetName,
    FeatureSpec,
    ScalingStrategy,
)
from bitcoin_aml.utils.logging import configure_logging, get_logger

# Descriptive statistics of the prepared dataset: sample and label counts,
# per-timestep evolution, feature summaries by group (over all nodes, labeled
# or not), and a check of how far the raw columns are already normalized.

_BLUE = "#2a78d6"
_ORANGE = "#eb6834"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build descriptive statistics of the prepared dataset."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/processed/elliptic_v1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/tables/descriptive"),
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("outputs/figures"),
    )
    return parser.parse_args()


def build_summary_counts(dataset: EllipticDataset) -> pd.DataFrame:
    n_total = dataset.labels.shape[0]
    n_illicit = int((dataset.labels == int(TransactionLabel.ILLICIT)).sum())
    n_licit = int((dataset.labels == int(TransactionLabel.LICIT)).sum())
    n_unknown = int((dataset.labels == int(TransactionLabel.UNKNOWN)).sum())
    n_labeled = n_illicit + n_licit

    return pd.DataFrame(
        [
            {"quantity": "total_nodes", "count": n_total, "share_of_total": 1.0},
            {"quantity": "labeled_nodes", "count": n_labeled, "share_of_total": n_labeled / n_total},
            {"quantity": "unknown_nodes", "count": n_unknown, "share_of_total": n_unknown / n_total},
            {"quantity": "licit_nodes", "count": n_licit, "share_of_total": n_licit / n_total},
            {"quantity": "illicit_nodes", "count": n_illicit, "share_of_total": n_illicit / n_total},
            {"quantity": "illicit_share_of_labeled", "count": n_illicit, "share_of_total": n_illicit / n_labeled},
            {"quantity": "total_edges", "count": int(dataset.edge_index.shape[1]), "share_of_total": np.nan},
        ]
    )


def build_per_timestep(dataset: EllipticDataset) -> pd.DataFrame:
    rows = []

    for timestep in sorted(np.unique(dataset.timestep).tolist()):
        mask = dataset.timestep == timestep
        labels = dataset.labels[mask]
        n_illicit = int((labels == int(TransactionLabel.ILLICIT)).sum())
        n_licit = int((labels == int(TransactionLabel.LICIT)).sum())
        n_labeled = n_illicit + n_licit

        rows.append(
            {
                "timestep": int(timestep),
                "n_nodes": int(mask.sum()),
                "n_labeled": n_labeled,
                "n_illicit": n_illicit,
                "n_licit": n_licit,
                "illicit_share_of_labeled": (n_illicit / n_labeled) if n_labeled > 0 else np.nan,
            }
        )

    return pd.DataFrame(rows)


def build_feature_summary(dataset: EllipticDataset) -> pd.DataFrame:
    rows = []

    for feature_set in (FeatureSetName.LOCAL_21, FeatureSetName.LOCAL_93, FeatureSetName.AGGREGATE_72):
        spec = FeatureSpec(feature_set=feature_set, scaling=ScalingStrategy.NONE)
        view = select_feature_view(dataset, spec)

        means = view.mean(axis=0)
        stds = view.std(axis=0)
        skew = _feature_skewness(view)
        skewed_mask = np.abs(skew) > _SKEW_THRESHOLD

        rows.append(
            {
                "feature_group": feature_set.value,
                "n_columns": view.shape[1],
                "mean_of_column_means": float(means.mean()),
                "mean_of_column_stds": float(stds.mean()),
                "min_column_std": float(stds.min()),
                "max_column_std": float(stds.max()),
                "n_skewed_columns": int(skewed_mask.sum()),
                "share_skewed_columns": float(skewed_mask.mean()),
                "max_abs_skew": float(np.abs(skew).max()),
            }
        )

    return pd.DataFrame(rows)


def build_normalization_check(dataset: EllipticDataset) -> pd.DataFrame:
    means = dataset.features.mean(axis=0)
    stds = dataset.features.std(axis=0)

    return pd.DataFrame(
        [
            {
                "n_columns": dataset.features.shape[1],
                "mean_of_column_means": float(means.mean()),
                "min_column_mean": float(means.min()),
                "max_column_mean": float(means.max()),
                "mean_of_column_stds": float(stds.mean()),
                "min_column_std": float(stds.min()),
                "max_column_std": float(stds.max()),
                "n_columns_mean_near_zero_abs_lt_0p05": int((np.abs(means) < 0.05).sum()),
                "n_columns_std_near_one_0p9_1p1": int(((stds > 0.9) & (stds < 1.1)).sum()),
            }
        ]
    )


def plot_sample_trend(per_timestep: pd.DataFrame, output: Path) -> None:
    fig, (ax_counts, ax_share) = plt.subplots(2, 1, figsize=(6.5, 5.5), sharex=True)

    ax_counts.plot(per_timestep["timestep"], per_timestep["n_nodes"], color=_BLUE, lw=2, label="all nodes")
    ax_counts.plot(
        per_timestep["timestep"], per_timestep["n_labeled"], color=_ORANGE, lw=2, ls="--", label="labeled nodes"
    )
    ax_counts.set_ylabel("node count")
    ax_counts.legend(frameon=False, loc="upper right")
    ax_counts.spines[["top", "right"]].set_visible(False)
    ax_counts.grid(axis="y", color="0.85", lw=0.8)

    ax_share.plot(
        per_timestep["timestep"],
        per_timestep["illicit_share_of_labeled"],
        color=_BLUE,
        lw=2,
        marker="o",
        ms=3,
    )
    ax_share.set_ylabel("illicit share\nof labeled nodes")
    ax_share.set_xlabel("timestep")
    ax_share.spines[["top", "right"]].set_visible(False)
    ax_share.grid(axis="y", color="0.85", lw=0.8)

    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def main() -> None:
    configure_logging()
    logger = get_logger(__name__)
    args = parse_args()

    dataset = EllipticRepository.load(args.dataset_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    per_timestep = build_per_timestep(dataset)

    tables = {
        "summary_counts.csv": build_summary_counts(dataset),
        "descriptives_per_timestep.csv": per_timestep,
        "feature_summary_by_group.csv": build_feature_summary(dataset),
        "pre_normalization_check.csv": build_normalization_check(dataset),
    }
    for filename, table in tables.items():
        table.to_csv(args.output_dir / filename, index=False)
        logger.info("wrote %s", args.output_dir / filename)

    plot_sample_trend(per_timestep, args.figures_dir / "sample_trend.png")
    logger.info("wrote %s", args.figures_dir / "sample_trend.png")


if __name__ == "__main__":
    main()
