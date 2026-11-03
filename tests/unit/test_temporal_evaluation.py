import math

import numpy as np

from bitcoin_aml.data.dataset import EllipticDataset
from bitcoin_aml.evaluation.metrics import compute_binary_metrics
from bitcoin_aml.evaluation.temporal import evaluate_per_timestep
from bitcoin_aml.splits.chronological import labeled_indices


def _scores_from_dataset(dataset: EllipticDataset, predict_idx: np.ndarray) -> np.ndarray:
    raw = dataset.features[predict_idx, 0]
    return (raw - raw.min()) / (raw.max() - raw.min())


def test_evaluate_per_timestep_matches_overall_and_groups_correctly(
    synthetic_elliptic_dataset: EllipticDataset,
) -> None:
    dataset = synthetic_elliptic_dataset

    all_idx = np.arange(dataset.num_nodes, dtype=np.int64)
    predict_idx = labeled_indices(dataset, all_idx)

    y_true = dataset.labels[predict_idx]
    scores = _scores_from_dataset(dataset, predict_idx)

    result = evaluate_per_timestep(dataset, predict_idx, y_true, scores, threshold=0.5)

    expected_overall = compute_binary_metrics(y_true, scores, threshold=0.5)
    assert result.overall == expected_overall

    observed_timesteps = sorted({int(t) for t in dataset.timestep[predict_idx]})
    assert sorted(result.per_timestep) == observed_timesteps

    total_tp = sum(m.true_positives for m in result.per_timestep.values())
    total_fp = sum(m.false_positives for m in result.per_timestep.values())
    total_tn = sum(m.true_negatives for m in result.per_timestep.values())
    total_fn = sum(m.false_negatives for m in result.per_timestep.values())

    assert total_tp == expected_overall.true_positives
    assert total_fp == expected_overall.false_positives
    assert total_tn == expected_overall.true_negatives
    assert total_fn == expected_overall.false_negatives


def test_single_class_timestep_yields_nan_average_precision(
    synthetic_elliptic_dataset: EllipticDataset,
) -> None:
    dataset = synthetic_elliptic_dataset

    all_idx = np.arange(dataset.num_nodes, dtype=np.int64)
    predict_idx = labeled_indices(dataset, all_idx)

    y_true = dataset.labels[predict_idx]
    scores = _scores_from_dataset(dataset, predict_idx)

    result = evaluate_per_timestep(dataset, predict_idx, y_true, scores, threshold=0.5)

    timesteps = dataset.timestep[predict_idx]
    single_class_timesteps = [
        int(ts)
        for ts in sorted({int(t) for t in timesteps})
        if len(set(y_true[timesteps == ts].tolist())) == 1
    ]

    assert single_class_timesteps, "fixture expected to contain a single-class timestep"

    for ts in single_class_timesteps:
        assert math.isnan(result.per_timestep[ts].average_precision)