import numpy as np
import pytest

from bitcoin_aml.models.base import ModelInput
from bitcoin_aml.models.tabular.xgboost import XGBoostAdapter


def _separable_model_input() -> ModelInput:
    x = np.array(
        [
            [0.0],
            [0.1],
            [0.2],
            [5.0],
            [5.1],
            [5.2],
        ],
        dtype=np.float32,
    )
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)

    fit_idx = np.array([0, 1, 3, 4], dtype=np.int64)
    predict_idx = np.array([2, 5], dtype=np.int64)

    return ModelInput(x=x, y=y, fit_idx=fit_idx, predict_idx=predict_idx)


def test_capabilities_are_feature_only() -> None:
    capabilities = XGBoostAdapter.capabilities

    assert capabilities.requires_graph is False
    assert capabilities.requires_directed_graph is False
    assert capabilities.support_inductive_inference is True


def test_fit_and_predict_scores_shape_and_range() -> None:
    data = _separable_model_input()
    model = XGBoostAdapter(n_estimators=10, max_depth=2)
    model.fit(data, seed=0)

    scores = model.predict_scores(data)

    assert scores.shape == (len(data.predict_idx),)
    assert np.all((scores >= 0.0) & (scores <= 1.0))


def test_predict_scores_ranks_illicit_class_higher() -> None:
    data = _separable_model_input()
    # min_child_weight=0: the default of 1 blocks any split on a fit set
    # this tiny (4-row) by design, since each candidate child's Hessian
    # sum falls below the threshold -- not representative of real runs
    model = XGBoostAdapter(n_estimators=10, max_depth=2, min_child_weight=0)
    model.fit(data, seed=0)

    scores = model.predict_scores(data)

    # predict_idx = [2 (licit, label 0), 5 (illicit, label 1)]
    assert scores[1] > scores[0]


def test_predict_scores_before_fit_raises() -> None:
    model = XGBoostAdapter()
    data = _separable_model_input()

    with pytest.raises(RuntimeError):
        model.predict_scores(data)


def _many_rows_model_input(n_per_class: int = 200) -> ModelInput:
    rng = np.random.RandomState(0)

    x_licit = rng.normal(loc=0.0, scale=1.0, size=(n_per_class, 10))
    x_illicit = rng.normal(loc=1.5, scale=1.0, size=(n_per_class, 10))
    x = np.concatenate([x_licit, x_illicit], axis=0).astype(np.float32)
    y = np.concatenate(
        [np.zeros(n_per_class, dtype=np.int64), np.ones(n_per_class, dtype=np.int64)]
    )

    idx = np.arange(2 * n_per_class)
    fit_idx = idx
    predict_idx = idx

    return ModelInput(x=x, y=y, fit_idx=fit_idx, predict_idx=predict_idx)


def test_seed_changes_scores_with_subsampling() -> None:
    # with subsample/colsample_bytree < 1.0, random_state controls which
    # rows/features each boosting round actually sees -- without them
    # (defaults of 1.0), xgboost's tree construction is deterministic given
    # the data and random_state has no observable effect at all, which is
    # exactly the zero-across-seed-variance anomaly this test guards against
    data = _many_rows_model_input()

    model_a = XGBoostAdapter(
        n_estimators=50, max_depth=3, subsample=0.8, colsample_bytree=0.8
    )
    model_a.fit(data, seed=42)
    scores_a = model_a.predict_scores(data)

    model_b = XGBoostAdapter(
        n_estimators=50, max_depth=3, subsample=0.8, colsample_bytree=0.8
    )
    model_b.fit(data, seed=43)
    scores_b = model_b.predict_scores(data)

    assert not np.allclose(scores_a, scores_b)
