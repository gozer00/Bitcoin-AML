import pytest

from bitcoin_aml.models.base import ModelAdapter
from bitcoin_aml.models.registry import create_model
from bitcoin_aml.models.tabular.logistic_regression import LogisticRegressionAdapter
from bitcoin_aml.models.tabular.mlp import MLPAdapter
from bitcoin_aml.models.tabular.random_forest import RandomForestAdapter
from bitcoin_aml.models.tabular.xgboost import XGBoostAdapter


def test_create_model_returns_registered_adapter() -> None:
    model = create_model("logistic_regression", {"max_iter": 50})

    assert isinstance(model, LogisticRegressionAdapter)
    assert isinstance(model, ModelAdapter)


def test_create_model_passes_through_parameters() -> None:
    model = create_model(
        "logistic_regression", {"max_iter": 50, "solver": "liblinear"}
    )

    assert model._hyperparameters == {"max_iter": 50, "solver": "liblinear"}


def test_create_model_unknown_name_raises() -> None:
    with pytest.raises(KeyError):
        create_model("does_not_exist", {})


@pytest.mark.parametrize(
    ("model_name", "adapter_cls"),
    [
        ("logistic_regression", LogisticRegressionAdapter),
        ("random_forest", RandomForestAdapter),
        ("xgboost", XGBoostAdapter),
        ("mlp", MLPAdapter),
    ],
)
def test_all_feature_only_baselines_are_registered(model_name, adapter_cls) -> None:
    model = create_model(model_name, {})

    assert isinstance(model, adapter_cls)
    assert isinstance(model, ModelAdapter)
    assert model.capabilities.requires_graph is False