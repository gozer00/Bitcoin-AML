from __future__ import annotations

from typing import Any

import numpy as np
from xgboost import XGBClassifier

from bitcoin_aml.data.schema import TransactionLabel
from bitcoin_aml.models.base import ModelAdapter, ModelCapabilities, ModelInput
from bitcoin_aml.models.registry import register_model
from bitcoin_aml.utils.logging import get_logger

logger = get_logger(__name__)


@register_model("xgboost")
class XGBoostAdapter(ModelAdapter):
    """
    tabular baseline wrapping xgboost.XGBClassifier
    """

    capabilities = ModelCapabilities(
        requires_graph=False,
        requires_directed_graph=False,
        support_inductive_inference=True,
    )

    def __init__(self, **hyperparameters: Any) -> None:
        self._hyperparameters = dict(hyperparameters)
        self._model: XGBClassifier | None = None

    def fit(self, data: ModelInput, *, seed: int) -> None:
        logger.info(
            "xgboost fit start samples=%d hyperparameters=%s",
            len(data.fit_idx),
            self._hyperparameters,
        )
        model = XGBClassifier(random_state=seed, **self._hyperparameters)
        model.fit(data.x[data.fit_idx], data.y[data.fit_idx])
        self._model = model
        logger.info("xgboost fit complete")

    def predict_scores(self, data: ModelInput) -> np.ndarray[np.floating]:
        if self._model is None:
            raise RuntimeError("model has not been fitted")

        illicit_columns = np.flatnonzero(
            self._model.classes_ == int(TransactionLabel.ILLICIT)
        )

        if len(illicit_columns) == 0:
            raise RuntimeError(
                "fitted model was not trained on any illicit-labeled observations"
            )

        logger.info("xgboost scoring samples=%d", len(data.predict_idx))
        probabilities = self._model.predict_proba(data.x[data.predict_idx])

        return probabilities[:, illicit_columns[0]].astype(np.float64)
