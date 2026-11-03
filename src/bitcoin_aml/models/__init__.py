# importing concrete adapters here registers them in MODEL_REGISTRY

from bitcoin_aml.models.tabular import logistic_regression  # noqa: F401
from bitcoin_aml.models.tabular import mlp  # noqa: F401
from bitcoin_aml.models.tabular import random_forest  # noqa: F401
from bitcoin_aml.models.tabular import xgboost  # noqa: F401
from bitcoin_aml.models.graph import gcn  # noqa: F401
from bitcoin_aml.models.graph import graphsage  # noqa: F401
from bitcoin_aml.models.graph import evolvegcn_h  # noqa: F401
from bitcoin_aml.models.graph import evolvegcn_o  # noqa: F401
from bitcoin_aml.models.graph import predecessor_gnn  # noqa: F401