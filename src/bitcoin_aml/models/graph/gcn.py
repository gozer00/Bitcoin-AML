from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from bitcoin_aml.models.base import ModelAdapter, ModelCapabilities, ModelInput
from bitcoin_aml.models.registry import register_model
from bitcoin_aml.utils.logging import get_logger
from bitcoin_aml.utils.resources import select_device, track_peak_cuda_memory

logger = get_logger(__name__)


@register_model("gcn")
class GCNAdapter(ModelAdapter):
    """
    hand-rolled, torch-native 2-layer graph convolutional network

    uses only per-node ops (linear, relu, dropout) -- deliberately no
    cross-node normalization layer (e.g. batch/graph norm) -- so a node's
    prediction depends only on itself and its graph neighborhood, never on
    which other nodes happen to be present in the batch. combined with the
    dataset invariant that Elliptic has no cross-timestep edges (see
    docs/experiment_protocols.md, "Graph-based models"), this keeps message
    passing from ever leaking future-timestep information into a
    currently-visible node's representation, regardless of fit_idx/predict_idx
    """

    capabilities = ModelCapabilities(
        requires_graph=True,
        requires_directed_graph=False,
        support_inductive_inference=True,
    )

    def __init__(self, **hyperparameters: Any) -> None:
        self._hyperparameters = dict(hyperparameters)
        self._model: _GCNModule | None = None
        self._device: torch.device | None = None
        self.last_peak_gpu_memory_gb: float | None = None

    def fit(self, data: ModelInput, *, seed: int) -> None:
        if data.edge_index is None:
            raise ValueError("gcn requires a graph -- data.edge_index is None")

        hyperparameters = dict(self._hyperparameters)
        num_layers = int(hyperparameters.pop("num_layers", 2))
        epochs = int(hyperparameters.pop("epochs", 200))
        learning_rate = float(hyperparameters.pop("learning_rate", 0.01))
        weight_decay = float(hyperparameters.pop("weight_decay", 0.0005))
        hidden_dim = int(hyperparameters.pop("hidden_dim", 32))
        dropout = float(hyperparameters.pop("dropout", 0.0))

        if hyperparameters:
            raise ValueError(f"unsupported gcn hyperparameters: {sorted(hyperparameters)}")

        logger.info(
            "gcn fit start samples=%d num_layers=%d hidden_dim=%d dropout=%.2f "
            "epochs=%d learning_rate=%.4f weight_decay=%.5f",
            len(data.fit_idx),
            num_layers,
            hidden_dim,
            dropout,
            epochs,
            learning_rate,
            weight_decay,
        )

        device = select_device()
        self._device = device

        x = torch.as_tensor(np.asarray(data.x), dtype=torch.float32, device=device)
        y = torch.as_tensor(np.asarray(data.y), dtype=torch.float32, device=device)
        fit_idx = torch.as_tensor(data.fit_idx, dtype=torch.long, device=device)

        adjacency = _normalized_adjacency(
            np.asarray(data.edge_index), num_nodes=x.shape[0], device=device
        )

        model = _GCNModule(
            in_dim=x.shape[1],
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        with track_peak_cuda_memory(logger, "gcn_fit") as peak_memory:
            model.train()

            for epoch in range(epochs):
                optimizer.zero_grad()
                logits = model(x, adjacency)
                loss = F.binary_cross_entropy_with_logits(
                    logits[fit_idx], y[fit_idx]
                )
                loss.backward()
                optimizer.step()

                if (epoch + 1) % 50 == 0 or epoch == epochs - 1:
                    logger.info(
                        "gcn fit epoch=%d/%d loss=%.4f", epoch + 1, epochs, loss.item()
                    )

        self.last_peak_gpu_memory_gb = peak_memory["peak_gb"]
        self._model = model
        logger.info("gcn fit complete")

    def predict_scores(self, data: ModelInput) -> np.ndarray[np.floating]:
        if self._model is None or self._device is None:
            raise RuntimeError("model has not been fitted")

        if data.edge_index is None:
            raise ValueError("gcn requires a graph -- data.edge_index is None")

        device = self._device
        x = torch.as_tensor(np.asarray(data.x), dtype=torch.float32, device=device)
        predict_idx = torch.as_tensor(data.predict_idx, dtype=torch.long, device=device)

        adjacency = _normalized_adjacency(
            np.asarray(data.edge_index), num_nodes=x.shape[0], device=device
        )

        logger.info("gcn scoring samples=%d", len(data.predict_idx))

        self._model.eval()

        with torch.no_grad():
            logits = self._model(x, adjacency)
            scores = torch.sigmoid(logits[predict_idx])

        return scores.detach().cpu().numpy().astype(np.float64)


class _GCNModule(torch.nn.Module):
    def __init__(
        self, *, in_dim: int, hidden_dim: int, num_layers: int, dropout: float
    ) -> None:
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [1]
        self._layers = torch.nn.ModuleList(
            [torch.nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers)]
        )
        self._dropout = dropout

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        h = x

        for index, layer in enumerate(self._layers):
            h = torch.sparse.mm(adjacency, h)
            h = layer(h)

            if index != len(self._layers) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self._dropout, training=self.training)

        return h.squeeze(-1)


def _normalized_adjacency(
    edge_index: np.ndarray, *, num_nodes: int, device: torch.device
) -> torch.Tensor:
    """
    symmetrically-normalized sparse adjacency with self-loops: D^-1/2 (A + I) D^-1/2
    """
    self_loops = np.arange(num_nodes, dtype=np.int64)
    rows = np.concatenate([edge_index[0], self_loops])
    cols = np.concatenate([edge_index[1], self_loops])

    degree = np.zeros(num_nodes, dtype=np.float64)
    np.add.at(degree, rows, 1.0)
    inv_sqrt_degree = np.zeros(num_nodes, dtype=np.float64)
    nonzero = degree > 0
    inv_sqrt_degree[nonzero] = 1.0 / np.sqrt(degree[nonzero])

    values = inv_sqrt_degree[rows] * inv_sqrt_degree[cols]

    indices = torch.as_tensor(np.stack([rows, cols]), dtype=torch.long, device=device)
    values_tensor = torch.as_tensor(values, dtype=torch.float32, device=device)

    return torch.sparse_coo_tensor(
        indices, values_tensor, size=(num_nodes, num_nodes), device=device
    ).coalesce()
