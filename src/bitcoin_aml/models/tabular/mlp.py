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


@register_model("mlp")
class MLPAdapter(ModelAdapter):
    """
    hand-rolled, torch-native feature-only MLP -- replaces an earlier
    sklearn.neural_network.MLPClassifier wrapper. That wrapper passed
    **hyperparameters straight through to sklearn with no validation at
    all (unlike every other adapter in this repo), and MLPClassifier is
    structurally incapable of two things this hand-rolled version now
    supports: dropout (sklearn's MLP has no such parameter, only L2 via
    `alpha`) and, more generally, the same pop-with-defaults/
    reject-unsupported-keys hyperparameter contract every torch adapter in
    this repo already has (gcn.py, graphsage.py, evolvegcn_h.py,
    evolvegcn_o.py, predecessor_gnn.py).

    architecturally this is gcn.py's `_GCNModule` with the graph-mixing
    step removed: a plain per-row classifier -- `requires_graph=False`,
    `data.edge_index` is never touched -- so there is no leakage argument
    to make here (no cross-node ops of any kind, graph or otherwise).

    the hyperparameter surface (`num_layers`, `hidden_dim`, `dropout`,
    `epochs`, `learning_rate`, `weight_decay`) is deliberately identical to
    gcn.py's, not a redesign: this makes the MLP baseline's architecture
    search directly comparable to gcn's, and keeps every torch adapter in
    this repo on one consistent hyperparameter contract. Uniform hidden
    width instead of sklearn's non-uniform `hidden_layer_sizes` tuples is
    part of that same consistency choice -- a different shape here would
    only benefit MLP alone, at the cost of comparability with the rest of
    the model family.
    """

    capabilities = ModelCapabilities(
        requires_graph=False,
        requires_directed_graph=False,
        support_inductive_inference=True,
    )

    def __init__(self, **hyperparameters: Any) -> None:
        self._hyperparameters = dict(hyperparameters)
        self._model: _MLPModule | None = None
        self._device: torch.device | None = None
        self.last_peak_gpu_memory_gb: float | None = None

    def fit(self, data: ModelInput, *, seed: int) -> None:
        hyperparameters = dict(self._hyperparameters)
        num_layers = int(hyperparameters.pop("num_layers", 2))
        epochs = int(hyperparameters.pop("epochs", 200))
        learning_rate = float(hyperparameters.pop("learning_rate", 0.01))
        weight_decay = float(hyperparameters.pop("weight_decay", 0.0005))
        hidden_dim = int(hyperparameters.pop("hidden_dim", 32))
        dropout = float(hyperparameters.pop("dropout", 0.0))

        if hyperparameters:
            raise ValueError(f"unsupported mlp hyperparameters: {sorted(hyperparameters)}")

        logger.info(
            "mlp fit start samples=%d num_layers=%d hidden_dim=%d dropout=%.2f "
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

        model = _MLPModule(
            in_dim=x.shape[1],
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        with track_peak_cuda_memory(logger, "mlp_fit") as peak_memory:
            model.train()

            for epoch in range(epochs):
                optimizer.zero_grad()
                logits = model(x)
                loss = F.binary_cross_entropy_with_logits(
                    logits[fit_idx], y[fit_idx]
                )
                loss.backward()
                optimizer.step()

                if (epoch + 1) % 50 == 0 or epoch == epochs - 1:
                    logger.info(
                        "mlp fit epoch=%d/%d loss=%.4f", epoch + 1, epochs, loss.item()
                    )

        self.last_peak_gpu_memory_gb = peak_memory["peak_gb"]
        self._model = model
        logger.info("mlp fit complete")

    def predict_scores(self, data: ModelInput) -> np.ndarray[np.floating]:
        if self._model is None or self._device is None:
            raise RuntimeError("model has not been fitted")

        device = self._device
        x = torch.as_tensor(np.asarray(data.x), dtype=torch.float32, device=device)
        predict_idx = torch.as_tensor(data.predict_idx, dtype=torch.long, device=device)

        logger.info("mlp scoring samples=%d", len(data.predict_idx))

        self._model.eval()

        with torch.no_grad():
            logits = self._model(x)
            scores = torch.sigmoid(logits[predict_idx])

        return scores.detach().cpu().numpy().astype(np.float64)


class _MLPModule(torch.nn.Module):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x

        for index, layer in enumerate(self._layers):
            h = layer(h)

            if index != len(self._layers) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self._dropout, training=self.training)

        return h.squeeze(-1)
