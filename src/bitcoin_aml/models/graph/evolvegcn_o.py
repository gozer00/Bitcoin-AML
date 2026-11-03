from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from bitcoin_aml.models.base import ModelAdapter, ModelCapabilities, ModelInput
from bitcoin_aml.models.registry import register_model
from bitcoin_aml.utils.logging import get_logger
from bitcoin_aml.utils.resources import (
    select_device,
    track_peak_cuda_memory,
    track_peak_host_memory,
)

logger = get_logger(__name__)


@register_model("evolvegcn_o")
class EvolveGCNOAdapter(ModelAdapter):
    """
    hand-rolled, torch-native EvolveGCN-O: a per-timestep GCN whose
    per-layer weight MATRICES are evolved forward through time by a
    self-recurrent matrix-LSTM (Pareja et al., "EvolveGCN: Evolving Graph
    Convolutional Networks for Dynamic Graphs", arXiv:1902.10191 / AAAI
    2020, Sec. 3.5: W_t^{(l)} = LSTM(W_{t-1}^{(l)}), where "current input
    X_t is the same as the past output H_{t-1}" -- i.e. the weight matrix
    plays both the LSTM's input and hidden-state role, with no external
    input at all), instead of being learned as static parameters the way
    gcn.py's are, and instead of taking node embeddings as input the way
    evolvegcn_h.py's GRU does (that's the paper's other variant, "-H" --
    see that file's docstring; a matrix-LSTM with no external input is
    specifically "-O", never "-H").

    per-timestep propagation reuses gcn.py's leakage-safety argument
    unchanged: only per-node ops (linear, relu, dropout), no cross-node
    normalization layer, so within one timestep a node's representation
    depends only on itself and its own (same-timestep) neighborhood. the
    *temporal* dimension adds a second, distinct invariant this adapter must
    additionally uphold: the weight matrix W_t (and its LSTM cell state
    C_t) is a pure function of W_{t-1}/C_{t-1} and the gate parameters --
    never of any later timestep, and never of node embeddings at all -- so
    the whole recurrence is causal. Concretely: both `fit` and
    `predict_scores` replay the recurrence from the dataset's earliest
    timestep forward, one timestep at a time, and never look ahead;
    perturbing a later timestep's features/edges/labels therefore cannot
    change W_t or any prediction at timestep <= t (see
    tests/regression/test_evolvegcn_o_no_future_leakage.py and
    docs/experiment_protocols.md, "Temporal graph models").

    the matrix-LSTM (`_MatrixLSTMCell`) evolves each layer's (in_dim x
    out_dim) weight matrix using four small (out_dim x out_dim) gate
    parameter matrices (forget/input/output/candidate), right-multiplied
    onto the weight state, plus a persistent (in_dim x out_dim) cell
    state threaded alongside it -- so the recurrent unit's parameter count
    is independent of in_dim, following the same motivation as the
    original EvolveGCN-O. Because the paper's input and hidden-state roles
    are literally the same tensor here (unlike evolvegcn_h.py's GRU, whose
    input and hidden state have different shapes and need separate gate
    matrices), each gate needs only one (out_dim x out_dim) matrix rather
    than the paper's redundant input/hidden pair -- a hand-rolled
    simplification of the paper's matrix-LSTM gate equations, not a
    literal reproduction, consistent with how gcn.py and graphsage.py are
    deliberately hand-rolled rather than built on a GNN library. It
    receives no external input signal (the "-O" variant): W_t depends only
    on W_{t-1}/C_{t-1} and the gate parameters, never on node embeddings.

    `fit`/`predict_scores` do not persist any recurrent state between calls:
    each call independently replays the whole weight-evolution sequence
    from the dataset's earliest timestep, using only the trained (or, for
    `predict_scores`, already-trained) parameters and whatever edge_index
    that call was given. This mirrors gcn.py/graphsage.py's "deterministic
    function of the visible graph + trained params" pattern, generalized
    over a timestep sequence instead of a single static graph, and avoids
    the complexity of threading hidden state across the fit/predict
    boundary.
    """

    capabilities = ModelCapabilities(
        requires_graph=True,
        requires_directed_graph=False,
        support_inductive_inference=True,
    )

    def __init__(self, **hyperparameters: Any) -> None:
        self._hyperparameters = dict(hyperparameters)
        self._model: _EvolveGCNOModule | None = None
        self._device: torch.device | None = None
        self.last_peak_gpu_memory_gb: float | None = None
        self.last_peak_host_memory_gb: float | None = None

    def fit(self, data: ModelInput, *, seed: int) -> None:
        if data.edge_index is None:
            raise ValueError("evolvegcn_o requires a graph -- data.edge_index is None")

        if data.timestep is None:
            raise ValueError(
                "evolvegcn_o requires data.timestep for temporal segmentation"
            )

        if len(data.fit_idx) == 0:
            raise ValueError("evolvegcn_o requires at least one fit_idx node")

        hyperparameters = dict(self._hyperparameters)
        num_layers = int(hyperparameters.pop("num_layers", 2))
        epochs = int(hyperparameters.pop("epochs", 100))
        learning_rate = float(hyperparameters.pop("learning_rate", 0.01))
        weight_decay = float(hyperparameters.pop("weight_decay", 0.0005))
        hidden_dim = int(hyperparameters.pop("hidden_dim", 16))
        dropout = float(hyperparameters.pop("dropout", 0.0))

        if hyperparameters:
            raise ValueError(
                f"unsupported evolvegcn_o hyperparameters: {sorted(hyperparameters)}"
            )

        device = select_device()
        self._device = device

        x = torch.as_tensor(np.asarray(data.x), dtype=torch.float32, device=device)
        y = torch.as_tensor(np.asarray(data.y), dtype=torch.float32, device=device)
        fit_idx = torch.as_tensor(data.fit_idx, dtype=torch.long, device=device)
        num_nodes = x.shape[0]
        timestep = np.asarray(data.timestep)
        edge_index = np.asarray(data.edge_index)

        t_start = int(timestep.min())
        t_end = int(timestep[data.fit_idx].max())

        timestep_structures = _build_timestep_structures(
            edge_index,
            timestep,
            num_nodes=num_nodes,
            t_start=t_start,
            t_end=t_end,
            device=device,
        )

        logger.info(
            "evolvegcn_o fit start samples=%d num_layers=%d hidden_dim=%d dropout=%.2f "
            "epochs=%d learning_rate=%.4f weight_decay=%.5f timesteps=%d",
            len(data.fit_idx),
            num_layers,
            hidden_dim,
            dropout,
            epochs,
            learning_rate,
            weight_decay,
            len(timestep_structures),
        )

        model = _EvolveGCNOModule(
            in_dim=x.shape[1],
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        with track_peak_cuda_memory(logger, "evolvegcn_o_fit") as peak_gpu_memory, \
                track_peak_host_memory(logger, "evolvegcn_o_fit") as peak_host_memory:
            model.train()

            for epoch in range(epochs):
                optimizer.zero_grad()
                full_logits = _run_sequence(
                    model, x, timestep_structures, num_nodes=num_nodes, device=device
                )
                loss = F.binary_cross_entropy_with_logits(
                    full_logits[fit_idx], y[fit_idx]
                )
                loss.backward()
                optimizer.step()

                if (epoch + 1) % 50 == 0 or epoch == epochs - 1:
                    logger.info(
                        "evolvegcn_o fit epoch=%d/%d loss=%.4f", epoch + 1, epochs, loss.item()
                    )

        self.last_peak_gpu_memory_gb = peak_gpu_memory["peak_gb"]
        self.last_peak_host_memory_gb = peak_host_memory["peak_gb"]
        self._model = model
        logger.info("evolvegcn_o fit complete")

    def predict_scores(self, data: ModelInput) -> np.ndarray[np.floating]:
        if self._model is None or self._device is None:
            raise RuntimeError("model has not been fitted")

        if data.edge_index is None:
            raise ValueError("evolvegcn_o requires a graph -- data.edge_index is None")

        if data.timestep is None:
            raise ValueError(
                "evolvegcn_o requires data.timestep for temporal segmentation"
            )

        if len(data.predict_idx) == 0:
            raise ValueError("evolvegcn_o requires at least one predict_idx node")

        device = self._device
        x = torch.as_tensor(np.asarray(data.x), dtype=torch.float32, device=device)
        predict_idx = torch.as_tensor(data.predict_idx, dtype=torch.long, device=device)
        num_nodes = x.shape[0]
        timestep = np.asarray(data.timestep)
        edge_index = np.asarray(data.edge_index)

        t_start = int(timestep.min())
        t_end = int(timestep[data.predict_idx].max())

        timestep_structures = _build_timestep_structures(
            edge_index,
            timestep,
            num_nodes=num_nodes,
            t_start=t_start,
            t_end=t_end,
            device=device,
        )

        logger.info(
            "evolvegcn_o scoring samples=%d timesteps=%d",
            len(data.predict_idx),
            len(timestep_structures),
        )

        self._model.eval()

        with torch.no_grad():
            full_logits = _run_sequence(
                self._model, x, timestep_structures, num_nodes=num_nodes, device=device
            )
            scores = torch.sigmoid(full_logits[predict_idx])

        return scores.detach().cpu().numpy().astype(np.float64)


class _MatrixLSTMCell(torch.nn.Module):
    """
    self-recurrent LSTM-style update over a weight MATRIX (not a flattened
    vector), following EvolveGCN-O's definition (Pareja et al. 2019/2020,
    Sec. 3.5): W_t = LSTM(W_{t-1}), with the previous weight matrix serving
    as both the "input" and the "previous hidden state" (the paper states
    this explicitly: no node embeddings are used). All four gates use
    small (out_dim x out_dim) parameter matrices, right-multiplied onto
    the (in_dim x out_dim) weight state, so parameter count is independent
    of in_dim. A persistent (in_dim x out_dim) cell state is threaded
    alongside the weight matrix across timesteps.
    """

    def __init__(self, out_dim: int) -> None:
        super().__init__()
        self._forget_gate = torch.nn.Parameter(_glorot(out_dim, out_dim))
        self._input_gate = torch.nn.Parameter(_glorot(out_dim, out_dim))
        self._output_gate = torch.nn.Parameter(_glorot(out_dim, out_dim))
        self._candidate_gate = torch.nn.Parameter(_glorot(out_dim, out_dim))

    def forward(
        self, weight: torch.Tensor, cell_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        forget = torch.sigmoid(weight @ self._forget_gate)
        input_ = torch.sigmoid(weight @ self._input_gate)
        output = torch.sigmoid(weight @ self._output_gate)
        candidate = torch.tanh(weight @ self._candidate_gate)

        new_cell_state = forget * cell_state + input_ * candidate
        new_weight = output * torch.tanh(new_cell_state)

        return new_weight, new_cell_state


class _EvolveGCNOModule(torch.nn.Module):
    def __init__(
        self, *, in_dim: int, hidden_dim: int, num_layers: int, dropout: float
    ) -> None:
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [1]
        self._dims = dims
        self._num_layers = num_layers
        self._dropout = dropout
        self._initial_weights = torch.nn.ParameterList(
            [torch.nn.Parameter(_glorot(dims[i], dims[i + 1])) for i in range(num_layers)]
        )
        self._lstm_cells = torch.nn.ModuleList(
            [_MatrixLSTMCell(dims[i + 1]) for i in range(num_layers)]
        )

    def initial_state(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        weights = list(self._initial_weights)
        cell_states = [torch.zeros_like(w) for w in weights]
        return weights, cell_states

    def evolve(
        self, weights: list[torch.Tensor], cell_states: list[torch.Tensor]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        evolved = [
            self._lstm_cells[layer](weights[layer], cell_states[layer])
            for layer in range(self._num_layers)
        ]
        return [w for w, _ in evolved], [c for _, c in evolved]

    def propagate_timestep(
        self, x_local: torch.Tensor, adjacency: torch.Tensor, weights: list[torch.Tensor]
    ) -> torch.Tensor:
        h = x_local

        for layer in range(self._num_layers):
            h = torch.sparse.mm(adjacency, h)
            h = h @ weights[layer]

            if layer != self._num_layers - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self._dropout, training=self.training)

        return h.squeeze(-1)


def _glorot(rows: int, cols: int) -> torch.Tensor:
    weight = torch.empty(rows, cols)
    torch.nn.init.xavier_uniform_(weight)
    return weight


def _run_sequence(
    model: _EvolveGCNOModule,
    x: torch.Tensor,
    timestep_structures: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    num_nodes: int,
    device: torch.device,
) -> torch.Tensor:
    """
    replays the weight-evolution recurrence forward across
    `timestep_structures` (already sorted ascending by timestep) and
    scatters each timestep's per-node logits back into a full-size (N,)
    tensor at that timestep's global node positions. Every node belongs to
    exactly one timestep (Elliptic invariant), so there are no id collisions
    to resolve in the scatter.
    """
    weights, cell_states = model.initial_state()

    if not timestep_structures:
        return torch.zeros(num_nodes, device=device)

    all_ids = []
    all_logits = []

    for global_ids, adjacency in timestep_structures:
        weights, cell_states = model.evolve(weights, cell_states)
        x_local = x[global_ids]
        logits = model.propagate_timestep(x_local, adjacency, weights)
        all_ids.append(global_ids)
        all_logits.append(logits)

    ids_cat = torch.cat(all_ids)
    logits_cat = torch.cat(all_logits)

    return torch.zeros(num_nodes, device=device).scatter(0, ids_cat, logits_cat)


def _build_timestep_structures(
    edge_index: np.ndarray,
    timestep: np.ndarray,
    *,
    num_nodes: int,
    t_start: int,
    t_end: int,
    device: torch.device,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """
    groups the (already phase-masked) edge_index by timestep and builds one
    self-loop symmetrically-normalized local adjacency per visible timestep
    in [t_start, t_end], ascending. Timesteps with zero nodes are skipped
    (they contribute nothing to the recurrence either way); every other
    visible timestep contributes exactly one recurrence step, even if it has
    no edges (isolated nodes still get a self-loop).
    """
    if t_end < t_start:
        raise ValueError("t_end must be >= t_start")

    in_range = (timestep >= t_start) & (timestep <= t_end)
    visible_timesteps = np.unique(timestep[in_range])

    structures = []

    for t in visible_timesteps:
        global_ids = np.flatnonzero(timestep == t)
        local_id_of = np.full(num_nodes, -1, dtype=np.int64)
        local_id_of[global_ids] = np.arange(len(global_ids), dtype=np.int64)

        if edge_index.size > 0:
            src_local = local_id_of[edge_index[0]]
            dst_local = local_id_of[edge_index[1]]
            edge_mask = (src_local >= 0) & (dst_local >= 0)
            local_edge_index = np.stack(
                [src_local[edge_mask], dst_local[edge_mask]]
            ).astype(np.int64)
        else:
            local_edge_index = np.zeros((2, 0), dtype=np.int64)

        adjacency = _normalized_adjacency_local(
            local_edge_index, num_nodes=len(global_ids), device=device
        )
        global_ids_tensor = torch.as_tensor(global_ids, dtype=torch.long, device=device)
        structures.append((global_ids_tensor, adjacency))

    return structures


def _normalized_adjacency_local(
    edge_index: np.ndarray, *, num_nodes: int, device: torch.device
) -> torch.Tensor:
    """
    symmetrically-normalized sparse adjacency with self-loops over a single
    timestep's local node set: D^-1/2 (A + I) D^-1/2 -- identical formula to
    gcn.py's `_normalized_adjacency`, applied per-timestep instead of once
    over the whole graph.
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
