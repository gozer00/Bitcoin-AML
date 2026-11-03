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


@register_model("evolvegcn_h")
class EvolveGCNHAdapter(ModelAdapter):
    """
    hand-rolled, torch-native EvolveGCN-H: a per-timestep GCN whose
    per-layer weight MATRICES are evolved forward through time by a GRU
    that takes that layer's OWN timestep-local node embeddings as input
    (Pareja et al., "EvolveGCN: Evolving Graph Convolutional Networks for
    Dynamic Graphs", arXiv:1902.10191 / AAAI 2020, Sec. 3.4:
    W_t^{(l)} = GRU(H_t^{(l)}, W_{t-1}^{(l)})) -- instead of being learned
    as static parameters the way gcn.py's are, and instead of evolving with
    no input signal at all the way evolvegcn_o.py's LSTM does (that's the
    paper's other variant, "-O" -- see that file's docstring; a matrix-GRU
    with node-embedding input is specifically "-H", never "-O").

    per-timestep propagation reuses gcn.py's intra-timestep leakage-safety
    argument unchanged: only per-node ops (linear, relu, dropout), no
    cross-node normalization layer, so within one timestep a node's
    representation depends only on itself and its own (same-timestep)
    neighborhood. the *temporal* dimension adds a second, distinct
    invariant this adapter must additionally uphold: the weight matrix W_t
    is a pure function of W_{t-1} and that timestep's own structure/
    features -- never of any later timestep -- so the whole recurrence is
    causal. This holds even though H_t^{(l)} (the GRU's input) is itself
    derived from node embeddings: H_t^{(l)} is computed purely from
    timestep t's own local subgraph (features/edges belonging to nodes at
    timestep t, propagated through already-evolved, already-causal
    weights from earlier layers/timesteps), so it carries no information
    from any later timestep. Concretely: both `fit` and `predict_scores`
    replay the recurrence from the dataset's earliest timestep forward,
    one timestep at a time, and never look ahead; perturbing a later
    timestep's features/edges/labels therefore cannot change W_t or any
    prediction at timestep <= t (see
    tests/regression/test_evolvegcn_h_no_future_leakage.py and
    docs/experiment_protocols.md, "Temporal graph models").

    node embeddings are summarized down to this layer's own in_dim via
    `_TopKPooling` -- a differentiable top-k row selection gated by
    tanh(score), the pooling trick the paper borrows from prior GNN
    pooling work (Sec. 3.4) -- since H_t^{(l)} (num_nodes x in_dim) and
    W_{t-1}^{(l)} (in_dim x out_dim) don't otherwise share a shape the GRU
    could combine. `_MatrixGRUCell` then applies the paper's four GRU
    equations directly: two gate-parameter matrices per gate (one for the
    pooled input, one for the hidden weight state), since -- unlike
    evolvegcn_o.py's self-referential "-O" cell, where input and hidden
    state are literally the same tensor and their gate matrices collapse
    into one -- H's input and hidden state have genuinely different shapes
    here and cannot be collapsed.

    evolution and propagation are interleaved layer by layer within a
    timestep (`propagate_timestep`), not evolved-all-then-propagated-all
    the way evolvegcn_o.py's independent-of-input cell can afford to:
    layer l's weight update needs layer l's OWN within-timestep input
    embedding, which only exists once layer l-1 has already been
    propagated at this same timestep.

    `fit`/`predict_scores` do not persist any recurrent state between
    calls: each call independently replays the whole weight-evolution
    sequence from the dataset's earliest timestep, using only the trained
    (or, for `predict_scores`, already-trained) parameters and whatever
    edge_index that call was given -- mirrors gcn.py/graphsage.py's
    "deterministic function of the visible graph + trained params"
    pattern, generalized over a timestep sequence instead of a single
    static graph.
    """

    capabilities = ModelCapabilities(
        requires_graph=True,
        requires_directed_graph=False,
        support_inductive_inference=True,
    )

    def __init__(self, **hyperparameters: Any) -> None:
        self._hyperparameters = dict(hyperparameters)
        self._model: _EvolveGCNHModule | None = None
        self._device: torch.device | None = None
        self.last_peak_gpu_memory_gb: float | None = None
        self.last_peak_host_memory_gb: float | None = None

    def fit(self, data: ModelInput, *, seed: int) -> None:
        if data.edge_index is None:
            raise ValueError("evolvegcn_h requires a graph -- data.edge_index is None")

        if data.timestep is None:
            raise ValueError(
                "evolvegcn_h requires data.timestep for temporal segmentation"
            )

        if len(data.fit_idx) == 0:
            raise ValueError("evolvegcn_h requires at least one fit_idx node")

        hyperparameters = dict(self._hyperparameters)
        num_layers = int(hyperparameters.pop("num_layers", 2))
        epochs = int(hyperparameters.pop("epochs", 100))
        learning_rate = float(hyperparameters.pop("learning_rate", 0.01))
        weight_decay = float(hyperparameters.pop("weight_decay", 0.0005))
        hidden_dim = int(hyperparameters.pop("hidden_dim", 16))
        dropout = float(hyperparameters.pop("dropout", 0.0))

        if hyperparameters:
            raise ValueError(
                f"unsupported evolvegcn_h hyperparameters: {sorted(hyperparameters)}"
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
            "evolvegcn_h fit start samples=%d num_layers=%d hidden_dim=%d dropout=%.2f "
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

        model = _EvolveGCNHModule(
            in_dim=x.shape[1],
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        with track_peak_cuda_memory(logger, "evolvegcn_h_fit") as peak_gpu_memory, \
                track_peak_host_memory(logger, "evolvegcn_h_fit") as peak_host_memory:
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
                        "evolvegcn_h fit epoch=%d/%d loss=%.4f", epoch + 1, epochs, loss.item()
                    )

        self.last_peak_gpu_memory_gb = peak_gpu_memory["peak_gb"]
        self.last_peak_host_memory_gb = peak_host_memory["peak_gb"]
        self._model = model
        logger.info("evolvegcn_h fit complete")

    def predict_scores(self, data: ModelInput) -> np.ndarray[np.floating]:
        if self._model is None or self._device is None:
            raise RuntimeError("model has not been fitted")

        if data.edge_index is None:
            raise ValueError("evolvegcn_h requires a graph -- data.edge_index is None")

        if data.timestep is None:
            raise ValueError(
                "evolvegcn_h requires data.timestep for temporal segmentation"
            )

        if len(data.predict_idx) == 0:
            raise ValueError("evolvegcn_h requires at least one predict_idx node")

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
            "evolvegcn_h scoring samples=%d timesteps=%d",
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


class _TopKPooling(torch.nn.Module):
    """
    EvolveGCN-H's node-embedding summarization (Pareja et al. 2019/2020,
    Sec. 3.4): projects each node's embedding onto a learned scoring
    vector, selects the top `k` (= this layer's in_dim) highest-scoring
    node rows, and gates each selected row by tanh(score) so gradients can
    flow through the otherwise non-differentiable top-k selection.

    if fewer than `k` nodes are visible at this timestep (a small toy
    graph, or a sparse real timestep), the result is zero-padded up to `k`
    rows -- the paper assumes a large graph and does not address this
    case explicitly.
    """

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self._score = torch.nn.Parameter(_glorot(in_dim, 1))

    def forward(self, embeddings: torch.Tensor, *, k: int) -> torch.Tensor:
        num_nodes = embeddings.shape[0]
        scores = (embeddings @ self._score).squeeze(-1) / (self._score.norm() + 1e-12)

        take = min(k, num_nodes)
        top_scores, top_indices = torch.topk(scores, take)
        selected = embeddings[top_indices] * torch.tanh(top_scores).unsqueeze(-1)

        if take < k:
            padding = torch.zeros(
                k - take, embeddings.shape[1], device=embeddings.device
            )
            selected = torch.cat([selected, padding], dim=0)

        return selected


class _MatrixGRUCell(torch.nn.Module):
    """
    EvolveGCN-H's GRU: W_t = GRU(pooled_input, W_{t-1}) (Pareja et al.
    2019/2020, Sec. 3.4) -- the weight matrix is the hidden state, this
    layer's pooled node-embedding input (see _TopKPooling) is the GRU's
    input signal. Each gate uses two (in_dim x out_dim) / (out_dim x
    out_dim) parameter matrices -- one for the pooled input, one for the
    hidden weight state -- since input and hidden state have genuinely
    different shapes here (unlike evolvegcn_o.py's self-referential "-O"
    cell, where they're the same tensor and collapse into one matrix per
    gate).
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self._update_input = torch.nn.Parameter(_glorot(in_dim, out_dim))
        self._update_hidden = torch.nn.Parameter(_glorot(out_dim, out_dim))
        self._update_bias = torch.nn.Parameter(torch.zeros(out_dim))

        self._reset_input = torch.nn.Parameter(_glorot(in_dim, out_dim))
        self._reset_hidden = torch.nn.Parameter(_glorot(out_dim, out_dim))
        self._reset_bias = torch.nn.Parameter(torch.zeros(out_dim))

        self._candidate_input = torch.nn.Parameter(_glorot(in_dim, out_dim))
        self._candidate_hidden = torch.nn.Parameter(_glorot(out_dim, out_dim))
        self._candidate_bias = torch.nn.Parameter(torch.zeros(out_dim))

    def forward(self, pooled_input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        update = torch.sigmoid(
            pooled_input @ self._update_input
            + weight @ self._update_hidden
            + self._update_bias
        )
        reset = torch.sigmoid(
            pooled_input @ self._reset_input
            + weight @ self._reset_hidden
            + self._reset_bias
        )
        candidate = torch.tanh(
            pooled_input @ self._candidate_input
            + (reset * weight) @ self._candidate_hidden
            + self._candidate_bias
        )
        return (1.0 - update) * weight + update * candidate


class _EvolveGCNHModule(torch.nn.Module):
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
        self._poolings = torch.nn.ModuleList(
            [_TopKPooling(dims[i]) for i in range(num_layers)]
        )
        self._gru_cells = torch.nn.ModuleList(
            [_MatrixGRUCell(dims[i], dims[i + 1]) for i in range(num_layers)]
        )

    def initial_weights(self) -> list[torch.Tensor]:
        return list(self._initial_weights)

    def propagate_timestep(
        self, x_local: torch.Tensor, adjacency: torch.Tensor, weights: list[torch.Tensor]
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        evolves each layer's weight using THIS layer's own timestep-local
        input embedding, then immediately propagates through that layer
        with the freshly evolved weight before moving to the next layer --
        evolution and propagation must interleave layer by layer, since
        layer l's own within-timestep input only exists once layer l-1 has
        already been propagated at this same timestep (unlike
        evolvegcn_o.py's independent-of-input cell, which can evolve every
        layer up front).
        """
        h = x_local
        new_weights = []

        for layer in range(self._num_layers):
            in_dim = self._dims[layer]
            pooled = self._poolings[layer](h, k=in_dim)
            new_weight = self._gru_cells[layer](pooled, weights[layer])
            new_weights.append(new_weight)

            h = torch.sparse.mm(adjacency, h)
            h = h @ new_weight

            if layer != self._num_layers - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self._dropout, training=self.training)

        return h.squeeze(-1), new_weights


def _glorot(rows: int, cols: int) -> torch.Tensor:
    weight = torch.empty(rows, cols)
    torch.nn.init.xavier_uniform_(weight)
    return weight


def _run_sequence(
    model: _EvolveGCNHModule,
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
    weights = model.initial_weights()

    if not timestep_structures:
        return torch.zeros(num_nodes, device=device)

    all_ids = []
    all_logits = []

    for global_ids, adjacency in timestep_structures:
        x_local = x[global_ids]
        logits, weights = model.propagate_timestep(x_local, adjacency, weights)
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
