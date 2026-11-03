from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import torch
import torch.nn.functional as F

from bitcoin_aml.graphs.predecessors import (
    build_incoming_csr_from_edges,
    build_predecessor_batch,
)
from bitcoin_aml.models.base import ModelAdapter, ModelCapabilities, ModelInput
from bitcoin_aml.models.registry import register_model
from bitcoin_aml.utils.logging import get_logger
from bitcoin_aml.utils.resources import select_device, track_peak_cuda_memory

logger = get_logger(__name__)


class _PredecessorGNNAdapterBase(ModelAdapter):
    """
    hand-rolled, torch-native, SAGE-style predecessor-transaction adapter:
    concat(self, mean(local predecessors)) aggregation per layer, over the
    LOCAL subgraph produced by graphs/predecessors.py's
    build_incoming_csr_from_edges + build_predecessor_batch -- not the
    global graph gcn.py/graphsage.py operate over.

    by default (the incoming view) this is a structural subset of what's
    already proven leakage-safe for gcn.py/graphsage.py
    (docs/experiment_protocols.md, "Graph-based models"). this
    adapter can also be deliberately run against the reversed or
    bidirectional views as ablations (see
    docs/experiment_protocols.md, "Predecessor-transaction models"),
    in which case argument (b) below no longer applies and leakage-safety
    rests on (a) and (c) alone -- both of which are direction-agnostic and
    hold under any view:
    (a) per-node ops only within each layer, no cross-node normalization --
        identical base argument to gcn.py/graphsage.py.
    (b) under the default incoming view, edges are restricted to the
        canonical directed (predecessor) graph, never the bidirectional
        view -- a strict subset of gcn/graphsage's edge set, so it cannot
        introduce leakage those models don't already have proven safe.
        this framing is specific to the incoming view: under the
        bidirectional-view ablation the edge set is no longer a subset of
        gcn/graphsage's edges, it IS that same edge set, so this argument
        does not apply there (leakage-safety for that configuration rests
        on (a) and (c) alone); under the reversed-view ablation the edge
        set is a same-size transposition of the canonical directed graph,
        not a subset of it either, for the same reason.
    (c) inherits resolve_phase_edge_index's fit-phase masking for free: the
        CSR/PredecessorBatch this adapter builds are derived from
        data.edge_index, which the runner has already phase-masked under
        strict_rolling before this adapter ever sees it -- any node beyond
        the fit-phase cutoff has zero edges touching it in that masked
        edge_index, so it can never appear in a fit-phase PredecessorBatch's
        global_node_ids. this holds regardless of edge direction: a
        reversed or bidirectional edge between two nodes is masked out by
        resolve_phase_edge_index exactly like a directed one, since masking
        is purely a function of node-timestep membership, never of edge
        direction. unlike evolvegcn_o, no second adapter-local bound
        is needed: each call processes only the single already-masked graph
        it was given, exactly like gcn.py/graphsage.py.

    three concrete subclasses fix `hops` (and therefore num_layers = hops)
    as a ClassVar rather than a runtime hyperparameter: configs/experiments/
    *.yaml select a model only by name with no per-experiment override of
    fixed_parameters (experiments/config.py::_load_model), and the
    one-hop/two-hop/three-hop variants need different fixed extraction
    depths, so -- exactly like graphsage.py's full_batch/sampled split --
    they need distinct registry names, each with its own
    configs/models/*.yaml.

    a single hyperparameter, `hops`, drives both build_predecessor_batch's
    extraction depth and the number of message-passing layers: num_layers
    is derived internally as self._hops, never exposed as an independently
    settable hyperparameter, avoiding a redundant knob that could disagree
    with the extraction depth. Neither `hops` nor `num_layers` is accepted
    from **hyperparameters -- passing either falls through to the ordinary
    unsupported-hyperparameter check below and raises.

    one full predecessor-batch is built per fit()/predict_scores() call,
    over the entirety of data.fit_idx / data.predict_idx -- no per-epoch
    minibatch resampling, no batch_size hyperparameter -- mirroring
    graphsage_full_batch's static-per-call adjacency precedent.
    """

    _hops: ClassVar[int]
    _name: ClassVar[str]

    capabilities = ModelCapabilities(
        requires_graph=True,
        requires_directed_graph=True,
        support_inductive_inference=True,
    )

    def __init__(self, **hyperparameters: Any) -> None:
        self._hyperparameters = dict(hyperparameters)
        self._model: _PredecessorGNNModule | None = None
        self._device: torch.device | None = None
        self.last_peak_gpu_memory_gb: float | None = None

    def fit(self, data: ModelInput, *, seed: int) -> None:
        if data.edge_index is None:
            raise ValueError(
                f"{self._name} requires a graph -- data.edge_index is None"
            )

        if len(data.fit_idx) == 0:
            raise ValueError(f"{self._name} requires at least one fit_idx node")

        hyperparameters = dict(self._hyperparameters)
        epochs = int(hyperparameters.pop("epochs", 200))
        learning_rate = float(hyperparameters.pop("learning_rate", 0.01))
        weight_decay = float(hyperparameters.pop("weight_decay", 0.0005))
        hidden_dim = int(hyperparameters.pop("hidden_dim", 32))
        dropout = float(hyperparameters.pop("dropout", 0.0))

        if hyperparameters:
            raise ValueError(
                f"unsupported {self._name} hyperparameters: {sorted(hyperparameters)}"
            )

        logger.info(
            "%s fit start samples=%d hops=%d hidden_dim=%d dropout=%.2f "
            "epochs=%d learning_rate=%.4f weight_decay=%.5f",
            self._name,
            len(data.fit_idx),
            self._hops,
            hidden_dim,
            dropout,
            epochs,
            learning_rate,
            weight_decay,
        )

        device = select_device()
        self._device = device

        x_full = torch.as_tensor(np.asarray(data.x), dtype=torch.float32, device=device)
        y = torch.as_tensor(np.asarray(data.y), dtype=torch.float32, device=device)
        num_nodes = x_full.shape[0]
        edge_index = np.asarray(data.edge_index)
        fit_idx = np.asarray(data.fit_idx, dtype=np.int64)

        csr = build_incoming_csr_from_edges(edge_index, num_nodes=num_nodes)
        batch = build_predecessor_batch(csr, target_nodes=fit_idx, hops=self._hops)

        global_ids = torch.as_tensor(batch.global_node_ids, dtype=torch.long, device=device)
        x_local = x_full[global_ids]
        adjacency = _local_mean_predecessor_adjacency(
            batch.local_edge_index, num_nodes=batch.num_nodes, device=device
        )
        target_local_ids = torch.as_tensor(
            batch.target_local_ids, dtype=torch.long, device=device
        )

        # build_predecessor_batch's returned target_nodes is
        # np.unique(fit_idx) -- sorted/deduplicated, not fit_idx's original
        # order -- so map each per-target local logit back to fit_idx's
        # original order before computing the loss against y[fit_idx].
        order_map = np.searchsorted(batch.target_nodes, fit_idx).astype(np.int64)
        order_map_t = torch.as_tensor(order_map, dtype=torch.long, device=device)
        fit_idx_t = torch.as_tensor(fit_idx, dtype=torch.long, device=device)

        model = _PredecessorGNNModule(
            in_dim=x_full.shape[1],
            hidden_dim=hidden_dim,
            num_layers=self._hops,
            dropout=dropout,
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        with track_peak_cuda_memory(logger, f"{self._name}_fit") as peak_memory:
            model.train()

            for epoch in range(epochs):
                optimizer.zero_grad()
                local_logits = model(x_local, adjacency)
                target_logits_sorted = local_logits[target_local_ids]
                logits_for_fit = target_logits_sorted[order_map_t]
                loss = F.binary_cross_entropy_with_logits(
                    logits_for_fit, y[fit_idx_t]
                )
                loss.backward()
                optimizer.step()

                if (epoch + 1) % 50 == 0 or epoch == epochs - 1:
                    logger.info(
                        "%s fit epoch=%d/%d loss=%.4f",
                        self._name,
                        epoch + 1,
                        epochs,
                        loss.item(),
                    )

        self.last_peak_gpu_memory_gb = peak_memory["peak_gb"]
        self._model = model
        logger.info("%s fit complete", self._name)

    def predict_scores(self, data: ModelInput) -> np.ndarray[np.floating]:
        if self._model is None or self._device is None:
            raise RuntimeError("model has not been fitted")

        if data.edge_index is None:
            raise ValueError(
                f"{self._name} requires a graph -- data.edge_index is None"
            )

        if len(data.predict_idx) == 0:
            raise ValueError(f"{self._name} requires at least one predict_idx node")

        device = self._device
        x_full = torch.as_tensor(np.asarray(data.x), dtype=torch.float32, device=device)
        num_nodes = x_full.shape[0]
        edge_index = np.asarray(data.edge_index)
        predict_idx = np.asarray(data.predict_idx, dtype=np.int64)

        csr = build_incoming_csr_from_edges(edge_index, num_nodes=num_nodes)
        batch = build_predecessor_batch(csr, target_nodes=predict_idx, hops=self._hops)

        global_ids = torch.as_tensor(batch.global_node_ids, dtype=torch.long, device=device)
        x_local = x_full[global_ids]
        adjacency = _local_mean_predecessor_adjacency(
            batch.local_edge_index, num_nodes=batch.num_nodes, device=device
        )
        target_local_ids = torch.as_tensor(
            batch.target_local_ids, dtype=torch.long, device=device
        )

        order_map = np.searchsorted(batch.target_nodes, predict_idx).astype(np.int64)
        order_map_t = torch.as_tensor(order_map, dtype=torch.long, device=device)

        logger.info(
            "%s scoring samples=%d hops=%d", self._name, len(data.predict_idx), self._hops
        )

        self._model.eval()

        with torch.no_grad():
            local_logits = self._model(x_local, adjacency)
            target_logits_sorted = local_logits[target_local_ids]
            logits_for_predict = target_logits_sorted[order_map_t]
            scores = torch.sigmoid(logits_for_predict)

        return scores.detach().cpu().numpy().astype(np.float64)


@register_model("predecessor_gnn_one_hop")
class PredecessorGNNOneHopAdapter(_PredecessorGNNAdapterBase):
    """one-hop predecessor-transaction GNN: single SAGE-style layer over each target's exact 1-hop predecessors"""

    _hops = 1
    _name = "predecessor_gnn_one_hop"


@register_model("predecessor_gnn_two_hop")
class PredecessorGNNTwoHopAdapter(_PredecessorGNNAdapterBase):
    """two-hop predecessor-transaction GNN: two stacked SAGE-style layers over each target's exact 2-hop predecessor union"""

    _hops = 2
    _name = "predecessor_gnn_two_hop"


@register_model("predecessor_gnn_three_hop")
class PredecessorGNNThreeHopAdapter(_PredecessorGNNAdapterBase):
    """three-hop predecessor-transaction GNN: three stacked SAGE-style layers over each target's exact 3-hop predecessor union"""

    _hops = 3
    _name = "predecessor_gnn_three_hop"


class _SAGELayer(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self._linear = torch.nn.Linear(2 * in_dim, out_dim)

    def forward(self, self_h: torch.Tensor, neighbor_h: torch.Tensor) -> torch.Tensor:
        return self._linear(torch.cat([self_h, neighbor_h], dim=-1))


class _PredecessorGNNModule(torch.nn.Module):
    def __init__(
        self, *, in_dim: int, hidden_dim: int, num_layers: int, dropout: float
    ) -> None:
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [1]
        self._layers = torch.nn.ModuleList(
            [_SAGELayer(dims[i], dims[i + 1]) for i in range(num_layers)]
        )
        self._dropout = dropout

    def forward(self, x_local: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        h = x_local

        for index, layer in enumerate(self._layers):
            neighbor_mean = torch.sparse.mm(adjacency, h)
            h = layer(h, neighbor_mean)

            if index != len(self._layers) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self._dropout, training=self.training)

        return h.squeeze(-1)


def _local_mean_predecessor_adjacency(
    local_edge_index: np.ndarray, *, num_nodes: int, device: torch.device
) -> torch.Tensor:
    """
    row-mean sparse adjacency over one PredecessorBatch's local ids: row v
    aggregates over its local predecessors u. rows = local_edge_index[1]
    (target/dest -- the node being updated), cols = local_edge_index[0]
    (source/predecessor -- whose h flows in), normalized by 1/indegree(v).

    this is the reverse row/col convention of graphsage.py's
    _mean_neighbor_adjacency of the same shape: that one always receives an
    already-symmetrized bidirectional edge_index, so direction is moot
    there. local_edge_index here is genuinely directed and predecessor-only
    -- reusing graphsage's rows=source convention unmodified would silently
    aggregate backwards (predecessors absorbing their successors' features,
    targets receiving nothing). No self-loops (self is handled separately
    via concat, as in gcn.py/graphsage.py).
    """
    dest = local_edge_index[1]
    source = local_edge_index[0]

    degree = np.zeros(num_nodes, dtype=np.float64)
    np.add.at(degree, dest, 1.0)
    inv_degree = np.zeros(num_nodes, dtype=np.float64)
    nonzero = degree > 0
    inv_degree[nonzero] = 1.0 / degree[nonzero]

    values = inv_degree[dest]

    indices = torch.as_tensor(np.stack([dest, source]), dtype=torch.long, device=device)
    values_tensor = torch.as_tensor(values, dtype=torch.float32, device=device)

    return torch.sparse_coo_tensor(
        indices, values_tensor, size=(num_nodes, num_nodes), device=device
    ).coalesce()
