from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import torch
import torch.nn.functional as F

from bitcoin_aml.models.base import ModelAdapter, ModelCapabilities, ModelInput
from bitcoin_aml.models.registry import register_model
from bitcoin_aml.utils.logging import get_logger
from bitcoin_aml.utils.resources import select_device, track_peak_cuda_memory

logger = get_logger(__name__)


class _GraphSAGEAdapterBase(ModelAdapter):
    """
    hand-rolled, torch-native SAGE-style adapter: concat(self, mean(neighbors))
    aggregation per layer, per-node ops only (linear, relu, dropout) --
    deliberately no cross-node normalization layer -- the same leakage
    argument and constraints as gcn.py (docs/experiment_protocols.md,
    "Graph-based models"): a node's representation depends only on itself
    and its own graph neighborhood, never on which other nodes happen to be
    present, so message passing cannot leak future-timestep information into
    a currently-visible node's representation given Elliptic's
    zero-cross-timestep-edges invariant.

    two concrete subclasses register two fixed modes rather than exposing
    `mode` as a runtime hyperparameter: `configs/experiments/*.yaml` can only
    select a model by name (see experiments/config.py::_load_model), with no
    per-experiment override of that model's fixed_parameters, so full_batch
    and sampled -- which require disjoint hyperparameters (fanout is
    meaningless for full_batch) -- need distinct registry names to each get
    their own configs/models/*.yaml.

    `_GraphSAGEFullBatchAdapter` ("graphsage_full_batch") aggregates over
    every edge in the provided edge_index, GCN-like.

    `_GraphSAGESampledAdapter` ("graphsage_sampled") aggregates over a fixed
    per-layer fan-out neighbor sample instead. Each node's sampled neighbors
    are a pure function of (seed, node id, layer, draw index, epoch, that
    node's own out-degree) -- see `_stable_neighbor_offsets` -- so sampling
    for one node can never be perturbed by the presence, degree, or absence
    of any other node in the graph, including a future-timestep node that
    the no-cross-timestep-edges invariant guarantees is never actually
    reachable from it anyway. This is a self-contained, adapter-local
    sampler -- deliberately not built on graphs/predecessors.py's CSR
    utilities, which serve the predecessor-transaction models.
    """

    _mode: ClassVar[str]

    capabilities = ModelCapabilities(
        requires_graph=True,
        requires_directed_graph=False,
        support_inductive_inference=True,
    )

    def __init__(self, **hyperparameters: Any) -> None:
        self._hyperparameters = dict(hyperparameters)
        self._model: _GraphSAGEModule | None = None
        self._device: torch.device | None = None
        self._fanout: list[int] | None = None
        self._num_layers: int | None = None
        self._seed: int | None = None
        self.last_peak_gpu_memory_gb: float | None = None

    def fit(self, data: ModelInput, *, seed: int) -> None:
        if data.edge_index is None:
            raise ValueError(
                f"{self._mode} graphsage requires a graph -- data.edge_index is None"
            )

        hyperparameters = dict(self._hyperparameters)
        num_layers = int(hyperparameters.pop("num_layers", 2))
        epochs = int(hyperparameters.pop("epochs", 200))
        learning_rate = float(hyperparameters.pop("learning_rate", 0.01))
        weight_decay = float(hyperparameters.pop("weight_decay", 0.0005))
        hidden_dim = int(hyperparameters.pop("hidden_dim", 32))
        dropout = float(hyperparameters.pop("dropout", 0.0))
        fanout = self._pop_fanout(hyperparameters, num_layers=num_layers)

        if hyperparameters:
            raise ValueError(
                f"unsupported graphsage hyperparameters: {sorted(hyperparameters)}"
            )

        logger.info(
            "graphsage fit start mode=%s samples=%d num_layers=%d hidden_dim=%d "
            "dropout=%.2f epochs=%d learning_rate=%.4f weight_decay=%.5f fanout=%s",
            self._mode,
            len(data.fit_idx),
            num_layers,
            hidden_dim,
            dropout,
            epochs,
            learning_rate,
            weight_decay,
            fanout,
        )

        device = select_device()
        self._device = device
        self._fanout = fanout
        self._num_layers = num_layers
        self._seed = seed

        x = torch.as_tensor(np.asarray(data.x), dtype=torch.float32, device=device)
        y = torch.as_tensor(np.asarray(data.y), dtype=torch.float32, device=device)
        fit_idx = torch.as_tensor(data.fit_idx, dtype=torch.long, device=device)
        num_nodes = x.shape[0]
        edge_index = np.asarray(data.edge_index)

        static_adjacency, neighbor_index = self._build_static_structures(
            edge_index, num_nodes=num_nodes, device=device
        )

        model = _GraphSAGEModule(
            in_dim=x.shape[1],
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        with track_peak_cuda_memory(logger, "graphsage_fit") as peak_memory:
            model.train()

            for epoch in range(epochs):
                adjacencies = self._adjacencies_for_epoch(
                    static_adjacency,
                    neighbor_index,
                    num_layers=num_layers,
                    fanout=fanout,
                    seed=seed,
                    epoch=epoch,
                    device=device,
                )

                optimizer.zero_grad()
                logits = model(x, adjacencies)
                loss = F.binary_cross_entropy_with_logits(
                    logits[fit_idx], y[fit_idx]
                )
                loss.backward()
                optimizer.step()

                if (epoch + 1) % 50 == 0 or epoch == epochs - 1:
                    logger.info(
                        "graphsage fit epoch=%d/%d loss=%.4f",
                        epoch + 1,
                        epochs,
                        loss.item(),
                    )

        self.last_peak_gpu_memory_gb = peak_memory["peak_gb"]
        self._model = model
        logger.info("graphsage fit complete mode=%s", self._mode)

    def predict_scores(self, data: ModelInput) -> np.ndarray[np.floating]:
        if self._model is None or self._device is None:
            raise RuntimeError("model has not been fitted")

        if data.edge_index is None:
            raise ValueError(
                f"{self._mode} graphsage requires a graph -- data.edge_index is None"
            )

        device = self._device
        x = torch.as_tensor(np.asarray(data.x), dtype=torch.float32, device=device)
        predict_idx = torch.as_tensor(data.predict_idx, dtype=torch.long, device=device)
        num_nodes = x.shape[0]
        edge_index = np.asarray(data.edge_index)

        static_adjacency, neighbor_index = self._build_static_structures(
            edge_index, num_nodes=num_nodes, device=device
        )
        adjacencies = self._adjacencies_for_epoch(
            static_adjacency,
            neighbor_index,
            num_layers=self._num_layers,
            fanout=self._fanout,
            seed=self._seed,
            epoch=_INFERENCE_EPOCH,
            device=device,
        )

        logger.info(
            "graphsage scoring mode=%s samples=%d", self._mode, len(data.predict_idx)
        )

        self._model.eval()

        with torch.no_grad():
            logits = self._model(x, adjacencies)
            scores = torch.sigmoid(logits[predict_idx])

        return scores.detach().cpu().numpy().astype(np.float64)

    def _pop_fanout(
        self, hyperparameters: dict[str, Any], *, num_layers: int
    ) -> list[int] | None:
        raise NotImplementedError

    def _build_static_structures(
        self, edge_index: np.ndarray, *, num_nodes: int, device: torch.device
    ) -> tuple[torch.Tensor | None, "_NeighborIndex | None"]:
        raise NotImplementedError

    def _adjacencies_for_epoch(
        self,
        static_adjacency: torch.Tensor | None,
        neighbor_index: "_NeighborIndex | None",
        *,
        num_layers: int,
        fanout: list[int] | None,
        seed: int,
        epoch: int,
        device: torch.device,
    ) -> list[torch.Tensor]:
        raise NotImplementedError


@register_model("graphsage_full_batch")
class GraphSAGEFullBatchAdapter(_GraphSAGEAdapterBase):
    """full-batch GraphSAGE: mean-aggregates over every edge, GCN-like"""

    _mode = "full_batch"

    def _pop_fanout(
        self, hyperparameters: dict[str, Any], *, num_layers: int
    ) -> list[int] | None:
        if "fanout" in hyperparameters:
            raise ValueError("fanout is only valid for graphsage_sampled")

        return None

    def _build_static_structures(
        self, edge_index: np.ndarray, *, num_nodes: int, device: torch.device
    ) -> tuple[torch.Tensor, None]:
        adjacency = _mean_neighbor_adjacency(edge_index, num_nodes=num_nodes, device=device)
        return adjacency, None

    def _adjacencies_for_epoch(
        self,
        static_adjacency: torch.Tensor | None,
        neighbor_index: "_NeighborIndex | None",
        *,
        num_layers: int,
        fanout: list[int] | None,
        seed: int,
        epoch: int,
        device: torch.device,
    ) -> list[torch.Tensor]:
        assert static_adjacency is not None
        return [static_adjacency] * num_layers


@register_model("graphsage_sampled")
class GraphSAGESampledAdapter(_GraphSAGEAdapterBase):
    """sampled GraphSAGE: fixed per-layer fan-out neighbor sampling"""

    _mode = "sampled"

    def _pop_fanout(
        self, hyperparameters: dict[str, Any], *, num_layers: int
    ) -> list[int]:
        fanout = hyperparameters.pop("fanout", None)

        if fanout is None:
            raise ValueError(
                "graphsage_sampled requires 'fanout' (one entry per layer)"
            )

        fanout = [int(value) for value in fanout]

        if len(fanout) != num_layers:
            raise ValueError(
                f"fanout must have num_layers={num_layers} entries, got {len(fanout)}"
            )

        if any(value <= 0 for value in fanout):
            raise ValueError("fanout entries must be positive")

        return fanout

    def _build_static_structures(
        self, edge_index: np.ndarray, *, num_nodes: int, device: torch.device
    ) -> tuple[None, "_NeighborIndex"]:
        return None, _build_neighbor_index(edge_index, num_nodes=num_nodes)

    def _adjacencies_for_epoch(
        self,
        static_adjacency: torch.Tensor | None,
        neighbor_index: "_NeighborIndex | None",
        *,
        num_layers: int,
        fanout: list[int] | None,
        seed: int,
        epoch: int,
        device: torch.device,
    ) -> list[torch.Tensor]:
        assert neighbor_index is not None and fanout is not None
        return [
            _sampled_adjacency_from_index(
                neighbor_index,
                fanout=fanout[layer_index],
                seed=seed,
                layer=layer_index,
                epoch=epoch,
                device=device,
            )
            for layer_index in range(num_layers)
        ]


_INFERENCE_EPOCH = -1


class _SAGELayer(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self._linear = torch.nn.Linear(2 * in_dim, out_dim)

    def forward(self, self_h: torch.Tensor, neighbor_h: torch.Tensor) -> torch.Tensor:
        return self._linear(torch.cat([self_h, neighbor_h], dim=-1))


class _GraphSAGEModule(torch.nn.Module):
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

    def forward(self, x: torch.Tensor, adjacencies: list[torch.Tensor]) -> torch.Tensor:
        h = x

        for index, (layer, adjacency) in enumerate(zip(self._layers, adjacencies)):
            neighbor_mean = torch.sparse.mm(adjacency, h)
            h = layer(h, neighbor_mean)

            if index != len(self._layers) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self._dropout, training=self.training)

        return h.squeeze(-1)


def _mean_neighbor_adjacency(
    edge_index: np.ndarray, *, num_nodes: int, device: torch.device
) -> torch.Tensor:
    """
    row-mean sparse adjacency (no self-loops -- self is handled separately
    via concat, not mixed into the neighbor term): D^-1 A
    """
    rows = edge_index[0]
    cols = edge_index[1]

    degree = np.zeros(num_nodes, dtype=np.float64)
    np.add.at(degree, rows, 1.0)
    inv_degree = np.zeros(num_nodes, dtype=np.float64)
    nonzero = degree > 0
    inv_degree[nonzero] = 1.0 / degree[nonzero]

    values = inv_degree[rows]

    indices = torch.as_tensor(np.stack([rows, cols]), dtype=torch.long, device=device)
    values_tensor = torch.as_tensor(values, dtype=torch.float32, device=device)

    return torch.sparse_coo_tensor(
        indices, values_tensor, size=(num_nodes, num_nodes), device=device
    ).coalesce()


class _NeighborIndex:
    """
    CSR-like cache of each node's out-neighbors, reused across every
    epoch/layer draw so the O(E log E) sort happens once per fit/predict
    call rather than once per epoch
    """

    __slots__ = ("node_ids", "starts", "degrees", "sorted_dst", "num_nodes")

    def __init__(
        self,
        node_ids: np.ndarray,
        starts: np.ndarray,
        degrees: np.ndarray,
        sorted_dst: np.ndarray,
        num_nodes: int,
    ) -> None:
        self.node_ids = node_ids
        self.starts = starts
        self.degrees = degrees
        self.sorted_dst = sorted_dst
        self.num_nodes = num_nodes


def _build_neighbor_index(edge_index: np.ndarray, *, num_nodes: int) -> _NeighborIndex:
    order = np.argsort(edge_index[0], kind="stable")
    sorted_src = edge_index[0][order]
    sorted_dst = edge_index[1][order]
    boundaries = np.searchsorted(sorted_src, np.arange(num_nodes + 1))
    degree = boundaries[1:] - boundaries[:-1]
    has_neighbors = degree > 0

    return _NeighborIndex(
        node_ids=np.nonzero(has_neighbors)[0],
        starts=boundaries[:-1][has_neighbors],
        degrees=degree[has_neighbors],
        sorted_dst=sorted_dst,
        num_nodes=num_nodes,
    )


def _stable_neighbor_offsets(
    node_ids: np.ndarray,
    degrees: np.ndarray,
    *,
    seed: int,
    layer: int,
    draw: int,
    epoch: int,
) -> np.ndarray:
    """
    deterministic per-node pseudo-random offsets into each node's own
    neighbor segment -- a pure (hashed) function of
    (seed, node id, layer, draw index, epoch, that node's own degree), never
    of any other node's presence or the total node count. This is what
    keeps sampled-mode leakage-safe: a shared mutable RNG stream advanced
    once per node would make node i's draw depend on how many nodes were
    sampled before it, which depends on which other nodes exist in the
    graph -- exactly the kind of cross-node coupling this must avoid.
    """
    # wraparound (mod 2**64) is intentional -- this is a splitmix64-style
    # hash finalizer, not an arithmetic accumulation
    with np.errstate(over="ignore"):
        key = (
            node_ids.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)
            + np.uint64(seed & 0xFFFFFFFFFFFFFFFF) * np.uint64(0xBF58476D1CE4E5B9)
            + np.uint64(layer) * np.uint64(0x94D049BB133111EB)
            + np.uint64(draw) * np.uint64(0xD6E8FEB86659FD93)
            + np.uint64((epoch + 1) & 0xFFFFFFFFFFFFFFFF) * np.uint64(0xA24BAED4963EE407)
        )
        key ^= key >> np.uint64(33)
        key *= np.uint64(0xFF51AFD7ED558CCD)
        key ^= key >> np.uint64(33)
        key *= np.uint64(0xC4CEB9FE1A85EC53)
        key ^= key >> np.uint64(33)

    fraction = (key & np.uint64(0xFFFFFFFF)).astype(np.float64) / float(0x100000000)
    offsets = np.floor(fraction * degrees.astype(np.float64)).astype(np.int64)

    return np.minimum(offsets, degrees - 1)


def _sampled_adjacency_from_index(
    index: _NeighborIndex,
    *,
    fanout: int,
    seed: int,
    layer: int,
    epoch: int,
    device: torch.device,
) -> torch.Tensor:
    if len(index.node_ids) == 0:
        indices_np = np.zeros((2, 0), dtype=np.int64)
        values_np = np.zeros((0,), dtype=np.float32)
    else:
        rows_list = []
        cols_list = []

        for draw in range(fanout):
            offsets = _stable_neighbor_offsets(
                index.node_ids, index.degrees, seed=seed, layer=layer, draw=draw, epoch=epoch
            )
            picked = index.sorted_dst[index.starts + offsets]
            rows_list.append(index.node_ids)
            cols_list.append(picked)

        rows = np.concatenate(rows_list)
        cols = np.concatenate(cols_list)
        indices_np = np.stack([rows, cols])
        values_np = np.full(len(rows), 1.0 / fanout, dtype=np.float32)

    indices = torch.as_tensor(indices_np, dtype=torch.long, device=device)
    values = torch.as_tensor(values_np, dtype=torch.float32, device=device)

    return torch.sparse_coo_tensor(
        indices, values, size=(index.num_nodes, index.num_nodes), device=device
    ).coalesce()
