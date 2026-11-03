"""
Runner-level regression coverage for GraphAccessPolicy wiring.

Complements the adapter-level no-future-leakage tests
(test_gcn_no_future_leakage.py, test_graphsage_no_future_leakage.py,
test_evolvegcn_o_no_future_leakage.py) by
verifying the actual guarantee end to end through ExperimentRunner:
strict_rolling must genuinely mask the fit-phase graph, and -- because
Elliptic has no cross-timestep edges and the graph adapters use only
per-node operations -- doing so must not change predictions relative to
full_transductive.
"""

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pytest

from bitcoin_aml.data.dataset import EllipticDataset
from bitcoin_aml.data.repository import ARRAY_FILES, DATASET_MANIFEST_FILENAME
from bitcoin_aml.experiments.runner import ExperimentRunner
from bitcoin_aml.experiments.specification import (
    DecisionRuleSpec,
    DecisionRuleType,
    EvaluationSpec,
    ExperimentSpec,
    ModelSelectionSpec,
    ModelSpec,
    ProtocolSpec,
    RefitSpec,
    SelectionDirection,
)
from bitcoin_aml.features.specification import (
    FeatureSetName,
    FeatureSpec,
    ScalingStrategy,
)
from bitcoin_aml.graphs.specification import GraphAccessPolicy, GraphSpec, GraphViewName
from bitcoin_aml.models.graph.gcn import GCNAdapter
from bitcoin_aml.models.graph.graphsage import (
    GraphSAGEFullBatchAdapter,
    GraphSAGESampledAdapter,
)
from bitcoin_aml.splits.specification import SplitSpec, TimestepRange
from bitcoin_aml.utils.hashing import sha256_file


def _write_dataset_to_disk(root: Path, dataset: EllipticDataset) -> None:
    root.mkdir(parents=True, exist_ok=True)

    arrays = {
        "tx_id": dataset.tx_id,
        "timestep": dataset.timestep,
        "features": dataset.features,
        "labels": dataset.labels,
        "edge_index": dataset.edge_index,
    }

    array_metadata = {}

    for logical_name, array in arrays.items():
        filename = ARRAY_FILES[logical_name]
        path = root / filename
        np.save(path, np.ascontiguousarray(array), allow_pickle=False)

        array_metadata[logical_name] = {
            "file": filename,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "sha256": sha256_file(path),
        }

    manifest = {"schema_version": 1, "arrays": array_metadata}

    with (root / DATASET_MANIFEST_FILENAME).open("w", encoding="utf-8") as file:
        json.dump(manifest, file)


def _graph_model_spec(
    dataset_version: str,
    *,
    graph_access: GraphAccessPolicy,
    model_name: str,
    fixed_parameters: Mapping[str, Any],
    search_space: Mapping[str, tuple[Any, ...]],
    seed: int,
) -> ExperimentSpec:
    protocol_name = (
        "thesis_strict_v1"
        if graph_access == GraphAccessPolicy.STRICT_ROLLING
        else "thesis_fulltransductive_v1"
    )

    protocol = ProtocolSpec(
        name=protocol_name,
        split=SplitSpec(
            train=TimestepRange(1, 29),
            validation=TimestepRange(30, 34),
            test=TimestepRange(35, 49),
        ),
        graph_access=graph_access,
        model_selection=ModelSelectionSpec(
            enabled=True,
            metric="average_precision",
            direction=SelectionDirection.MAXIMIZE,
        ),
        refit=RefitSpec(enabled=True, include_validation=True, refit_processing=True),
        decision_rules=DecisionRuleSpec(
            type=DecisionRuleType.FIXED_THRESHOLD, threshold=0.5
        ),
        evaluation=EvaluationSpec(),
    )

    model = ModelSpec(
        name=model_name,
        fixed_parameters=fixed_parameters,
        search_space=search_space,
    )

    return ExperimentSpec(
        name=f"synthetic_{model_name}_{graph_access.value}",
        dataset_version=dataset_version,
        protocol=protocol,
        features=FeatureSpec(
            feature_set=FeatureSetName.LOCAL_93, scaling=ScalingStrategy.STANDARD
        ),
        graph=GraphSpec(view=GraphViewName.DIRECTED),
        model=model,
        seeds=(seed,),
    )


_GCN_PARAMETERS = dict(
    num_layers=2,
    hidden_dim=4,
    dropout=0.0,
    epochs=10,
    learning_rate=0.05,
    weight_decay=0.0,
)

_PREDECESSOR_GNN_PARAMETERS = dict(
    hidden_dim=4,
    dropout=0.0,
    epochs=10,
    learning_rate=0.05,
    weight_decay=0.0,
)


def test_strict_rolling_fit_phase_excludes_future_edges(
    tmp_path: Path,
    synthetic_elliptic_dataset: EllipticDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = synthetic_elliptic_dataset
    dataset_version = "synthetic_v1"
    data_root = tmp_path / "data"
    _write_dataset_to_disk(data_root / dataset_version, dataset)

    captured_edge_indices: list[np.ndarray] = []
    original_fit = GCNAdapter.fit

    def spy_fit(self, data, *, seed):
        captured_edge_indices.append(np.asarray(data.edge_index).copy())
        return original_fit(self, data, seed=seed)

    monkeypatch.setattr(GCNAdapter, "fit", spy_fit)

    runner = ExperimentRunner(output_root=tmp_path / "outputs", data_root=data_root)
    spec = _graph_model_spec(
        dataset_version,
        graph_access=GraphAccessPolicy.STRICT_ROLLING,
        model_name="gcn",
        fixed_parameters=_GCN_PARAMETERS,
        search_space={},
        seed=7,
    )
    runner.run(spec)

    # one fit-phase call for model selection, one for refit
    assert len(captured_edge_indices) == 2

    for edge_index in captured_edge_indices:
        assert edge_index.size > 0
        assert np.all(dataset.timestep[edge_index[0]] <= 34)
        assert np.all(dataset.timestep[edge_index[1]] <= 34)


def test_full_transductive_fit_phase_sees_full_graph(
    tmp_path: Path,
    synthetic_elliptic_dataset: EllipticDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = synthetic_elliptic_dataset
    dataset_version = "synthetic_v1"
    data_root = tmp_path / "data"
    _write_dataset_to_disk(data_root / dataset_version, dataset)

    captured_edge_indices: list[np.ndarray] = []
    original_fit = GCNAdapter.fit

    def spy_fit(self, data, *, seed):
        captured_edge_indices.append(np.asarray(data.edge_index).copy())
        return original_fit(self, data, seed=seed)

    monkeypatch.setattr(GCNAdapter, "fit", spy_fit)

    runner = ExperimentRunner(output_root=tmp_path / "outputs", data_root=data_root)
    spec = _graph_model_spec(
        dataset_version,
        graph_access=GraphAccessPolicy.FULL_TRANSDUCTIVE,
        model_name="gcn",
        fixed_parameters=_GCN_PARAMETERS,
        search_space={},
        seed=7,
    )
    runner.run(spec)

    assert len(captured_edge_indices) == 2

    for edge_index in captured_edge_indices:
        np.testing.assert_array_equal(np.sort(edge_index, axis=1), np.sort(dataset.edge_index, axis=1))


_LEAKAGE_EQUIVALENCE_CASES = [
    pytest.param("gcn", _GCN_PARAMETERS, id="gcn"),
    pytest.param("graphsage_full_batch", _GCN_PARAMETERS, id="graphsage_full_batch"),
    pytest.param(
        "graphsage_sampled",
        {**_GCN_PARAMETERS, "fanout": [2, 2]},
        id="graphsage_sampled",
    ),
    pytest.param("evolvegcn_h", _GCN_PARAMETERS, id="evolvegcn_h"),
    pytest.param("evolvegcn_o", _GCN_PARAMETERS, id="evolvegcn_o"),
    pytest.param(
        "predecessor_gnn_one_hop", _PREDECESSOR_GNN_PARAMETERS, id="predecessor_gnn_one_hop"
    ),
    pytest.param(
        "predecessor_gnn_two_hop", _PREDECESSOR_GNN_PARAMETERS, id="predecessor_gnn_two_hop"
    ),
]


@pytest.mark.parametrize(("model_name", "fixed_parameters"), _LEAKAGE_EQUIVALENCE_CASES)
def test_strict_and_transductive_test_predictions_match(
    tmp_path: Path,
    synthetic_elliptic_dataset: EllipticDataset,
    model_name: str,
    fixed_parameters: Mapping[str, Any],
) -> None:
    """
    The falsifiable claim documented in docs/experiment_protocols.md,
    "Graph-based models": under matched hyperparameters and seed,
    strict_rolling and full_transductive must produce statistically
    indistinguishable -- here, numerically identical -- test predictions,
    because masking training-time edges to future timesteps is a no-op
    given Elliptic's zero-cross-timestep-edges invariant and the adapters'
    per-node-only-ops constraint.
    """
    dataset = synthetic_elliptic_dataset
    dataset_version = "synthetic_v1"
    data_root = tmp_path / "data"
    _write_dataset_to_disk(data_root / dataset_version, dataset)

    scores_by_policy = {}

    for graph_access in (
        GraphAccessPolicy.STRICT_ROLLING,
        GraphAccessPolicy.FULL_TRANSDUCTIVE,
    ):
        runner = ExperimentRunner(
            output_root=tmp_path / f"outputs_{graph_access.value}",
            data_root=data_root,
        )
        spec = _graph_model_spec(
            dataset_version,
            graph_access=graph_access,
            model_name=model_name,
            fixed_parameters=fixed_parameters,
            search_space={},
            seed=11,
        )
        (result,) = runner.run(spec)
        scores_by_policy[graph_access] = result.test_metrics.overall

    strict = scores_by_policy[GraphAccessPolicy.STRICT_ROLLING]
    transductive = scores_by_policy[GraphAccessPolicy.FULL_TRANSDUCTIVE]

    assert strict.true_positives == transductive.true_positives
    assert strict.false_positives == transductive.false_positives
    assert strict.true_negatives == transductive.true_negatives
    assert strict.false_negatives == transductive.false_negatives
    assert strict.average_precision == pytest.approx(transductive.average_precision)
