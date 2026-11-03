from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from bitcoin_aml.artifacts.manifest import build_run_manifest
from bitcoin_aml.artifacts.predictions_store import save_predictions
from bitcoin_aml.artifacts.run_store import save_run
from bitcoin_aml.data.dataset import EllipticDataset
from bitcoin_aml.data.repository import EllipticRepository
from bitcoin_aml.evaluation.temporal import TemporalMetrics, evaluate_per_timestep
from bitcoin_aml.experiments.config import (
    experiment_spec_to_dict,
    hash_experiment_spec,
    resolve_dataset_root,
)
from bitcoin_aml.experiments.model_selection import select_best_hyperparameters
from bitcoin_aml.experiments.specification import ExperimentSpec
from bitcoin_aml.features.preprocessing import fit_feature_preprocessor
from bitcoin_aml.features.selectors import select_feature_view
from bitcoin_aml.graphs.policies import resolve_phase_edge_index
from bitcoin_aml.graphs.views import edge_index_for_view
from bitcoin_aml.models.base import ModelInput
from bitcoin_aml.models.registry import MODEL_REGISTRY, create_model
from bitcoin_aml.splits.chronological import build_chronological_split, labeled_indices
from bitcoin_aml.splits.leakage import (
    assert_supervised_indices_labeled,
    validate_temporal_split,
)
from bitcoin_aml.utils.logging import get_logger
from bitcoin_aml.utils.seeding import set_global_seed

# ------------------------------------------------------
# orchestrates experiments with access to:
# data
# splits
# features
# graphs
# models
# evaluation
# artifacts
# ------------------------------------------------------

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class RunResult:
    run_id: str
    config_hash: str
    seed: int
    manifest_path: Path
    predictions_path: Path
    test_metrics: TemporalMetrics


class ExperimentRunner:
    """
    orchestrates one resolved experiment: dataset load, chronological
    split, model selection, final refit, one untouched test evaluation,
    and run persistence -- once per configured seed
    """

    def __init__(
        self,
        *,
        output_root: str | Path = Path("outputs/runs"),
        data_root: str | Path = Path("data/processed"),
    ) -> None:
        self._output_root = Path(output_root)
        self._data_root = Path(data_root)

    def run(self, spec: ExperimentSpec) -> tuple[RunResult, ...]:
        if not spec.protocol.model_selection.enabled:
            raise NotImplementedError(
                "ExperimentRunner currently requires model selection to be enabled"
            )

        if not spec.protocol.refit.enabled:
            raise NotImplementedError(
                "ExperimentRunner currently requires refit to be enabled"
            )

        config_hash = hash_experiment_spec(spec)
        resolved_config = experiment_spec_to_dict(spec)

        logger.info(
            "resolved experiment name=%s model=%s config_hash=%s seeds=%s",
            spec.name,
            spec.model.name,
            config_hash,
            list(spec.seeds),
        )

        dataset_root = resolve_dataset_root(
            spec.dataset_version, data_root=self._data_root
        )
        dataset = EllipticRepository.load(dataset_root)
        dataset_manifest = EllipticRepository.read_manifest(dataset_root)
        logger.info(
            "loaded dataset version=%s num_nodes=%d",
            spec.dataset_version,
            dataset.num_nodes,
        )

        split = build_chronological_split(dataset, spec.protocol.split)
        validate_temporal_split(dataset, spec.protocol.split, split)
        logger.info(
            "built and validated chronological split train=%s validation=%s test=%s",
            spec.protocol.split.train.values(),
            None if split.validation is None else spec.protocol.split.validation.values(),
            spec.protocol.split.test.values(),
        )

        if split.validation is None:
            raise ValueError(
                "model selection requires a validation split"
            )

        raw_features = select_feature_view(dataset, spec.features)
        edge_index = (
            edge_index_for_view(dataset, spec.graph)
            if _model_requires_graph(spec.model.name)
            else None
        )

        train_fit_idx = labeled_indices(dataset, split.train)
        validation_idx = labeled_indices(dataset, split.validation)
        test_idx = labeled_indices(dataset, split.test)

        assert_supervised_indices_labeled(dataset, train_fit_idx, name="train_fit")
        assert_supervised_indices_labeled(dataset, validation_idx, name="validation")
        assert_supervised_indices_labeled(dataset, test_idx, name="test")

        results = []

        for seed in spec.seeds:
            logger.info("starting run seed=%d (%d/%d)", seed, len(results) + 1, len(spec.seeds))
            set_global_seed(seed)

            results.append(
                self._run_single_seed(
                    dataset=dataset,
                    dataset_manifest=dataset_manifest,
                    raw_features=raw_features,
                    edge_index=edge_index,
                    train_fit_idx=train_fit_idx,
                    validation_idx=validation_idx,
                    test_idx=test_idx,
                    spec=spec,
                    config_hash=config_hash,
                    resolved_config=resolved_config,
                    seed=seed,
                )
            )

        return tuple(results)

    def _run_single_seed(
        self,
        *,
        dataset: EllipticDataset,
        dataset_manifest: Mapping[str, Any],
        raw_features: np.ndarray,
        edge_index: np.ndarray | None,
        train_fit_idx: np.ndarray,
        validation_idx: np.ndarray,
        test_idx: np.ndarray,
        spec: ExperimentSpec,
        config_hash: str,
        resolved_config: Mapping[str, Any],
        seed: int,
    ) -> RunResult:
        all_indices = np.arange(dataset.num_nodes)

        fit_edge_index: np.ndarray | None = None
        eval_edge_index: np.ndarray | None = None

        if edge_index is not None:
            fit_cutoff_timestep = spec.protocol.split.validation.end

            fit_edge_index = resolve_phase_edge_index(
                dataset,
                edge_index,
                access_policy=spec.protocol.graph_access,
                phase="fit",
                fit_cutoff_timestep=fit_cutoff_timestep,
            )
            eval_edge_index = resolve_phase_edge_index(
                dataset,
                edge_index,
                access_policy=spec.protocol.graph_access,
                phase="eval",
                fit_cutoff_timestep=fit_cutoff_timestep,
            )

            logger.info(
                "seed=%d graph_access=%s fit_cutoff_timestep=%d "
                "fit_edges=%d eval_edges=%d",
                seed,
                spec.protocol.graph_access.value,
                fit_cutoff_timestep,
                fit_edge_index.shape[1],
                eval_edge_index.shape[1],
            )

        # --- model selection: fit + transform on labeled train only, evaluate on validation ---
        logger.info(
            "seed=%d state=model_selection start train_fit=%d validation=%d",
            seed,
            len(train_fit_idx),
            len(validation_idx),
        )
        selection_preprocessor = fit_feature_preprocessor(
            raw_features, train_fit_idx, strategy=spec.features.scaling
        )
        selection_x = selection_preprocessor.transform_rows(raw_features, all_indices)

        selection_input = ModelInput(
            x=selection_x,
            y=dataset.labels,
            fit_idx=train_fit_idx,
            predict_idx=validation_idx,
            edge_index=fit_edge_index,
            timestep=dataset.timestep,
        )

        selection_result = select_best_hyperparameters(
            model_name=spec.model.name,
            fixed_parameters=spec.model.fixed_parameters,
            search_space=spec.model.search_space,
            train_data=selection_input,
            metric=spec.protocol.model_selection.metric,
            direction=spec.protocol.model_selection.direction,
            threshold=spec.protocol.decision_rules.threshold,
            seed=seed,
        )

        logger.info(
            "seed=%d state=model_selection complete best_%s=%.4f best_parameters=%s",
            seed,
            spec.protocol.model_selection.metric,
            selection_result.best_metric_value,
            dict(selection_result.best_parameters),
        )

        # the model-selection preprocessing object and model are discarded
        # here; only the selected hyperparameters carry forward

        # --- final refit: fresh preprocessing + model on labeled train+validation (timesteps 1-34) ---
        refit_fit_idx = np.union1d(train_fit_idx, validation_idx)

        logger.info(
            "seed=%d state=final_refit start fit=%d predict=%d",
            seed,
            len(refit_fit_idx),
            len(test_idx),
        )
        refit_preprocessor = fit_feature_preprocessor(
            raw_features, refit_fit_idx, strategy=spec.features.scaling
        )
        refit_x = refit_preprocessor.transform_rows(raw_features, all_indices)

        final_parameters = {
            **spec.model.fixed_parameters,
            **selection_result.best_parameters,
        }
        final_model = create_model(spec.model.name, final_parameters)

        refit_fit_input = ModelInput(
            x=refit_x,
            y=dataset.labels,
            fit_idx=refit_fit_idx,
            predict_idx=test_idx,
            edge_index=fit_edge_index,
            timestep=dataset.timestep,
        )
        final_model.fit(refit_fit_input, seed=seed)
        logger.info("seed=%d state=final_refit complete", seed)

        # --- final, untouched test evaluation (timesteps 35-49) ---
        # always uses the full, unrestricted graph regardless of policy: test
        # nodes need their real same-timestep neighbors to be predicted on
        # meaningfully (see docs/experiment_protocols.md, "Graph-based models")
        logger.info("seed=%d state=test_evaluation start predict=%d", seed, len(test_idx))
        refit_eval_input = ModelInput(
            x=refit_x,
            y=dataset.labels,
            fit_idx=refit_fit_idx,
            predict_idx=test_idx,
            edge_index=eval_edge_index,
            timestep=dataset.timestep,
        )
        scores = final_model.predict_scores(refit_eval_input)
        y_true = dataset.labels[test_idx]

        test_metrics = evaluate_per_timestep(
            dataset,
            test_idx,
            y_true,
            scores,
            threshold=spec.protocol.decision_rules.threshold,
        )
        logger.info(
            "seed=%d state=test_evaluation complete precision=%.4f recall=%.4f "
            "f1=%.4f average_precision=%.4f",
            seed,
            test_metrics.overall.precision,
            test_metrics.overall.recall,
            test_metrics.overall.f1,
            test_metrics.overall.average_precision,
        )

        # --- persist ---
        created_at = datetime.now(timezone.utc).isoformat()
        run_id = _build_run_id(
            experiment_name=spec.name,
            config_hash=config_hash,
            seed=seed,
            created_at=created_at,
        )

        manifest = build_run_manifest(
            run_id=run_id,
            experiment_name=spec.name,
            config_hash=config_hash,
            resolved_config=resolved_config,
            dataset_version=spec.dataset_version,
            dataset_manifest=dataset_manifest,
            seed=seed,
            selected_hyperparameters=selection_result.best_parameters,
            model_selection_metric=spec.protocol.model_selection.metric,
            model_selection_metric_value=selection_result.best_metric_value,
            test_metrics=test_metrics,
            created_at=created_at,
            peak_gpu_memory_gb=getattr(final_model, "last_peak_gpu_memory_gb", None),
            peak_host_memory_gb=getattr(final_model, "last_peak_host_memory_gb", None),
        )

        manifest_path = save_run(self._output_root, manifest)
        logger.info("seed=%d state=persisted run_id=%s manifest=%s", seed, run_id, manifest_path)

        predictions_path = save_predictions(
            self._output_root,
            run_id,
            tx_id=dataset.tx_id[test_idx],
            timestep=dataset.timestep[test_idx],
            y_true=y_true,
            scores=scores,
        )
        logger.info(
            "seed=%d state=persisted run_id=%s predictions=%s",
            seed,
            run_id,
            predictions_path,
        )

        return RunResult(
            run_id=run_id,
            config_hash=config_hash,
            seed=seed,
            manifest_path=manifest_path,
            predictions_path=predictions_path,
            test_metrics=test_metrics,
        )


def _model_requires_graph(model_name: str) -> bool:
    adapter_cls = MODEL_REGISTRY.get(model_name)

    if adapter_cls is None:
        raise KeyError(f"unknown model {model_name!r}")

    return adapter_cls.capabilities.requires_graph


def _build_run_id(
    *, experiment_name: str, config_hash: str, seed: int, created_at: str
) -> str:
    timestamp = created_at.replace(":", "").replace(".", "").replace("+", "")
    return f"{experiment_name}_{config_hash[:12]}_seed{seed}_{timestamp}"