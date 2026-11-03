from __future__ import annotations

import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bitcoin_aml.data.dataset import EllipticDataset
from bitcoin_aml.data.repository import (
    ARRAY_FILES,
    DATASET_MANIFEST_FILENAME,
    DatasetIntegrityError,
    EllipticRepository,
)
from bitcoin_aml.data.schema import (
    ELLIPTIC_NUM_AGGREGATE_FEATURES,
    ELLIPTIC_NUM_FEATURES,
    ELLIPTIC_NUM_LOCAL_FEATURES,
    RAW_CLASS_MAPPING,
)
from bitcoin_aml.utils.hashing import sha256_file

# -------------------------------------------------------
# workflow of the preparation process
# 1. reads raw Elliptic data files
# 2. validates their correctness
# 3. maps txIds to contiguous node IDs (zero-based)
# 4. converts labels to (-1, 0, 1)
# 5. constructs five numpy arrays (tx_ids, timesteps, features, labels, edge_indices)
# 6. stores these arrays into .npy files together with manifest for later checks
# -------------------------------------------------------

logger = logging.getLogger(__name__)

# predefined filenames of original Elliptic files
RAW_FILES = {
    "features": "elliptic_txs_features.csv",
    "classes": "elliptic_txs_classes.csv",
    "edges": "elliptic_txs_edgelist.csv",
}


def prepare_elliptic_dataset(
        raw_dir: str | Path,
        output_dir: str | Path,
        *,
        dataset_version: str = "elliptic_v1",
        force: bool = False,
) -> Path:
    """
    builds the canonical Elliptic dataset
    if force=True, the dataset will be reloaded guaranteed
        - raw source files are expected in raw_dir
        - output files will be stored as in output_dir:
            * edge_index.npy
            * features.npy
            * labels.npy
            * timestep.npy
            * tx_id.npy
            * manifest.json
    """

    # path objects
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)

    # version check
    if not dataset_version:
        raise ValueError("dataset_version must be set")

    # dictionary for raw filepaths
    raw_paths = _resolve_raw_paths(raw_dir)

    # dictionary for file hashes
    raw_hashes = {name: sha256_file(path) for name, path in raw_paths.items()}

    # fast exit if prepared data already exists and raw hashes align
    if output_dir.exists() and not force:
        _reuse_or_reject_existing_dataset(
            output_dir=output_dir,
            raw_hashes=raw_hashes,
        )
        logger.info(
            f"Existing processed dataset is valid and will be reused: {output_dir}"
        )
        return output_dir

    # call for building dataset object (nodes, edges, labels)
    logger.info(f"Reading and validating raw Elliptic data files")
    dataset = _build_canonical_dataset(raw_paths)

    # ensure output directory is built
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    # ensure temporary output directory is build
    # we only write output if final checks are passed
    temporary_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.",
            dir=output_dir.parent,
        )
    )

    try:
        _save_arrays(dataset, temporary_dir)

        manifest = _build_manifest(
            dataset=dataset,
            dataset_version=dataset_version,
            raw_paths=raw_paths,
            raw_hashes=raw_hashes,
            processed_dir=temporary_dir,
        )

        _write_manifest(
            manifest,
            temporary_dir / DATASET_MANIFEST_FILENAME,
        )

        # verify the whole persisted repr. before saving
        EllipticRepository.verify(temporary_dir)

        if output_dir.exists():
            shutil.rmtree(output_dir)

        temporary_dir.rename(output_dir)

    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise

    logger.info(
        f"Preparation of Elliptic dataset completed at {output_dir}",
    )

    return output_dir

def _resolve_raw_paths(raw_dir:Path) -> dict[str, Path]:
    """
    returns all files in directory if named correctly
    raises error if:
        - directory does not contain correctly named Elliptic files
    """

    # build name-filepath dictionary
    paths = {
        logical_name: raw_dir / filename
        for logical_name, filename in RAW_FILES.items()
    }

    # identify and catch missing files
    missing = [
        path for path in paths.values() if not path.is_file()
    ]

    if missing:
        formatted = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"missing raw files: {formatted}")

    return paths

def _reuse_or_reject_existing_dataset(
        output_dir: Path,
        raw_hashes: dict[str, str],
) -> None:
    """
    checks whether an already built manifest in output_dir holds equal hashes to raw_hashes
    :param output_dir:
    :param raw_hashes:
    :return:
    """
    try:
        manifest = EllipticRepository.read_manifest(output_dir)
    except DatasetIntegrityError as exc:
        raise DatasetIntegrityError(
            f"{output_dir} exists but not valid. For rebuild, set force=True"
        ) from exc

    manifest_raw_files = manifest.get("source", {}).get("raw_files", {})

    stored_hashes = {
        logical_name: metadata.get("sha256")
        for logical_name, metadata in manifest_raw_files.items()
        if isinstance(metadata, dict)
    }

    if stored_hashes != raw_hashes:
        raise DatasetIntegrityError(
            f"{output_dir} was produced from differen raw input bytes. "
            "Use a new dataset version of rebuild by setting force=True"
        )

    # trail basis call for loading the already existing dataset
    try:
        EllipticRepository.verify(output_dir)
    except DatasetIntegrityError as exc:
        raise DatasetIntegrityError(
            f"{output_dir} corresponds to given raw input but fails integrity check."
            "Rebuild by setting force=True"
        ) from exc

def _build_canonical_dataset(
        raw_paths: dict[str, Path],
) -> EllipticDataset:
    """

    :param raw_paths:
    :return:
    """
    tx_id, timestep, features = _load_features(
        raw_paths["features"]
    )

    labels = _load_and_align_classes(
        raw_paths["classes"],
        tx_id,
    )

    edges = _load_and_map_edges(
        raw_paths["edges"],
        tx_id,
        timestep,
    )

    # make storage dtypes explicit
    tx_id = np.ascontiguousarray(tx_id, dtype=np.int64)
    timestep = np.ascontiguousarray(timestep, dtype=np.int16)
    features = np.ascontiguousarray(features, dtype=np.float32)
    labels = np.ascontiguousarray(labels, dtype=np.int8)
    edge_index = np.ascontiguousarray(edges, dtype=np.int64)

    return EllipticDataset(
        tx_id=tx_id,
        timestep=timestep,
        features=features,
        labels=labels,
        edge_index=edge_index
    )

def _load_features(
        path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    :param path:
    :return: three numpy arrays (tx_ids, timestep, features), which serve as the canonical data representation
    """
    frame = pd.read_csv(path, header=None)

    expected_columns = 2 + ELLIPTIC_NUM_FEATURES

    if frame.shape[1] != expected_columns:
        f"Expected {expected_columns} columns in feature file, "
        f"received {frame.shape[1]}"

    if frame.shape[0] == 0:
        raise ValueError(
            f"feature file is empty"
        )

    tx_id = _to_integer_array(
        frame.iloc[:, 0],
        name="feature txId",
        dtype=np.int64,
    )

    timestep = _to_integer_array(
        frame.iloc[:, 1],
        name="timestep",
        dtype=np.int16
    )

    if len(np.unique(tx_id)) != len(tx_id):
        raise ValueError("feature file contains duplicate txIds")

    try:
        features = frame.iloc[:, 2:].to_numpy(
            dtype=np.float32,
            copy=True
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "feature matrix contains non numeric values"
        ) from exc

    if not np.isfinite(features).all():
        raise ValueError("feature matrix contains NaNs or infinite values")

    return tx_id, timestep, features

def _load_and_align_classes(
        path: Path,
        tx_id: np.ndarray,
) -> np.ndarray:
    """
    :param path:
    :param tx_id:
    :return: numpy array with transaction classes
    """
    frame = pd.read_csv(path)

    _require_exact_columns(
        frame,
        expected={"txId", "class"},
        source_name="class file"
    )

    class_tx_id = _to_integer_array(
        frame["txId"],
        name="class txId",
        dtype=np.int64,
    )

    if len(np.unique(class_tx_id)) != len(class_tx_id):
        raise ValueError("class file contains duplicate txIds")

    missing_class_ids = np.setdiff1d(
        tx_id,
        class_tx_id,
        assume_unique=True,
    )

    extra_class_ids = np.setdiff1d(
        class_tx_id,
        tx_id,
        assume_unique=True,
    )

    if len(missing_class_ids) > 0:
        raise ValueError(
            f"missing classes for {missing_class_ids[:10].tolist()}"
        )

    if len(extra_class_ids) > 0:
        raise ValueError(
            f"extra classes for {extra_class_ids[:10].tolist()}"
        )

    normalized = frame["class"].astype(str).str.strip().str.lower()

    unsupported = sorted(set(normalized.unique()) - set(RAW_CLASS_MAPPING))

    if unsupported:
        raise ValueError(
            f"unsupported raw class labels: {unsupported}"
        )

    canonical_labels = normalized.map(lambda value: int(RAW_CLASS_MAPPING[value]))

    labels_by_tx = pd.Series(
        canonical_labels.to_numpy(dtype=np.int8),
        index=class_tx_id
    )

    labels = labels_by_tx.reindex(tx_id)

    if labels.isna().any():
        raise RuntimeError(
            "internal error while aligning class labels to feature rows"
        )

    return labels.to_numpy(dtype=np.int8, copy=True)

def _load_and_map_edges(
        path: Path,
        tx_id: np.ndarray,
        timestep: np.ndarray,
) -> np.ndarray:
    """
    :param path:
    :param tx_id:
    :param timestep:
    :return: 2-dimensional numpy array with two rows: first row is source tx, second row is target tx.
    """
    frame = pd.read_csv(path)

    _require_exact_columns(
        frame,
        expected={"txId1", "txId2"},
        source_name="edge file"
    )

    source_tx = _to_integer_array(
        frame["txId1"],
        name="edge source txId",
        dtype=np.int64,
    )

    target_tx = _to_integer_array(
        frame["txId2"],
        name="edge target txId",
        dtype=np.int64,
    )

    if np.any(source_tx == target_tx):
        raise ValueError("raw edgelist contains self-edges")

    edge_pairs = pd.DataFrame(
        {
            "source": source_tx,
            "target": target_tx,
        }
    )

    if edge_pairs.duplicated().any():
        raise ValueError("raw edgelist contains duplicate directed edges")

    node_index_by_tx = pd.Series(
        np.arange(len(tx_id), dtype=np.int64),
        index=tx_id,
    )

    source_index = node_index_by_tx.reindex(source_tx)
    target_index = node_index_by_tx.reindex(target_tx)

    if source_index.isna().any():
        missing = np.unique(source_tx[source_index.isna().to_numpy()])
        raise ValueError(
            "edge sources contain transactions that are not present in features: "
            f"{missing[:10].tolist()}"
        )

    if target_index.isna().any():
        missing = np.unique(target_tx[target_index.isna().to_numpy()])
        raise ValueError(
            "edge targets contain transactions that are not present in features: "
            f"{missing[:10].tolist()}"
        )

    edge_index = np.vstack(
        (
            source_index.to_numpy(dtype=np.int64),
            target_index.to_numpy(dtype=np.int64),
        )
    )

    if edge_index.size > 0:
        source_timestep = timestep[edge_index[0]]
        target_timestep = timestep[edge_index[1]]

        if np.any(source_timestep != target_timestep):
            raise ValueError("raw Elliptic edge between time slices")

    return edge_index

def _to_integer_array(
        series: pd.Series,
        *,
        name: str,
        dtype: np.dtype,
) -> np.ndarray:
    try:
        numeric = pd.to_numeric(series, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} contains non-numeric values"
        ) from exc

    values = numeric.to_numpy()

    if not np.isfinite(values).all():
        raise ValueError(
            f"{name} contains NaN or infinite values"
        )

    return values.astype(dtype, copy=True)

def _require_exact_columns(
        frame: pd.DataFrame,
        *,
        expected: set[str],
        source_name: str,
) -> None:
    actual = set(frame.columns)

    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = set(actual - expected)

        raise ValueError(
            f"invalid columns in {source_name}. "
            f"Missing: {missing}; unexpected: {unexpected}"
        )

def _save_arrays(
        dataset: EllipticDataset,
        directory: Path,
) -> None:
    arrays = {
        "tx_id": dataset.tx_id,
        "timestep": dataset.timestep,
        "features": dataset.features,
        "labels": dataset.labels,
        "edge_index": dataset.edge_index,
    }

    for logical_name, array in arrays.items():
        path = directory / ARRAY_FILES[logical_name]
        np.save(path, np.ascontiguousarray(array), allow_pickle=False)

def _build_manifest(
        *,
        dataset: EllipticDataset,
        dataset_version: str,
        raw_paths: dict[str, Path],
        raw_hashes: dict[str, str],
        processed_dir: Path,
) -> dict[str, Any]:
    arrays: dict[str, dict[str, Any]] = {}

    for logical_name, filename in ARRAY_FILES.items():
        path = processed_dir / filename
        array = np.load(path, allow_pickle=False, mmap_mode="r")

        arrays[logical_name] = {
            "file": filename,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "sha256": sha256_file(path),
        }

    labels = dataset.labels

    return {
        "schema_version": 1,
        "dataset_version": dataset_version,
        "source": {
            "raw_files": {
                logical_name: {
                    "file": raw_paths[logical_name].name,
                    "sha256": raw_hashes[logical_name],
                }
                for logical_name in RAW_FILES
            },
            "raw_class_mapping": {
                "1": "illicit",
                "2": "licit",
                "unknown": "unknown",
            },
            "edge_direction": "txId1 -> txId2",
        },
        "canonical": {
            "node_indexing": (
                "zero-based index preserving feature-file row order"
            ),
            "label_mapping": {
                "-1": "unknown",
                "0": "licit",
                "1": "illicit",
            },
            "feature_groups": {
                "local_93": {
                    "start_index": 0,
                    "end_index": (
                        ELLIPTIC_NUM_LOCAL_FEATURES - 1
                    ),
                    "count": ELLIPTIC_NUM_LOCAL_FEATURES
                },
                "aggregate_72": {
                    "start_index": ELLIPTIC_NUM_LOCAL_FEATURES,
                    "end_index": ELLIPTIC_NUM_FEATURES - 1,
                    "count": ELLIPTIC_NUM_AGGREGATE_FEATURES,
                },
            },
            "counts": {
                "nodes": dataset.num_nodes,
                "edges": dataset.num_edges,
                "features": dataset.num_features,
                "unknown": int(np.sum(labels == -1)),
                "licit": int(np.sum(labels == 0)),
                "illicit": int(np.sum(labels == 1)),
            },
            "cross_timestep_edges": 0,
        },
        "arrays": arrays
    }

def _write_manifest(
        manifest: dict[str, Any],
        path: Path,
) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(
            manifest,
            file,
            indent=2,
            sort_keys=True,
        )
        file.write("\n")
