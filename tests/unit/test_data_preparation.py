from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bitcoin_aml.data.prepare import prepare_elliptic_dataset
from bitcoin_aml.data.repository import (
    DatasetIntegrityError,
    EllipticRepository,
)


def _write_valid_raw_dataset(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True)

    # Two transactions per timestep -> 98 nodes.
    timesteps = np.repeat(
        np.arange(1, 50, dtype=np.int64),
        2,
    )

    tx_id = np.arange(
        10_000,
        10_000 + len(timesteps),
        dtype=np.int64,
    )

    feature_values = np.zeros(
        (len(tx_id), 165),
        dtype=np.float64,
    )

    # Add non-constant deterministic values.
    feature_values[:, 0] = timesteps
    feature_values[:, 1] = np.arange(len(tx_id))

    feature_frame = pd.DataFrame(feature_values)
    feature_frame.insert(0, "timestep", timesteps)
    feature_frame.insert(0, "txId", tx_id)

    feature_frame.to_csv(
        raw_dir / "elliptic_txs_features.csv",
        header=False,
        index=False,
    )

    raw_labels = np.array(
        [
            "1" if i % 3 == 0
            else "2" if i % 3 == 1
            else "unknown"
            for i in range(len(tx_id))
        ],
        dtype=object,
    )

    class_frame = pd.DataFrame(
        {
            "txId": tx_id,
            "class": raw_labels,
        }
    )

    class_frame.to_csv(
        raw_dir / "elliptic_txs_classes.csv",
        index=False,
    )

    # One directed edge within every timestep:
    # first transaction -> second transaction.
    source = tx_id[0::2]
    target = tx_id[1::2]

    edge_frame = pd.DataFrame(
        {
            "txId1": source,
            "txId2": target,
        }
    )

    edge_frame.to_csv(
        raw_dir / "elliptic_txs_edgelist.csv",
        index=False,
    )


def test_prepare_and_load_roundtrip(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed" / "elliptic_v1"

    _write_valid_raw_dataset(raw_dir)

    prepare_elliptic_dataset(
        raw_dir,
        output_dir,
    )

    dataset = EllipticRepository.load(output_dir)

    assert dataset.num_nodes == 98
    assert dataset.num_features == 165
    assert dataset.num_edges == 49

    assert dataset.tx_id.dtype == np.int64
    assert dataset.timestep.dtype == np.int16
    assert dataset.features.dtype == np.float32
    assert dataset.labels.dtype == np.int8
    assert dataset.edge_index.dtype == np.int64

    assert set(np.unique(dataset.labels)) == {-1, 0, 1}

    assert np.all(
        dataset.timestep[dataset.edge_index[0]]
        ==
        dataset.timestep[dataset.edge_index[1]]
    )


def test_preparation_is_idempotent(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed" / "elliptic_v1"

    _write_valid_raw_dataset(raw_dir)

    prepare_elliptic_dataset(
        raw_dir,
        output_dir,
    )

    manifest_path = output_dir / "manifest.json"
    feature_path = output_dir / "features.npy"

    original_manifest = manifest_path.read_bytes()
    original_mtime = feature_path.stat().st_mtime_ns

    prepare_elliptic_dataset(
        raw_dir,
        output_dir,
    )

    assert manifest_path.read_bytes() == original_manifest
    assert feature_path.stat().st_mtime_ns == original_mtime


def test_cross_timestep_edge_is_rejected(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed" / "elliptic_v1"

    _write_valid_raw_dataset(raw_dir)

    edge_path = raw_dir / "elliptic_txs_edgelist.csv"
    edges = pd.read_csv(edge_path)

    # Transaction 10000 is timestep 1.
    # Transaction 10002 is timestep 2.
    edges.loc[0, "txId2"] = 10_002
    edges.to_csv(edge_path, index=False)

    with pytest.raises(
        ValueError,
        match="raw Elliptic edge between time slices",
    ):
        prepare_elliptic_dataset(
            raw_dir,
            output_dir,
        )


def test_missing_class_is_rejected(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed" / "elliptic_v1"

    _write_valid_raw_dataset(raw_dir)

    class_path = raw_dir / "elliptic_txs_classes.csv"
    classes = pd.read_csv(class_path)

    classes.iloc[:-1].to_csv(
        class_path,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="missing classes *",
    ):
        prepare_elliptic_dataset(
            raw_dir,
            output_dir,
        )


def test_processed_file_tampering_is_detected(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed" / "elliptic_v1"

    _write_valid_raw_dataset(raw_dir)

    prepare_elliptic_dataset(
        raw_dir,
        output_dir,
    )

    feature_path = output_dir / "features.npy"

    # Change the file bytes without modifying the manifest.
    with feature_path.open("ab") as file:
        file.write(b"tampered")

    with pytest.raises(
        DatasetIntegrityError,
        match="SHA-256 hash mismatch for features.npy",
    ):
        EllipticRepository.load(output_dir)