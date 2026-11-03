from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from bitcoin_aml.data.dataset import EllipticDataset
from bitcoin_aml.utils.hashing import sha256_file

# ----------------------------------------------------
# handler for loading and saving
# single gateway to processed data
# ----------------------------------------------------

DATASET_MANIFEST_FILENAME = "manifest.json"

ARRAY_FILES = {
    "tx_id": "tx_id.npy",
    "timestep": "timestep.npy",
    "features": "features.npy",
    "labels": "labels.npy",
    "edge_index": "edge_index.npy",
}


class DatasetIntegrityError(RuntimeError):
    """
    Raised when persisted dataset artifacts fail integrity checks
    """

class EllipticRepository:
    """
    gateway for persistence of the canonical processed Elliptic dataset
    """

    @staticmethod
    def read_manifest(root: str | Path) -> dict[str, Any]:
        root = Path(root)
        manifest_path = root / DATASET_MANIFEST_FILENAME

        if not manifest_path.is_file():
            raise DatasetIntegrityError(
                f"dataset manifest file {manifest_path} does not exist"
            )

        try:
            with manifest_path.open("r", encoding="utf-8") as file:
                return json.load(file)
        except json.JSONDecodeError as exc:
            raise DatasetIntegrityError(
                f"invalid dataset manifest file {manifest_path}"
            ) from exc

    @classmethod
    def load(
            cls,
            root: str | Path,
            *,
            verify_hashes: bool = True,
            mmap_mode: str | None = "r",
    ) -> EllipticDataset:
        root = Path(root)
        manifest = cls.read_manifest(root)

        if manifest.get("schema_version") != 1:
            raise DatasetIntegrityError(
                "dataset manifest version is not supported"
            )

        array_metadata = manifest.get("arrays")

        if not isinstance(array_metadata, dict):
            raise DatasetIntegrityError(
                "dataset manifest does not contain array metadata"
            )

        arrays: dict[str, np.ndarray] = {}

        for logical_name, expected_filename in ARRAY_FILES.items():
            metadata = array_metadata.get(logical_name)

            if not isinstance(metadata, dict):
                raise DatasetIntegrityError(
                    f"missing metadata for array {logical_name}"
                )

            filename = metadata.get("file")

            if filename != expected_filename:
                raise DatasetIntegrityError(
                    f"file {filename!r} does not match expected {logical_name}"
                )

            path = root / filename

            if not path.is_file():
                raise DatasetIntegrityError(
                    f"processed dataset file is missing: {filename}"
                )

            if verify_hashes:
                expected_hash = metadata.get("sha256")
                actual_hash = sha256_file(path)

                if expected_hash != actual_hash:
                    raise DatasetIntegrityError(
                        f"SHA-256 hash mismatch for {path.name}"
                    )

            try:
                array = np.load(
                    path,
                    allow_pickle=False,
                    mmap_mode=mmap_mode,
                )
            except Exception as exc:
                raise DatasetIntegrityError(
                    f"could not load {path}"
                ) from exc

            expected_shape = tuple(metadata.get("shape", ()))
            expected_dtype = metadata.get("dtype")

            if array.shape != expected_shape:
                raise DatasetIntegrityError(
                    f"shape mismatch for {path.name}: "
                    f"expected {expected_shape}, received {array.shape}"
                )

            if str(array.dtype) != expected_dtype:
                raise DatasetIntegrityError(
                    f"dtype mismatch for {path}: "
                    f"expected {expected_dtype}, received {array.dtype}"

                )

            arrays[logical_name] = array

        return EllipticDataset(
            tx_id=arrays["tx_id"],
            timestep=arrays["timestep"],
            features=arrays["features"],
            labels=arrays["labels"],
            edge_index=arrays["edge_index"]
        )

    @classmethod
    def verify(cls, root: str | Path) -> None:
        """
        verifies persisted files and dataset contract
        """

        dataset = cls.load(
            root,
            verify_hashes=True,
            mmap_mode="r",
        )
        del dataset

