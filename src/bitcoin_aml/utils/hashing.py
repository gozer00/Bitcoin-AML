from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

def sha256_json(obj: Any) -> str:
    """
    deterministic hash of a JSON-serializable structure
    (dicts are hashed with keys sorted, independent of insertion order)
    """
    payload = json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(payload).hexdigest()

def sha256_file(
        path: str | Path,
        *,
        chunk_size: int = 4 * 1024 * 1024
) -> str:
    """
    memory efficient file hashing
    """

    file_path = Path(path)

    if not file_path.is_file():
        raise FileNotFoundError(file_path)

    hasher = hashlib.sha256()

    with file_path.open("rb") as file:
        while chunk := file.read(chunk_size):
            hasher.update(chunk)

    return hasher.hexdigest()