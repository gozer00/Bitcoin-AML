# persistence of run manifests under outputs/runs/

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

RUN_MANIFEST_FILENAME = "manifest.json"


def save_run(output_root: str | Path, manifest: dict[str, Any]) -> Path:
    """
    writes a run manifest to {output_root}/{run_id}/manifest.json

    never overwrites an existing run directory
    """
    run_id = manifest.get("run_id")

    if not run_id:
        raise ValueError("manifest is missing a non-empty 'run_id'")

    run_dir = Path(output_root) / run_id

    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")

    run_dir.mkdir(parents=True)

    manifest_path = run_dir / RUN_MANIFEST_FILENAME

    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, sort_keys=True)
        file.write("\n")

    return manifest_path