# persistence of raw per-node test-set predictions under outputs/runs/

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

PREDICTIONS_FILENAME = "predictions.npz"


def save_predictions(
    output_root: str | Path,
    run_id: str,
    *,
    tx_id: NDArray[np.integer],
    timestep: NDArray[np.integer],
    y_true: NDArray[np.integer],
    scores: NDArray[np.floating],
) -> Path:
    """
    writes raw per-node test-set predictions to
    {output_root}/{run_id}/predictions.npz

    must be called after save_run(output_root, manifest) has already
    created {output_root}/{run_id}/ for this run_id -- raises if that
    directory does not yet exist, rather than creating it itself, so
    directory-creation/overwrite-guard logic lives in exactly one place
    (run_store.save_run)
    """
    run_dir = Path(output_root) / run_id

    if not run_dir.is_dir():
        raise FileNotFoundError(
            f"run directory does not exist (call save_run first): {run_dir}"
        )

    predictions_path = run_dir / PREDICTIONS_FILENAME

    np.savez_compressed(
        predictions_path,
        tx_id=np.asarray(tx_id),
        timestep=np.asarray(timestep),
        y_true=np.asarray(y_true),
        scores=np.asarray(scores),
    )

    return predictions_path


def load_predictions(run_dir: str | Path) -> dict[str, NDArray]:
    """
    reads {run_dir}/predictions.npz back into a plain dict of arrays
    (tx_id, timestep, y_true, scores), the counterpart to save_predictions
    """
    predictions_path = Path(run_dir) / PREDICTIONS_FILENAME

    with np.load(predictions_path) as loaded:
        return {key: loaded[key] for key in loaded.files}
