# device selection and GPU/host peak-memory tracking

from __future__ import annotations

import logging
import resource
from contextlib import contextmanager
from typing import Iterator

import torch

_BYTES_PER_GB = 1024**3
_KB_PER_GB = 1024**2


def select_device() -> torch.device:
    """
    returns the cuda device if available, else cpu
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


@contextmanager
def track_peak_cuda_memory(
    logger: logging.Logger,
    label: str,
) -> Iterator[dict[str, float]]:
    """
    tracks peak CUDA allocation across the wrapped block

    yields a dict with a single "peak_gb" key that is populated on exit
    (0.0 when CUDA is unavailable, so this degrades cleanly on CPU-only
    machines rather than raising)
    """
    cuda_available = torch.cuda.is_available()

    if cuda_available:
        torch.cuda.reset_peak_memory_stats()

    result: dict[str, float] = {"peak_gb": 0.0}

    try:
        yield result
    finally:
        if cuda_available:
            result["peak_gb"] = torch.cuda.max_memory_allocated() / _BYTES_PER_GB
            logger.info("%s peak_gpu_memory=%.4f GB", label, result["peak_gb"])
        else:
            logger.info("%s peak_gpu_memory=0.0 GB (cpu)", label)


@contextmanager
def track_peak_host_memory(
    logger: logging.Logger,
    label: str,
) -> Iterator[dict[str, float]]:
    """
    reports peak host RSS (resident set size) for the current process

    yields a dict with a single "peak_gb" key, populated on exit.

    unlike `track_peak_cuda_memory`, this cannot be reset/scoped to just the
    wrapped block: `getrusage(RUSAGE_SELF).ru_maxrss` is a monotonically
    non-decreasing, process-wide high-water mark since process start (POSIX
    has no per-block equivalent of CUDA's `reset_peak_memory_stats`). The
    reported value is therefore an upper bound that also includes whatever
    the process had already allocated before entering this block, not this
    block's own contribution in isolation -- still useful as a ceiling check
    (e.g. against a fixed RAM budget) if this call runs early/in isolation,
    but not a precise per-block delta.
    """
    result: dict[str, float] = {"peak_gb": 0.0}

    try:
        yield result
    finally:
        peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result["peak_gb"] = peak_kb / _KB_PER_GB
        logger.info("%s peak_host_memory=%.4f GB (process-wide high-water mark)", label, result["peak_gb"])
