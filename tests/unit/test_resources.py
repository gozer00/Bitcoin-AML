import logging

import torch

from bitcoin_aml.utils.resources import select_device, track_peak_cuda_memory

logger = logging.getLogger(__name__)


def test_select_device_returns_cpu_without_cuda() -> None:
    if torch.cuda.is_available():
        return

    assert select_device() == torch.device("cpu")


def test_track_peak_cuda_memory_does_not_raise_and_reports_zero_on_cpu() -> None:
    if torch.cuda.is_available():
        return

    with track_peak_cuda_memory(logger, "test") as peak_memory:
        pass

    assert peak_memory["peak_gb"] == 0.0
