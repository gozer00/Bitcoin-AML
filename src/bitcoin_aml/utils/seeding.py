# deterministic seeding helpers

from __future__ import annotations

import random

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """
    seeds python's random module, numpy's global RNG, and torch (cpu + cuda)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_rng(seed: int) -> np.random.Generator:
    """
    returns an independent, seeded numpy random generator
    """
    return np.random.default_rng(seed)