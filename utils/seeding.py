"""Reproducibility helpers for Python and optional numerical frameworks."""

from __future__ import annotations

import importlib
import importlib.util
import os
import random
from typing import Any


def seed_everything(seed: int) -> dict[str, Any]:
    """Seed available random-number generators without requiring ML packages.

    NumPy and PyTorch are supported when installed. They are imported lazily so
    the Stage 1 smoke test remains lightweight and usable without PyTorch.
    """

    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    seeded = {"seed": seed, "python": True, "numpy": False, "torch": False}

    if importlib.util.find_spec("numpy") is not None:
        np = importlib.import_module("numpy")
        np.random.seed(seed)
        seeded["numpy"] = True

    if importlib.util.find_spec("torch") is not None:
        torch = importlib.import_module("torch")
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        seeded["torch"] = True

    return seeded
