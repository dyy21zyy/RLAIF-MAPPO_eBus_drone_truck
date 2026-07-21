"""Shared helpers for commands that require the optional PyTorch runtime."""

from __future__ import annotations

import subprocess
import sys

PYTORCH_REQUIRED_MESSAGE = (
    "PyTorch is required for reward-model training/evaluation. "
    "Install torch>=2.0,<3.0 and rerun this command."
)


def is_torch_runtime_available() -> bool:
    """Return whether PyTorch can be imported in a separate runtime process."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import torch"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def is_missing_torch_error(exc: ModuleNotFoundError) -> bool:
    """Return whether an import failure is specifically the optional torch dependency."""
    return exc.name == "torch" or (exc.name is not None and exc.name.startswith("torch."))
