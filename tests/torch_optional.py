from __future__ import annotations

import pytest

from rlaif.torch_runtime import is_torch_runtime_available


def import_optional_torch(*, allow_module_level: bool = False):
    if not is_torch_runtime_available():
        pytest.skip(
            "torch is installed but unavailable in this environment",
            allow_module_level=allow_module_level,
        )
    try:
        return pytest.importorskip("torch")
    except OSError as exc:
        pytest.skip(
            f"torch is installed but unavailable: {exc}",
            allow_module_level=allow_module_level,
        )
