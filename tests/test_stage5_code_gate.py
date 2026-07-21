"""Dependency-light checks for the Stage 5 Code Gate."""

from __future__ import annotations

import importlib.util
import subprocess
import sys

import pytest

from rlaif.torch_runtime import PYTORCH_REQUIRED_MESSAGE, is_torch_runtime_available


def test_stage5_cli_modules_import_without_torch_runtime() -> None:
    __import__("experiments.train_reward_model")
    __import__("experiments.evaluate_reward_model")
    __import__("experiments.smoke_test_reward_model")


@pytest.mark.skipif(is_torch_runtime_available(), reason="torch runtime is available")
@pytest.mark.parametrize(
    "arguments",
    [
        ["-m", "experiments.train_reward_model"],
        ["-m", "experiments.evaluate_reward_model", "--checkpoint", "missing.pt"],
        ["-m", "experiments.smoke_test_reward_model"],
    ],
)
def test_stage5_runtime_commands_fail_gracefully_without_torch(arguments: list[str]) -> None:
    result = subprocess.run([sys.executable, *arguments], capture_output=True, text=True, check=False)
    assert result.returncode == 3
    assert PYTORCH_REQUIRED_MESSAGE in result.stderr
    assert "Traceback" not in result.stderr
