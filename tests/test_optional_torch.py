from __future__ import annotations

import importlib.machinery
import subprocess

import pytest

from utils.seeding import seed_everything


def test_seed_everything_treats_broken_torch_as_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    real_find_spec = importlib.util.find_spec
    real_import_module = importlib.import_module

    def fake_find_spec(name: str):
        if name == "torch":
            return importlib.machinery.ModuleSpec("torch", loader=None)
        return real_find_spec(name)

    def fake_import_module(name: str, package: str | None = None):
        if name == "torch":
            raise OSError("broken torch dll")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    seeded = seed_everything(7)

    assert seeded["torch"] is False


def test_torch_runtime_probe_reports_failed_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    from rlaif.torch_runtime import is_torch_runtime_available

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args, 1, stdout="", stderr="broken torch dll")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert is_torch_runtime_available() is False


def test_torch_runtime_probe_reports_successful_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    from rlaif.torch_runtime import is_torch_runtime_available

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args, 0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert is_torch_runtime_available() is True


def test_import_optional_torch_skips_without_importing_when_probe_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.torch_optional import import_optional_torch

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args, 1, stdout="", stderr="broken torch dll")

    def fail_importorskip(name: str):
        raise AssertionError(f"should not import {name} after a failed probe")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(pytest, "importorskip", fail_importorskip)

    with pytest.raises(pytest.skip.Exception):
        import_optional_torch()
