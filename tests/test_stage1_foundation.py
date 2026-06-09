"""Regression tests for the Stage 1 project foundation."""

from __future__ import annotations

import importlib
import logging
import random
import socket
from pathlib import Path

import pytest

from experiments.smoke_test_project import run_smoke_test
from utils.config import ConfigError, ensure_project_directories, load_config
from utils.logger import configure_logger
from utils.seeding import seed_everything


@pytest.mark.parametrize("contents", ["", "   \n\t\n"])
def test_load_config_rejects_blank_file(tmp_path: Path, contents: str) -> None:
    config_path = tmp_path / "blank.yaml"
    config_path.write_text(contents, encoding="utf-8")

    with pytest.raises(ConfigError, match="empty"):
        load_config(config_path)


@pytest.mark.parametrize(
    "contents",
    [
        "{}\n",
        "[unclosed\n",
        "- one\n- two\n",
    ],
)
def test_load_config_rejects_invalid_or_non_mapping_roots(
    tmp_path: Path, contents: str
) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(contents, encoding="utf-8")

    with pytest.raises(ConfigError):
        load_config(config_path)


def test_load_config_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="does not exist"):
        load_config(tmp_path / "missing.yaml")


def test_load_config_loads_valid_mapping(tmp_path: Path) -> None:
    config_path = tmp_path / "valid.yaml"
    config_path.write_text("project:\n  seed: 42\n", encoding="utf-8")

    assert load_config(config_path) == {"project": {"seed": 42}}


def test_ensure_project_directories_creates_runtime_tree(tmp_path: Path) -> None:
    created = ensure_project_directories(tmp_path)

    assert set(created) == {
        "data/raw",
        "data/processed",
        "logs",
        "outputs",
        "checkpoints",
    }
    assert all(path.is_dir() for path in created.values())


def test_configure_logger_writes_file_without_duplicate_handlers(
    tmp_path: Path,
) -> None:
    logger_name = "tests.stage1.logger"
    log_path = tmp_path / "logs" / "stage1.log"

    logger = configure_logger(logger_name, log_path, console=False)
    logger = configure_logger(logger_name, log_path, console=False)
    project_handlers = [
        handler
        for handler in logger.handlers
        if getattr(handler, "_project_handler", False)
    ]

    assert len(project_handlers) == 1
    logger.info("Stage 1 logger regression marker")
    for handler in logger.handlers:
        handler.flush()
    assert "Stage 1 logger regression marker" in log_path.read_text(encoding="utf-8")

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    logging.Logger.manager.loggerDict.pop(logger_name, None)


def test_seed_everything_is_deterministic() -> None:
    seed_everything(12345)
    first_sequence = [random.random() for _ in range(5)]

    seed_everything(12345)

    assert [random.random() for _ in range(5)] == first_sequence


@pytest.mark.parametrize(
    "module_name",
    [
        "data_pipeline.placeholder",
        "envs.placeholder",
        "models.placeholder",
        "training.placeholder",
    ],
)
def test_future_stage_placeholders_remain_unimplemented(module_name: str) -> None:
    module = importlib.import_module(module_name)

    assert module.IMPLEMENTED is False


def test_smoke_test_does_not_access_network(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def reject_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("Stage 1 smoke test attempted network access")

    monkeypatch.setattr(socket, "socket", reject_network)
    monkeypatch.chdir(tmp_path)

    result = run_smoke_test(Path(__file__).parents[1] / "configs/shanghai_small.yaml")

    assert result["city"] == "shanghai_yangpu"
    assert result["placeholder_modules"] == [
        "data_pipeline.placeholder",
        "envs.placeholder",
        "models.placeholder",
        "training.placeholder",
    ]
