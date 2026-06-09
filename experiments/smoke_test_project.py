"""Offline smoke test for the Stage 1 project skeleton."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import random
from pathlib import Path
from typing import Sequence

from utils.config import PROJECT_ROOT, ensure_project_directories, load_config
from utils.logger import configure_logger
from utils.seeding import seed_everything


PLACEHOLDER_MODULES = (
    "data_pipeline.placeholder",
    "envs.placeholder",
    "models.placeholder",
    "training.placeholder",
)
REQUIRED_CONFIG_SECTIONS = (
    "project",
    "city",
    "network",
    "bus",
    "truck",
    "station",
    "parcel",
    "reward",
    "ppo",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/shanghai_small.yaml",
        help="Path to the Stage 1 YAML configuration.",
    )
    return parser.parse_args(argv)


def run_smoke_test(config_path: str | Path) -> dict[str, object]:
    config = load_config(config_path)
    missing_sections = [key for key in REQUIRED_CONFIG_SECTIONS if key not in config]
    assert not missing_sections, f"Missing config sections: {missing_sections}"

    folders = ensure_project_directories(PROJECT_ROOT)
    assert all(path.is_dir() for path in folders.values())

    log_path = folders["logs"] / "smoke_test_project.log"
    logger = configure_logger("project.smoke_test", log_path, console=False)
    marker = "Stage 1 smoke test logger check"
    logger.info(marker)
    for handler in logger.handlers:
        handler.flush()
    assert log_path.is_file()
    assert marker in log_path.read_text(encoding="utf-8")

    seed = int(config["project"]["seed"])
    seeded_backends = seed_everything(seed)
    first_python_value = random.random()
    seed_everything(seed)
    assert random.random() == first_python_value

    if importlib.util.find_spec("numpy") is not None:
        np = importlib.import_module("numpy")
        seed_everything(seed)
        first_numpy_value = float(np.random.random())
        seed_everything(seed)
        assert float(np.random.random()) == first_numpy_value

    imported = [importlib.import_module(name).__name__ for name in PLACEHOLDER_MODULES]

    return {
        "config": str(config_path),
        "city": config["city"]["name"],
        "folders": [str(path) for path in folders.values()],
        "log_file": str(log_path),
        "seeded_backends": seeded_backends,
        "placeholder_modules": imported,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    results = run_smoke_test(args.config)
    print("Stage 1 smoke test passed.")
    for key, value in results.items():
        print(f"- {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
