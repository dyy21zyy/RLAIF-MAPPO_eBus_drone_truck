"""Shared helpers for original-scale real-transit smoke tests."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from utils.config import PROJECT_ROOT, load_config

TRANSIT_FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures" / "transit"


def smoke_config(config_path: str | Path, output_root: str | Path) -> Path:
    """Create a temporary config using committed smoke transit fixtures."""

    output_root = Path(output_root)
    config = deepcopy(load_config(config_path))
    config["reference"]["defaults_config"] = str(PROJECT_ROOT / "configs" / "original_ebus_drone_defaults.yaml")
    config["city"]["name"] = "smoke_original_scale_real_transit"
    config["project"]["output_dir"] = str(output_root / "outputs")
    config["project"]["log_dir"] = str(output_root / "logs")
    config["project"]["checkpoint_dir"] = str(output_root / "checkpoints")
    config["transit"]["stops_csv"] = str(TRANSIT_FIXTURE_DIR / "real_bus_stops.csv")
    config["transit"]["trips_csv"] = str(TRANSIT_FIXTURE_DIR / "real_bus_trips.csv")
    config["transit"]["stop_times_csv"] = str(TRANSIT_FIXTURE_DIR / "real_bus_stop_times.csv")
    config["transit"]["allow_synthetic_timetable_if_missing"] = False
    path = output_root / "original_scale_real_transit_smoke_config.json"
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return path
