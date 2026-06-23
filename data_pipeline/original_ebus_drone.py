"""Load and validate inherited defaults from the original eBus-Drone project."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from utils.config import PROJECT_ROOT, load_config


REQUIRED_TOP_LEVEL_KEYS = {
    "scale",
    "bus",
    "charging",
    "parcel",
    "drone",
    "battery",
    "power",
    "reward",
    "sources",
}


def _resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def load_original_ebus_drone_defaults(path: str | Path) -> dict[str, Any]:
    """Return the machine-readable inherited eBus-Drone defaults.

    The file is intentionally checked into this repository so experiment builds
    do not silently infer or fabricate previous-paper settings. The reference
    repo path is still documented for traceability.
    """

    resolved = _resolve_project_path(path)
    defaults = load_config(resolved)
    missing = REQUIRED_TOP_LEVEL_KEYS - set(defaults)
    if missing:
        raise ValueError(f"Original eBus-Drone defaults are missing sections: {sorted(missing)}")
    scale = defaults["scale"]
    for key in (
        "num_stops",
        "num_integrated_stations",
        "num_parcels",
        "service_horizon_min",
        "bus_operation_horizon_min",
        "planned_headway_min",
    ):
        if key not in scale:
            raise ValueError(f"Original eBus-Drone defaults are missing scale.{key}")
    return defaults


def source_entry(defaults: dict[str, Any], field: str, value: Any, notes: str | None = None) -> dict[str, Any]:
    """Build a provenance entry for an inherited setting."""

    source = defaults.get("sources", {}).get(field, defaults.get("sources", {}).get(field.split(".", 1)[0], {}))
    return {
        "field": field,
        "value": value,
        "source_type": source.get("source_type", "original_ebus_drone"),
        "source_file": source.get("source_file", "../eBus-Drone"),
        "notes": notes if notes is not None else source.get("notes", ""),
    }
