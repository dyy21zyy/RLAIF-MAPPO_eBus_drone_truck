"""Versioned scenario manifest writer with artifact hashes."""
from __future__ import annotations

import hashlib, json, subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCENARIO_SCHEMA_VERSION = 1
SEED_NAMES = ("network_seed", "parcel_seed", "passenger_seed", "travel_time_seed", "initial_bus_energy_seed", "station_base_load_seed")

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def sha256_json(data: Any) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def resolve_seeds(config: dict[str, Any]) -> dict[str, int]:
    seeds = dict(config.get("seeds", {}))
    base = int(config.get("project", {}).get("seed", 0))
    for i, name in enumerate(SEED_NAMES):
        seeds.setdefault(name, base + i * 1009)
    return {name: int(seeds[name]) for name in SEED_NAMES}

def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None

def write_scenario_manifest(output_dir: Path, scenario_id: str, config: dict[str, Any], artifacts: dict[str, str], provenance: dict[str, Any] | None = None) -> dict[str, Any]:
    artifact_hashes = {k: sha256_file(output_dir / v) for k, v in sorted(artifacts.items()) if (output_dir / v).is_file()}
    manifest = {
        "scenario_id": scenario_id,
        "scenario_schema_version": SCENARIO_SCHEMA_VERSION,
        "resolved_configuration": config,
        "configuration_sha256": sha256_json(config),
        "random_seeds": resolve_seeds(config),
        "generated_artifact_paths": artifacts,
        "artifact_sha256": artifact_hashes,
        "parameter_provenance": provenance or config.get("parameter_provenance", {}),
        "generation_timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
    }
    (output_dir / "scenario_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest
