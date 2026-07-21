"""Validation helpers for Phase 0 paper parameter and MAPPO configs."""

from __future__ import annotations

from collections.abc import Mapping

PARAMETER_CATEGORIES = {"literature_adapted", "project_extension", "real_input", "fallback_only", "derived"}

REQUIRED_SECTIONS = {
    "scenario", "network", "time", "bus_schedule", "bus", "passenger",
    "parcel", "truck", "drone", "drone_battery", "station",
}


def flatten_parameter_keys(config: Mapping[str, object]) -> set[str]:
    keys: set[str] = set()
    for section, value in config.items():
        if isinstance(value, Mapping):
            for key in value:
                keys.add(f"{section}.{key}")
        else:
            keys.add(str(section))
    return keys


def _positive(config: Mapping[str, object], section: str, key: str) -> None:
    value = config[section][key]  # type: ignore[index]
    if float(value) <= 0:
        raise ValueError(f"{section}.{key} must be positive")


def validate_paper_config(config: Mapping[str, object]) -> None:
    missing = REQUIRED_SECTIONS - set(config)
    if missing:
        raise ValueError(f"Missing sections: {sorted(missing)}")
    if len(config["network"]["integrated_station_stop_indices"]) != config["network"]["num_integrated_stations"]:  # type: ignore[index]
        raise ValueError("Integrated station count mismatch")
    if config["bus_schedule"]["freight_trip_count"] > config["bus_schedule"]["scheduled_trip_count"]:  # type: ignore[index]
        raise ValueError("freight trips cannot exceed scheduled trips")
    for section, key in [
        ("scenario", "num_parcels"), ("network", "num_stops"),
        ("truck", "weight_capacity_kg"), ("truck", "volume_capacity_m3"),
        ("drone_battery", "charging_slots"), ("station", "power_capacity_kw"),
    ]:
        _positive(config, section, key)


def validate_mappo_config(config: Mapping[str, object]) -> None:
    if config.get("mode") not in {"environment_reward", "rlaif_reward"}:
        raise ValueError("MAPPO config mode must be environment_reward or rlaif_reward")
    training = config.get("training")
    networks = config.get("networks")
    if not isinstance(training, Mapping) or not isinstance(networks, Mapping):
        raise ValueError("MAPPO config requires training and networks sections")
    if int(training.get("total_episodes", 0)) <= 20:
        raise ValueError("Formal MAPPO config must not be the 20-episode smoke config")
    forbidden = {"epsilon", "epsilon_greedy", "replay_buffer", "target_q_network", "polyak", "fixed_nine_action_output"}
    present = forbidden & set(training)
    if present:
        raise ValueError(f"DDQN-only settings are forbidden in MAPPO config: {sorted(present)}")
    for agent in ["assignment", "truck", "bus", "station"]:
        if f"{agent}_hidden_dims" not in networks:
            raise ValueError(f"Missing {agent} network dimensions")


def validate_parameter_provenance(parameter_keys: set[str], provenance: Mapping[str, object]) -> None:
    missing = parameter_keys - set(provenance)
    if missing:
        raise ValueError(f"Missing provenance for: {sorted(missing)}")
    for key in parameter_keys:
        entry = provenance[key]
        if not isinstance(entry, Mapping):
            raise ValueError(f"Provenance entry for {key} must be a mapping")
        category = entry.get("category")
        if category not in PARAMETER_CATEGORIES:
            raise ValueError(f"Invalid provenance category for {key}: {category}")
        if not entry.get("source"):
            raise ValueError(f"Missing source text for {key}")
