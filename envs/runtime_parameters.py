"""Runtime operational-parameter resolution for delivery environments."""
from __future__ import annotations

from typing import Any


class RuntimeParameterError(ValueError):
    """Raised when a formal run is missing a required operational parameter."""


SMOKE_DEFAULTS = {
    "network.drone_payload_kg": 5.0,
    "network.drone_radius_km": 8.0,
    "network.max_drone_round_trip_min": 120.0,
    "network.drone_speed_kmph": 40.0,
    "station.battery_charging_power_kw": 2.0,
    "station.battery_charging_duration_min": 45.0,
    "station.power_capacity_kw": 1100.0,
    "station.charging_slots": 6,
    "bus.charging_power_kw": 500.0,
}


def run_classification(env: Any) -> str:
    return str(getattr(env, "run_classification", None) or getattr(env, "config", {}).get("run_classification", "smoke")).lower()


def _formal(env: Any) -> bool:
    return run_classification(env) == "formal"


def _cfg(env: Any, section: str, key: str, *, default_key: str | None = None) -> float:
    config = getattr(env, "config", {}) or {}
    try:
        value = config.get(section, {}).get(key, None)
    except AttributeError as exc:
        raise RuntimeParameterError(f"invalid config section {section!r}") from exc
    if value is None:
        dk = default_key or f"{section}.{key}"
        if _formal(env):
            raise RuntimeParameterError(
                f"run classification=formal invalid field={section}.{key} actual value=None required value=resolved numeric parameter"
            )
        value = SMOKE_DEFAULTS[dk]
    if isinstance(value, bool):
        raise RuntimeParameterError(f"{section}.{key} must be numeric, got boolean")
    return float(value)


def current_station_base_load_kw(env: Any, station_id: str, time_min: float | None = None) -> float:
    t = float(getattr(env, "now_min", 0.0) if time_min is None else time_min)
    profile = getattr(env, "station_base_load_profile", None)
    if profile is None:
        if _formal(env):
            raise RuntimeParameterError(
                "run classification=formal invalid field=station_base_load_profile actual value=None required value=time-varying profile"
            )
        return _cfg(env, "station", "base_load_kw", default_key="station.power_capacity_kw") * 0.0
    return float(profile.load_at(station_id, t))


def drone_payload_capacity_kg(env: Any) -> float:
    return _cfg(env, "network", "drone_payload_kg")


def drone_service_radius_km(env: Any) -> float:
    return _cfg(env, "network", "drone_radius_km")


def maximum_drone_round_trip_min(env: Any) -> float:
    return _cfg(env, "network", "max_drone_round_trip_min")


def drone_speed_kmph(env: Any) -> float:
    return _cfg(env, "network", "drone_speed_kmph")


def station_charging_slot_count(env: Any, station: Any) -> int:
    value = getattr(station, "charging_slots", None)
    if value is None:
        value = getattr(env, "config", {}).get("station", {}).get("charging_slots")
    if value is None:
        if _formal(env):
            raise RuntimeParameterError("run classification=formal invalid field=station.charging_slots actual value=None required value=integer")
        value = SMOKE_DEFAULTS["station.charging_slots"]
    return int(value)


def battery_charging_power_kw(env: Any, station: Any) -> float:
    value = getattr(station, "battery_power_kw", None)
    return float(value) if value is not None else _cfg(env, "station", "battery_charging_power_kw")


def battery_charging_duration_min(env: Any, station: Any) -> float:
    value = getattr(station, "battery_charge_duration_min", None)
    return float(value) if value is not None else _cfg(env, "station", "battery_charging_duration_min")


def station_power_capacity_kw(env: Any, station: Any) -> float:
    value = getattr(station, "power_capacity_kw", None)
    return float(value) if value is not None else _cfg(env, "station", "power_capacity_kw")


def bus_charging_power_kw(env: Any) -> float:
    return _cfg(env, "bus", "charging_power_kw")
