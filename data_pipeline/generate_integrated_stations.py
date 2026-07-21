"""Integrated station selection from ordered bus stops."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from data_pipeline.common import write_csv

STATION_COLUMNS = ["station_id", "stop_id", "lat", "lon", "charger_num", "locker_capacity_kg", "drones_num", "initial_full_batteries", "power_capacity_kw", "battery_charging_power_kw", "battery_charging_duration_min"]


def select_integrated_stations(stops: list[dict[str, Any]], config: dict[str, Any], output_dir: Path | None = None) -> list[dict[str, Any]]:
    ordered = sorted(stops, key=lambda stop: int(stop["stop_sequence"]))
    count = int(config["network"]["num_integrated_stations"])
    if count > len(ordered) or count < 1:
        raise ValueError("num_integrated_stations must be between 1 and the number of bus stops")
    configured = config.get("network", {}).get("integrated_station_stop_indices")
    indices = [max(0, min(len(ordered) - 1, int(i) - 1)) for i in configured] if configured else ([0] if count == 1 else [round(i * (len(ordered) - 1) / (count - 1)) for i in range(count)])
    count = len(indices)
    station_config = config["station"]
    stations = []
    for number, index in enumerate(indices, start=1):
        stop = ordered[index]
        stations.append({"station_id": f"station_{number:02d}", "stop_id": stop["stop_id"],
                         "lat": float(stop["lat"]), "lon": float(stop["lon"]),
                         "charger_num": station_config["chargers_per_station"],
                         "locker_capacity_kg": station_config["locker_capacity_kg"],
                         "drones_num": station_config["drones_per_station"],
                         "initial_full_batteries": station_config["initial_full_batteries"],
                         "power_capacity_kw": station_config["power_capacity_kw"],
                         "battery_charging_power_kw": station_config["battery_charging_power_kw"],
                         "battery_charging_duration_min": station_config["battery_charging_duration_min"]})
    if output_dir:
        write_csv(output_dir / "integrated_stations.csv", stations, STATION_COLUMNS)
    return stations
