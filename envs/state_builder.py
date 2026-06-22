"""Shared objective feature construction for assignment observations and RLAIF data."""

from __future__ import annotations

from statistics import fmean
from typing import Any


def action_name(env: Any, action_id: int) -> str:
    """Return the stable Stage 3 assignment action name."""
    station_count = len(env.station_ids)
    if action_id == 0:
        return "TD"
    if action_id <= station_count:
        return f"TBD_{env.station_ids[action_id - 1]}"
    return f"TLD_{env.station_ids[action_id - station_count - 1]}"


def build_assignment_features(env: Any, parcel: Any) -> list[float]:
    """Build the normalized Stage 3 assignment observation vector."""
    horizon = max(env.horizon_min, 1.0)
    return [
        env.now_min / horizon,
        parcel.weight_kg / max(float(env.config["truck"]["capacity_kg"]), 1.0),
        max(0.0, parcel.deadline_min - env.now_min) / horizon,
        parcel.priority / 3.0,
        float(parcel.drone_feasible),
        min(env.truck_available_min, default=env.horizon_min) / horizon,
    ]


def _drone_time(env: Any, station_id: str, parcel_id: str) -> float:
    distance_km = float(
        env.drone_distance_m[env.drone_row_index[station_id], env.drone_column_index[parcel_id]]
    ) / 1000.0
    return 2.0 * distance_km / float(env.config["network"]["drone_speed_kmph"]) * 60.0 + float(
        env.config["network"]["customer_service_time_min"]
    )


def _station_power_margin(env: Any, station: Any) -> float:
    battery_load = len(station.battery_ready_min) * station.battery_power_kw
    bus_load = len([end for end in station.active_bus_charges if end > env.now_min]) * float(
        env.config["bus"]["charging_power_kw"]
    )
    return station.power_capacity_kw - float(env.config["station"]["base_load_kw"]) - battery_load - bus_load


def _next_bus_details(env: Any, station_id: str, parcel: Any) -> tuple[float, float, float, str | None]:
    trip = env._next_freight_trip(env.now_min, station_id, parcel.weight_kg)
    if trip is None:
        return 0.0, 0.0, 0.0, None
    trip_id, arrival = trip
    rows = env.trip_stop_times[trip_id]
    departure = float(rows[0]["departure_time"])
    remaining = float(env.config["bus"]["freight_capacity_kg"]) - env.bus_freight_kg[trip_id]
    return max(0.0, departure - env.now_min), max(0.0, arrival - departure), remaining, trip_id


def build_candidate_action_features(env: Any, parcel: Any, action_id: int, feasible: bool) -> dict[str, Any]:
    """Build objective estimates for one action without assigning a preference score."""
    station_count = len(env.station_ids)
    truck_capacity = float(env.config["truck"]["capacity_kg"])
    reasons: list[str] = []
    if parcel.weight_kg > truck_capacity:
        reasons.append("truck_capacity_exceeded")
    estimated_truck_distance = 0.0
    estimated_truck_time = 0.0
    estimated_bus_wait = 0.0
    estimated_bus_linehaul = 0.0
    estimated_drone_time = 0.0
    estimated_locker_load = 0.0
    estimated_power_margin = 0.0

    if action_id == 0:
        depot = env.truck_location_index["depot_01"]
        customer = env.truck_location_index[parcel.parcel_id]
        estimated_truck_distance = float(env.truck_distance_m[depot, customer]) / 1000.0
        estimated_truck_time = float(env.truck_time_min[depot, customer])
        start = max(env.now_min, min(env.truck_available_min)) + parcel.weight_kg * float(
            env.config["truck"]["loading_time_min_per_kg"]
        )
        delivery = start + estimated_truck_time + float(env.config["network"]["customer_service_time_min"])
    else:
        station_index = (action_id - 1) % station_count
        station_id = env.station_ids[station_index]
        station = env.stations[station_id]
        estimated_drone_time = _drone_time(env, station_id, parcel.parcel_id)
        estimated_locker_load = station.locker_load_kg + parcel.weight_kg
        estimated_power_margin = _station_power_margin(env, station)
        if not parcel.drone_feasible:
            reasons.append("parcel_not_drone_feasible")
        if parcel.weight_kg > float(env.config["network"]["drone_payload_kg"]):
            reasons.append("drone_payload_exceeded")
        if station.drones <= 0:
            reasons.append("no_station_drone")
        if station.full_batteries <= 0:
            reasons.append("no_full_battery_now")
        if action_id <= station_count:
            estimated_bus_wait, estimated_bus_linehaul, bus_remaining, trip_id = _next_bus_details(
                env, station_id, parcel
            )
            if parcel.weight_kg > float(env.config["bus"]["freight_capacity_kg"]):
                reasons.append("bus_freight_capacity_exceeded")
            if trip_id is None:
                reasons.append("no_feasible_freight_bus")
            elif parcel.weight_kg > bus_remaining:
                reasons.append("selected_bus_capacity_exceeded")
            delivery = env.now_min + estimated_bus_wait + estimated_bus_linehaul + estimated_drone_time / 2.0
        else:
            depot = env.truck_location_index["depot_01"]
            station_location = env.truck_location_index[station_id]
            estimated_truck_distance = float(env.truck_distance_m[depot, station_location]) / 1000.0
            estimated_truck_time = float(env.truck_time_min[depot, station_location])
            start = max(env.now_min, min(env.truck_available_min)) + parcel.weight_kg * float(
                env.config["truck"]["loading_time_min_per_kg"]
            )
            delivery = start + estimated_truck_time + parcel.weight_kg * float(
                env.config["truck"]["unloading_time_min_per_kg"]
            ) + estimated_drone_time / 2.0
    if not feasible and not reasons:
        reasons.append("masked_by_environment")
    return {
        "action_id": action_id,
        "action_name": action_name(env, action_id),
        "feasible_flag": bool(feasible),
        "estimated_delivery_time": float(delivery),
        "estimated_lateness": max(0.0, float(delivery) - parcel.deadline_min),
        "estimated_truck_distance": float(estimated_truck_distance),
        "estimated_truck_time": float(estimated_truck_time),
        "estimated_bus_wait_time": float(estimated_bus_wait),
        "estimated_bus_linehaul_time": float(estimated_bus_linehaul),
        "estimated_drone_time": float(estimated_drone_time),
        "estimated_locker_load_after_assignment": float(estimated_locker_load),
        "estimated_station_power_margin": float(estimated_power_margin),
        "infeasibility_reasons": reasons,
    }


def build_system_summary(env: Any, parcel: Any) -> dict[str, Any]:
    feasible_trips = []
    next_arrivals = []
    remaining_capacities = []
    for station_id in env.station_ids:
        wait, linehaul, remaining, trip_id = _next_bus_details(env, station_id, parcel)
        if trip_id is not None:
            feasible_trips.append(trip_id)
            next_arrivals.append(env.now_min + wait)
            remaining_capacities.append(remaining)
    stations = list(env.stations.values())
    return {
        "idle_truck_count": sum(time <= env.now_min for time in env.truck_available_min),
        "earliest_truck_available_time": float(min(env.truck_available_min)),
        "average_truck_capacity_remaining": float(env.config["truck"]["capacity_kg"]),
        "terminal_queue_length": sum(len(ids) for ids in env.pending_bus_parcels.values()),
        "next_freight_bus_arrival_time": float(min(next_arrivals)) if next_arrivals else 0.0,
        "feasible_freight_bus_count_before_deadline": sum(
            env.now_min <= arrival <= parcel.deadline_min for arrival in next_arrivals
        ),
        "average_bus_freight_remaining_capacity": fmean(remaining_capacities) if remaining_capacities else 0.0,
        "global_locker_occupancy_mean": fmean(
            station.locker_load_kg / max(station.locker_capacity_kg, 1.0) for station in stations
        ),
        "global_idle_drones": sum(sum(value <= env.now_min for value in station.drone_available_min) for station in stations),
        "global_full_batteries": sum(station.full_batteries for station in stations),
        "global_station_power_margin_mean": fmean(_station_power_margin(env, station) for station in stations),
    }


def build_station_states(env: Any, parcel: Any) -> list[dict[str, Any]]:
    rows = []
    for station_id in env.station_ids:
        station = env.stations[station_id]
        distance = float(
            env.drone_distance_m[env.drone_row_index[station_id], env.drone_column_index[parcel.parcel_id]]
        ) / 1000.0
        wait, linehaul, remaining, trip_id = _next_bus_details(env, station_id, parcel)
        rows.append({
            "station_id": station_id,
            "distance_customer_to_station": distance,
            "drone_round_trip_time": _drone_time(env, station_id, parcel.parcel_id),
            "drone_feasible_from_station": bool(
                parcel.drone_feasible and parcel.weight_kg <= float(env.config["network"]["drone_payload_kg"])
            ),
            "locker_remaining_capacity": station.locker_capacity_kg - station.locker_load_kg,
            "locker_occupancy_ratio": station.locker_load_kg / max(station.locker_capacity_kg, 1.0),
            "idle_drones": sum(value <= env.now_min for value in station.drone_available_min),
            "full_batteries": station.full_batteries,
            "station_power_margin": _station_power_margin(env, station),
            "next_feasible_bus_arrival_to_station": env.now_min + wait + linehaul if trip_id else 0.0,
            "bus_freight_remaining_capacity_to_station": remaining,
        })
    return rows
