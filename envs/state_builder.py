"""Shared objective feature construction for assignment observations and RLAIF data."""

from __future__ import annotations

from statistics import fmean
from typing import Any


ASSIGNMENT_GLOBAL_FEATURE_NAMES = (
    "time_norm",
    "deadline_remaining_norm",
    "weight_norm",
    "volume_norm",
    "priority_urgent",
    "priority_normal",
    "priority_low",
    "drone_feasible_global",
    "depot_to_customer_time_norm",
    "nearest_station_distance_norm",
    "idle_truck_count_norm",
    "earliest_truck_available_time_norm",
    "avg_truck_capacity_remaining_norm",
    "terminal_queue_length_norm",
    "next_freight_bus_arrival_time_norm",
    "feasible_freight_bus_count_norm",
    "avg_bus_freight_remaining_capacity_norm",
)

ASSIGNMENT_STATION_FEATURE_NAMES = (
    "station_customer_distance_norm",
    "station_drone_round_trip_time_norm",
    "station_drone_feasible",
    "locker_remaining_capacity_norm",
    "locker_occupancy_ratio",
    "idle_drones_norm",
    "full_batteries_norm",
    "station_power_margin_norm",
    "next_feasible_bus_wait_time_norm",
    "bus_freight_remaining_capacity_to_station_norm",
)

CANDIDATE_ACTION_FEATURE_NAMES = (
    "action_type_TD",
    "action_type_TBD",
    "action_type_TLD",
    "action_station_index_norm",
    "feasible_flag",
    "estimated_delivery_time_norm",
    "estimated_lateness_norm",
    "estimated_truck_distance_norm",
    "estimated_truck_time_norm",
    "estimated_bus_wait_time_norm",
    "estimated_bus_linehaul_time_norm",
    "estimated_drone_time_norm",
    "estimated_locker_load_after_assignment_norm",
    "estimated_station_power_margin_norm",
)


def assignment_feature_names(station_ids: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """Return the ordered shared assignment schema for sorted station IDs."""
    station_names = tuple(
        f"{station_id}.{feature_name}"
        for station_id in sorted(station_ids)
        for feature_name in ASSIGNMENT_STATION_FEATURE_NAMES
    )
    return ASSIGNMENT_GLOBAL_FEATURE_NAMES + station_names


def action_name(env: Any, action_id: int) -> str:
    """Return the stable Stage 3 assignment action name."""
    station_count = len(env.station_ids)
    if action_id == 0:
        return "TD"
    if action_id <= station_count:
        return f"TBD_{env.station_ids[action_id - 1]}"
    return f"TLD_{env.station_ids[action_id - station_count - 1]}"


def build_assignment_features(env: Any, parcel: Any) -> list[float]:
    """Build the normalized Stage 3 assignment observation vector.

    The MVP has no parcel-volume vehicle constraint or persistent per-trip truck
    payload yet. Volume is therefore normalized by the largest instance parcel,
    and average truck capacity is approximated as full for each one-parcel trip.
    Both approximations keep a stable slot until richer physical data is added.
    """
    horizon = max(env.horizon_min, 1.0)
    truck_capacity = max(float(env.config["truck"]["capacity_kg"]), 1.0)
    bus_capacity = max(float(env.config["bus"]["freight_capacity_kg"]), 1.0)
    parcel_count = max(len(env.parcels), 1)
    trip_count = max(len(env.trip_rows), 1)
    volume_scale = max((float(row["volume"]) for row in env.parcel_rows), default=1.0)
    depot = env.truck_location_index["depot_01"]
    customer = env.truck_location_index[parcel.parcel_id]
    station_distances = [
        float(env.drone_distance_m[env.drone_row_index[station_id], env.drone_column_index[parcel.parcel_id]])
        / 1000.0
        for station_id in sorted(env.station_ids)
    ]
    bus_details = [
        _next_bus_details(env, station_id, parcel)
        for station_id in sorted(env.station_ids)
    ]
    feasible_details = [details for details in bus_details if details[3] is not None]
    earliest_bus_arrival = min(
        (wait + linehaul for wait, linehaul, _remaining, _trip_id in feasible_details),
        default=horizon,
    )
    feasible_trip_ids = {trip_id for _wait, _linehaul, _remaining, trip_id in feasible_details}
    remaining_capacities = [remaining for _wait, _linehaul, remaining, _trip_id in feasible_details]
    truck_count = max(int(env.config["truck"]["num_trucks"]), 1)
    global_features = [
        env.now_min / horizon,
        max(0.0, parcel.deadline_min - env.now_min) / horizon,
        parcel.weight_kg / truck_capacity,
        parcel.volume / max(volume_scale, 1.0),
        float(parcel.priority == 3),
        float(parcel.priority == 2),
        float(parcel.priority == 1),
        float(any(env._station_can_serve_by_drone(parcel, station_id) for station_id in env.station_ids)),
        float(env.truck_time_min[depot, customer]) / horizon,
        min(station_distances, default=0.0) / max(float(env.config["network"]["drone_radius_km"]), 1.0),
        sum(available <= env.now_min for available in env.truck_available_min) / truck_count,
        min(env.truck_available_min, default=env.horizon_min) / horizon,
        (
            fmean(truck.remaining_capacity_kg for truck in env.trucks) / truck_capacity
            if env.trucks else 0.0
        ),
        sum(len(parcel_ids) for parcel_ids in env.pending_bus_parcels.values()) / parcel_count,
        earliest_bus_arrival / horizon,
        len(feasible_trip_ids) / trip_count,
        (fmean(remaining_capacities) if remaining_capacities else 0.0) / bus_capacity,
    ]
    station_features: list[float] = []
    for station_id in sorted(env.station_ids):
        station = env.stations[station_id]
        distance = float(
            env.drone_distance_m[
                env.drone_row_index[station_id], env.drone_column_index[parcel.parcel_id]
            ]
        ) / 1000.0
        wait, linehaul, remaining, trip_id = _next_bus_details(env, station_id, parcel)
        station_features.extend([
            distance / max(float(env.config["network"]["drone_radius_km"]), 1.0),
            _drone_time(env, station_id, parcel.parcel_id)
            / max(float(env.config["network"]["max_drone_round_trip_min"]), 1.0),
            float(env._station_can_serve_by_drone(parcel, station_id)),
            max(0.0, station.locker_capacity_kg - station.locker_load_kg)
            / max(station.locker_capacity_kg, 1.0),
            station.locker_load_kg / max(station.locker_capacity_kg, 1.0),
            sum(value <= env.now_min for value in station.drone_available_min) / max(station.drones, 1),
            station.full_batteries / max(float(env.config["station"]["initial_full_batteries"]), 1.0),
            _station_power_margin(env, station) / max(station.power_capacity_kw, 1.0),
            (wait + linehaul) / horizon if trip_id else 1.0,
            remaining / bus_capacity if trip_id else 0.0,
        ])
    return [float(value) for value in global_features + station_features]


def _drone_time(env: Any, station_id: str, parcel_id: str) -> float:
    distance_km = float(
        env.drone_distance_m[env.drone_row_index[station_id], env.drone_column_index[parcel_id]]
    ) / 1000.0
    return 2.0 * distance_km / float(env.config["network"]["drone_speed_kmph"]) * 60.0 + float(
        env.config["network"]["customer_service_time_min"]
    )


def _station_power_margin(env: Any, station: Any) -> float:
    battery_load = sum(
        start <= env.now_min < end for start, end in station.active_battery_charges
    ) * station.battery_power_kw
    bus_load = len([end for end in station.active_bus_charges if end > env.now_min]) * float(
        env.config["bus"]["charging_power_kw"]
    )
    return station.power_capacity_kw - float(env.config["station"]["base_load_kw"]) - battery_load - bus_load


def _next_bus_details(env: Any, station_id: str, parcel: Any) -> tuple[float, float, float, str | None]:
    trip = env._next_freight_trip(
        env.now_min,
        station_id,
        parcel.weight_kg,
        min(parcel.deadline_min, env.horizon_min),
    )
    if trip is None:
        return 0.0, 0.0, 0.0, None
    trip_id, arrival = trip
    rows = env.trip_stop_times[trip_id]
    departure = float(rows[0]["departure_time"])
    remaining = float(env.config["bus"]["freight_capacity_kg"]) - env.bus_freight_kg[trip_id]
    return max(0.0, departure - env.now_min), max(0.0, arrival - departure), remaining, trip_id


def build_candidate_action_features(env: Any, parcel: Any, action_id: int, feasible: bool) -> dict[str, Any]:
    """Build the ordered, normalized objective context for one candidate action."""
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
    station = None
    is_td = action_id == 0
    is_tbd = 0 < action_id <= station_count
    is_tld = action_id > station_count
    station_index = None if is_td else (action_id - 1) % station_count

    if is_td:
        depot = env.truck_location_index["depot_01"]
        customer = env.truck_location_index[parcel.parcel_id]
        estimated_truck_distance = float(env.truck_distance_m[depot, customer]) / 1000.0
        estimated_truck_time = float(env.truck_time_min[depot, customer])
        start = max(env.now_min, min(env.truck_available_min, default=env.horizon_min)) + parcel.weight_kg * float(
            env.config["truck"]["loading_time_min_per_kg"]
        )
        delivery = start + estimated_truck_time + float(env.config["network"]["customer_service_time_min"])
    else:
        assert station_index is not None
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
        if is_tbd:
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
            start = max(env.now_min, min(env.truck_available_min, default=env.horizon_min)) + parcel.weight_kg * float(
                env.config["truck"]["loading_time_min_per_kg"]
            )
            delivery = start + estimated_truck_time + parcel.weight_kg * float(
                env.config["truck"]["unloading_time_min_per_kg"]
            ) + estimated_drone_time / 2.0
    if not feasible and not reasons:
        reasons.append("masked_by_environment")
    horizon = max(float(env.horizon_min), 1.0)
    max_truck_distance_km = max(float(env.truck_distance_m.max()) / 1000.0, 1.0)
    locker_capacity = max(float(station.locker_capacity_kg), 1.0) if station is not None else 1.0
    power_capacity = max(float(station.power_capacity_kw), 1.0) if station is not None else 1.0
    return {
        "action_id": action_id,
        "action_name": action_name(env, action_id),
        "action_type_TD": float(is_td),
        "action_type_TBD": float(is_tbd),
        "action_type_TLD": float(is_tld),
        "action_station_index_norm": (
            0.0 if station_index is None else float(station_index + 1) / max(station_count, 1)
        ),
        "feasible_flag": float(feasible),
        "estimated_delivery_time_norm": float(delivery) / horizon,
        "estimated_lateness_norm": max(0.0, float(delivery) - parcel.deadline_min) / horizon,
        "estimated_truck_distance_norm": float(estimated_truck_distance) / max_truck_distance_km,
        "estimated_truck_time_norm": float(estimated_truck_time) / horizon,
        "estimated_bus_wait_time_norm": float(estimated_bus_wait) / horizon,
        "estimated_bus_linehaul_time_norm": float(estimated_bus_linehaul) / horizon,
        "estimated_drone_time_norm": float(estimated_drone_time) / horizon,
        "estimated_locker_load_after_assignment_norm": float(estimated_locker_load) / locker_capacity,
        "estimated_station_power_margin_norm": float(estimated_power_margin) / power_capacity,
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
        "earliest_truck_available_time": float(min(env.truck_available_min, default=env.horizon_min)),
        "average_truck_capacity_remaining": (
            fmean(truck.remaining_capacity_kg for truck in env.trucks) if env.trucks else 0.0
        ),
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
    for station_id in sorted(env.station_ids):
        station = env.stations[station_id]
        distance = float(
            env.drone_distance_m[env.drone_row_index[station_id], env.drone_column_index[parcel.parcel_id]]
        ) / 1000.0
        wait, linehaul, remaining, trip_id = _next_bus_details(env, station_id, parcel)
        rows.append({
            "station_id": station_id,
            "distance_customer_to_station": distance,
            "drone_round_trip_time": _drone_time(env, station_id, parcel.parcel_id),
            "drone_feasible_from_station": env._station_can_serve_by_drone(parcel, station_id),
            "locker_remaining_capacity": station.locker_capacity_kg - station.locker_load_kg,
            "locker_occupancy_ratio": station.locker_load_kg / max(station.locker_capacity_kg, 1.0),
            "idle_drones": sum(value <= env.now_min for value in station.drone_available_min),
            "full_batteries": station.full_batteries,
            "station_power_margin": _station_power_margin(env, station),
            "next_feasible_bus_arrival_to_station": env.now_min + wait + linehaul if trip_id else 0.0,
            "bus_freight_remaining_capacity_to_station": remaining,
        })
    return rows
