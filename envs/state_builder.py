"""Shared objective feature construction for assignment observations and RLAIF data."""

from __future__ import annotations

from statistics import fmean
from typing import Any

from envs.decision_schema import ActionCandidate, DecisionSurface
from envs.action_generators.bus_loading_actions import generate_bus_loading_candidates
from envs.action_generators.bus_charging_actions import generate_bus_charging_candidates
from envs.action_generators.station_actions import generate_station_operation_candidates, projected_load


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

TRUCK_FEATURE_NAMES = (
    "time_norm",
    "available_now",
    "pending_task_count_norm",
    "capacity_norm",
)

BUS_LOADING_FEATURE_NAMES = (
    "time_norm",
    "ready_parcel_count_norm",
    "freight_load_norm",
    "capacity_remaining_norm",
)

BUS_CHARGING_FEATURE_NAMES = (
    "time_norm",
    "soc_norm",
    "delay_norm",
    "locker_load_norm",
    "full_batteries_norm",
    "freight_load_norm",
    "total_onboard_passengers_norm",
    "remaining_passenger_capacity_norm",
    "current_stop_waiting_passengers_norm",
    "downstream_waiting_summary_norm",
    "boarding_count_norm",
    "alighting_count_norm",
    "current_waiting_passenger_minutes_norm",
    "current_onboard_additional_delay_passenger_minutes_norm",
    "expected_boarding_time_norm",
    "expected_alighting_time_norm",
)

STATION_OPERATION_FEATURE_NAMES = (
    "time_norm", "waiting_parcels", "earliest_deadline_slack",
    "available_drones", "busy_drones", "full_batteries", "depleted_batteries",
    "charging_batteries", "charging_slots_available", "locker_load",
    "locker_capacity", "active_bus_chargers", "station_base_load", "power_margin",
    "projected_future_drone_returns", "projected_future_charge_completions",
)

COMMON_CANDIDATE_FEATURE_NAMES = (
    "action_type_id",
    "estimated_time_norm",
    "estimated_lateness_norm",
    "capacity_after_norm",
    "resource_margin_norm",
    "idle_flag",
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


def build_assignment_decision_surface(env: Any, parcel: Any, mask: list[bool]) -> DecisionSurface:
    """Return the four-agent candidate surface for parcel assignment."""

    candidates = []
    for action_id, feasible in enumerate(mask):
        features = build_candidate_action_features(env, parcel, action_id, bool(feasible))
        candidates.append(
            ActionCandidate(
                action_id=action_id,
                action_type=str(features["action_name"]).split("_", 1)[0],
                entity_id=str(features["action_name"]),
                description=str(features["action_name"]),
                features={key: float(features[key]) for key in CANDIDATE_ACTION_FEATURE_NAMES},
                feasible=bool(feasible),
                reasons=tuple(str(reason) for reason in features.get("infeasibility_reasons", ())),
            )
        )
    return DecisionSurface(
        agent_id="assignment",
        event_type="PARCEL_RELEASE",
        entity_id=parcel.parcel_id,
        features=build_assignment_features(env, parcel),
        feature_names=assignment_feature_names(env.station_ids),
        candidates=candidates,
    )


def _common_candidate(
    action_id: int,
    action_type: str,
    entity_id: str,
    description: str,
    *,
    action_type_id: float,
    estimated_time_norm: float = 0.0,
    estimated_lateness_norm: float = 0.0,
    capacity_after_norm: float = 0.0,
    resource_margin_norm: float = 0.0,
    idle_flag: float = 0.0,
    feasible: bool = True,
    reasons: tuple[str, ...] = (),
) -> ActionCandidate:
    return ActionCandidate(
        action_id=action_id,
        action_type=action_type,
        entity_id=entity_id,
        description=description,
        features={
            "action_type_id": action_type_id,
            "estimated_time_norm": estimated_time_norm,
            "estimated_lateness_norm": estimated_lateness_norm,
            "capacity_after_norm": capacity_after_norm,
            "resource_margin_norm": resource_margin_norm,
            "idle_flag": idle_flag,
        },
        feasible=feasible,
        reasons=reasons,
    )


def build_truck_decision_surface(env: Any, truck: Any) -> DecisionSurface:
    """Build batched parcel-pool candidates for a truck-availability decision."""
    from envs.action_generators.truck_batch_actions import generate_truck_batch_candidates

    horizon = max(float(env.horizon_min), 1.0)
    cfg = env.config.get("truck", {})
    weight_capacity = max(float(cfg.get("weight_capacity_kg", cfg.get("capacity_kg", 100.0))), 1.0)
    volume_capacity = max(float(cfg.get("volume_capacity_m3", 1.0)), 1.0)
    waiting = [p for p in env.parcels.values() if p.status == "WAITING_TRUCK"]
    urgent = [p for p in waiting if p.deadline_min - env.now_min <= 60.0]
    terminal = [p for p in waiting if p.mode == "TBD"]
    station = [p for p in waiting if p.mode == "TLD"]
    direct = [p for p in waiting if p.mode == "TD"]
    earliest_slack = min((p.deadline_min - env.now_min for p in waiting), default=horizon)
    available = truck.available_time <= env.now_min + 1e-9
    features = [
        env.now_min / horizon,
        float(available),
        len(waiting) / max(len(env.parcels), 1),
        len(urgent) / max(len(env.parcels), 1),
        earliest_slack / horizon,
        len(terminal) / max(len(env.parcels), 1),
        len(station) / max(len(env.parcels), 1),
        len(direct) / max(len(env.parcels), 1),
        weight_capacity / weight_capacity,
        volume_capacity / volume_capacity,
        len([t for t in env.trucks if t.status != "idle"]) / max(len(env.trucks), 1),
    ]
    candidates=[]
    for i, cand in enumerate(generate_truck_batch_candidates(env, truck)):
        candidates.append(ActionCandidate(
            action_id=i, action_type="idle" if cand.idle_flag else "truck_batch",
            entity_id=cand.candidate_id, description=f"{cand.heuristic_source}: {','.join(cand.parcel_ids) or 'idle'}",
            features=cand.feature_dict(env), feasible=cand.feasible and (available or cand.idle_flag),
            reasons=cand.infeasibility_reasons if cand.feasible else cand.infeasibility_reasons,
        ))
    return DecisionSurface(
        agent_id="truck", event_type="TRUCK_AVAILABLE", entity_id=truck.truck_id,
        features=features,
        feature_names=("time_norm","available_now","waiting_parcel_count","urgent_waiting_parcel_count","earliest_deadline_slack","terminal_feeder_count","station_feeder_count","direct_delivery_count","truck_weight_capacity","truck_volume_capacity","active_truck_route_summaries"),
        candidates=candidates,
    )

def build_bus_loading_decision_surface(env: Any, trip_id: str) -> DecisionSurface:
    """Build bounded physical-bus loading-batch candidates."""
    horizon = max(float(env.horizon_min), 1.0)
    capacity = max(float(env.config["bus"].get("freight_capacity_kg", 20.0)), 1.0)
    bus_id = env.trip_to_bus.get(trip_id, trip_id)
    bus = env.physical_buses.get(bus_id)
    candidates = []
    generated = generate_bus_loading_candidates(env, trip_id)
    for i, cand in enumerate(generated):
        candidates.append(ActionCandidate(
            action_id=i, action_type="idle" if cand.idle_flag else "loading_batch",
            entity_id=cand.candidate_id, description=f"{cand.heuristic_source}: {','.join(cand.parcel_ids) or 'idle'}",
            features=cand.feature_dict(), feasible=cand.feasible, reasons=cand.infeasibility_reasons,
        ))
    terminal_ready = sum(1 for p in env.parcels.values() if getattr(p, "status", None) == "AT_BUS_TERMINAL" and getattr(p, "mode", None) == "TBD")
    current_load = float(env.bus_freight_kg.get(trip_id, 0.0))
    target_profile = len({sid for cand in generated for sid in cand.target_station_ids})
    return DecisionSurface(
        agent_id="bus", event_type="BUS_TERMINAL_DEPARTURE", entity_id=trip_id,
        features=[env.now_min / horizon, current_load / capacity, max(0.0, capacity-current_load)/capacity, terminal_ready / max(len(env.parcels), 1), float(getattr(getattr(bus, "passenger_manifest", None), "total_onboard_passengers", 0)) / max(float(env.config["bus"].get("bus_capacity_passenger",80)),1.0), float(getattr(bus, "soc_kwh", 0.0)) / max(float(env.config["bus"].get("bus_battery_kwh",160.0)),1.0), target_profile / max(len(env.station_ids),1)],
        feature_names=("time_norm","freight_load_norm","capacity_remaining_norm","terminal_ready_parcel_count_norm","onboard_passengers_norm","soc_norm","target_station_profile_norm"),
        candidates=candidates,
    )

def build_bus_charging_decision_surface(env: Any, event: Any) -> DecisionSurface:
    """Build masked physical-bus flash-charging candidates."""
    trip_id = event.payload["trip_id"]; station_id = event.payload["station_id"]
    station = env.stations[station_id]; bus = env.physical_buses[env.trip_to_bus[trip_id]]
    horizon = max(float(env.horizon_min), 1.0); battery = max(float(env.config["bus"].get("bus_battery_kwh",160.0)), 1.0)
    generated = generate_bus_charging_candidates(env, event)
    candidates=[ActionCandidate(i, "charge" if c.duration_sec else "no_charge", c.candidate_id, f"charge {c.duration_sec} seconds", c.feature_dict(), c.feasible, c.infeasibility_reasons) for i,c in enumerate(generated)]
    waiting = getattr(env.passenger_stops.get(env.trip_stop_times[trip_id][event.payload["stop_index"]]["stop_id"]), "total_waiting", 0)
    return DecisionSurface(
        agent_id="bus", event_type="BUS_STATION_ARRIVAL", entity_id=f"{trip_id}:{station_id}",
        features=[env.now_min/horizon, bus.soc_kwh/battery, max(0.0,bus.soc_kwh-bus.minimum_safe_energy_kwh)/battery, bus.schedule_delay_min/horizon, float(event.payload.get("unloading_delay_min",0.0))/horizon, float(bus.passenger_manifest.total_onboard_passengers)/max(float(env.config["bus"].get("bus_capacity_passenger",80)),1.0), float(waiting)/max(float(env.config["bus"].get("bus_capacity_passenger",80)),1.0), (station.power_capacity_kw-env._station_load_kw(station,env.now_min))/max(station.power_capacity_kw,1.0)],
        feature_names=("time_norm","soc_norm","safety_energy_margin_norm","current_delay_norm","unloading_time_norm","passenger_load_norm","stop_queue_norm","station_power_margin_norm"),
        candidates=candidates,
    )


def build_station_decision_surface(env: Any, station_id: str) -> DecisionSurface:
    """Build bounded joint station dispatch/charging candidates."""
    station = env.stations[station_id]
    waiting = list(getattr(env, "waiting_station_parcels", {}).get(station_id, []))
    horizon = max(float(env.horizon_min), 1.0)
    candidates: list[ActionCandidate] = []
    generated = generate_station_operation_candidates(env, station_id)
    for idx, cand in enumerate(generated):
        candidates.append(ActionCandidate(
            idx,
            "idle" if cand.idle_flag else "dispatch_drone",
            station_id,
            f"{cand.heuristic_source}: {len(cand.dispatches)} dispatch(es), {len(cand.battery_ids_to_start_charging)} charge start(s)",
            {
                "action_type_id": 0.0 if cand.idle_flag else 1.0,
                "dispatch_count": float(len(cand.dispatches)),
                "charge_start_count": float(len(cand.battery_ids_to_start_charging)),
                "estimated_time_norm": (max(cand.estimated_return_times) if cand.estimated_return_times else env.now_min) / horizon,
                "estimated_lateness_norm": (sum(cand.estimated_parcel_lateness) / max(len(cand.estimated_parcel_lateness), 1)) / horizon,
                "capacity_after_norm": max(0.0, station.locker_capacity_kg - station.locker_load_kg) / max(station.locker_capacity_kg, 1.0),
                "resource_margin_norm": cand.power_margin / max(station.power_capacity_kw, 1.0),
                "full_batteries_remaining": float(cand.full_batteries_remaining),
                "depleted_batteries_remaining": float(cand.depleted_batteries_remaining),
                "available_drones_remaining": float(cand.available_drones_remaining),
                "charging_slots_used_after_action": float(cand.charging_slots_used_after_action),
                "projected_station_load": cand.projected_station_load,
                "projected_overload": cand.projected_overload,
                "power_margin": cand.power_margin,
                "expected_overload_duration": cand.expected_overload_duration,
                "idle_flag": 1.0 if cand.idle_flag else 0.0,
                "dispatch_payload": list(cand.dispatches),
                "charge_payload": list(cand.battery_ids_to_start_charging),
            },
            cand.feasible,
            cand.infeasibility_reasons,
        ))
    drone_states = getattr(station, "drone_states", [])
    battery_states = getattr(station, "battery_states", [])
    available_drones = sum(d.status == "AVAILABLE" and d.available_time_min <= env.now_min + 1e-9 for d in drone_states)
    busy_drones = len(drone_states) - available_drones
    full = sum(b.status == "FULL" for b in battery_states)
    depleted = sum(b.status == "DEPLETED" for b in battery_states)
    charging = sum(b.status == "CHARGING" for b in battery_states)
    slots = min(6, int(getattr(station, "charging_slots", 6)))
    active_bus = sum(end > env.now_min + 1e-9 for end in getattr(station, "active_bus_charges", []))
    load = projected_load(env, station, 0)
    deadlines = [env.parcels[pid].deadline_min - env.now_min for pid in waiting]
    future_returns = sum(d.available_time_min > env.now_min + 1e-9 for d in drone_states)
    future_completions = sum((b.charge_completion_time_min or 0.0) > env.now_min + 1e-9 for b in battery_states)
    return DecisionSurface(
        agent_id="station",
        event_type="STATION_OPERATION",
        entity_id=station_id,
        features=[
            env.now_min / horizon, len(waiting), min(deadlines) if deadlines else 0.0,
            available_drones, busy_drones, full, depleted, charging, max(0, slots - charging),
            station.locker_load_kg, station.locker_capacity_kg, active_bus,
            float(getattr(station, "base_load_kw", env.config["station"].get("base_load_kw", 0.0))),
            station.power_capacity_kw - load, future_returns, future_completions,
        ],
        feature_names=STATION_OPERATION_FEATURE_NAMES,
        candidates=candidates,
    )

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
