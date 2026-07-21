"""Bounded joint station dispatch and battery-charging candidates."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

EPS = 1e-9
MAX_CHARGING_SLOTS = 6
DEFAULT_MAX_CANDIDATES = 16
DRONE_PAYLOAD_KG = 5.0
DRONE_RADIUS_KM = 8.0
MAX_ROUND_TRIP_MIN = 120.0
CUSTOMER_SERVICE_MIN = 1.0
PREP_MIN = 0.0
POST_TURNAROUND_MIN = 0.0
CHARGING_DURATION_MIN = 45.0
CHARGING_POWER_KW = 2.0
STATION_POWER_CAPACITY_KW = 1100.0

@dataclass(frozen=True)
class StationOperationCandidate:
    candidate_id: str
    dispatches: tuple[tuple[str, str, str], ...]
    battery_ids_to_start_charging: tuple[str, ...]
    estimated_delivery_times: tuple[float, ...]
    estimated_return_times: tuple[float, ...]
    estimated_parcel_lateness: tuple[float, ...]
    full_batteries_remaining: int
    depleted_batteries_remaining: int
    available_drones_remaining: int
    charging_slots_used_after_action: int
    projected_station_load: float
    projected_overload: float
    power_margin: float
    expected_overload_duration: float
    feasible: bool
    infeasibility_reasons: tuple[str, ...] = ()
    heuristic_source: str = "unknown"
    idle_flag: bool = False


def _cfg(env: Any, section: str, key: str, default: float) -> float:
    try:
        return float(env.config.get(section, {}).get(key, default))
    except Exception:
        return default


def _station_base_load(env: Any, station: Any) -> float:
    return float(getattr(station, "base_load_kw", _cfg(env, "station", "base_load_kw", 0.0)))


def drone_mission_times(env: Any, station_id: str, parcel_id: str, start: float) -> tuple[float, float, float, float]:
    distance_km = float(env.drone_distance_m[env.drone_row_index[station_id], env.drone_column_index[parcel_id]]) / 1000.0
    speed = max(_cfg(env, "network", "drone_speed_kmph", 40.0), EPS)
    prep = _cfg(env, "network", "drone_pre_departure_time_min", PREP_MIN)
    service = _cfg(env, "network", "customer_service_time_min", CUSTOMER_SERVICE_MIN)
    turnaround = _cfg(env, "network", "drone_turnaround_time_min", POST_TURNAROUND_MIN)
    one_way = distance_km / speed * 60.0
    delivery = start + prep + one_way + service
    ret = delivery + one_way
    resource = ret + turnaround
    return delivery, ret, resource, distance_km


def available_drones(station: Any, now: float) -> list[Any]:
    if hasattr(station, "drone_states"):
        return [d for d in station.drone_states if d.status == "AVAILABLE" and d.available_time_min <= now + EPS]
    return [type("DroneRef", (), {"drone_id": f"drone_{i:03d}", "available_time_min": t, "status": "AVAILABLE"}) for i, t in enumerate(getattr(station, "drone_available_min", [])) if t <= now + EPS]


def batteries_by_status(station: Any, status: str) -> list[Any]:
    if hasattr(station, "battery_states"):
        return [b for b in station.battery_states if b.status == status]
    count = int(getattr(station, "full_batteries" if status == "FULL" else "depleted_batteries", 0))
    return [type("BatteryRef", (), {"battery_id": f"{status.lower()}_{i:03d}", "status": status}) for i in range(count)]


def active_charging_count(station: Any, now: float) -> int:
    if hasattr(station, "battery_states"):
        return sum(b.status == "CHARGING" for b in station.battery_states)
    return sum(start <= now + EPS and end > now + EPS for start, end in getattr(station, "active_battery_charges", []))


def projected_load(env: Any, station: Any, starts: int = 0) -> float:
    now = float(getattr(env, "now_min", 0.0))
    bus_kw = _cfg(env, "bus", "charging_power_kw", 300.0)
    bus = sum(end > now + EPS for end in getattr(station, "active_bus_charges", [])) * bus_kw
    battery = (active_charging_count(station, now) + starts) * float(getattr(station, "battery_power_kw", _cfg(env, "station", "battery_charging_power_kw", CHARGING_POWER_KW)))
    return _station_base_load(env, station) + bus + battery


def _dispatch_pattern(env: Any, station_id: str, drones: list[Any], full: list[Any], waiting: list[str], heuristic: str) -> list[tuple[str,str,str]]:
    station = env.stations[station_id]
    feasible = []
    for pid in waiting:
        p = env.parcels[pid]
        delivery, ret, resource, dist = drone_mission_times(env, station_id, pid, env.now_min)
        rt = resource - env.now_min
        if p.status == "WAITING_DRONE" and p.station_id == station_id and p.weight_kg <= DRONE_PAYLOAD_KG + EPS and dist <= DRONE_RADIUS_KM + EPS and rt <= MAX_ROUND_TRIP_MIN + EPS:
            feasible.append((pid, p, delivery, resource, dist, max(0.0, delivery - p.deadline_min)))
    if heuristic == "earliest_deadline": feasible.sort(key=lambda x: (x[1].deadline_min, x[0]))
    elif heuristic == "highest_priority": feasible.sort(key=lambda x: (-x[1].priority, x[1].deadline_min, x[0]))
    elif heuristic == "shortest_mission": feasible.sort(key=lambda x: (x[4], x[0]))
    elif heuristic == "minimum_estimated_lateness": feasible.sort(key=lambda x: (x[5], x[1].deadline_min, x[0]))
    elif heuristic == "battery_conservative": feasible = feasible[:max(0, len(full)-1)]
    elif heuristic == "future_capacity_preserving": feasible = feasible[:max(1, min(len(drones), len(full))//2)]
    matches=[]
    for d,b,item in zip(drones, full, feasible):
        matches.append((d.drone_id, item[0], b.battery_id))
    return matches


def generate_station_operation_candidates(env: Any, station_id: str, max_candidates: int | None = None) -> list[StationOperationCandidate]:
    station = env.stations[station_id]; now=float(env.now_min)
    max_candidates = int(max_candidates or env.config.get("station", {}).get("max_operation_candidates", DEFAULT_MAX_CANDIDATES))
    waiting = list(getattr(env, "waiting_station_parcels", {}).get(station_id, []))
    drones = sorted(available_drones(station, now), key=lambda d: d.drone_id)
    full = sorted(batteries_by_status(station, "FULL"), key=lambda b: b.battery_id)
    depleted = sorted(batteries_by_status(station, "DEPLETED"), key=lambda b: b.battery_id)
    slots = min(MAX_CHARGING_SLOTS, int(getattr(station, "charging_slots", MAX_CHARGING_SLOTS)))
    free_slots = max(0, slots - active_charging_count(station, now))
    patterns=[("idle", [])]
    for h in ["earliest_deadline","highest_priority","shortest_mission","minimum_estimated_lateness","maximum_cardinality","battery_conservative","future_capacity_preserving"]:
        patterns.append((h, _dispatch_pattern(env, station_id, drones, full, waiting, h)))
    seen=set(); out=[]
    capacity=float(getattr(station,"power_capacity_kw",STATION_POWER_CAPACITY_KW))
    for h, dispatches in patterns:
        charge_options=[(), tuple(b.battery_id for b in depleted[:1]), tuple(b.battery_id for b in depleted[:free_slots])]
        reserve=max(0, int(env.config.get("station",{}).get("full_battery_reserve_target", 0))- (len(full)-len(dispatches)))
        charge_options.append(tuple(b.battery_id for b in depleted[:min(free_slots, reserve)]))
        for charges in charge_options:
            key=(tuple(dispatches), tuple(charges))
            if key in seen: continue
            seen.add(key)
            reasons=[]
            if len(charges) > free_slots: reasons.append("charging_slots_unavailable")
            delivery=[]; returns=[]; late=[]
            for _,pid,_ in dispatches:
                de,_,re,_=drone_mission_times(env, station_id, pid, now); delivery.append(de); returns.append(re); late.append(max(0.0, de-env.parcels[pid].deadline_min))
            load=projected_load(env, station, len(charges)); overload=max(0.0, load-capacity)
            out.append(StationOperationCandidate(f"station_{len(out)}", tuple(dispatches), tuple(charges), tuple(delivery), tuple(returns), tuple(late), len(full)-len(dispatches), len(depleted)-len(charges), len(drones)-len(dispatches), active_charging_count(station, now)+len(charges), load, overload, capacity-load, CHARGING_DURATION_MIN if overload else 0.0, not reasons, tuple(reasons), h, not dispatches and not charges))
            if len(out)>=max_candidates: return out
    return out
