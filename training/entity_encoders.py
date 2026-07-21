"""Fixed-dimensional entity-set encoders for the centralized MAPPO critic."""
from __future__ import annotations
from statistics import fmean
from typing import Any

ENTITY_ENCODER_SCHEMA_VERSION = 1

def _stats(values):
    vals = [float(v) for v in values]
    if not vals:
        return [0.0, 0.0, 0.0]
    return [fmean(vals), max(vals), float(len(vals))]

def encode_entity_critic_state(env: Any) -> list[float]:
    """Pool variable entity sets into a fixed-length critic vector.

    Uses mean/max/count summaries, so the output dimension is invariant to the
    number of parcels, trucks, physical buses, and stations.
    """
    horizon = max(float(getattr(env, "horizon_min", 1.0)), 1.0)
    parcels = list(getattr(env, "parcels", {}).values())
    statuses = ["UNRELEASED","PENDING_ASSIGNMENT","WAITING_TRUCK","ONBOARD_TRUCK","AT_BUS_TERMINAL","ONBOARD_BUS","AT_STATION","ONBOARD_DRONE","DELIVERED","FAILED"]
    parcel_dist = [sum(getattr(p,"status","") == s for p in parcels) / max(len(parcels), 1) for s in statuses]
    deadlines = _stats([(getattr(p,"deadline_min",0.0)-getattr(env,"now_min",0.0))/horizon for p in parcels])
    lateness = _stats([max(0.0, (getattr(p,"delivered_time_min", None) or getattr(env,"now_min",0.0))-getattr(p,"deadline_min",0.0))/horizon for p in parcels])
    trucks = list(getattr(env, "trucks", []))
    truck_stats = _stats([getattr(t,"available_time",0.0)/horizon for t in trucks]) + _stats([len(getattr(t,"onboard_parcels",[])) for t in trucks]) + _stats([getattr(t,"total_travel_time",0.0)/horizon for t in trucks])
    buses = list(getattr(env, "physical_buses", {}).values())
    cap = max(float(getattr(env, "config", {}).get("bus", {}).get("bus_battery_kwh", 160.0)), 1.0)
    bus_stats = _stats([getattr(b,"soc_kwh",0.0)/cap for b in buses]) + _stats([getattr(b,"schedule_delay_min",0.0)/horizon for b in buses]) + _stats([getattr(getattr(b,"passenger_manifest",None),"total_onboard_passengers",0.0) for b in buses])
    queues = list(getattr(env, "passenger_stops", {}).values())
    passenger_stats = _stats([getattr(q,"total_waiting",0.0) for q in queues])
    stations = list(getattr(env, "stations", {}).values())
    station_stats = []
    station_stats += _stats([s.locker_load_kg / max(s.locker_capacity_kg,1.0) for s in stations])
    station_stats += _stats([sum(d.status == "AVAILABLE" for d in getattr(s,"drone_states",[])) for s in stations])
    station_stats += _stats([sum(b.status == "FULL" for b in getattr(s,"battery_states",[])) for s in stations])
    station_stats += _stats([sum(b.status == "CHARGING" for b in getattr(s,"battery_states",[]))/max(getattr(s,"charging_slots",1),1) for s in stations])
    future = list(getattr(env, "events", []))
    future_stats = _stats([(e.time_min-getattr(env,"now_min",0.0))/horizon for e in future]) + [sum(getattr(e,"kind","")==k for e in future)/max(len(future),1) for k in ("parcel_release","truck_available","bus_departure","bus_arrival","station_operation")]
    return [float(getattr(env,"now_min",0.0))/horizon, *parcel_dist, *deadlines, *lateness, *truck_stats, *bus_stats, *passenger_stats, *station_stats, *future_stats]
