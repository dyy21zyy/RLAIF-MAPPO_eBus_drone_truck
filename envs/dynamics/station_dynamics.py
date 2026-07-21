"""Station operation dynamics for explicit drones and batteries."""
from __future__ import annotations

from envs.action_generators.station_actions import CHARGING_DURATION_MIN, CHARGING_POWER_KW, MAX_CHARGING_SLOTS, drone_mission_times

def start_battery_charging(env, station, battery, now):
    if battery.status != "DEPLETED":
        raise ValueError("battery_not_depleted")
    active = sum(b.status == "CHARGING" for b in station.battery_states)
    if active >= min(MAX_CHARGING_SLOTS, station.charging_slots):
        raise ValueError("charging_slots_unavailable")
    battery.status = "CHARGING"
    battery.charge_start_time_min = now
    battery.charge_completion_time_min = now + getattr(station, "battery_charge_duration_min", CHARGING_DURATION_MIN)
    station.active_battery_charges.append((now, battery.charge_completion_time_min))
    env._push(battery.charge_completion_time_min, "battery_ready", {"station_id": station.station_id, "battery_id": battery.battery_id})

def complete_battery_charging(station, battery):
    battery.status = "FULL"
    battery.charge_start_time_min = None
    battery.charge_completion_time_min = None

def dispatch_drone(env, station, drone, parcel, battery, now):
    if drone.status != "AVAILABLE" or drone.available_time_min > now + 1e-9: raise ValueError("drone_unavailable")
    if parcel.status != "WAITING_DRONE" or parcel.station_id != station.station_id: raise ValueError("parcel_unavailable")
    if battery.status != "FULL": raise ValueError("battery_not_full")
    delivery, ret, resource, _ = drone_mission_times(env, station.station_id, parcel.parcel_id, now)
    drone.status="IN_MISSION"; drone.active_parcel_id=parcel.parcel_id; drone.active_battery_id=battery.battery_id; drone.available_time_min=resource
    battery.status="IN_USE"; battery.assigned_drone_id=drone.drone_id
    parcel.status="ONBOARD_DRONE"; station.locker_load_kg=max(0.0, station.locker_load_kg-parcel.weight_kg)
    env._push(delivery, "parcel_delivery", {"parcel_id": parcel.parcel_id})
    env._push(ret, "drone_return", {"station_id": station.station_id, "drone_id": drone.drone_id, "battery_id": battery.battery_id})
