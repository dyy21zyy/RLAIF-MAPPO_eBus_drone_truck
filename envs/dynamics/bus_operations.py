"""Runtime helpers for physical-bus freight and passenger-aware charging."""
from __future__ import annotations
from envs.action_generators.bus_charging_actions import energy_added_kwh

def station_unloading_time_min(weight_kg:float)->float: return float(weight_kg)*6.0/60.0

def passenger_delay_cost(duration_min:float,onboard:int,waiting:int,weight:float=1.0)->float: return float(duration_min)*float(onboard+waiting)*float(weight)
