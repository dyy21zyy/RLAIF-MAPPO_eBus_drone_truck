"""Flash-charging action candidates for physical buses."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
ACTIONS_SEC=(0,15,30,45,60,75,90,105,120)
@dataclass(frozen=True)
class BusChargingCandidate:
    candidate_id:str; duration_sec:int; physical_bus_id:str; station_id:str; energy_added_kwh:float; projected_soc_kwh:float; projected_station_load_kw:float; projected_overload_kw:float; feasible:bool; infeasibility_reasons:tuple[str,...]; idle_flag:bool=False
    def feature_dict(self): return {"parcel_count":0.0,"total_weight_kg":0.0,"freight_capacity_utilization":0.0,"loading_time_min":0.0,"maximum_single_station_unload_kg":0.0,"estimated_lateness_min":0.0,"estimated_passenger_impact_min":0.0,"duration_sec":self.duration_sec,"energy_added_kwh":self.energy_added_kwh,"projected_soc_kwh":self.projected_soc_kwh,"projected_station_load_kw":self.projected_station_load_kw,"projected_overload_kw":self.projected_overload_kw,"idle_flag":float(self.idle_flag)}
def energy_added_kwh(duration_sec:int,power_kw:float=500.0,efficiency:float=0.95)->float: return float(power_kw)*(float(duration_sec)/3600.0)*float(efficiency)
def generate_bus_charging_candidates(env,event)->list[BusChargingCandidate]:
    trip_id=event.payload["trip_id"]; station_id=event.payload["station_id"]; bus_id=env.trip_to_bus[trip_id]; bus=env.physical_buses[bus_id]; station=env.stations[station_id]
    active=[e for e in station.active_bus_charges if e>env.now_min+1e-9]; charger_available=len(active)<2
    current_load=env._station_load_kw(station,env.now_min) if hasattr(env,"_station_load_kw") else 0.0
    out=[]
    for u in getattr(env,"config",{}).get("bus",{}).get("charging_actions_sec", ACTIONS_SEC):
        u=int(u); added=energy_added_kwh(u, float(env.config["bus"].get("charging_power_kw",500.0)), float(env.config["bus"].get("charging_efficiency",0.95))); proj=bus.soc_kwh+added; reasons=[]
        if u>0 and not charger_available: reasons.append("charger_unavailable")
        if proj>float(env.config["bus"].get("bus_battery_kwh",160.0))+1e-9: reasons.append("overcharge")
        pload=current_load+(float(env.config["bus"].get("charging_power_kw",500.0)) if u>0 else 0.0); overload=max(0.0,pload-float(getattr(station,"power_capacity_kw",1100.0)))
        out.append(BusChargingCandidate(f"charge_{u}s",u,bus_id,station_id,added,proj,pload,overload,not reasons,tuple(reasons),u==0))
    return out
