"""Passenger-aware bus freight loading candidate generation."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

EPS=1e-9
BUS_FREIGHT_CAPACITY_KG=20.0
STATION_UNLOAD_LIMIT_KG=10.0

@dataclass(frozen=True)
class BusLoadingCandidate:
    candidate_id:str; physical_bus_id:str; trip_id:str; parcel_ids:tuple[str,...]; target_station_ids:tuple[str,...]
    total_weight_kg:float; freight_capacity_utilization:float; loading_time_min:float
    unload_weight_by_station:dict[str,float]; estimated_unloading_time_by_station:dict[str,float]
    maximum_single_station_unload_kg:float; estimated_lateness_min:float; estimated_passenger_impact_min:float
    feasible:bool; infeasibility_reasons:tuple[str,...]=(); heuristic_source:str="unknown"; idle_flag:bool=False
    def feature_dict(self)->dict[str,float]:
        return {"parcel_count":len(self.parcel_ids),"total_weight_kg":self.total_weight_kg,"freight_capacity_utilization":self.freight_capacity_utilization,"loading_time_min":self.loading_time_min,"maximum_single_station_unload_kg":self.maximum_single_station_unload_kg,"estimated_lateness_min":self.estimated_lateness_min,"estimated_passenger_impact_min":self.estimated_passenger_impact_min,"duration_sec":0.0,"energy_added_kwh":0.0,"projected_soc_kwh":0.0,"projected_station_load_kw":0.0,"projected_overload_kw":0.0,"idle_flag":float(self.idle_flag)}

def _cfg(env,key,default): return float(getattr(env,"config",{}).get("bus",{}).get(key,default))
def _trip_rows(env,trip_id): return list(getattr(env,"trip_stop_times",{}).get(trip_id,[]))
def downstream_station_ids(env, trip_id:str)->list[str]:
    return [env.stop_to_station[r["stop_id"]] for r in _trip_rows(env,trip_id) if r.get("stop_id") in getattr(env,"stop_to_station",{})]

def eligible_parcel_ids(env, trip_id:str, physical_bus_id:str|None=None, cutoff_min:float|None=None)->list[str]:
    cutoff = float(getattr(env,"now_min",0.0) if cutoff_min is None else cutoff_min)
    downstream=set(downstream_station_ids(env,trip_id)); onboard=set(getattr(getattr(env,"physical_buses",{}).get(physical_bus_id) if physical_bus_id else None,"onboard_parcel_ids",[]))
    reserved=set(onboard)
    for ids in getattr(env,"pending_bus_parcels",{}).values(): reserved.update(ids)
    ready=[]
    for pid,p in sorted(getattr(env,"parcels",{}).items()):
        if getattr(p,"status",None)!="AT_BUS_TERMINAL" or getattr(p,"mode",None)!="TBD": continue
        sid=getattr(p,"station_id",None) or getattr(p,"target_station_id",None) or getattr(p,"nearest_station_id",None)
        if sid not in downstream: continue
        if pid in reserved: continue
        if getattr(p,"truck_terminal_arrival_min", getattr(p,"terminal_arrival_min", cutoff)) > cutoff + EPS: continue
        ready.append(pid)
    return ready

def _make(env, trip_id, bus_id, pids, source, idle=False):
    cap=min(_cfg(env,"freight_capacity_kg",20.0), BUS_FREIGHT_CAPACITY_KG); load=float(getattr(env,"bus_freight_kg",{}).get(trip_id,0.0)); unload_sec=_cfg(env,"unloading_time_sec_per_kg",6.0); unload_min_per_kg=_cfg(env,"station_unloading_time_min_per_kg",unload_sec/60.0)
    loading_per=_cfg(env,"terminal_loading_time_min_per_kg",0.0); reasons=[]; by={}
    weight=sum(float(env.parcels[pid].weight_kg) for pid in pids)
    if load+weight > cap+EPS: reasons.append("bus_freight_capacity_exceeded")
    downstream=set(downstream_station_ids(env,trip_id))
    for pid in pids:
        p=env.parcels[pid]; sid=getattr(p,"station_id",None) or getattr(p,"target_station_id",None) or getattr(p,"nearest_station_id",None)
        if sid not in downstream: reasons.append("target_not_downstream")
        by[sid]=by.get(sid,0.0)+float(p.weight_kg)
    for sid,w in by.items():
        if w > STATION_UNLOAD_LIMIT_KG+EPS: reasons.append("station_unload_limit_exceeded")
    maxu=max(by.values(), default=0.0); unload_time={s:w*unload_min_per_kg for s,w in by.items()}
    lateness=sum(max(0.0, getattr(env,"now_min",0.0)+weight*loading_per-getattr(env.parcels[pid],"deadline_min",1e9)) for pid in pids)
    impact=(weight*loading_per + sum(unload_time.values()))*getattr(getattr(getattr(env,"physical_buses",{}).get(bus_id),"passenger_manifest",None),"total_onboard_passengers",0)
    return BusLoadingCandidate(("idle" if idle else source)+":"+("-".join(pids) if pids else "none"), bus_id, trip_id, tuple(pids), tuple(sorted(by)), weight, (load+weight)/max(cap,1.0), weight*loading_per, by, unload_time, maxu, lateness, impact, not reasons, tuple(sorted(set(reasons))), source, idle)

def generate_bus_loading_candidates(env, trip_id:str)->list[BusLoadingCandidate]:
    bus_id=getattr(env,"trip_to_bus",{}).get(trip_id, trip_id); elig=eligible_parcel_ids(env,trip_id,bus_id)
    parcels=[env.parcels[pid] for pid in elig]
    strategies=[]
    strategies.append(("idle", []))
    strategies.append(("earliest-deadline-first", [p.parcel_id for p in sorted(parcels,key=lambda p:(p.deadline_min,p.parcel_id))]))
    strategies.append(("highest-priority-first", [p.parcel_id for p in sorted(parcels,key=lambda p:(-p.priority,p.deadline_min,p.parcel_id))]))
    strategies.append(("maximum-weight-utilization", [p.parcel_id for p in sorted(parcels,key=lambda p:(-p.weight_kg,p.parcel_id))]))
    for sid in sorted(set(getattr(p,"station_id",None) for p in parcels)): strategies.append(("same-target-station consolidation", [p.parcel_id for p in parcels if getattr(p,"station_id",None)==sid]))
    strategies.append(("station-balanced batch", [p.parcel_id for p in sorted(parcels,key=lambda p:(getattr(p,"station_id",''),p.deadline_min,p.parcel_id))]))
    strategies.append(("minimum-unloading-delay batch", [p.parcel_id for p in sorted(parcels,key=lambda p:(p.weight_kg,p.deadline_min,p.parcel_id))]))
    strategies.append(("minimum-estimated-lateness batch", [p.parcel_id for p in sorted(parcels,key=lambda p:(max(0,getattr(env,'now_min',0)-p.deadline_min),p.deadline_min,p.parcel_id))]))
    out=[]; seen=set(); cap=min(_cfg(env,"freight_capacity_kg",20.0),20.0)
    for source, order in strategies:
        batch=[]; by={}; total=0.0
        for pid in order:
            p=env.parcels[pid]; sid=getattr(p,"station_id",None); w=float(p.weight_kg)
            if total+w<=cap+EPS and by.get(sid,0)+w<=STATION_UNLOAD_LIMIT_KG+EPS:
                batch.append(pid); total+=w; by[sid]=by.get(sid,0)+w
        cand=_make(env,trip_id,bus_id,batch,source,source=="idle")
        key=cand.parcel_ids
        if key not in seen: seen.add(key); out.append(cand)
    return sorted(out,key=lambda c:(not c.idle_flag, c.candidate_id))
