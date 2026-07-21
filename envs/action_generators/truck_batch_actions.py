"""Side-effect-free bounded truck batch candidate generation."""
from __future__ import annotations
from dataclasses import dataclass
from math import isfinite
from typing import Any, Callable

from envs.route_planning import build_route, STOP_CUSTOMER, STOP_BUS_TERMINAL, STOP_INTEGRATED_STATION

@dataclass(frozen=True)
class TruckBatchCandidate:
    candidate_id: str; truck_id: str; parcel_ids: tuple[str, ...]; ordered_route_stops: tuple[Any, ...]
    total_weight_kg: float; total_volume_m3: float; weight_utilization: float; volume_utilization: float
    estimated_distance_km: float; estimated_travel_time_min: float; estimated_loading_time_min: float
    estimated_unloading_time_min: float; estimated_completion_time_min: float; estimated_lateness_min: float
    remaining_weight_capacity_kg: float; remaining_volume_capacity_m3: float
    number_of_direct_customers: int; number_of_terminal_deliveries: int; number_of_station_deliveries: int
    feasible: bool; infeasibility_reasons: tuple[str, ...]; heuristic_source: str; idle_flag: bool

    def feature_dict(self, env: Any) -> dict[str, float]:
        h=max(float(getattr(env,"horizon_min",1.0)),1.0); maxd=max(float(getattr(env,"truck_distance_m").max())/1000.0,1.0)
        return {"batch_size":len(self.parcel_ids)/max(int(env.config.get("truck",{}).get("max_batch_size",1)),1),"weight_utilization":self.weight_utilization,"volume_utilization":self.volume_utilization,"distance":self.estimated_distance_km/maxd,"travel_time":self.estimated_travel_time_min/h,"estimated_lateness":self.estimated_lateness_min/h,"direct_customer_count":self.number_of_direct_customers/max(len(self.parcel_ids),1),"terminal_delivery_count":self.number_of_terminal_deliveries/max(len(self.parcel_ids),1),"station_delivery_count":self.number_of_station_deliveries/max(len(self.parcel_ids),1),"route_stops":len(self.ordered_route_stops)/max(len(self.parcel_ids)+1,1),"idle_flag":float(self.idle_flag)}

def _truck_cfg(env: Any, key: str, default: float) -> float:
    cfg=env.config.get("truck",{})
    if key=="weight_capacity_kg": return float(cfg.get("weight_capacity_kg", cfg.get("capacity_kg", default)))
    if key=="volume_capacity_m3": return float(cfg.get("volume_capacity_m3", default))
    return float(cfg.get(key, default))

def eligible_parcels(env: Any) -> list[Any]:
    reserved=set().union(*(getattr(t,"onboard_parcels",[]) for t in getattr(env,"trucks",[]))) if getattr(env,"trucks",None) else set()
    out=[]
    for p in env.parcels.values():
        if getattr(p,"status",None)=="WAITING_TRUCK" and float(getattr(p,"release_time_min",0.0)) <= env.now_min + 1e-9 and getattr(p,"mode",None) in {"TD","TBD","TLD"} and p.parcel_id not in reserved:
            out.append(p)
    return sorted(out, key=lambda p:p.parcel_id)

def _make(env:Any, truck:Any, parcels:list[Any], source:str, cid:int)->TruckBatchCandidate:
    wc=_truck_cfg(env,"weight_capacity_kg",100.0); vc=_truck_cfg(env,"volume_capacity_m3",1.0); maxn=int(_truck_cfg(env,"max_batch_size",10))
    tw=sum(float(p.weight_kg) for p in parcels); tv=sum(float(getattr(p,"volume",0.0)) for p in parcels)
    reasons=[]
    if tw>wc+1e-9: reasons.append("weight_capacity_exceeded")
    if tv>vc+1e-9: reasons.append("volume_capacity_exceeded")
    if len(parcels)>maxn: reasons.append("batch_size_exceeded")
    route=build_route(env, truck.current_location_id, parcels, bool(env.config.get("truck",{}).get("return_to_depot",True)))
    loading=sum(float(p.weight_kg) for p in parcels)*_truck_cfg(env,"loading_time_min_per_kg", env.config.get("truck",{}).get("loading_time_min",0.0))
    unloading=sum(float(p.weight_kg) for p in parcels)*_truck_cfg(env,"unloading_time_min_per_kg", env.config.get("truck",{}).get("unloading_time_min",0.0))
    completion=env.now_min+loading+route.total_travel_time_min+unloading
    late=sum(max(0.0, completion-float(p.deadline_min))*float(getattr(p,"priority",1)) for p in parcels)
    stops=route.stops
    return TruckBatchCandidate(f"{truck.truck_id}:{source}:{cid}",truck.truck_id,tuple(p.parcel_id for p in sorted(parcels,key=lambda p:p.parcel_id)),stops,tw,tv,tw/wc if wc else 0,tv/vc if vc else 0,route.total_distance_km,route.total_travel_time_min,loading,unloading,completion,late,max(0,wc-tw),max(0,vc-tv),sum(s.stop_type==STOP_CUSTOMER for s in stops),sum(s.stop_type==STOP_BUS_TERMINAL for s in stops),sum(s.stop_type==STOP_INTEGRATED_STATION for s in stops),not reasons,tuple(reasons),source,False)

def _greedy(env, ordered):
    wc=_truck_cfg(env,"weight_capacity_kg",100); vc=_truck_cfg(env,"volume_capacity_m3",1); maxn=int(_truck_cfg(env,"max_batch_size",10)); res=[]; w=v=0.0
    for p in ordered:
        pw=float(p.weight_kg); pv=float(getattr(p,"volume",0.0))
        if len(res)<maxn and w+pw<=wc+1e-9 and v+pv<=vc+1e-9:
            res.append(p); w+=pw; v+=pv
    return res

def generate_truck_batch_candidates(env:Any, truck:Any)->list[TruckBatchCandidate]:
    cap=int(_truck_cfg(env,"max_batch_candidates",12)); elig=eligible_parcels(env); out=[]; seen=set()
    idle=TruckBatchCandidate(f"{truck.truck_id}:idle",truck.truck_id,(),(),0,0,0,0,0,0,0,0,env.now_min,0,_truck_cfg(env,"weight_capacity_kg",100),_truck_cfg(env,"volume_capacity_m3",1),0,0,0,True,(),"idle",True)
    heur=[("earliest_deadline",lambda p:(p.deadline_min,p.parcel_id)),("highest_priority",lambda p:(-p.priority,p.deadline_min,p.parcel_id)),("nearest_neighbor",lambda p:(build_route(env, truck.current_location_id,[p],False).total_distance_km,p.parcel_id)),("same_bus_terminal",lambda p:(p.mode!="TBD",p.station_id or "",p.parcel_id)),("same_station",lambda p:(p.mode!="TLD",p.station_id or "",p.parcel_id)),("max_weight_utilization",lambda p:(-p.weight_kg,p.parcel_id)),("max_volume_utilization",lambda p:(-getattr(p,"volume",0),p.parcel_id)),("minimum_lateness",lambda p:(max(0,env.now_min-p.deadline_min),p.parcel_id)),("mixed_destination",lambda p:({"TD":0,"TBD":1,"TLD":2}.get(p.mode,9),p.parcel_id))]
    for name,key in heur:
        batch=_greedy(env, sorted(elig,key=key))
        if not batch: continue
        cand=_make(env,truck,batch,name,len(out)); sig=(cand.parcel_ids, tuple((s.stop_id,s.stop_type,s.parcel_ids) for s in cand.ordered_route_stops))
        if sig not in seen: seen.add(sig); out.append(cand)
        if len(out)>=cap: break
    limited = out[: max(cap - 1, 0)]
    limited.append(idle)
    return limited
