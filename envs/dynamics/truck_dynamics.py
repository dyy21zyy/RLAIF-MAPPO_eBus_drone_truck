"""Batched truck route event application."""
from __future__ import annotations
from typing import Any


def apply_truck_batch(env: Any, truck: Any, candidate: Any) -> float:
    if candidate.idle_flag:
        return 0.0
    parcels=[env.parcels[pid] for pid in candidate.parcel_ids]
    for p in parcels:
        p.status="ONBOARD_TRUCK"; p.truck_id=truck.truck_id
    truck.status="traveling"; truck.onboard_parcels=list(candidate.parcel_ids); truck.remaining_capacity_kg=candidate.remaining_weight_capacity_kg
    truck.route_history.append([truck.current_location_id]+[s.stop_id for s in candidate.ordered_route_stops])
    t=env.now_min + candidate.estimated_loading_time_min
    env._push(env.now_min, "truck_departure", {"truck_id":truck.truck_id,"candidate_id":candidate.candidate_id})
    current=truck.current_location_id
    # derive per-leg times from matrix
    for stop in candidate.ordered_route_stops:
        if current!=stop.stop_id:
            t += float(env.truck_time_min[env.truck_location_index[current], env.truck_location_index[stop.stop_id]])
        env._push(t, "truck_arrive_stop", {"truck_id":truck.truck_id,"stop_id":stop.stop_id,"stop_type":stop.stop_type,"parcel_ids":list(stop.parcel_ids)})
        unload=sum(env.parcels[pid].weight_kg for pid in stop.parcel_ids)*float(env.config.get("truck",{}).get("unloading_time_min_per_kg", env.config.get("truck",{}).get("unloading_time_min",0.0)))
        t += unload
        env._push(t, "truck_unload", {"truck_id":truck.truck_id,"stop_id":stop.stop_id,"stop_type":stop.stop_type,"parcel_ids":list(stop.parcel_ids)})
        current=stop.stop_id
    truck.available_time=t; truck.total_distance += candidate.estimated_distance_km; truck.total_travel_time += candidate.estimated_travel_time_min
    env._push(t, "truck_route_complete", {"truck_id":truck.truck_id,"final_location_id":current,"candidate_id":candidate.candidate_id})
    idx=env.trucks.index(truck)
    if hasattr(env, "pending_truck_decision_min"):
        env.pending_truck_decision_min[idx]=t
    env._push(t,"truck_available",{"truck_index":idx,"truck_id":truck.truck_id})
    env.truck_dispatch_count=getattr(env,"truck_dispatch_count",0)+1
    env.truck_parcels_routed=getattr(env,"truck_parcels_routed",0)+len(candidate.parcel_ids)
    env.truck_weight_utilization_sum=getattr(env,"truck_weight_utilization_sum",0.0)+candidate.weight_utilization
    env.truck_volume_utilization_sum=getattr(env,"truck_volume_utilization_sum",0.0)+candidate.volume_utilization
    cfg=env.config.get("truck",{})
    cost=float(cfg.get("fixed_dispatch_cost",0.0))+candidate.estimated_distance_km*float(cfg.get("cost_per_km",0.0))+(candidate.estimated_travel_time_min+candidate.estimated_loading_time_min+candidate.estimated_unloading_time_min)*float(cfg.get("cost_per_min",0.0))
    return env._charge_cost("truck_cost", cost)
