"""Build physical bus circulation artifacts from a scheduled timetable."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from typing import Any, Sequence
from data_pipeline.common import write_csv, write_json
from envs.dynamics.bus_circulation import calculate_physical_fleet_size, build_trip_to_bus_mapping, sample_initial_energy

def build_bus_circulation(trips:list[dict[str,Any]], stop_times:list[dict[str,Any]], config:dict[str,Any], output_dir:Path)->dict[str,Any]:
    bus=config.setdefault("bus", {})
    schedule=config.get("bus_schedule", {})
    headway=float(schedule.get("planned_headway_min", bus.get("headway_min", 10.0)))
    if "non_service_relocation_time_min" in bus and "relocation_time_min" in bus and float(bus["non_service_relocation_time_min"]) != float(bus["relocation_time_min"]):
        raise ValueError("conflicting bus relocation aliases")
    if "minimum_layover_time_min" in bus and "minimum_layover_min" in bus and float(bus["minimum_layover_time_min"]) != float(bus["minimum_layover_min"]):
        raise ValueError("conflicting bus layover aliases")
    relocation=float(bus.get("non_service_relocation_time_min", config.get("bus_schedule", {}).get("relocation_time_min", bus.get("relocation_time_min", 5.0))))
    layover=float(bus.get("minimum_layover_time_min", config.get("bus_schedule", {}).get("minimum_layover_min", bus.get("minimum_layover_min", 2.0))))
    bus["non_service_relocation_time_min"] = relocation
    bus["minimum_layover_time_min"] = layover
    fleet=calculate_physical_fleet_size(trips, stop_times, headway, relocation, layover)
    mapping=build_trip_to_bus_mapping(trips, stop_times, fleet["physical_bus_count"], relocation, layover)
    ids=[f"bus_{i:03d}" for i in range(fleet["physical_bus_count"])]
    seed=int(config.get("seeds", {}).get("initial_bus_energy_seed", config.get("project", {}).get("seed", 0)))
    energies=sample_initial_energy(ids, seed, float(bus.get("bus_battery_kwh", bus.get("battery_capacity_kwh", 160.0))))
    first_origin={r["bus_id"]: next((st["stop_id"] for st in sorted(stop_times, key=lambda x:int(x.get("stop_sequence",0))) if st["trip_id"]==r["trip_id"]), "terminal") for r in mapping if int(r["sequence_index"])==0}
    physical=[{"bus_id":bid,"initial_location_id":first_origin.get(bid,"terminal"),"battery_capacity_kwh":float(bus.get("bus_battery_kwh",160.0)),"minimum_safe_energy_kwh":float(bus.get("bus_min_soc_kwh",40.0)),"initial_energy_seed_reference":"seeds.initial_bus_energy_seed","initial_soc_kwh":round(energies[bid],6)} for bid in ids]
    write_csv(output_dir/"physical_buses.csv", physical, list(physical[0].keys()))
    write_csv(output_dir/"trip_to_bus.csv", mapping, ["trip_id","bus_id","sequence_index","scheduled_start_min","scheduled_end_min","previous_trip_id","next_trip_id","relocation_time_min","minimum_layover_min"])
    doc={"schema_version":1,"physical_fleet_size":fleet["physical_bus_count"],"nominal_cycle_time_components":fleet,"mapping":mapping,"non_service_relocation_time_min": relocation,"minimum_layover_time_min": layover,"provenance":{"non_service_relocation_time_min":"project_extension","minimum_layover_time_min":"project_extension","initial_locations":"nominal_circulation_first_origin"}}
    write_json(output_dir/"bus_circulation.json", doc)
    return {"physical_buses":physical,"trip_to_bus":mapping,"bus_circulation":doc}

def _read(p):
    with open(p, newline='', encoding='utf-8') as f: return list(csv.DictReader(f))
def main(argv:Sequence[str]|None=None)->int:
    ap=argparse.ArgumentParser(); ap.add_argument("--input-dir",type=Path,required=True); ap.add_argument("--config",type=Path,required=True); ns=ap.parse_args(argv)
    from utils.config import load_config
    build_bus_circulation(_read(ns.input_dir/"bus_trips.csv"), _read(ns.input_dir/"bus_stop_times.csv"), load_config(ns.config), ns.input_dir); return 0
if __name__=="__main__": raise SystemExit(main())
