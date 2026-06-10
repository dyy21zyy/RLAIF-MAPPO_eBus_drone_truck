"""Run config-driven Stage 8 sensitivity variants; smoke mode selects a tiny subset."""
from __future__ import annotations
import argparse, copy, json, tempfile
from pathlib import Path
from data_pipeline.build_instance import build_instance
from evaluation.aggregator import aggregate_directory
from evaluation.runner import EvaluationRunner
from evaluation.reporting import write_records_csv
from utils.config import load_config

PATHS={"parcel_demand_level":("parcel","num_parcels"),"truck_fleet_size":("truck","num_trucks"),"integrated_station_count":("network","num_integrated_stations"),"locker_capacity":("station","locker_capacity_kg"),"initial_full_batteries":("station","initial_full_batteries"),"station_power_capacity":("station","power_capacity_kw"),"bus_headway":("bus","headway_min"),"drone_radius":("network","drone_radius_km"),"urgent_parcel_ratio":("parcel","tight_deadline_ratio")}

def run_sensitivity(config):
    exp=config["experiment"]; base=load_config(exp.get("base_config","configs/shanghai_small.yaml")); rows=[]
    dimensions=list(config.get("dimensions",{}).items()); dimensions=dimensions[:1] if exp.get("smoke") else dimensions
    with tempfile.TemporaryDirectory(prefix="stage8_sensitivity_") as temp:
        for dimension,values in dimensions:
            selected=list(values)[:1] if exp.get("smoke") else list(values)
            for value in selected:
                variant=copy.deepcopy(base); section,key=PATHS[dimension]; variant[section][key]=value
                config_path=Path(temp)/f"{dimension}_{value}.json"; config_path.write_text(json.dumps(variant),encoding="utf-8")
                built=build_instance(config_path,fallback=True,output_root=Path(temp)/"instances"/f"{dimension}_{value}")
                run_exp=dict(exp); run_exp["name"]=f"{exp['name']}__{dimension}_{value}"; run_exp["instance_name"]=f"{dimension}_{value}"
                for method in config.get("methods",[]):
                    rows.extend(EvaluationRunner(run_exp,Path(built["output_directory"])/"instance.json",method,int(exp.get("seeds",[0])[0])).run_many(exp.get("seeds",[0])))
    write_records_csv(Path(exp["output_dir"])/"episodes.csv",rows)
    aggregate_directory(Path(exp["output_dir"])/"raw",Path(exp["output_dir"])/"summary"); return rows

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--config",type=Path,required=True); args=parser.parse_args(argv); rows=run_sensitivity(load_config(args.config)); print(f"Processed {len(rows)} sensitivity episodes."); return 1 if any(r["status"]=="failed" for r in rows) else 0
if __name__=="__main__": raise SystemExit(main())
