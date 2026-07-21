from __future__ import annotations
import argparse, copy, json, tempfile
from pathlib import Path
from data_pipeline.build_instance import build_instance
from evaluation.scenario_bank import scenario_id, freeze_scenario, write_bank_manifest, validate_disjoint_banks
from utils.config import load_config
SIZES={"small":{"parcel":{"num_parcels":30},"network":{"num_integrated_stations":6},"bus":{"headway_min":15,"scheduled_trips":24,"freight_trips":8}},"medium":{"parcel":{"num_parcels":60},"network":{"num_integrated_stations":8},"bus":{"headway_min":10,"scheduled_trips":36,"freight_trips":12}},"large":{"parcel":{"num_parcels":90},"network":{"num_integrated_stations":8},"bus":{"headway_min":8,"scheduled_trips":45,"freight_trips":16}}}
def deep_update(d,u):
    for k,v in u.items(): d.setdefault(k,{}).update(v) if isinstance(v,dict) else d.__setitem__(k,v)
    return d
def generate(config):
    root=Path(config.get('output_root','data/scenarios')); base=load_config(config.get('base_config','configs/shanghai_small.yaml')); manifests=[]
    with tempfile.TemporaryDirectory(prefix='scenario_bank_') as tmp:
        for bank, spec in config.get('banks',{}).items():
            scenarios=[]; count=int(spec.get('count',1)); offset=int(spec.get('seed_offset',0)); size=spec.get('size','small')
            for i in range(count):
                cfg=copy.deepcopy(base); deep_update(cfg,SIZES[size]); seed_tuple={"network_seed":offset+i*101,"parcel_seed":offset+i*101+1,"passenger_seed":offset+i*101+2,"travel_time_seed":offset+i*101+3,"initial_bus_energy_seed":offset+i*101+4,"station_base_load_seed":offset+i*101+5}; cfg['seeds']=seed_tuple; cfg.setdefault('project',{})['seed']=offset+i*101
                cp=Path(tmp)/f'{bank}_{i}.json'; cp.write_text(json.dumps(cfg),encoding='utf-8')
                built=build_instance(cp,fallback=True,output_root=Path(tmp)/'built'/bank/str(i)); sid=scenario_id(bank,i,seed_tuple)
                scenarios.append(freeze_scenario(Path(built['output_directory'])/'instance.json', root/bank/sid, sid, seed_tuple, cfg))
            manifests.append(write_bank_manifest(root/bank,bank,scenarios,config))
    validate_disjoint_banks(manifests); return manifests
def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument('--config',type=Path,default=Path('configs/paper/scenario_banks.yaml')); args=ap.parse_args(argv); ms=generate(load_config(args.config)); print(json.dumps({m['bank']:m['size'] for m in ms},indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
