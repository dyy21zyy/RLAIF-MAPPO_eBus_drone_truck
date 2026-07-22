from __future__ import annotations
import argparse, json
from pathlib import Path
from data_pipeline.build_instance import build_instance
from evaluation.scenario_bank import freeze_scenario, scenario_id, write_bank_manifest, sha256_json
from utils.config import load_config

def build_bank(config_path: str|Path, split:str, count:int, seed_start:int, output: str|Path, *, force:bool=False):
    out=Path(output)
    manifest=out/'scenario_bank_manifest.json'
    if manifest.exists() and not force: raise FileExistsError(f'{manifest} exists; pass --force to overwrite')
    out.mkdir(parents=True,exist_ok=True)
    cfg=load_config(config_path)
    scenarios=[]
    for i in range(int(count)):
        seed=seed_start+i; seed_tuple={'base':seed,'parcel':seed*10+1,'passenger':seed*10+2,'timetable':seed*10+3,'physical_bus':seed*10+4,'station_load':seed*10+5}
        sid=scenario_id(split,i,seed_tuple)
        # deterministic smoke generation; formal users invoke this offline before evaluation.
        built=build_instance(config_path,fallback=True,output_root=out/'_generated'/sid)
        scenarios.append(freeze_scenario(Path(built['output_directory'])/'instance.json', out/sid, sid, seed_tuple, cfg))
    ids=[s['scenario_id'] for s in scenarios]
    if len(ids)!=len(set(ids)): raise ValueError('duplicate scenario IDs generated')
    return write_bank_manifest(out,split,scenarios,{'config_path':str(config_path),'config_hash':sha256_json(cfg),'split':split,'count':count,'seed_start':seed_start})

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--split',required=True,choices=['train','validation','test']); p.add_argument('--count',type=int,required=True); p.add_argument('--seed-start',type=int,required=True); p.add_argument('--output',required=True); p.add_argument('--force',action='store_true')
    a=p.parse_args(argv); m=build_bank(a.config,a.split,a.count,a.seed_start,a.output,force=a.force); print(json.dumps({'manifest':str(Path(a.output)/'scenario_bank_manifest.json'),'scenario_count':m['scenario_count'],'bank_hash':m['bank_hash']},indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
