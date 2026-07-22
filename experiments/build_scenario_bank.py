from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
from data_pipeline.build_instance import build_instance
from data_pipeline.scenario_seeds import derive_scenario_seed_tuple
from evaluation.scenario_bank import freeze_scenario, scenario_id, write_bank_manifest, sha256_json, load_bank_manifest
from utils.config import load_config

def _parse_seeds(text: str|None, count:int, start:int):
    if text:
        vals=[int(x.strip()) for x in text.split(',') if x.strip()]
        if len(vals)!=count: raise ValueError('--explicit-seeds length must equal --count')
    else:
        vals=list(range(start,start+count))
    return [derive_scenario_seed_tuple(v) for v in vals]

def build_bank(config_path: str|Path, split:str, count:int, seed_start:int, output: str|Path, *, explicit_seeds:str|None=None, fallback:bool=False, run_classification:str='formal', force:bool=False):
    if run_classification=='formal' and fallback: raise ValueError('formal scenario-bank generation cannot use fallback')
    out=Path(output); manifest=out/'scenario_bank_manifest.json'
    if manifest.exists() and not force: raise FileExistsError(f'{manifest} exists; pass --force to overwrite')
    if force and out.exists(): shutil.rmtree(out)
    out.mkdir(parents=True,exist_ok=True)
    cfg=load_config(config_path)
    generation_config_path = cfg.get("env", {}).get("config_path", config_path) if isinstance(cfg, dict) else config_path
    scenarios=[]
    for i,t in enumerate(_parse_seeds(explicit_seeds,count,seed_start)):
        seeds=t.to_dict(); sid=scenario_id(split,i,seeds)
        built=build_instance(generation_config_path, fallback=fallback, output_root=out/'_generated'/sid, seed_overrides=t)
        scenarios.append(freeze_scenario(Path(built['output_directory'])/'instance.json', out/sid, sid, seeds, cfg, split=split, run_classification=run_classification, fallback=fallback))
    return write_bank_manifest(out,split,scenarios,{'config_path':str(config_path),'config_hash':sha256_json(cfg),'split':split,'count':count,'seed_start':seed_start}, run_classification=run_classification)

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--split',required=True,choices=['train','validation','test']); p.add_argument('--count',type=int,required=True); p.add_argument('--seed-start',type=int,default=0); p.add_argument('--explicit-seeds'); p.add_argument('--output',required=True); p.add_argument('--fallback',action='store_true'); p.add_argument('--run-classification',default='formal',choices=['formal','diagnostic','smoke']); p.add_argument('--validate-only',action='store_true'); p.add_argument('--force',action='store_true')
    a=p.parse_args(argv)
    if a.validate_only:
        m=load_bank_manifest(a.output); print(json.dumps({'manifest':str(Path(a.output)/'scenario_bank_manifest.json'),'scenario_count':m['scenario_count'],'bank_hash':m['bank_hash']},indent=2)); return 0
    m=build_bank(a.config,a.split,a.count,a.seed_start,a.output,explicit_seeds=a.explicit_seeds,fallback=a.fallback,run_classification=a.run_classification,force=a.force)
    print(json.dumps({'manifest':str(Path(a.output)/'scenario_bank_manifest.json'),'scenario_count':m['scenario_count'],'bank_hash':m['bank_hash']},indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
