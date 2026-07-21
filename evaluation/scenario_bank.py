"""Frozen scenario-bank utilities for paper experiments."""
from __future__ import annotations
import hashlib, json, shutil
from pathlib import Path
from typing import Any
SCHEMA_VERSION=1
BANKS=("train","validation","test")

def sha256_file(path: Path)->str:
    h=hashlib.sha256();
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def sha256_json(data: Any)->str:
    return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(",",":"),default=str).encode()).hexdigest()

def scenario_id(bank:str,index:int,seeds:dict[str,int])->str:
    return f"{bank}_{index:04d}_{sha256_json(seeds)[:10]}"

def write_bank_manifest(bank_dir: Path, bank:str, scenarios:list[dict[str,Any]], config:dict[str,Any]|None=None)->dict[str,Any]:
    ids=[s['scenario_id'] for s in scenarios]
    if len(ids)!=len(set(ids)): raise ValueError(f"duplicate scenario_id in {bank}")
    manifest={"schema_version":SCHEMA_VERSION,"bank":bank,"size":len(scenarios),"configuration_hash":sha256_json(config or {}),"scenarios":scenarios}
    bank_dir.mkdir(parents=True,exist_ok=True); (bank_dir/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    return manifest

def load_bank_manifest(path: str|Path)->dict[str,Any]:
    p=Path(path); p=p/'manifest.json' if p.is_dir() else p
    return json.loads(p.read_text(encoding='utf-8'))

def validate_disjoint_banks(manifests:list[dict[str,Any]])->None:
    seen_ids={}; seen_seeds={}
    for m in manifests:
        bank=m.get('bank','unknown')
        for s in m.get('scenarios',[]):
            sid=s['scenario_id']; seed_tuple=tuple(sorted((s.get('seed_tuple') or {}).items()))
            if sid in seen_ids: raise ValueError(f"scenario_id {sid} appears in both {seen_ids[sid]} and {bank}")
            if seed_tuple and seed_tuple in seen_seeds: raise ValueError(f"seed_tuple for {sid} duplicates {seen_seeds[seed_tuple]}")
            seen_ids[sid]=bank; seen_seeds[seed_tuple]=sid

def freeze_scenario(source_instance: Path, dest_dir: Path, scenario_id_value:str, seed_tuple:dict[str,int], config:dict[str,Any])->dict[str,Any]:
    dest_dir.mkdir(parents=True,exist_ok=True); target=dest_dir/'instance.json'; shutil.copyfile(source_instance,target)
    data_hash=sha256_file(target)
    return {"scenario_id":scenario_id_value,"configuration_hash":sha256_json(config),"seed_tuple":dict(seed_tuple),"artifacts":{"instance.json":data_hash},"artifact_hashes":{"instance.json":data_hash},"size":target.stat().st_size,"schema_version":SCHEMA_VERSION,"path":str(target)}
