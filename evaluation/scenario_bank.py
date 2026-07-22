"""Frozen scenario-bank utilities for formal paired evaluation."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib, json, shutil, subprocess
from pathlib import Path
from typing import Any
SCHEMA_VERSION=2
BANKS=('train','validation','test')

@dataclass(frozen=True)
class FrozenScenario:
    scenario_id: str; split: str; instance_path: str; scenario_manifest_path: str; artifact_hashes: dict[str,str]; seed_tuple: dict[str,int]; config_hash: str
    @property
    def instance_hash(self): return self.artifact_hashes.get('instance.json','')
    @property
    def scenario_manifest_hash(self): return self.artifact_hashes.get('scenario_manifest.json','')

@dataclass(frozen=True)
class ScenarioBank:
    bank_id: str; split: str; scenarios: tuple[FrozenScenario,...]; bank_hash: str

def sha256_file(path: str|Path)->str:
    h=hashlib.sha256();
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

def sha256_json(data: Any)->str:
    return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':'),default=str).encode()).hexdigest()

def git_commit()->str:
    try: return subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    except Exception: return 'unknown'

def scenario_id(split:str,index:int,seeds:dict[str,int])->str:
    return f'{split}_{index:04d}_{sha256_json(seeds)[:10]}'

def _manifest_path(p: str|Path)->Path:
    p=Path(p); return p/'scenario_bank_manifest.json' if p.is_dir() else p

def write_bank_manifest(bank_dir: Path, bank:str, scenarios:list[dict[str,Any]], config:dict[str,Any]|None=None)->dict[str,Any]:
    ids=[s['scenario_id'] for s in scenarios]
    if len(ids)!=len(set(ids)): raise ValueError(f'duplicate scenario_id in {bank}')
    instance_hashes={s['scenario_id']:s.get('artifact_hashes',{}).get('instance.json') for s in scenarios}
    scenario_manifest_hashes={s['scenario_id']:s.get('artifact_hashes',{}).get('scenario_manifest.json') for s in scenarios}
    base_hash=sha256_json(config or {})
    body={'schema_version':SCHEMA_VERSION,'bank_id':f'{bank}-{base_hash[:12]}','bank':bank,'split':bank,'scenario_count':len(scenarios),'size':len(scenarios),'scenario_ids':ids,'scenario_manifest_hashes':scenario_manifest_hashes,'instance_hashes':instance_hashes,'base_config_hash':base_hash,'generation_commit':git_commit(),'scenarios':scenarios}
    body['bank_hash']=sha256_json({k:v for k,v in body.items() if k!='bank_hash'})
    bank_dir.mkdir(parents=True,exist_ok=True)
    (bank_dir/'scenario_bank_manifest.json').write_text(json.dumps(body,indent=2,sort_keys=True)+'\n')
    (bank_dir/'manifest.json').write_text(json.dumps(body,indent=2,sort_keys=True)+'\n')
    return body

def load_bank_manifest(path: str|Path)->dict[str,Any]:
    return json.loads(_manifest_path(path).read_text())

def load_scenario_bank(path: str|Path)->ScenarioBank:
    m=load_bank_manifest(path); root=_manifest_path(path).parent; scenarios=[]
    for s in m.get('scenarios',[]):
        inst=Path(s.get('instance_path') or s.get('path') or root/s['scenario_id']/'instance.json')
        sm=Path(s.get('scenario_manifest_path') or root/s['scenario_id']/'scenario_manifest.json')
        scenarios.append(FrozenScenario(s['scenario_id'],s.get('split',m.get('split',m.get('bank'))),str(inst),str(sm),dict(s.get('artifact_hashes',{})),dict(s.get('seed_tuple',{})),s.get('config_hash',s.get('configuration_hash',m.get('base_config_hash','')))))
    return ScenarioBank(m.get('bank_id',m.get('bank','bank')),m.get('split',m.get('bank','')),tuple(scenarios),m.get('bank_hash',''))

def verify_scenario_hashes(s: FrozenScenario) -> None:
    for name,expected in s.artifact_hashes.items():
        p=Path(s.instance_path).parent/name
        if not p.is_file(): raise FileNotFoundError(f'missing scenario artifact: {p}')
        actual=sha256_file(p)
        if actual != expected: raise ValueError(f'artifact hash mismatch for {s.scenario_id}/{name}: {actual} != {expected}')

def load_frozen_instance(s: FrozenScenario)->dict[str,Any]:
    verify_scenario_hashes(s)
    return json.loads(Path(s.instance_path).read_text())

def freeze_scenario(source_instance: Path, dest_dir: Path, scenario_id_value:str, seed_tuple:dict[str,int], config:dict[str,Any])->dict[str,Any]:
    dest_dir.mkdir(parents=True,exist_ok=True)
    target=dest_dir/'instance.json'; shutil.copyfile(source_instance,target)
    manifest={'scenario_id':scenario_id_value,'split':dest_dir.parent.name,'seed_tuple':dict(seed_tuple),'base_configuration_hash':sha256_json(config),'generation_code_commit':git_commit(),'creation_timestamp':'deterministic-smoke'}
    (dest_dir/'scenario_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    artifacts={'instance.json':sha256_file(target),'scenario_manifest.json':sha256_file(dest_dir/'scenario_manifest.json')}
    (dest_dir/'artifact_hashes.json').write_text(json.dumps(artifacts,indent=2,sort_keys=True)+'\n')
    artifacts['artifact_hashes.json']=sha256_file(dest_dir/'artifact_hashes.json')
    return {'scenario_id':scenario_id_value,'split':manifest['split'],'config_hash':sha256_json(config),'configuration_hash':sha256_json(config),'seed_tuple':dict(seed_tuple),'artifact_hashes':artifacts,'instance_path':str(target),'scenario_manifest_path':str(dest_dir/'scenario_manifest.json'),'path':str(target),'schema_version':SCHEMA_VERSION}

def validate_disjoint_banks(manifests:list[dict[str,Any]], *, allow_duplicate_artifacts:bool=False)->None:
    seen_ids={}; seen_artifacts={}
    for m in manifests:
        split=m.get('split',m.get('bank','unknown'))
        for s in m.get('scenarios',[]):
            sid=s['scenario_id']
            if sid in seen_ids: raise ValueError(f'scenario_id {sid} appears in both {seen_ids[sid]} and {split}')
            seen_ids[sid]=split
            ih=s.get('artifact_hashes',{}).get('instance.json')
            if ih and not allow_duplicate_artifacts:
                if ih in seen_artifacts: raise ValueError(f'duplicate scenario artifact hash across splits: {sid} and {seen_artifacts[ih]}')
                seen_artifacts[ih]=sid
