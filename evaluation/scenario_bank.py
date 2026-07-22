"""Frozen self-contained scenario-bank utilities."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib, json, shutil, subprocess
from pathlib import Path
from typing import Any
import yaml
SCHEMA_VERSION=3
BANKS=('train','validation','test')
DYNAMIC_KEYS={'parcels','passenger_arrivals','passenger_stop_rates','passenger_temporal_profile','station_base_load','physical_buses'}

def sha256_file(path: str|Path)->str:
    h=hashlib.sha256()
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

def _canonical_instance(instance: dict[str,Any]) -> dict[str,Any]:
    x=json.loads(json.dumps(instance, sort_keys=True, default=str))
    x.pop('output_directory', None)
    if isinstance(x.get('config_snapshot'), dict):
        x['config_snapshot'].pop('output_directory', None)
    return x

def _artifact_hashes(root: Path, artifacts: dict[str,str]) -> dict[str,str]:
    out={}
    for _k,name in sorted(artifacts.items()):
        p=root/name
        if not p.is_file(): raise FileNotFoundError(f'missing referenced artifact {_k}: {p}')
        out[name]=sha256_file(p)
    for extra in ('instance.yaml','resolved_scenario_config.yaml'):
        if (root/extra).is_file(): out[extra]=sha256_file(root/extra)
    return out

def _dynamic_static_hashes(artifacts: dict[str,str], hashes: dict[str,str]) -> tuple[dict[str,str],dict[str,str]]:
    dyn={}; stat={}
    for key,name in artifacts.items():
        if name not in hashes: continue
        (dyn if key in DYNAMIC_KEYS else stat)[key]=hashes[name]
    return dyn, stat

@dataclass(frozen=True)
class FrozenScenario:
    scenario_id: str; split: str; instance_path: str; scenario_manifest_path: str; artifact_hashes: dict[str,str]; seed_tuple: dict[str,int]; config_hash: str; scenario_content_hash: str=''
    @property
    def instance_hash(self): return self.artifact_hashes.get('instance.json','')
    @property
    def scenario_manifest_hash(self): return self.artifact_hashes.get('scenario_manifest.json','')
@dataclass(frozen=True)
class ScenarioBank:
    bank_id: str; split: str; scenarios: tuple[FrozenScenario,...]; bank_hash: str

def freeze_scenario(source_instance: Path, dest_dir: Path, scenario_id_value:str, seed_tuple:dict[str,int], config:dict[str,Any], *, split: str|None=None, run_classification: str='diagnostic', fallback: bool|None=None)->dict[str,Any]:
    source_instance=Path(source_instance); src=source_instance.parent; dest_dir=Path(dest_dir)
    if dest_dir.exists(): shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True,exist_ok=True)
    inst=json.loads(source_instance.read_text())
    artifacts=dict(inst.get('artifacts',{}))
    for key,name in artifacts.items():
        sp=src/name
        if not sp.is_file(): raise FileNotFoundError(f'missing referenced artifact {key}: {sp}')
        shutil.copy2(sp, dest_dir/name)
    if (src/'resolved_scenario_config.yaml').is_file(): shutil.copy2(src/'resolved_scenario_config.yaml', dest_dir/'resolved_scenario_config.yaml')
    inst['output_directory']='.'
    inst['scenario_id']=scenario_id_value
    inst['scenario_split']=split or dest_dir.parent.name
    inst['artifacts']={k:Path(v).name for k,v in artifacts.items()}
    inst.setdefault('scenario_seed_tuple', dict(seed_tuple))
    inst['seeds_used']=dict(seed_tuple)
    (dest_dir/'instance.json').write_text(json.dumps(inst,indent=2,sort_keys=True)+'\n')
    (dest_dir/'instance.yaml').write_text(json.dumps(inst,indent=2,sort_keys=True)+'\n')
    art_hashes=_artifact_hashes(dest_dir, inst['artifacts'])
    art_hashes['instance.json']=sha256_file(dest_dir/'instance.json'); art_hashes['instance.yaml']=sha256_file(dest_dir/'instance.yaml')
    dyn,stat=_dynamic_static_hashes(inst['artifacts'], art_hashes)
    resolved_hash=art_hashes.get('resolved_scenario_config.yaml', sha256_json(config))
    instance_content_hash=sha256_json(_canonical_instance(inst))
    content_hash=sha256_json({'resolved_config_hash':resolved_hash,'seed_tuple':dict(seed_tuple),'artifact_hashes':art_hashes,'instance':_canonical_instance(inst)})
    manifest={'scenario_schema_version':SCHEMA_VERSION,'scenario_id':scenario_id_value,'split':inst['scenario_split'],'scenario_content_hash':content_hash,'resolved_config_hash':resolved_hash,'seed_tuple':dict(seed_tuple),'instance_hash':art_hashes['instance.json'],'instance_content_hash':instance_content_hash,'artifact_hashes':art_hashes,'dynamic_artifact_hashes':dyn,'static_artifact_hashes':stat,'generation_code_commit':git_commit(),'source_base_config_hash':sha256_json(config),'run_classification':run_classification,'fallback': bool(fallback) if fallback is not None else inst.get('mode')=='fallback','provenance':{'source_instance_hash':sha256_file(source_instance),'source_generation_directory':str(src)}}
    (dest_dir/'scenario_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    art_hashes['scenario_manifest.json']=sha256_file(dest_dir/'scenario_manifest.json')
    (dest_dir/'artifact_hashes.json').write_text(json.dumps(art_hashes,indent=2,sort_keys=True)+'\n')
    art_hashes['artifact_hashes.json']=sha256_file(dest_dir/'artifact_hashes.json')
    return {'schema_version':SCHEMA_VERSION,'scenario_id':scenario_id_value,'split':inst['scenario_split'],'config_hash':sha256_json(config),'configuration_hash':sha256_json(config),'seed_tuple':dict(seed_tuple),'artifact_hashes':art_hashes,'dynamic_artifact_hashes':dyn,'static_artifact_hashes':stat,'scenario_content_hash':content_hash,'resolved_config_hash':resolved_hash,'instance_hash':art_hashes['instance.json'],'instance_path':str(dest_dir/'instance.json'),'scenario_manifest_path':str(dest_dir/'scenario_manifest.json'),'path':str(dest_dir/'instance.json')}

def _uniqueness(scenarios:list[dict[str,Any]], key:str)->dict[str,int]:
    names=set().union(*(s.get(key,{}) for s in scenarios)) if scenarios else set()
    return {n:len({s.get(key,{}).get(n) for s in scenarios if s.get(key,{}).get(n)}) for n in sorted(names)}

def write_bank_manifest(bank_dir: Path, bank:str, scenarios:list[dict[str,Any]], config:dict[str,Any]|None=None, *, run_classification:str='diagnostic')->dict[str,Any]:
    ids=[s['scenario_id'] for s in scenarios]
    if len(ids)!=len(set(ids)): raise ValueError(f'duplicate scenario_id in {bank}')
    seeds=[json.dumps(s.get('seed_tuple',{}),sort_keys=True) for s in scenarios]
    if len(seeds)!=len(set(seeds)): raise ValueError('duplicate seed tuple in bank')
    contents=[s.get('scenario_content_hash') for s in scenarios]
    if any(contents) and len(contents)!=len(set(contents)): raise ValueError('duplicate scenario content hash in bank')
    inst=[s.get('instance_hash') or s.get('artifact_hashes',{}).get('instance.json') for s in scenarios]
    if any(inst) and len(inst)!=len(set(inst)): raise ValueError('duplicate instance hash in bank')
    dyn_summary=_uniqueness(scenarios,'dynamic_artifact_hashes')
    if len(scenarios)>1 and dyn_summary and max(dyn_summary.values()) <= 1: raise ValueError('all dynamic artifact hashes are identical')
    base_hash=sha256_json(config or {})
    body={'schema_version':SCHEMA_VERSION,'bank_id':f'{bank}-{base_hash[:12]}','bank':bank,'split':bank,'run_classification':run_classification,'scenario_count':len(scenarios),'size':len(scenarios),'scenario_ids':ids,'scenario_content_hashes':{s['scenario_id']:s.get('scenario_content_hash') for s in scenarios},'instance_hashes':{s['scenario_id']:s.get('instance_hash') or s.get('artifact_hashes',{}).get('instance.json') for s in scenarios},'resolved_config_hashes':{s['scenario_id']:s.get('resolved_config_hash') for s in scenarios},'seed_tuples':{s['scenario_id']:s.get('seed_tuple',{}) for s in scenarios},'dynamic_artifact_uniqueness_summary':dyn_summary,'static_artifact_reuse_summary':_uniqueness(scenarios,'static_artifact_hashes'),'base_config_hash':base_hash,'base_config_path':(config or {}).get('config_path'),'generation_commit':git_commit(),'scenario_generation_method':'seeded_build_instance_freeze','scenarios':scenarios}
    body['bank_hash']=sha256_json({k:v for k,v in body.items() if k not in {'bank_hash','generation_commit'}})
    bank_dir.mkdir(parents=True,exist_ok=True)
    (bank_dir/'scenario_bank_manifest.json').write_text(json.dumps(body,indent=2,sort_keys=True)+'\n')
    (bank_dir/'manifest.json').write_text(json.dumps(body,indent=2,sort_keys=True)+'\n')
    return body

def load_bank_manifest(path: str|Path)->dict[str,Any]: return json.loads(_manifest_path(path).read_text())
def load_scenario_bank(path: str|Path)->ScenarioBank:
    m=load_bank_manifest(path); root=_manifest_path(path).parent; scenarios=[]
    for s in m.get('scenarios',[]):
        inst=Path(s.get('instance_path') or s.get('path') or root/s['scenario_id']/'instance.json')
        if not inst.is_absolute() and not inst.exists(): inst=root/inst
        sm=Path(s.get('scenario_manifest_path') or root/s['scenario_id']/'scenario_manifest.json')
        if not sm.is_absolute() and not sm.exists(): sm=root/sm
        scenarios.append(FrozenScenario(s['scenario_id'],s.get('split',m.get('split',m.get('bank'))),str(inst),str(sm),dict(s.get('artifact_hashes',{})),dict(s.get('seed_tuple',{})),s.get('config_hash',m.get('base_config_hash','')),s.get('scenario_content_hash','')))
    return ScenarioBank(m.get('bank_id',m.get('bank','bank')),m.get('split',m.get('bank','')),tuple(scenarios),m.get('bank_hash',''))
def verify_scenario_hashes(s: FrozenScenario) -> None:
    root=Path(s.instance_path).parent
    for name,expected in s.artifact_hashes.items():
        p=root/name
        if not p.is_file(): raise FileNotFoundError(f'missing scenario artifact: {p}')
        actual=sha256_file(p)
        if actual != expected: raise ValueError(f'artifact hash mismatch for {s.scenario_id}/{name}: {actual} != {expected}')
def load_frozen_instance(s: FrozenScenario)->dict[str,Any]: verify_scenario_hashes(s); return json.loads(Path(s.instance_path).read_text())
def validate_disjoint_banks(manifests:list[dict[str,Any]], *, allow_duplicate_artifacts:bool=False)->None:
    seen={'scenario_id':{},'seed':{},'content':{},'dynamic':{},'artifact':{}}
    for m in manifests:
        split=m.get('split',m.get('bank','unknown'))
        for s in m.get('scenarios',[]):
            vals={'scenario_id':s['scenario_id'],'seed':(json.dumps(s.get('seed_tuple',{}),sort_keys=True) if s.get('seed_tuple') else None),'content':s.get('scenario_content_hash'), 'artifact': (s.get('artifact_hashes',{}) or {}).get('instance.json') if not allow_duplicate_artifacts else None}
            dyn=json.dumps(s.get('dynamic_artifact_hashes',{}),sort_keys=True); vals['dynamic']=dyn if dyn!='{}' else None
            for kind,val in vals.items():
                if not val: continue
                if val in seen[kind]: raise ValueError(f'duplicate {kind} across splits: {s["scenario_id"]} and {seen[kind][val]}')
                seen[kind][val]=s['scenario_id']
