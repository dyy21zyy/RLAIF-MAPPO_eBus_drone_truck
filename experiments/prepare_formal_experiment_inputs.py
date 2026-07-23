"""Prepare ignored final formal scenario banks, reward scales, and runtime configs."""
from __future__ import annotations
import argparse,json,shutil,hashlib
from pathlib import Path
from typing import Any
import yaml
from experiments.build_scenario_bank import build_bank
from experiments.estimate_reward_reference_scales import run_estimation
from evaluation.scenario_bank import load_bank_manifest, validate_disjoint_banks, verify_scenario_hashes, load_scenario_bank, sha256_file
from training.config_resolver import resolve_mappo_training_config

COUNTS={"train":300,"validation":60,"test":100}
SEED_OFFSETS={"train":0,"validation":100000,"test":200000}
PLACEHOLDERS=("REPLACE_WITH_REAL_HASH","REPLACE_WITH_FINAL_HASH","PLACEHOLDER","TBD")

def _real_hash(v:str|None)->bool:
    return bool(v) and len(str(v))>=32 and not any(p in str(v) for p in PLACEHOLDERS)

def _namespace(template:Path)->int:
    cfg=yaml.safe_load(template.read_text()) or {}
    return int(cfg["seed_protocol"]["namespaces"]["scenario_generation"]["value"])

def _validate_bank(path:Path, split:str, count:int)->dict[str,Any]:
    m=load_bank_manifest(path)
    if m.get('split')!=split or int(m.get('scenario_count',-1))!=count: raise ValueError(f"bad {split} count/split")
    if not _real_hash(m.get('bank_hash')): raise ValueError(f"placeholder bank hash for {split}")
    for s in load_scenario_bank(path).scenarios: verify_scenario_hashes(s)
    return m

def _hash_reward_artifact(path:Path)->str:
    payload=json.loads(path.read_text())
    clean=dict(payload); clean.pop('artifact_hash',None)
    return hashlib.sha256(json.dumps(clean,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def _write_resolved_configs(root:Path, banks:dict[str,dict[str,Any]], scale_path:Path, scale_hash:str, reward_models:dict[str,dict[str,str]])->dict[str,str]:
    out=root/'configs'; out.mkdir(parents=True,exist_ok=True); written={}
    for key,path in {'mappo_env':'configs/paper/train_mappo_env.yaml','mappo_rlaif_assignment':'configs/paper/train_mappo_rlaif_assignment.yaml','mappo_rlaif_all':'configs/paper/train_mappo_rlaif_all.yaml'}.items():
        cfg=yaml.safe_load(Path(path).read_text())
        cfg['env'].update({'scenario_bank_manifest':str(root/'scenarios/train/scenario_bank_manifest.json'),'expected_split':'train','expected_bank_hash':banks['train']['bank_hash'],'validation_scenario_bank_manifest':str(root/'scenarios/validation/scenario_bank_manifest.json'),'expected_validation_bank_hash':banks['validation']['bank_hash'],'test_scenario_bank_manifest':str(root/'scenarios/test/scenario_bank_manifest.json'),'expected_test_bank_hash':banks['test']['bank_hash']})
        cfg['reward']['scale_artifact']=str(scale_path); cfg['reward']['scale_artifact_hash']=scale_hash; cfg['reward']['expected_training_scenario_bank_hash']=banks['train']['bank_hash']
        for a, agent in cfg.get('rlaif',{}).get('agents',{}).items():
            if agent.get('enabled'):
                if a in reward_models:
                    agent['checkpoint']=reward_models[a]['path']; agent['checkpoint_hash']=reward_models[a]['hash']
                else:
                    agent['checkpoint_hash']='MISSING_FORMAL_REWARD_CHECKPOINT_'+a.upper()
        resolved=resolve_mappo_training_config(cfg, output_root_override=root/key)
        text=yaml.safe_dump(resolved,sort_keys=False)
        if any(p in text for p in PLACEHOLDERS): raise ValueError(f"unresolved placeholder in {key}")
        dest=out/f'{key}.yaml'; dest.write_text(text); written[key]=str(dest)
    cfg=yaml.safe_load(Path('configs/paper/train_assignment_ppo.yaml').read_text())
    cfg['env']['train_scenario_bank_manifest']=str(root/'scenarios/train/scenario_bank_manifest.json'); cfg['env']['expected_train_bank_hash']=banks['train']['bank_hash']
    cfg['env']['validation_scenario_bank_manifest']=str(root/'scenarios/validation/scenario_bank_manifest.json'); cfg['env']['expected_validation_bank_hash']=banks['validation']['bank_hash']
    cfg['env']['test_scenario_bank_manifest']=str(root/'scenarios/test/scenario_bank_manifest.json'); cfg['env']['expected_test_bank_hash']=banks['test']['bank_hash']
    text=yaml.safe_dump(cfg,sort_keys=False)
    if any(p in text for p in PLACEHOLDERS): raise ValueError('unresolved placeholder in assignment_ppo')
    dest=out/'assignment_ppo.yaml'; dest.write_text(text); written['assignment_ppo']=str(dest)
    return written

def prepare(output_root:Path, force:bool=False, counts:dict[str,int]|None=None)->dict[str,Any]:
    counts=counts or COUNTS
    if output_root.exists() and force: shutil.rmtree(output_root)
    output_root.mkdir(parents=True,exist_ok=True)
    base=Path('configs/paper/base_medium.yaml'); ns=_namespace(Path('configs/paper/final_experiment_freeze.template.yaml'))
    banks={}
    for split,count in counts.items():
        out=output_root/'scenarios'/split
        build_bank(base, split, int(count), ns+SEED_OFFSETS[split], out, fallback=False, run_classification='formal', force=force)
        banks[split]=_validate_bank(out, split, int(count))
    validate_disjoint_banks([banks['train'],banks['validation'],banks['test']])
    scale_path=output_root/'reward_scales'/'final_reward_reference_scales.json'
    
    scale_cfg=yaml.safe_load(Path('configs/paper/reward_scale_estimation.yaml').read_text()) or {}
    scale_cfg.setdefault('scenario_bank', {})['expected_bank_hash']=banks['train']['bank_hash']
    scale_cfg_path=output_root/'configs'/'reward_scale_estimation.resolved.yaml'; scale_cfg_path.parent.mkdir(parents=True,exist_ok=True); scale_cfg_path.write_text(yaml.safe_dump(scale_cfg,sort_keys=False))
    run_estimation(output_root/'scenarios/train/scenario_bank_manifest.json',scale_cfg_path,scale_path,run_classification='formal',force=True)
    scale_hash=_hash_reward_artifact(scale_path)
    scale=json.loads(scale_path.read_text())
    if scale.get('training_scenario_bank_hash')!=banks['train']['bank_hash']: raise ValueError('reward scale train-bank hash mismatch')
    status={k:{'scale':v,'valid':isinstance(v,(int,float)) and v>0} for k,v in scale.get('scales',{}).items()}
    if not all(v['valid'] for v in status.values()): raise ValueError('non-positive reward scale')
    reward_models={}
    rmroot=output_root/'reward_models'
    if rmroot.exists():
        for p in rmroot.glob('reward_*.pt'):
            reward_models[p.stem.removeprefix('reward_')]={'path':str(p),'hash':sha256_file(p)}
    configs=_write_resolved_configs(output_root,banks,scale_path,scale_hash,reward_models)
    manifest={'scenario_banks':{s:{'path':str(output_root/'scenarios'/s/'scenario_bank_manifest.json'),'count':banks[s]['scenario_count'],'bank_hash':banks[s]['bank_hash']} for s in banks},'reward_scale':{'path':str(scale_path),'artifact_hash':scale_hash,'training_bank_hash':scale['training_scenario_bank_hash'],'estimator':scale.get('estimator'),'component_validation_status':status},'resolved_configs':configs,'reward_models':reward_models,'rlaif_status':'available' if 'assignment' in reward_models else 'RLAIF_BLOCKED_MISSING_FORMAL_ASSIGNMENT_REWARD_CHECKPOINT'}
    (output_root/'formal_input_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True))
    print(json.dumps(manifest,indent=2,sort_keys=True)); return manifest

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--output-root',type=Path,default=Path('results/formal')); p.add_argument('--force',action='store_true')
    a=p.parse_args(argv); prepare(a.output_root,a.force); return 0
if __name__=='__main__': raise SystemExit(main())
