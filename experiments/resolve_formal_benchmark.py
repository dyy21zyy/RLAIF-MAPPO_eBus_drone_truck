"""Resolve formal benchmark config from real policy and reward artifacts."""
from __future__ import annotations
import argparse, json, sys, hashlib
from pathlib import Path
from typing import Any
import yaml
from evaluation.formal_policy_registry import LEARNED_METHODS, get_formal_policy_spec, validate_policy_checkpoint, validate_unique_learned_checkpoints
from rlaif.runtime_agent_reward_model import RuntimeAgentRewardModel
from training.event_schema import REQUIRED_EVENT_COVERAGE

PLACEHOLDERS=("REPLACE_WITH","PLACEHOLDER","MISSING_FORMAL","TBD","UNKNOWN")
REQUIRED_REWARDS={'mappo_rlaif_assignment':('assignment',),'mappo_rlaif_all':('assignment','truck','bus','station')}

def sha(path:Path)->str:
    h=hashlib.sha256();
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

def load_yaml(p:Path)->dict[str,Any]: return yaml.safe_load(p.read_text()) or {}

def _find_policy(root:Path, method:str, seed:int)->Path|None:
    pats=[f'{method}_seed_{seed}.pt', f'{method}/**/*seed*{seed}*.pt', f'**/{method}*seed*{seed}*.pt']
    for pat in pats:
        hits=[p for p in root.glob(pat) if p.is_file()]
        if hits: return sorted(hits)[0]
    return None

def _reward_path(root:Path, agent:str)->Path:
    return root/'reward_models'/f'reward_{agent}.pt'

def _validate_reward(root:Path, agent:str)->dict[str,Any]:
    p=_reward_path(root,agent)
    if not p.is_file(): raise RuntimeError(f'missing reward checkpoint for {agent}: {p}')
    digest=sha(p)
    m=RuntimeAgentRewardModel.from_checkpoint(p, expected_agent_type=agent, expected_event_types=sorted(REQUIRED_EVENT_COVERAGE[agent]), expected_checkpoint_hash=digest, formal_mode=True)
    return {'path':str(p),'hash':digest,'agent_type':m.agent_type,'supported_event_types':list(m.compatible_event_types)}

def resolve(template:Path, artifact_root:Path, output:Path)->dict[str,Any]:
    cfg=load_yaml(template); seeds=[int(s) for s in cfg.get('training_seeds',[1,2,3])]
    missing=[]; specs=[]; method_entries=[]
    reward_records={a:_validate_reward(artifact_root,a) for a in ('assignment','truck','bus','station') if _reward_path(artifact_root,a).exists()}
    for method in LEARNED_METHODS:
        checkpoints={}; metas={}
        for seed in seeds:
            p=_find_policy(artifact_root,method,seed)
            if p is None:
                missing.append(f'{method} seed {seed}')
                continue
            spec=get_formal_policy_spec(method).with_checkpoint(str(p), training_seed=seed)
            meta=validate_policy_checkpoint(spec,p)
            checkpoints[str(seed)]={'path':str(p),'hash':meta['checkpoint_hash']}
            metas[str(seed)]={k:meta.get(k) for k in ('algorithm','training_seed','training_scenario_bank_hash','validation_scenario_bank_hash','reward_scale_hash','rlaif_scope','enabled_reward_agents','reward_checkpoint_hashes','observation_schema','action_schema')}
            specs.append(spec)
        entry={'method_id':method,'policy_checkpoints':checkpoints,'checkpoint_lineage':metas}
        if method in REQUIRED_REWARDS:
            r={}
            for agent in REQUIRED_REWARDS[method]:
                rec=reward_records.get(agent) or _validate_reward(artifact_root,agent)
                r[agent]=rec
            entry['reward_checkpoints']=r
        method_entries.append(entry)
    if missing:
        raise RuntimeError('missing policy checkpoints: '+', '.join(missing))
    validate_unique_learned_checkpoints(specs)
    # preserve heuristic methods, replace learned methods fail-closed
    cfg['methods']=[m for m in cfg.get('methods',[]) if m.get('method_id') not in LEARNED_METHODS]+method_entries
    test_manifest=artifact_root/'scenarios'/'test'/'scenario_bank_manifest.json'
    cfg.setdefault('scenario_bank',{})
    if test_manifest.exists():
        cfg['scenario_bank']['manifest']=str(test_manifest)
        cfg['scenario_bank']['expected_bank_hash']=sha(test_manifest)
    else:
        cfg['scenario_bank']['expected_bank_hash']=cfg['scenario_bank'].get('expected_bank_hash_resolved','missing-test-bank')
    cfg['benchmark_output_root']=str(artifact_root/'benchmark')
    text=yaml.safe_dump(cfg,sort_keys=False)
    if any(p in text for p in PLACEHOLDERS): raise RuntimeError('resolved benchmark config contains placeholder')
    output.parent.mkdir(parents=True,exist_ok=True); output.write_text(text)
    return {'status':'resolved','output':str(output),'methods':[m['method_id'] for m in method_entries],'training_seeds':seeds}

def main(argv=None)->int:
    ap=argparse.ArgumentParser(); ap.add_argument('--template',type=Path,required=True); ap.add_argument('--artifact-root',type=Path,required=True); ap.add_argument('--output',type=Path,required=True)
    ns=ap.parse_args(argv)
    try: print(json.dumps(resolve(ns.template,ns.artifact_root,ns.output),indent=2,sort_keys=True)); return 0
    except Exception as exc: print(f'formal benchmark resolution failed: {exc}',file=sys.stderr); return 2
if __name__=='__main__': raise SystemExit(main())
