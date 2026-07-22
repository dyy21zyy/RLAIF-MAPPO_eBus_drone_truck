from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import Any
from utils.config import load_config
from evaluation.scenario_bank import load_bank_manifest
from evaluation.formal_policy_registry import get_formal_policy_spec, validate_unique_learned_checkpoints, validate_policy_checkpoint, FormalPolicySpec, PolicyCheckpointValidationError
from training.reward_model_wrapper import load_strict_agent_reward_checkpoint, RewardCheckpointError

CONFIG_VALID_ARTIFACTS_MISSING='CONFIG_VALID_ARTIFACTS_MISSING'; CONFIG_INVALID='CONFIG_INVALID'; ARTIFACTS_INCOMPATIBLE='ARTIFACTS_INCOMPATIBLE'; READY_FOR_FORMAL_EVALUATION='READY_FOR_FORMAL_EVALUATION'

def _methods(cfg): return cfg.get('methods') or cfg.get('policy_matrix',{}).get('policies',[])
def _spec_from_method(m):
    mid=m.get('method_id') or m.get('name')
    aliases={'four_agent_env_mappo':'mappo_env','assignment_only_rlaif_mappo':'mappo_rlaif_assignment','full_multi_agent_rlaif_mappo':'mappo_rlaif_all','integrated_rule_based_heuristic':'integrated_rule_based'}
    mid=aliases.get(mid,mid); base=get_formal_policy_spec(mid)
    return FormalPolicySpec(base.method_id,base.display_name,base.policy_type,m.get('checkpoint') or m.get('policy_checkpoint'),base.expected_algorithm,base.expected_rlaif_scope,base.enabled_reward_agents,m.get('reward_checkpoints') or ({'assignment':m.get('reward_model_checkpoint')} if m.get('reward_model_checkpoint') else {}),m.get('training_seed'),cfg_run_classification=='formal')

def validate_readiness(config_path: str|Path)->dict[str,Any]:
    global cfg_run_classification
    cfg=load_config(config_path); cfg_run_classification=cfg.get('run_classification',cfg.get('experiment',{}).get('run_classification','smoke' if 'smoke' in str(config_path) else 'formal'))
    issues=[]; missing=[]; incompatible=[]
    if cfg_run_classification=='formal' and cfg.get('fallback',cfg.get('experiment',{}).get('fallback',False)): issues.append('fallback enabled blocks formal readiness')
    if cfg_run_classification=='formal' and not cfg.get('paired_evaluation',True): issues.append('paired evaluation must be enabled')
    sb=cfg.get('scenario_bank') or {}; manifest=sb.get('manifest')
    if not manifest or not Path(manifest).is_file(): missing.append(f'scenario bank missing: {manifest}')
    else:
        try:
            m=load_bank_manifest(manifest)
            if sb.get('expected_count') is not None and int(m.get('scenario_count',m.get('size',0))) != int(sb['expected_count']): incompatible.append('scenario bank expected_count mismatch')
            if sb.get('split') and m.get('split',m.get('bank')) != sb.get('split'): incompatible.append('scenario bank split mismatch')
        except Exception as exc: incompatible.append(f'scenario bank invalid: {exc}')
    specs=[]
    for m in _methods(cfg):
        try:
            s=_spec_from_method(m); specs.append(s)
            if s.policy_checkpoint:
                if not Path(s.policy_checkpoint).is_file(): missing.append(f'policy checkpoint missing: {s.policy_checkpoint}')
                else: validate_policy_checkpoint(s,s.policy_checkpoint)
            elif s.policy_type not in ('heuristic',): missing.append(f'policy checkpoint missing for {s.method_id}')
            for a in s.enabled_reward_agents:
                ck=(s.reward_checkpoints or {}).get(a)
                if not ck or not Path(ck).is_file(): missing.append(f'reward checkpoint missing for {s.method_id}/{a}: {ck}')
                else: load_strict_agent_reward_checkpoint(ck,agent_type=a,formal=(cfg_run_classification=='formal'))
        except (ValueError, PolicyCheckpointValidationError, RewardCheckpointError) as exc:
            incompatible.append(str(exc))
    try: validate_unique_learned_checkpoints(specs)
    except Exception as exc: incompatible.append(str(exc))
    if issues: status=CONFIG_INVALID
    elif incompatible: status=ARTIFACTS_INCOMPATIBLE
    elif missing: status=CONFIG_VALID_ARTIFACTS_MISSING
    else: status=READY_FOR_FORMAL_EVALUATION
    return {'status':status,'issues':issues,'missing_artifacts':missing,'incompatible_artifacts':incompatible,'run_classification':cfg_run_classification}

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--report-only',action='store_true'); p.add_argument('--strict',action='store_true')
    a=p.parse_args(argv); r=validate_readiness(a.config); print(json.dumps(r,indent=2)); return 0 if (a.report_only or r['status']==READY_FOR_FORMAL_EVALUATION) else 2
if __name__=='__main__': raise SystemExit(main())
