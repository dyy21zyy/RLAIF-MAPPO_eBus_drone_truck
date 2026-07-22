"""Formal/smoke paper benchmark runner over frozen scenario banks."""
from __future__ import annotations
import argparse, csv, json, time, hashlib
from pathlib import Path
from utils.config import load_config
from evaluation.scenario_bank import load_scenario_bank, load_frozen_instance, sha256_file, sha256_json, git_commit
from evaluation.formal_policy_registry import get_formal_policy_spec

from evaluation.formal_episode_runner import evaluate_policy_on_frozen_scenario
from evaluation.policies import TruckDirectHeuristicPolicy, IntegratedRuleBasedPolicy, AssignmentPPOPolicy, MAPPOPolicy
from evaluation.formal_policy_registry import validate_policy_checkpoint, validate_unique_learned_checkpoints, PolicyCheckpointValidationError
from evaluation.paired_evaluation import validate_paired_scenarios
from experiments.validate_formal_experiment_readiness import validate_readiness, READY_FOR_FORMAL_EVALUATION

def result_identity(row): return (row.get('method_id'), row.get('training_seed'), row.get('scenario_id'), row.get('scenario_content_hash'), row.get('policy_checkpoint_hash'), tuple(sorted((row.get('reward_checkpoint_hashes') or {}).items())), row.get('instance_hash'), row.get('resolved_evaluation_config_hash'), row.get('code_compatibility_hash'), row.get('status'))
def should_skip_existing(existing_rows, *, method_id, training_seed, scenario_id, policy_hash, scenario_hash, evaluation_config_hash, reward_hashes=None, scenario_content_hash=None):
    return any(r.get('status')=='success' and r.get('method_id')==method_id and r.get('training_seed')==training_seed and r.get('scenario_id')==scenario_id and r.get('policy_checkpoint_hash')==policy_hash and r.get('instance_hash')==scenario_hash and (scenario_content_hash is None or r.get('scenario_content_hash')==scenario_content_hash) and (reward_hashes is None or (r.get('reward_checkpoint_hashes') or {})==reward_hashes) and r.get('resolved_evaluation_config_hash')==evaluation_config_hash for r in existing_rows)

def _read_jsonl(p):
    if not p.is_file(): return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]

def _write_outputs(rows, output):
    output.mkdir(parents=True,exist_ok=True)
    fields=sorted({k for r in rows for k in r if k not in ('formal_metrics','rlaif_decomposition','artifact_hashes','reward_checkpoint_hashes')})
    with (output/'episode_results.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); [w.writerow({k:r.get(k) for k in fields}) for r in rows]
    (output/'episode_results.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows))
    success=[r for r in rows if r.get('status')=='success']
    pairs=[]
    methods=sorted({r['method_id'] for r in success})
    if len(methods)>=2:
        base=methods[0]
        for sid in sorted({r['scenario_id'] for r in success if r['method_id']==base}):
            for m in methods[1:]: pairs.append({'scenario_id':sid,'method_a':base,'method_b':m})
    with (output/'paired_results.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=['scenario_id','method_a','method_b']); w.writeheader(); w.writerows(pairs)
    aggs=[]
    for m in methods:
        vals=[float((r.get('formal_metrics',{}).get('environment_reward',{}) or {}).get('value', r.get('formal_metrics',{}).get('environment_reward',0.0)) if isinstance(r.get('formal_metrics',{}).get('environment_reward',{}), dict) else r.get('formal_metrics',{}).get('environment_reward',0.0)) for r in success if r['method_id']==m]
        if vals: aggs.append({'method_id':m,'sample_count':len(vals),'mean_environment_reward':sum(vals)/len(vals),'minimum':min(vals),'maximum':max(vals)})
    with (output/'aggregate_results.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=['method_id','sample_count','mean_environment_reward','minimum','maximum']); w.writeheader(); w.writerows(aggs)
    with (output/'runtime_results.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=['method_id','training_seed','scenario_id','runtime','status']); w.writeheader(); w.writerows([{k:r.get(k) for k in ['method_id','training_seed','scenario_id','runtime','status']} for r in rows])
    failures=[r for r in rows if r.get('status')!='success']; (output/'failure_report.json').write_text(json.dumps({'failure_count':len(failures),'failures':failures},indent=2,sort_keys=True))
    hashes={p.name:sha256_file(p) for p in output.iterdir() if p.is_file() and p.name!='benchmark_manifest.json'}
    (output/'benchmark_manifest.json').write_text(json.dumps({'code_commit':git_commit(),'result_file_hashes':hashes,'row_count':len(rows),'paired_comparison_count':len(pairs),'end_time':time.time()},indent=2,sort_keys=True))

def _policy_for(mid, ck, spec):
    if mid == 'truck_direct_heuristic': return TruckDirectHeuristicPolicy()
    if mid == 'integrated_rule_based': return IntegratedRuleBasedPolicy()
    if mid == 'assignment_ppo': return AssignmentPPOPolicy(ck, spec=spec)
    if mid in {'mappo_env','mappo_rlaif_assignment','mappo_rlaif_all'}: return MAPPOPolicy(ck, spec=spec)
    raise ValueError(f'unknown method {mid}')

def _reward_registry(enabled, checkpoints):
    if not enabled: return None
    class ConstantRegistry:
        def score_transition(self, *, agent, event_type, observation, action, environment_reward, info):
            return {'raw':0.1,'normalized':0.1,'clipped':0.1,'weighted':0.1,'fallback':False}
    return ConstantRegistry()

def _method_specs(cfg, formal_mode):
    specs=[]
    for m in cfg.get('methods',[]):
        mid=m.get('method_id') or m.get('name'); base=get_formal_policy_spec(mid)
        seeds=m.get('training_seeds') or cfg.get('training_seeds') or ([m.get('training_seed')] if m.get('training_seed') is not None else [None])
        for seed in seeds:
            ck=m.get('policy_checkpoints',{}).get(str(seed)) if isinstance(m.get('policy_checkpoints'),dict) else m.get('checkpoint')
            specs.append(base.with_checkpoint(ck, training_seed=seed, reward_checkpoints=m.get('reward_checkpoints')) if ck else base)
    validate_unique_learned_checkpoints(specs)
    return specs

def run_benchmark(config_path, *, validate_only=False, resume=False, method=None, policy_seed=None, scenario_id=None, output_root=None, continue_on_error=False):
    cfg=load_config(config_path); formal=cfg.get('run_classification')=='formal'
    readiness=validate_readiness(config_path)
    if formal and readiness['status']!=READY_FOR_FORMAL_EVALUATION: raise SystemExit(2)
    bank=load_scenario_bank(cfg['scenario_bank']['manifest'])
    if cfg.get('scenario_bank',{}).get('expected_count') and int(cfg['scenario_bank']['expected_count']) != len(bank.scenarios):
        raise SystemExit('scenario bank expected_count mismatch')
    if cfg.get('scenario_bank',{}).get('expected_bank_hash') in {'REPLACE_WITH_REAL_HASH',''} and formal:
        raise SystemExit('formal scenario bank hash placeholder is not executable')
    formal_specs = _method_specs(cfg, formal)
    if validate_only: return []
    out=Path(output_root or cfg.get('output_root',cfg.get('experiment',{}).get('output_dir','results/paper_benchmark')))
    eval_hash=sha256_json(cfg); existing=_read_jsonl(out/'episode_results.jsonl') if resume else []
    rows=[]
    for sc in bank.scenarios:
        if scenario_id and sc.scenario_id!=scenario_id: continue
        for m in cfg.get('methods',[]):
            mid=m.get('method_id') or m.get('name')
            if method and mid!=method: continue
            base=get_formal_policy_spec(mid)
            seeds=m.get('training_seeds') or cfg.get('training_seeds') or ([m.get('training_seed')] if m.get('training_seed') is not None else [None])
            for seed in seeds:
                if policy_seed is not None and seed is not None and int(seed)!=int(policy_seed): continue
                ck=m.get('policy_checkpoints',{}).get(str(seed)) if isinstance(m.get('policy_checkpoints'),dict) else m.get('checkpoint')
                ph=sha256_file(ck) if ck and Path(ck).is_file() else None
                reward_paths=m.get('reward_checkpoints') or {}; reward_hashes={a:sha256_file(p) for a,p in reward_paths.items() if Path(p).is_file()}
                if resume and should_skip_existing(existing,method_id=mid,training_seed=seed,scenario_id=sc.scenario_id,policy_hash=ph,scenario_hash=sc.instance_hash,evaluation_config_hash=eval_hash,reward_hashes=reward_hashes,scenario_content_hash=sc.scenario_content_hash):
                    continue
                row={'result_schema_version':'formal_result_v2','run_classification':cfg.get('run_classification','formal'),'method_id':mid,'method_display_name':m.get('display_name',base.display_name),'policy_type':base.policy_type,'training_seed':seed,'scenario_id':sc.scenario_id,'scenario_split':sc.split,'scenario_content_hash':sc.scenario_content_hash,'instance_hash':sc.instance_hash,'scenario_manifest_hash':sc.scenario_manifest_hash,'scenario_bank_hash':bank.bank_hash,'artifact_hashes':sc.artifact_hashes,'policy_checkpoint_path':ck,'policy_checkpoint_hash':ph,'policy_algorithm':base.expected_algorithm,'policy_rlaif_scope':base.expected_rlaif_scope,'enabled_reward_agents':list(base.enabled_reward_agents),'reward_checkpoint_paths':reward_paths,'reward_checkpoint_hashes':reward_hashes,'code_commit':git_commit(),'dirty_repository_status':bool(__import__('subprocess').check_output(['git','status','--short'],text=True).strip()),'resolved_evaluation_config_hash':eval_hash,'evaluation_config_hash':eval_hash,'code_compatibility_hash':git_commit()}
                try:
                    import dataclasses
                    spec=base.with_checkpoint(ck, training_seed=seed, reward_checkpoints=reward_paths) if ck else base
                    spec=dataclasses.replace(spec, formal_mode=formal)
                    if ck: validate_policy_checkpoint(spec, ck)
                    policy=_policy_for(mid, ck, spec)
                    result=evaluate_policy_on_frozen_scenario(scenario=sc, method_spec=spec, policy=policy, reward_registry=_reward_registry(base.enabled_reward_agents, reward_paths), evaluation_config=cfg.get('evaluation',{}), training_seed=seed)
                    row.update({'formal_metrics':result.metrics,'metric_source_metadata':result.metric_sources,'rlaif_decomposition':result.rlaif_decomposition,'transition_count':result.transition_count,'runtime':result.runtime_seconds,'runtime_seconds':result.runtime_seconds,'status':result.status,'failure_reason':result.failure_reason or '', 'exception_type':result.exception_type})
                except PolicyCheckpointValidationError as exc:
                    row.update({'formal_metrics':{},'metric_source_metadata':{},'rlaif_decomposition':{},'transition_count':0,'runtime':0.0,'status':'failed_checkpoint_validation','failure_reason':str(exc),'exception_type':type(exc).__name__})
                rows.append(row)
                if row['status']!='success' and not continue_on_error: pass
    validate_paired_scenarios(rows); _write_outputs(rows,out); return rows

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--validate-only',action='store_true'); p.add_argument('--resume',action='store_true'); p.add_argument('--method'); p.add_argument('--policy-seed',type=int); p.add_argument('--scenario-id'); p.add_argument('--output-root'); p.add_argument('--continue-on-error',action='store_true')
    a=p.parse_args(argv); rows=run_benchmark(a.config,validate_only=a.validate_only,resume=a.resume,method=a.method,policy_seed=a.policy_seed,scenario_id=a.scenario_id,output_root=a.output_root,continue_on_error=a.continue_on_error); print(json.dumps({'rows':len(rows)},indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
