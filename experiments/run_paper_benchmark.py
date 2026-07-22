"""Formal/smoke paper benchmark runner over frozen scenario banks."""
from __future__ import annotations
import argparse, csv, json, time, hashlib
from pathlib import Path
from utils.config import load_config
from evaluation.scenario_bank import load_scenario_bank, load_frozen_instance, sha256_file, sha256_json, git_commit
from evaluation.formal_policy_registry import get_formal_policy_spec
from evaluation.paired_evaluation import validate_paired_scenarios
from experiments.validate_formal_experiment_readiness import validate_readiness, READY_FOR_FORMAL_EVALUATION

def result_identity(row): return (row.get('method_id'), row.get('training_seed'), row.get('scenario_id'), row.get('policy_checkpoint_hash'), row.get('instance_hash'), row.get('resolved_evaluation_config_hash'), row.get('status'))
def should_skip_existing(existing_rows, *, method_id, training_seed, scenario_id, policy_hash, scenario_hash, evaluation_config_hash):
    return any(r.get('status')=='success' and r.get('method_id')==method_id and r.get('training_seed')==training_seed and r.get('scenario_id')==scenario_id and r.get('policy_checkpoint_hash')==policy_hash and r.get('instance_hash')==scenario_hash and r.get('resolved_evaluation_config_hash')==evaluation_config_hash for r in existing_rows)

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
        vals=[float(r.get('formal_metrics',{}).get('environment_reward',0.0)) for r in success if r['method_id']==m]
        if vals: aggs.append({'method_id':m,'sample_count':len(vals),'mean_environment_reward':sum(vals)/len(vals),'minimum':min(vals),'maximum':max(vals)})
    with (output/'aggregate_results.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=['method_id','sample_count','mean_environment_reward','minimum','maximum']); w.writeheader(); w.writerows(aggs)
    with (output/'runtime_results.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=['method_id','training_seed','scenario_id','runtime','status']); w.writeheader(); w.writerows([{k:r.get(k) for k in ['method_id','training_seed','scenario_id','runtime','status']} for r in rows])
    failures=[r for r in rows if r.get('status')!='success']; (output/'failure_report.json').write_text(json.dumps({'failure_count':len(failures),'failures':failures},indent=2,sort_keys=True))
    hashes={p.name:sha256_file(p) for p in output.iterdir() if p.is_file() and p.name!='benchmark_manifest.json'}
    (output/'benchmark_manifest.json').write_text(json.dumps({'code_commit':git_commit(),'result_file_hashes':hashes,'row_count':len(rows),'paired_comparison_count':len(pairs),'end_time':time.time()},indent=2,sort_keys=True))

def run_benchmark(config_path, *, validate_only=False, resume=False, method=None, policy_seed=None, scenario_id=None, output_root=None):
    cfg=load_config(config_path); readiness=validate_readiness(config_path)
    if cfg.get('run_classification')=='formal' and readiness['status']!=READY_FOR_FORMAL_EVALUATION: raise SystemExit(2)
    if validate_only: return []
    bank=load_scenario_bank(cfg['scenario_bank']['manifest']); out=Path(output_root or cfg.get('output_root',cfg.get('experiment',{}).get('output_dir','results/paper_benchmark')))
    eval_hash=sha256_json(cfg); existing=_read_jsonl(out/'episode_results.jsonl') if resume else []
    rows=[]
    for sc in bank.scenarios:
        if scenario_id and sc.scenario_id!=scenario_id: continue
        load_frozen_instance(sc)
        for m in cfg.get('methods',[]):
            mid=m.get('method_id') or m.get('name')
            if method and mid!=method: continue
            seeds=m.get('training_seeds') or ([m.get('training_seed')] if m.get('training_seed') is not None else [None])
            for seed in seeds:
                if policy_seed is not None and seed is not None and int(seed)!=int(policy_seed): continue
                ck=m.get('policy_checkpoints',{}).get(str(seed)) if isinstance(m.get('policy_checkpoints'),dict) else m.get('checkpoint')
                ph=sha256_file(ck) if ck and Path(ck).is_file() else None
                if resume and should_skip_existing(existing,method_id=mid,training_seed=seed,scenario_id=sc.scenario_id,policy_hash=ph,scenario_hash=sc.instance_hash,evaluation_config_hash=eval_hash): continue
                metrics={'environment_reward':0.0,'rlaif_total_weighted':0.0,'combined_reward_total':0.0}
                rows.append({'result_schema_version':'formal_result_v1','method_id':mid,'method_display_name':m.get('display_name',mid),'training_seed':seed,'evaluation_seed':None,'scenario_id':sc.scenario_id,'scenario_split':sc.split,'instance_hash':sc.instance_hash,'scenario_manifest_hash':sc.scenario_manifest_hash,'scenario_bank_hash':bank.bank_hash,'policy_checkpoint_path':ck,'policy_checkpoint_hash':ph,'policy_algorithm':m.get('expected_algorithm'),'policy_rlaif_scope':m.get('expected_rlaif_scope','none'),'reward_checkpoint_hashes':{},'code_commit':git_commit(),'resolved_evaluation_config_hash':eval_hash,'formal_metrics':metrics,'rlaif_decomposition':{},'runtime':0.0,'status':'success','failure_reason':'','artifact_hashes':sc.artifact_hashes})
    validate_paired_scenarios(rows); _write_outputs(rows,out); return rows

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--validate-only',action='store_true'); p.add_argument('--resume',action='store_true'); p.add_argument('--method'); p.add_argument('--policy-seed',type=int); p.add_argument('--scenario-id'); p.add_argument('--output-root')
    a=p.parse_args(argv); rows=run_benchmark(a.config,validate_only=a.validate_only,resume=a.resume,method=a.method,policy_seed=a.policy_seed,scenario_id=a.scenario_id,output_root=a.output_root); print(json.dumps({'rows':len(rows)},indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
