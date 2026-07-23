from __future__ import annotations
import argparse, json, sys, shutil
from pathlib import Path
import yaml
from utils.config import load_config
from evaluation.experiment_jobs import make_job_identity, sha256_file, run_subprocess, ExperimentSubprocessError, write_job_outputs, canonical_json_hash
from evaluation.scenario_bank import load_bank_manifest
from evaluation.experiment_aggregation import aggregate_compatible, paired_differences
VALID_MODES={'retrain_and_evaluate','fixed_policy_evaluate'}
class MatrixValidationError(ValueError): pass

def _set_path(d,path,val):
    cur=d
    parts=path.split('.')
    for p in parts[:-1]: cur=cur.setdefault(p,{})
    cur[parts[-1]]=val

def _load_manifest_hash(p):
    if not p: return None
    m=load_bank_manifest(p); return m.get('bank_hash')

def validate_matrix(cfg):
    if cfg.get('experiment_kind')!='ablation': raise MatrixValidationError('experiment_kind must be ablation')
    ids=[v.get('variant_id') for v in cfg.get('variants',[])]
    if len(ids)!=len(set(ids)): raise MatrixValidationError('duplicate variant_id')
    if not cfg.get('baseline_variant_id') and 'assignment_rlaif_baseline' not in ids: raise MatrixValidationError('missing baseline variant')
    for v in cfg.get('variants',[]):
        if v.get('execution_mode') not in VALID_MODES: raise MatrixValidationError('unknown execution_mode')
        if v['execution_mode']=='retrain_and_evaluate' and not v.get('training_config'): raise MatrixValidationError('retraining variant requires training_config')
        if v['execution_mode']=='fixed_policy_evaluate' and not v.get('policy_checkpoint'): raise MatrixValidationError('fixed-policy variant requires checkpoint')
        if v.get('declared_override_paths') and not v.get('config_overrides'): raise MatrixValidationError('identical ablation config')
    return True

def _resolve_training(base, variant, seed, paths, out):
    cfg=load_config(base)
    cfg.setdefault('env',{})['scenario_bank_manifest']=str(paths['train_manifest']); cfg['env']['expected_split']='train'; cfg['env']['expected_bank_hash']=paths['train_hash']
    for k,v in (variant.get('config_overrides') or {}).items(): _set_path(cfg,k,v)
    cfg.setdefault('output',{})['output_root']=str(out); cfg['output']['checkpoint_path']=str(out/'policy.pt'); cfg['output']['training_log_path']=str(out/'training.csv'); cfg['output']['eval_path']=str(out/'eval.json'); cfg['output']['resolved_config_path']=str(out/'resolved_training.yaml')
    cfg.setdefault('training',{})['seed']=seed
    return cfg

def _benchmark_config(matrix, variant, seed, ckpt, paths, out):
    return {'run_classification':matrix.get('run_classification','formal'),'scenario_bank':{'manifest':str(paths['test_manifest']),'split':'test','expected_bank_hash':paths['test_hash']},'training_seeds':[seed],'methods':[{'method_id':variant.get('benchmark_method_id',variant['variant_id']),'display_name':variant.get('display_name'), 'policy_checkpoints':{str(seed):str(ckpt)}, 'reward_checkpoints':variant.get('reward_checkpoints',{})}], 'paired_evaluation':True, 'fallback':False, 'fail_on_missing_artifact': matrix.get('run_classification')=='formal'}

def run_matrix(config, *, validate_only=False, variant_filter=None, seed_filter=None, resume=False, output_root=None, continue_on_error=False, skip_training=False, force=False):
    matrix=load_config(config); validate_matrix(matrix)
    paths={k:Path(v) for k,v in matrix.get('scenario_banks',{}).items()}
    paths.update({k.replace('manifest','hash'):_load_manifest_hash(v) for k,v in paths.items()})
    if validate_only: print(json.dumps({'status':'validated','job_count':len(matrix['variants'])*len(matrix.get('training_seeds',[None]))})); return []
    root=Path(output_root or matrix.get('output_root','results/formal/ablation'))
    if force and root.exists(): shutil.rmtree(root)
    for sub in ['resolved_configs','training','evaluation','aggregation']: (root/sub).mkdir(parents=True,exist_ok=True)
    rows=[]
    for v in matrix['variants']:
        if variant_filter and v['variant_id']!=variant_filter: continue
        for seed in matrix.get('training_seeds',[None]):
            if seed_filter is not None and int(seed)!=int(seed_filter): continue
            job_id=canonical_json_hash({'kind':'ablation','variant':v['variant_id'],'seed':seed})[:16]
            tdir=root/'training'/v['variant_id']/str(seed); edir=root/'evaluation'/v['variant_id']/str(seed); rdir=root/'resolved_configs'/v['variant_id']/str(seed); rdir.mkdir(parents=True,exist_ok=True)
            identity=make_job_identity(experiment_kind='ablation', variant_id=v['variant_id'], training_seed=seed, training_config_hash=sha256_file(v.get('training_config')), benchmark_config_hash=canonical_json_hash(v), train_bank_hash=paths.get('train_hash'), validation_bank_hash=paths.get('validation_hash'), test_bank_hash=paths.get('test_hash'), reward_scale_hash=v.get('reward_scale_hash'), reward_checkpoint_hashes=v.get('reward_checkpoint_hashes'), code_compatibility_hash='source')
            row={'job_id':job_id,'variant_id':v['variant_id'],'method_id':v.get('benchmark_method_id',v['variant_id']),'experiment_kind':'ablation','run_classification':matrix.get('run_classification'),'training_seed':seed,'identity':identity,'status':'validated','test_bank_hash':paths.get('test_hash')}
            try:
                ckpt=Path(v.get('policy_checkpoint') or tdir/'policy.pt')
                if v['execution_mode']=='retrain_and_evaluate' and not skip_training:
                    tc=_resolve_training(v['training_config'],v,int(seed),paths,tdir); (rdir/'training.yaml').write_text(yaml.safe_dump(tc,sort_keys=False))
                    row['status']='training_running'; run_subprocess([sys.executable,'-m','experiments.train_mappo_async','--config',str(rdir/'training.yaml'),'--seed',str(seed),'--output-root',str(tdir)],stage='training')
                    if not ckpt.is_file(): raise RuntimeError(f'missing checkpoint: {ckpt}')
                    row['policy_checkpoint_hash']=sha256_file(ckpt); row['status']='training_success'
                bc=_benchmark_config(matrix,v,int(seed),ckpt,paths,edir); (rdir/'benchmark.yaml').write_text(yaml.safe_dump(bc,sort_keys=False))
                row['status']='evaluation_running'; run_subprocess([sys.executable,'-m','experiments.run_paper_benchmark','--config',str(rdir/'benchmark.yaml'),'--output-root',str(edir),'--continue-on-error'],stage='evaluation')
                erows=[json.loads(l) for l in (edir/'episode_results.jsonl').read_text().splitlines() if l.strip()]
                row['status']='evaluation_success'; row['evaluation_row_count']=len(erows); row['transition_count']=sum(int(e.get('transition_count',0)) for e in erows); row['environment_reward']=sum(float((e.get('formal_metrics',{}).get('environment_reward',{}) or {}).get('value',0) if isinstance(e.get('formal_metrics',{}).get('environment_reward',{}),dict) else e.get('formal_metrics',{}).get('environment_reward',0)) for e in erows)/max(1,len(erows))
            except Exception as exc:
                row.update({'status':'training_failed' if row.get('status')=='training_running' else 'evaluation_failed','exception_type':type(exc).__name__,'failure_reason':str(exc)})
                if isinstance(exc, ExperimentSubprocessError): row.update(exc.record)
                if not continue_on_error: rows.append(row); write_job_outputs(root,rows); raise SystemExit(1)
            rows.append(row)
    aggs=aggregate_compatible(rows); (root/'aggregation'/'aggregate_results.json').write_text(json.dumps(aggs,indent=2,sort_keys=True,default=str))
    base=matrix.get('baseline_variant_id','assignment_rlaif_baseline'); pairs=paired_differences(rows, baseline_selector=lambda r:r.get('variant_id')==base, comparison_selector=lambda r:r.get('variant_id')!=base); (root/'aggregation'/'paired_results.json').write_text(json.dumps(pairs,indent=2,sort_keys=True,default=str))
    checkpoint_hashes={r.get('variant_id'):r.get('policy_checkpoint_hash') for r in rows if r.get('policy_checkpoint_hash')}
    gate={'baseline_variant':base,'comparison_variants':[v.get('variant_id') for v in matrix.get('variants',[]) if v.get('variant_id')!=base],'training_seeds':matrix.get('training_seeds',[None]),'checkpoint_hashes':checkpoint_hashes,'test_bank_hash':paths.get('test_hash'),'expected_jobs':len(matrix.get('variants',[]))*len(matrix.get('training_seeds',[None])),'successful_jobs':sum(1 for r in rows if r.get('status')=='evaluation_success'),'failed_jobs':[r for r in rows if r.get('status','').endswith('failed')],'paired_scenario_count':pairs.get('summary',{}).get('paired_sample_count',0),'publication_eligible':False}
    (root/'ablation_gate_report.json').write_text(json.dumps(gate,indent=2,sort_keys=True,default=str))
    (root/'experiment_manifest.json').write_text(json.dumps({'run_classification':matrix.get('run_classification'),'publication_eligible':False,'job_count':len(rows)},indent=2))
    write_job_outputs(root,rows); return rows

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--validate-only',action='store_true'); p.add_argument('--variant'); p.add_argument('--seed',type=int); p.add_argument('--resume',action='store_true'); p.add_argument('--output-root'); p.add_argument('--continue-on-error',action='store_true'); p.add_argument('--skip-training',action='store_true'); p.add_argument('--force',action='store_true')
    a=p.parse_args(argv); rows=run_matrix(a.config, validate_only=a.validate_only, variant_filter=a.variant, seed_filter=a.seed, resume=a.resume, output_root=a.output_root, continue_on_error=a.continue_on_error, skip_training=a.skip_training, force=a.force); print(json.dumps({'rows':len(rows)})); return 0
if __name__=='__main__': raise SystemExit(main())
