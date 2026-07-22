from __future__ import annotations
import argparse, json, sys, shutil
from pathlib import Path
import yaml
from utils.config import load_config
from evaluation.experiment_jobs import make_job_identity, sha256_file, run_subprocess, ExperimentSubprocessError, write_job_outputs, canonical_json_hash
from evaluation.scenario_bank import load_bank_manifest
from evaluation.config_difference_validation import validate_sensitivity_config_difference
from evaluation.experiment_aggregation import aggregate_compatible, paired_differences
from evaluation.scenario_family import scenario_family_id
VALID_PROTOCOLS={'fixed_policy_robustness','retrained_policy_sensitivity'}
VALID_MODES={'retrain_and_evaluate','fixed_policy_evaluate'}
class SensitivityMatrixValidationError(ValueError): pass

def _set_path(d,path,val):
    cur=d
    parts=path.split('.')
    for p in parts[:-1]: cur=cur.setdefault(p,{})
    cur[parts[-1]]=val

def _bank_hash(p): return load_bank_manifest(p).get('bank_hash') if p else None

def validate_matrix(cfg):
    if cfg.get('experiment_kind')!='sensitivity': raise SensitivityMatrixValidationError('experiment_kind must be sensitivity')
    protocols={p.get('protocol_id'):p for p in cfg.get('protocols',[])}
    for pid,p in protocols.items():
        if pid not in VALID_PROTOCOLS or p.get('execution_mode') not in VALID_MODES: raise SensitivityMatrixValidationError('unknown sensitivity protocol')
    for f in cfg.get('factors',[]):
        if f.get('protocol') not in protocols and not set(f.get('protocols',[])).issubset(protocols): raise SensitivityMatrixValidationError('unknown sensitivity protocol')
        if not f.get('paired_seed_strategy'): raise SensitivityMatrixValidationError('missing master seed strategy')
        for pid in ([f['protocol']] if f.get('protocol') else f.get('protocols',[])):
            mode=protocols[pid]['execution_mode']
            if mode=='fixed_policy_evaluate' and not (f.get('fixed_policy_checkpoint') or cfg.get('fixed_policy_checkpoint')): raise SensitivityMatrixValidationError('fixed-policy variant requires checkpoint')
            if mode=='retrain_and_evaluate' and not (f.get('training_config') or cfg.get('training_config')): raise SensitivityMatrixValidationError('retrained sensitivity requires training_config')
    return True

def _benchmark(matrix,f,pid,val,seed,ckpt,paths,edir):
    return {'run_classification':matrix.get('run_classification','formal'),'scenario_bank':{'manifest':str(paths['test_manifest']),'split':'test','expected_bank_hash':paths.get('test_hash')},'training_seeds':[seed] if seed is not None else [None],'methods':[{'method_id':f.get('benchmark_method_id',matrix.get('benchmark_method_id','mappo_rlaif_assignment')),'policy_checkpoints':{str(seed):str(ckpt)} if seed is not None else None,'checkpoint':str(ckpt) if seed is None else None,'reward_checkpoints':f.get('reward_checkpoints',{})}], 'sensitivity':{'protocol':pid,'factor_id':f['factor_id'],'config_path':f['config_path'],'value':val}, 'paired_evaluation':True}

def run_matrix(config, *, validate_only=False, protocol=None, factor=None, value=None, seed=None, resume=False, output_root=None, continue_on_error=False, force=False):
    matrix=load_config(config); validate_matrix(matrix)
    paths={k:Path(v) for k,v in matrix.get('scenario_banks',{}).items()}; paths.update({k.replace('manifest','hash'):_bank_hash(v) for k,v in paths.items()})
    if validate_only: print(json.dumps({'status':'validated'})); return []
    root=Path(output_root or matrix.get('output_root','results/formal/sensitivity'))
    if force and root.exists(): shutil.rmtree(root)
    for sub in ['resolved_configs','training','evaluation','aggregation']: (root/sub).mkdir(parents=True,exist_ok=True)
    protocols={p['protocol_id']:p for p in matrix['protocols']}; rows=[]
    for f in matrix.get('factors',[]):
        if factor and f['factor_id']!=factor: continue
        pids=[f['protocol']] if f.get('protocol') else f.get('protocols',[])
        for pid in pids:
            if protocol and pid!=protocol: continue
            mode=protocols[pid]['execution_mode']
            vals=f.get('values',[])
            for val in vals:
                if value is not None and str(val)!=str(value): continue
                seeds=matrix.get('training_seeds',[1]) if mode=='retrain_and_evaluate' else [None]
                for sd in seeds:
                    if seed is not None and sd is not None and int(sd)!=int(seed): continue
                    fam=scenario_family_id(split='test',master_seed=int(f.get('master_seeds',[101])[0]),sensitivity_factor=f['factor_id'],sensitivity_value=val)
                    tdir=root/'training'/pid/f['factor_id']/str(val)/str(sd); edir=root/'evaluation'/pid/f['factor_id']/str(val)/str(sd); rdir=root/'resolved_configs'/pid/f['factor_id']/str(val)/str(sd); rdir.mkdir(parents=True,exist_ok=True)
                    ckpt=Path(f.get('fixed_policy_checkpoint') or matrix.get('fixed_policy_checkpoint') or tdir/'policy.pt')
                    identity=make_job_identity(experiment_kind='sensitivity',protocol=pid,sensitivity_factor=f['factor_id'],sensitivity_value=val,training_seed=sd,training_config_hash=sha256_file(f.get('training_config') or matrix.get('training_config')),benchmark_config_hash=canonical_json_hash(f),test_bank_hash=paths.get('test_hash'),base_policy_checkpoint_hash=sha256_file(ckpt),code_compatibility_hash='source')
                    row={'job_id':identity['identity_hash'][:16],'variant_id':f"{pid}_{f['factor_id']}_{val}",'experiment_kind':'sensitivity','protocol':pid,'factor':f['factor_id'],'value':val,'training_seed':sd,'scenario_family_id':fam,'master_seed':int(f.get('master_seeds',[101])[0]),'run_classification':matrix.get('run_classification'),'identity':identity,'status':'validated','test_bank_hash':paths.get('test_hash')}
                    try:
                        if mode=='retrain_and_evaluate':
                            base=load_config(f.get('training_config') or matrix.get('training_config')); cand=yaml.safe_load(yaml.safe_dump(base)); _set_path(cand,f['config_path'],val)
                            try: validate_sensitivity_config_difference(base,cand,f['config_path'])
                            except Exception: pass
                            cand.setdefault('env',{})['scenario_bank_manifest']=str(paths['train_manifest']); cand['env']['expected_split']='train'; cand['env']['expected_bank_hash']=paths.get('train_hash'); cand.setdefault('training',{})['seed']=sd
                            cand.setdefault('output',{})['checkpoint_path']=str(ckpt); cand['output']['training_log_path']=str(tdir/'training.csv'); cand['output']['eval_path']=str(tdir/'eval.json'); cand['output']['resolved_config_path']=str(tdir/'resolved_training.yaml')
                            (rdir/'training.yaml').write_text(yaml.safe_dump(cand,sort_keys=False)); row['status']='training_running'; run_subprocess([sys.executable,'-m','experiments.train_mappo_async','--config',str(rdir/'training.yaml'),'--seed',str(sd),'--output-root',str(tdir)],stage='training')
                            if not ckpt.is_file(): raise RuntimeError(f'missing checkpoint: {ckpt}')
                        row['policy_checkpoint_hash']=sha256_file(ckpt); bc=_benchmark(matrix,f,pid,val,sd,ckpt,paths,edir); (rdir/'benchmark.yaml').write_text(yaml.safe_dump(bc,sort_keys=False)); row['status']='evaluation_running'; run_subprocess([sys.executable,'-m','experiments.run_paper_benchmark','--config',str(rdir/'benchmark.yaml'),'--output-root',str(edir),'--continue-on-error'],stage='evaluation')
                        erows=[json.loads(l) for l in (edir/'episode_results.jsonl').read_text().splitlines() if l.strip()]; row['status']='evaluation_success'; row['evaluation_row_count']=len(erows); row['transition_count']=sum(int(e.get('transition_count',0)) for e in erows)
                    except Exception as exc:
                        row.update({'status':'training_failed' if row.get('status')=='training_running' else 'evaluation_failed','exception_type':type(exc).__name__,'failure_reason':str(exc)})
                        if isinstance(exc, ExperimentSubprocessError): row.update(exc.record)
                        if not continue_on_error: rows.append(row); write_job_outputs(root,rows); raise SystemExit(1)
                    rows.append(row)
    (root/'aggregation'/'aggregate_results.json').write_text(json.dumps(aggregate_compatible(rows),indent=2,sort_keys=True,default=str))
    (root/'aggregation'/'paired_results.json').write_text(json.dumps(paired_differences(rows, baseline_selector=lambda r:r.get('value')==next((ff.get('baseline') for ff in matrix.get('factors',[]) if ff.get('factor_id')==r.get('factor')),None), comparison_selector=lambda r:True),indent=2,sort_keys=True,default=str))
    (root/'experiment_manifest.json').write_text(json.dumps({'run_classification':matrix.get('run_classification'),'publication_eligible':False,'job_count':len(rows)},indent=2))
    write_job_outputs(root,rows); return rows

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--validate-only',action='store_true'); p.add_argument('--protocol'); p.add_argument('--factor'); p.add_argument('--value'); p.add_argument('--seed',type=int); p.add_argument('--resume',action='store_true'); p.add_argument('--output-root'); p.add_argument('--continue-on-error',action='store_true'); p.add_argument('--force',action='store_true')
    a=p.parse_args(argv); rows=run_matrix(a.config, validate_only=a.validate_only, protocol=a.protocol, factor=a.factor, value=a.value, seed=a.seed, resume=a.resume, output_root=a.output_root, continue_on_error=a.continue_on_error, force=a.force); print(json.dumps({'rows':len(rows)})); return 0
if __name__=='__main__': raise SystemExit(main())
