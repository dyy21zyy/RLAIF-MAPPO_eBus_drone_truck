from __future__ import annotations
import argparse, json, shutil, hashlib
from pathlib import Path
import yaml
from experiments.build_scenario_bank import build_bank
from envs.reward_scales import canonical_payload_hash
from envs.reward_components import REWARD_COMPONENTS

def _hash(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def _write_scale(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload={'artifact_type':'reward_reference_scales','artifact_version':1,'run_classification':'formal','validation_status':'passed','training_scenario_bank_hash':'diagnostic','component_order':list(REWARD_COMPONENTS),'scales':{c:1.0 for c in REWARD_COMPONENTS},'components':{c:{'status':'estimated'} for c in REWARD_COMPONENTS}}
    payload['artifact_hash']='pending'; clean=dict(payload); clean.pop('artifact_hash', None); payload['artifact_hash']=canonical_payload_hash(payload); path.write_text(json.dumps(payload,indent=2,sort_keys=True)); return payload['artifact_hash']

def _write_dummy_checkpoint(path, algorithm='four_agent_asynchronous_mappo_rlaif_assignment', scope='assignment'):
    path.parent.mkdir(parents=True, exist_ok=True)
    data={'metadata':{'checkpoint_schema_version':'diagnostic_v1','algorithm':algorithm,'rlaif_scope':scope,'enabled_reward_agents':['assignment'] if scope=='assignment' else [],'training_seed':1,'run_classification':'diagnostic','training_scenario_bank_hash':'diagnostic','resolved_training_config_hash':'diagnostic','code_commit':'diagnostic','reward_checkpoint_hashes':{'assignment':'diagnostic'} if scope=='assignment' else {}}}
    try:
        import torch; torch.save(data,path)
    except Exception: path.write_text(json.dumps(data))

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--output-root',default='results/diagnostic/experiment_matrices'); p.add_argument('--force',action='store_true')
    a=p.parse_args(argv); root=Path(a.output_root)
    if a.force and root.exists(): shutil.rmtree(root)
    root.mkdir(parents=True,exist_ok=True)
    base_env=yaml.safe_load(Path('configs/paper/base_medium.yaml').read_text()) or {}
    scale_path=root/'reward_scales'/'reward_reference_scales.json'; scale_hash=_write_scale(scale_path)
    if isinstance(base_env.get('reward'), dict):
        base_env['reward']['scale_artifact']=str(scale_path); base_env['reward']['scale_artifact_hash']=scale_hash; base_env['reward'].pop('expected_training_scenario_bank_hash', None)
    (root/'diagnostic_base.yaml').write_text(yaml.safe_dump(base_env, sort_keys=False))
    for split,count,start in [('train',2,100),('validation',1,200),('test',2,300)]:
        build_bank(root/'diagnostic_base.yaml', split, count, start, root/'scenarios'/split, fallback=False, run_classification='diagnostic', force=True)
    cfgdir=root/'configs'; cfgdir.mkdir(exist_ok=True)
    for src,name,mode in [('configs/paper/train_mappo_env.yaml','train_mappo_env.yaml','environment_reward'),('configs/paper/train_mappo_rlaif_assignment.yaml','train_mappo_rlaif_assignment.yaml','rlaif_reward')]:
        cfg=yaml.safe_load(Path('configs/paper/train_mappo_env.yaml').read_text()) or {}; overlay=yaml.safe_load(Path(src).read_text()) or {}; cfg.update({k:v for k,v in overlay.items() if k not in {'env','training','networks','output','reward'}}); cfg['run_classification']='diagnostic'; cfg['mode']=mode; cfg.setdefault('env',{})['fallback']=False; cfg['env']['config_path']=str(root/'diagnostic_base.yaml'); cfg.setdefault('training',{}).update({'total_episodes':2,'rollout_episodes':1,'ppo_epochs':1,'batch_size':2,'seed':1}); cfg['reward']={'passenger_delay':1.0,'bus_operating_delay':1.0,'parcel_lateness':1.0,'energy_cost':0.2,'power_overload':1.0,'bus_battery_violation':5.0,'locker_overflow':1.0,'truck_cost':0.0,'undelivered':10.0,'battery_shortage':1.0,'infeasible_action':5.0,'apply_reference_scales':True,'scale_artifact':str(scale_path),'scale_artifact_hash':scale_hash}; cfg.setdefault('output',{})['output_root']=str(root/'runtime'/name)
        if mode=='rlaif_reward':
            r=root/'reward_models'/'reward_assignment.pt'; _write_dummy_checkpoint(r); cfg.setdefault('rlaif',{})['enabled']=True; cfg['rlaif']['scope']='assignment'; cfg['rlaif']['agents']={'assignment': {'enabled': False}}
        (cfgdir/name).write_text(yaml.safe_dump(cfg,sort_keys=False))
    _write_dummy_checkpoint(root/'fixed_policy'/'policy.pt')
    for tmpl,out in [('configs/diagnostic/ablation_matrix.template.yaml','ablation_matrix.resolved.yaml'),('configs/diagnostic/sensitivity_matrix.template.yaml','sensitivity_matrix.resolved.yaml')]:
        Path(root/out).write_text(Path(tmpl).read_text().replace('__OUTPUT_ROOT__',str(root)))
    (root/'diagnostic_manifest.json').write_text(json.dumps({'run_classification':'diagnostic','publication_eligible':False},indent=2)); print(json.dumps({'output_root':str(root)})); return 0
if __name__=='__main__': raise SystemExit(main())
