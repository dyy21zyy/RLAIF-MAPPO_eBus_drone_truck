import json, copy, math
from pathlib import Path
import pytest, yaml, torch

from evaluation.preformal_gate import PreformalGate
from experiments.prepare_formal_experiment_inputs import _write_resolved_configs
from training.config_resolver import resolve_mappo_training_config
from training.ppo_trainer import save_assignment_checkpoint, load_assignment_checkpoint
from training.assignment_ppo import AssignmentActorCritic

SHARED = ["env.config_path","env.fallback","training.total_episodes","training.rollout_episodes","training.optimizer","training.lr_actor","training.lr_critic","training.gamma","training.gae_lambda","training.clip_eps","training.ppo_epochs","training.batch_size","training.entropy_coef","training.value_coef","training.max_grad_norm","training.event_time_reference_min","networks","reward"]

def get(d,path):
    for p in path.split('.'):
        d=d[p]
    return d

def enabled(cfg):
    return [a for a,c in cfg.get('rlaif',{}).get('agents',{}).items() if c.get('enabled')]

def test_complete_rlaif_configs_share_mappo_fields():
    env=yaml.safe_load(Path('configs/paper/train_mappo_env.yaml').read_text())
    ass=yaml.safe_load(Path('configs/paper/train_mappo_rlaif_assignment.yaml').read_text())
    allc=yaml.safe_load(Path('configs/paper/train_mappo_rlaif_all.yaml').read_text())
    assert enabled(env)==[]
    assert enabled(ass)==['assignment']
    assert set(enabled(allc))=={'assignment','truck','bus','station'}
    assert allc['rlaif']['agents']['bus']['supported_event_types']==['BUS_TERMINAL_DEPARTURE','BUS_STATION_ARRIVAL']
    for cfg in (ass,allc):
        assert cfg['run_classification']=='formal' and cfg['mode']=='rlaif_reward'
        assert cfg['env']['fallback'] is False
        assert cfg['rlaif']['fallback_to_env_reward'] is False
        assert cfg['rlaif']['fail_on_invalid_reward_model'] is True
        for key in SHARED:
            assert get(cfg,key)==get(env,key), key
        resolve_mappo_training_config(cfg)

def test_formal_assignment_ppo_config_contract():
    cfg=yaml.safe_load(Path('configs/paper/train_assignment_ppo.yaml').read_text())
    assert cfg['run_classification']=='formal'
    assert cfg['algorithm']=='assignment_ppo'
    assert cfg['env']['fallback'] is False
    assert cfg['training']['training_seeds']==[1,2,3]
    assert cfg['training']['total_episodes']==3000
    assert cfg['env']['train_scenario_bank_manifest'] != cfg['env'].get('test_scenario_bank_manifest')
    assert set(cfg['fixed_baseline_policies'])=={'truck','bus','station'}

def test_runtime_config_writer_rejects_placeholders(tmp_path):
    banks={s:{'bank_hash':('a'*64 if s=='train' else 'b'*64 if s=='validation' else 'c'*64),'scenario_count':1} for s in ['train','validation','test']}
    scale=tmp_path/'scale.json'; scale.write_text(json.dumps({'scales':{'x':1},'training_scenario_bank_hash':'a'*64}))
    written=_write_resolved_configs(tmp_path,banks,scale,'d'*64,{})
    for p in written.values():
        txt=Path(p).read_text()
        assert 'REPLACE_WITH' not in txt and 'PLACEHOLDER' not in txt and 'TBD' not in txt

def test_assignment_checkpoint_lineage_and_reload(tmp_path):
    cfg=yaml.safe_load(Path('configs/paper/train_assignment_ppo.yaml').read_text())
    cfg['env']['expected_train_bank_hash']='a'*64; cfg['env']['expected_validation_bank_hash']='b'*64
    cfg['training']['seed']=7
    model=AssignmentActorCritic(3,2,[4])
    opt=torch.optim.Adam(model.parameters(), lr=1e-3)
    path=tmp_path/'ckpt.pt'
    save_assignment_checkpoint(path, model, opt, cfg)
    loaded, ckpt=load_assignment_checkpoint(path)
    assert loaded.action_dim==2
    assert ckpt['algorithm']=='assignment_ppo'
    assert ckpt['lineage']['algorithm_identity']=='assignment_ppo'
    assert ckpt['lineage']['training_seed']==7
    assert ckpt['lineage']['train_bank_hash']=='a'*64
    assert ckpt['lineage']['validation_bank_hash']=='b'*64

def test_subprocess_failure_propagates_to_overall(tmp_path):
    cfg={'run_classification':'formal','experiment_stage':'preformal','publication_eligible':False,'output_root':str(tmp_path/'results/formal/gate'),'commands':{'repository_verification':['python','-c','import sys; sys.exit(3)']}}
    report=PreformalGate(cfg).run()
    assert report['stage_statuses']['repository_verification']=='failed'
    assert report['overall_status'].startswith('BLOCKED')
