from __future__ import annotations
import json, math
from pathlib import Path
import yaml
import pytest

from envs.reward_components import REWARD_COMPONENTS
from evaluation.preformal_gate import PreformalGate
from experiments import prepare_formal_experiment_inputs as prep
from training.ppo_trainer import load_assignment_checkpoint, save_assignment_checkpoint
from training.assignment_ppo import AssignmentActorCritic


def load_yaml(path):
    return yaml.safe_load(Path(path).read_text()) or {}


def _shared(cfg):
    return {k: cfg[k] for k in ('env','training','networks','reward') if k in cfg}


def test_formal_mappo_rlaif_configs_complete_and_scoped():
    env = load_yaml('configs/paper/train_mappo_env.yaml')
    assn = load_yaml('configs/paper/train_mappo_rlaif_assignment.yaml')
    allc = load_yaml('configs/paper/train_mappo_rlaif_all.yaml')
    for cfg in (assn, allc):
        for section in ('run_classification','mode','env','training','networks','output','reward','rlaif'):
            assert section in cfg
        assert cfg['run_classification'] == 'formal'
        assert cfg['mode'] == 'rlaif_reward'
        assert cfg['env']['fallback'] is False
        assert cfg['rlaif']['enabled'] is True
        assert cfg['rlaif']['fallback_to_env_reward'] is False
        assert cfg['rlaif']['fail_on_invalid_reward_model'] is True
        assert cfg['training'] == env['training']
        assert cfg['networks'] == env['networks']
        assert cfg['reward'] == env['reward']
    assert env['rlaif']['enabled'] is False
    assert [a for a,r in assn['rlaif']['agents'].items() if r.get('enabled')] == ['assignment']
    assert set(a for a,r in allc['rlaif']['agents'].items() if r.get('enabled')) == {'assignment','truck','bus','station'}
    ckpts = [r['checkpoint'] for r in allc['rlaif']['agents'].values() if r.get('enabled')]
    assert len(ckpts) == len(set(ckpts))
    assert set(allc['rlaif']['agents']['bus']['supported_event_types']) == {'BUS_TERMINAL_DEPARTURE','BUS_STATION_ARRIVAL'}


def test_formal_assignment_ppo_config_contract():
    cfg = load_yaml('configs/paper/train_assignment_ppo.yaml')
    assert cfg['run_classification'] == 'formal'
    assert cfg['algorithm'] == 'assignment_ppo'
    assert cfg['env']['config_path'] == 'configs/paper/base_medium.yaml'
    assert cfg['env']['fallback'] is False
    assert cfg['training']['training_seeds'] == [1,2,3]
    assert cfg['training']['total_episodes'] == 3000
    assert 'test_scenario_bank_manifest' in cfg['env']
    assert cfg['env']['train_scenario_bank_manifest'] != cfg['env']['test_scenario_bank_manifest']
    assert set(cfg['fixed_baseline_policies']) == {'truck','bus','station'}


def test_runtime_config_resolution_rejects_placeholders(tmp_path):
    m = {s:{'bank_hash':f'{s}hash','scenario_count':1} for s in ('train','validation','test')}
    scale = {'artifact_hash':'scalehash','training_scenario_bank_hash':'trainhash'}
    out = tmp_path/'results/formal'
    dst = prep._resolve_config(Path('configs/paper/train_mappo_env.yaml'), out/'cfg.yaml', m, scale, out, {})
    text = dst.read_text()
    assert not prep.PLACEHOLDER_RE.search(text)
    assert 'trainhash' in text and 'scalehash' in text


def test_reward_scale_validation_train_lineage(tmp_path):
    path = tmp_path/'scale.json'
    components = {c:{'scale':1.0,'status':'observed_positive'} for c in REWARD_COMPONENTS}
    path.write_text(json.dumps({'artifact_hash':'abc123','training_scenario_bank_hash':'trainhash','components':components}))
    assert prep._validate_reward_scale(path, 'trainhash')['artifact_hash'] == 'abc123'
    with pytest.raises(ValueError):
        prep._validate_reward_scale(path, 'other')


def test_preformal_subprocess_failure_propagates(tmp_path):
    cfg = {'run_classification':'formal','experiment_stage':'preformal','publication_eligible':False,'output_root':str(tmp_path/'results/preformal/gate'), 'commands': {'repository_verification':['python','-c','import sys; sys.exit(7)']}}
    report = PreformalGate(cfg).run()
    assert report['stage_statuses']['repository_verification'] == 'failed'
    assert report['overall_status'] in {'BLOCKED_SCENARIO_BANK','BLOCKED_REWARD_SCALE','BLOCKED_REWARD_MODELS','BLOCKED_POLICY_TRAINING','BLOCKED_BENCHMARK'}
    assert report['required_stage_failures']


def test_assignment_ppo_checkpoint_lineage_and_reload(tmp_path):
    torch = pytest.importorskip('torch')
    model = AssignmentActorCritic(3, 2, [4])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    train = tmp_path/'train.json'; val = tmp_path/'val.json'; envcfg = tmp_path/'env.yaml'
    train.write_text('{}'); val.write_text('{}'); envcfg.write_text('env: true')
    cfg = {'env': {'config_path': str(envcfg), 'train_scenario_bank_manifest': str(train), 'expected_train_bank_hash': 'trainhash', 'validation_scenario_bank_manifest': str(val), 'expected_validation_bank_hash': 'valhash'}, 'training': {'seed': 2, 'total_episodes': 1}, 'policy': {'hidden_dims': [4]}, 'bus_baseline': {'name':'uniform_30'}}
    ckpt = tmp_path/'policy.pt'
    save_assignment_checkpoint(ckpt, model, opt, cfg)
    loaded, payload = load_assignment_checkpoint(ckpt)
    assert loaded.action_dim == 2
    lineage = payload['lineage']
    assert lineage['algorithm_identity'] == 'assignment_ppo'
    assert lineage['training_seed'] == 2
    assert lineage['train_bank_hash'] == 'trainhash'
    assert lineage['validation_bank_hash'] == 'valhash'
    assert lineage['code_commit']
