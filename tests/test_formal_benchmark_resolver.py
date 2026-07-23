import json, pytest
from pathlib import Path
from experiments import resolve_formal_benchmark as r


def _write_test_bank(root: Path, bank_hash: str = "canonical-test-bank") -> Path:
    d=root/"scenarios"/"test"; d.mkdir(parents=True, exist_ok=True)
    p=d/"scenario_bank_manifest.json"
    p.write_text(json.dumps({"split":"test","bank":"test","scenario_count":100,"bank_hash":bank_hash,"scenarios":[]})+"\n")
    return p


def test_missing_policy_checkpoint_failure_lists_method_seed(tmp_path, monkeypatch):
    monkeypatch.setattr(r, '_validate_reward', lambda root, agent: {'path':str(root/f'reward_{agent}.pt'),'hash':'h','agent_type':agent,'supported_event_types':[]})
    with pytest.raises(RuntimeError, match='assignment_ppo seed 1'):
        r.resolve(Path('configs/paper/benchmark.yaml'), tmp_path, tmp_path/'b.yaml')
    assert not (tmp_path/'b.yaml').exists()


def test_benchmark_resolution_with_three_seeds_and_rewards(tmp_path, monkeypatch):
    def fake_validate(spec, path):
        seed=int(str(path).split('_seed_')[-1].split('.')[0])
        return {'checkpoint_hash':'h'+str(path), 'algorithm':spec.expected_algorithm, 'training_seed':seed, 'training_scenario_bank_hash':'tr','validation_scenario_bank_hash':'va','reward_scale_hash':'rs','rlaif_scope':spec.expected_rlaif_scope,'enabled_reward_agents':spec.enabled_reward_agents,'reward_checkpoint_hashes':{a:'rh' for a in spec.enabled_reward_agents},'observation_schema':'obs','action_schema':'act','run_classification':'formal','checkpoint_schema_version':4,'validation_status':'passed','resolved_training_config_hash':'cfg','code_commit':'c'}
    monkeypatch.setattr(r, 'validate_policy_checkpoint', fake_validate)
    monkeypatch.setattr(r, '_validate_reward', lambda root, agent: {'path':str(root/f'reward_{agent}.pt'),'hash':'rh','agent_type':agent,'supported_event_types':['BUS_TERMINAL_DEPARTURE','BUS_STATION_ARRIVAL'] if agent=='bus' else []})
    _write_test_bank(tmp_path)
    for method in r.LEARNED_METHODS:
        for seed in (1,2,3): (tmp_path/f'{method}_seed_{seed}.pt').write_text('x')
    out=tmp_path/'benchmark.yaml'; res=r.resolve(Path('configs/paper/benchmark.yaml'), tmp_path, out)
    text=out.read_text(); assert 'REPLACE_WITH' not in text
    data=__import__('yaml').safe_load(text)
    learned=[m for m in data['methods'] if m['method_id'] in r.LEARNED_METHODS]
    assert len(learned)==4 and all(set(m['policy_checkpoints'])=={'1','2','3'} for m in learned)
    assert res['training_seeds']==[1,2,3]


def test_benchmark_missing_test_bank_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(r, '_validate_reward', lambda root, agent: {'path':str(root/f'reward_{agent}.pt'),'hash':'h','agent_type':agent,'supported_event_types':[]})
    def fake_validate(spec, path):
        return {'checkpoint_hash':str(path), 'training_seed':1}
    monkeypatch.setattr(r, 'validate_policy_checkpoint', fake_validate)
    for method in r.LEARNED_METHODS:
        for seed in (1,2,3): (tmp_path/f'{method}_seed_{seed}.pt').write_text('x')
    with pytest.raises(RuntimeError, match='missing formal test scenario-bank manifest'):
        r.resolve(Path('configs/paper/benchmark.yaml'), tmp_path, tmp_path/'b.yaml')
    assert not (tmp_path/'b.yaml').exists()

def test_benchmark_uses_canonical_test_bank_hash(tmp_path, monkeypatch):
    def fake_validate(spec, path):
        seed=int(str(path).split('_seed_')[-1].split('.')[0])
        return {'checkpoint_hash':'h'+str(path), 'algorithm':spec.expected_algorithm, 'training_seed':seed, 'training_scenario_bank_hash':'tr','validation_scenario_bank_hash':'va','reward_scale_hash':'rs','rlaif_scope':spec.expected_rlaif_scope,'enabled_reward_agents':spec.enabled_reward_agents,'reward_checkpoint_hashes':{},'observation_schema':'obs','action_schema':'act'}
    monkeypatch.setattr(r, 'validate_policy_checkpoint', fake_validate)
    monkeypatch.setattr(r, '_validate_reward', lambda root, agent: {'path':str(root/f'reward_{agent}.pt'),'hash':'rh','agent_type':agent,'supported_event_types':[]})
    manifest=_write_test_bank(tmp_path, 'canonical-bank-hash')
    for method in r.LEARNED_METHODS:
        for seed in (1,2,3): (tmp_path/f'{method}_seed_{seed}.pt').write_text('x')
    r.resolve(Path('configs/paper/benchmark.yaml'), tmp_path, tmp_path/'b.yaml')
    data=__import__('yaml').safe_load((tmp_path/'b.yaml').read_text())
    assert data['scenario_bank']['expected_bank_hash'] == 'canonical-bank-hash'
    assert data['scenario_bank']['manifest_file_hash'] == r.sha256_file(manifest)
