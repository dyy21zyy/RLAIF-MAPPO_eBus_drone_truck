from utils.config import load_config
from training.config_resolver import resolve_mappo_training_config

def test_environment_formal_config_resolves_every_required_runtime_field():
    r=resolve_mappo_training_config(load_config('configs/paper/train_mappo_env.yaml'))
    assert r['env']['config_path'] and isinstance(r['env']['fallback'], bool)
    for k in ['seed','total_episodes','rollout_episodes','optimizer','lr_actor','lr_critic','gamma','gae_lambda','clip_eps','ppo_epochs','batch_size','entropy_coef','value_coef','max_grad_norm','event_time_reference_min']:
        assert k in r['training']
    for k in ['checkpoint_path','training_log_path','eval_path','resolved_config_path']:
        assert 'seed_1' in r['output'][k]

def test_rlaif_formal_config_resolves_config_only_without_checkpoints():
    r=resolve_mappo_training_config(load_config('configs/paper/train_mappo_rlaif.yaml'))
    assert r['rlaif']['enabled'] is True
    assert set(r['rlaif']['agents']) == {'assignment','truck','bus','station'}

def test_seed_override_replaces_seed_and_paths():
    r=resolve_mappo_training_config(load_config('configs/paper/train_mappo_env.yaml'), seed_override=3)
    assert r['training']['seed']==3 and 'seed_3' in r['output']['checkpoint_path']
