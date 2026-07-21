from utils.config import load_config
from experiments.train_policy_matrix import build_training_seed_matrix

def test_three_training_seeds_produce_distinct_outputs():
    m=build_training_seed_matrix(load_config('configs/paper/train_mappo_env.yaml'))
    assert [r['seed'] for r in m['runs']] == [1,2,3]
    paths=[r[k] for r in m['runs'] for k in ('checkpoint_path','training_log_path','eval_path','resolved_config_path')]
    assert len(paths)==len(set(paths))
