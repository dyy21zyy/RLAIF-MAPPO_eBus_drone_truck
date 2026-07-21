import pytest
from experiments.train_policy_matrix import validate_policy_matrix

def test_rlaif_and_non_rlaif_require_different_checkpoints():
    cfg={"policies":[{"name":"mappo_env","checkpoint":"same.pt"},{"name":"mappo_rlaif_assignment","checkpoint":"same.pt","rlaif_enabled":True,"reward_checkpoints":{"assignment":"r.pt"}},{"name":"mappo_rlaif_all","checkpoint":"all.pt","rlaif_enabled":True,"reward_checkpoints":{"assignment":"r.pt"}}]}
    with pytest.raises(ValueError): validate_policy_matrix(cfg)
