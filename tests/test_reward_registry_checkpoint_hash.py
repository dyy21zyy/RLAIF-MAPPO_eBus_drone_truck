import pytest
from rlaif.reward_registry import RewardRegistry
from training.reward_model_wrapper import RewardCheckpointCompatibilityError
from tests.rlaif_runtime_test_utils import cfg

def test_hash_mismatch_fails(tmp_path):
    c=cfg(tmp_path,'assignment'); c['rlaif']['agents']['assignment']['checkpoint_hash']='bad'
    with pytest.raises(RewardCheckpointCompatibilityError): RewardRegistry(c)
