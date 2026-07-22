import pytest
from rlaif.reward_registry import RewardRegistry
from tests.rlaif_runtime_test_utils import cfg

def test_formal_fallback_and_partial_full_rejected(tmp_path):
    c=cfg(tmp_path,'assignment',classification='formal',validation='passed'); c['rlaif']['fallback_to_env_reward']=True
    with pytest.raises(ValueError): RewardRegistry(c)
    c=cfg(tmp_path,'all',classification='formal',validation='passed'); c['rlaif']['agents']['bus']['enabled']=False
    with pytest.raises(ValueError): RewardRegistry(c)
