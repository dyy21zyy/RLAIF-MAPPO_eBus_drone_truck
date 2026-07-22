from rlaif.reward_registry import RewardRegistry
from tests.rlaif_runtime_test_utils import cfg

def test_full_scope_loads_four(tmp_path):
    r=RewardRegistry(cfg(tmp_path,'all'))
    assert set(r.models)=={'assignment','truck','bus','station'}
