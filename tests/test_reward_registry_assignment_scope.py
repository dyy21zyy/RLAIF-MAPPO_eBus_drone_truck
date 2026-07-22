from rlaif.reward_registry import RewardRegistry
from tests.rlaif_runtime_test_utils import cfg

def test_environment_and_assignment_scope(tmp_path):
    assert RewardRegistry({'rlaif':{'enabled':False}}).models == {}
    r=RewardRegistry(cfg(tmp_path,'assignment'))
    assert set(r.models)=={'assignment'}
