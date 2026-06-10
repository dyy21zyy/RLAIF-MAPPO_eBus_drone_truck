"""Stage 7 asynchronous control and reward integration tests."""
from pathlib import Path
import pytest
from training.mappo_async import transition_reward, validate_decision
from training.reward_model_wrapper import RewardModelCheckpointError, RewardModelWrapper
from utils.config import load_config
ROOT=Path(__file__).parents[1]

def test_config_loading():
    config=load_config(ROOT/'configs/train_mappo_async.yaml')
    assert config['training']['gamma']==.99 and config['rlaif']['enabled'] is False

def test_agent_event_pairs_are_asynchronous():
    validate_decision('assignment','PARCEL_ARRIVAL'); validate_decision('bus','BUS_ARRIVAL')
    with pytest.raises(ValueError): validate_decision('bus','PARCEL_ARRIVAL')

def test_disabled_rlaif_needs_no_checkpoint(tmp_path):
    wrapper=RewardModelWrapper(tmp_path/'missing.pt',enabled=False)
    assert transition_reward('assignment',2.0,wrapper)==(2.0,0.0)

def test_enabled_rlaif_requires_real_checkpoint(tmp_path):
    with pytest.raises(RewardModelCheckpointError): RewardModelWrapper(tmp_path/'missing.pt',enabled=True)

def test_rlaif_only_assignment_and_never_bus():
    class Wrapper:
        enabled=True
        def score(self,*args): return 3.0
    assert transition_reward('assignment',2.0,Wrapper(),lambda_rlaif=.5,state_features=[0],action_features=[0],action_id=0)==(3.5,3.0)
    assert transition_reward('bus',2.0,Wrapper(),lambda_rlaif=99)==(2.0,0.0)

def test_smoke_contract():
    from experiments.smoke_test_mappo_async import run_smoke_test
    result=run_smoke_test()
    assert result['skipped'] or (result['assignment_transitions']>0 and result['bus_transitions']>0)
