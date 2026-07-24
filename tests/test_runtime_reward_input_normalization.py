from rlaif.runtime_agent_reward_model import RuntimeAgentRewardModel
from tests.rlaif_runtime_test_utils import make_ckpt

def test_checkpoint_input_normalization_changes_score(tmp_path):
    p1=make_ckpt(tmp_path/'a.pt', state_mean=[0,0], cand_mean=[0,0])
    p2=make_ckpt(tmp_path/'b.pt', state_mean=[10,10], cand_mean=[10,10])
    a=RuntimeAgentRewardModel.from_checkpoint(p1, expected_agent_type='assignment', formal_mode=False)
    b=RuntimeAgentRewardModel.from_checkpoint(p2, expected_agent_type='assignment', formal_mode=False)
    assert a.score(state_features=[1,2],candidate_features=[3,4],event_type='PARCEL_RELEASE').raw_score != b.score(state_features=[1,2],candidate_features=[3,4],event_type='PARCEL_RELEASE').raw_score

import pytest
from envs.state_builder import BUS_LOADING_FEATURE_NAMES, BUS_CHARGING_FEATURE_NAMES, CANONICAL_BUS_REWARD_STATE_FEATURE_NAMES, COMMON_CANDIDATE_FEATURE_NAMES
from training.reward_model_wrapper import RewardCheckpointCompatibilityError


def test_runtime_bus_accepts_event_specific_raw_state_schemas(tmp_path):
    p = make_ckpt(tmp_path/'bus.pt', agent='bus', state_names=CANONICAL_BUS_REWARD_STATE_FEATURE_NAMES, cand_names=COMMON_CANDIDATE_FEATURE_NAMES)
    rm = RuntimeAgentRewardModel.from_checkpoint(p, expected_agent_type='bus', formal_mode=False)
    cand = [0.1] * len(COMMON_CANDIDATE_FEATURE_NAMES)
    a = rm.score(state_features=[0.1] * len(BUS_LOADING_FEATURE_NAMES), candidate_features=cand, event_type='BUS_TERMINAL_DEPARTURE')
    b = rm.score(state_features=[0.2] * len(BUS_CHARGING_FEATURE_NAMES), candidate_features=cand, event_type='BUS_STATION_ARRIVAL')
    assert a.raw_score == pytest.approx(float(a.raw_score))
    assert b.raw_score == pytest.approx(float(b.raw_score))


def test_runtime_bus_rejects_malformed_raw_event_dimension(tmp_path):
    p = make_ckpt(tmp_path/'bus.pt', agent='bus', state_names=CANONICAL_BUS_REWARD_STATE_FEATURE_NAMES, cand_names=COMMON_CANDIDATE_FEATURE_NAMES)
    rm = RuntimeAgentRewardModel.from_checkpoint(p, expected_agent_type='bus', formal_mode=False)
    with pytest.raises(RewardCheckpointCompatibilityError, match='raw BUS_TERMINAL_DEPARTURE'):
        rm.score(state_features=[0.1] * (len(BUS_LOADING_FEATURE_NAMES) + 1), candidate_features=[0.1] * len(COMMON_CANDIDATE_FEATURE_NAMES), event_type='BUS_TERMINAL_DEPARTURE')


def test_runtime_non_bus_dimension_behavior_unchanged(tmp_path):
    p = make_ckpt(tmp_path/'truck.pt', agent='truck')
    rm = RuntimeAgentRewardModel.from_checkpoint(p, expected_agent_type='truck', formal_mode=False)
    rm.score(state_features=[1, 2], candidate_features=[3, 4], event_type='TRUCK_AVAILABLE')
    with pytest.raises(RewardCheckpointCompatibilityError, match='state_features dimension mismatch'):
        rm.score(state_features=[1, 2, 3], candidate_features=[3, 4], event_type='TRUCK_AVAILABLE')
