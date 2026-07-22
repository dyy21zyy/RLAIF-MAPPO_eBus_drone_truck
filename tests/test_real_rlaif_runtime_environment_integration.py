import math
from rlaif.reward_registry import RewardRegistry
from tests.rlaif_runtime_test_utils import cfg

def test_four_smoke_models_score_real_decision_event_shapes(tmp_path):
    r=RewardRegistry(cfg(tmp_path,'all'))
    events={'assignment':'PARCEL_RELEASE','truck':'TRUCK_AVAILABLE','bus':'BUS_TERMINAL_DEPARTURE','station':'STATION_OPERATION'}
    for a,e in events.items():
        c=r.score_transition(agent_type=a,event_type=e,environment_reward=1.0,state_features=[0.1,0.2],candidate_features=[0.3,0.4])
        assert math.isfinite(c.raw_learned_reward) and math.isfinite(c.normalized_learned_reward)
        assert abs(c.clipped_learned_reward) <= .5 and not c.used_fallback and c.total_reward == c.environment_reward + c.weighted_learned_contribution
    c=r.score_transition(agent_type='bus',event_type='BUS_STATION_ARRIVAL',environment_reward=1.0,state_features=[0.1,0.2],candidate_features=[0.3,0.4])
    assert math.isfinite(c.raw_learned_reward)
