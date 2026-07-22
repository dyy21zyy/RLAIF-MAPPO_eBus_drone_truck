import pytest
from rlaif.reward_model_dataset import build_reward_pair_dataset
from tests.preference_v3_fixtures import rec

def bus(event,p): return rec(preference_id=p,agent_type='bus',event_type=event,original_candidate_a_id='a'+p,original_candidate_b_id='b'+p,displayed_first_candidate_id='a'+p,displayed_second_candidate_id='b'+p)
def test_bus_accepts_terminal_loading_rows(): assert len(build_reward_pair_dataset([bus('BUS_TERMINAL_DEPARTURE','1')],agent_type='bus'))==1
def test_bus_accepts_station_charging_rows(): assert len(build_reward_pair_dataset([bus('BUS_STATION_ARRIVAL','1')],agent_type='bus'))==1
def test_bus_rejects_unrelated_events():
    with pytest.raises(ValueError): build_reward_pair_dataset([rec(agent_type='bus',event_type='PARCEL_RELEASE')],agent_type='bus')
def test_formal_bus_validation_fails_when_missing_required_event():
    with pytest.raises(ValueError): build_reward_pair_dataset([bus('BUS_TERMINAL_DEPARTURE','1')],agent_type='bus',require_bus_event_coverage=True)
