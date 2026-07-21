from tests.test_station_dispatch_matching import env
from envs.action_generators.station_actions import generate_station_operation_candidates

def test_candidate_generation_side_effect_free_bounded_idle():
    e=env(); before=[b.status for b in e.stations['s'].battery_states]
    c=generate_station_operation_candidates(e,'s',max_candidates=5)
    assert len(c)<=5; assert any(x.idle_flag for x in c); assert [b.status for b in e.stations['s'].battery_states]==before
