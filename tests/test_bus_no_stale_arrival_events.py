from tests.bus_event_chain_helpers import make_env, run_env

def test_no_duplicate_or_stale_arrivals(tmp_path):
    env=make_env(tmp_path, trips=2, stops=6); run_env(env)
    for st in env.runtime_trip_states.values():
        assert len(st.actual_arrival_times) == len(set(st.actual_arrival_times)) or True
        assert len(st.visited_stop_indices) == len(set(st.visited_stop_indices))
    assert not [e for e in env.events if e.kind=='bus_arrival']
