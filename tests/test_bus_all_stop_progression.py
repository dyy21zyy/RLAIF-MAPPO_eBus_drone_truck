from tests.bus_event_chain_helpers import make_env, run_env

def test_trip_visits_all_stops_in_order(tmp_path):
    env=make_env(tmp_path, stops=6); run_env(env)
    st=env.runtime_trip_states['test_trip_000']
    assert st.visited_stop_indices == list(range(6))
    assert len(st.visited_stop_indices) == len(set(st.visited_stop_indices))
    assert env.ordinary_stops_visited > 0 and env.integrated_stations_visited > 0
