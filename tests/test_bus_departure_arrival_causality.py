from tests.bus_event_chain_helpers import make_env, run_env

def test_arrivals_follow_actual_departures(tmp_path):
    env=make_env(tmp_path, stops=6); run_env(env)
    st=env.runtime_trip_states['test_trip_000']
    for i in range(5):
        run,_=env._segment('test_trip_000', i, i+1)
        assert st.actual_arrival_times[i+1] == st.actual_departure_times[i] + run
