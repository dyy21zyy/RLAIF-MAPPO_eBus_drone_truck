from tests.bus_event_chain_helpers import make_env, run_env

def test_terminal_loading_delay_shifts_first_arrival(tmp_path):
    env=make_env(tmp_path, stops=6); run_env(env)
    st=env.runtime_trip_states['test_trip_000']; run,_=env._segment('test_trip_000',0,1)
    assert st.actual_arrival_times[1] == st.actual_departure_times[0] + run
