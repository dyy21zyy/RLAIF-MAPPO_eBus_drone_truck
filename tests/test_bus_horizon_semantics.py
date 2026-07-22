from tests.bus_event_chain_helpers import make_env, run_env

def test_360_operation_and_480_delivery_horizons(tmp_path):
    env=make_env(tmp_path, trips=2, stops=6)
    env.trip_stop_times['late']=[dict(env.trip_stop_times['test_trip_000'][0], trip_id='late', arrival_time='360.0', departure_time='360.0')]
    run_env(env)
    assert env.horizon_min == 480.0
    assert 'late' not in env.runtime_trip_states
