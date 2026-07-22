from tests.bus_event_chain_helpers import make_env, run_env

def test_every_segment_consumes_energy_once(tmp_path):
    env=make_env(tmp_path, trips=2, stops=6); run_env(env)
    assert env.bus_segment_count == 10
    expected=sum(env._segment(t,i,i+1)[1]*env.config['bus'].get('bus_energy_kwh_per_km',1.6) for t in env.runtime_trip_states for i in range(5))
    assert abs(env.bus_propulsion_energy_kwh-expected) < 1e-6
