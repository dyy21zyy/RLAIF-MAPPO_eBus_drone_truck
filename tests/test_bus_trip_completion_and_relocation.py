from tests.bus_event_chain_helpers import make_env, run_env

def test_trip_completion_relocation_layover_and_soc_persistence(tmp_path):
    env=make_env(tmp_path, trips=2, stops=6); run_env(env)
    a=env.runtime_trip_states['test_trip_000']; b=env.runtime_trip_states['test_trip_001']; bus=env.physical_buses['bus_000']
    assert a.completed and b.completed
    assert env.bus_relocation_energy_kwh > 0
    assert bus.next_available_time_min >= b.completion_time_min
    assert bus.soc_kwh < 150.0
