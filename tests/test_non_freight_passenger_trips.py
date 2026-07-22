from tests.bus_event_chain_helpers import make_env, run_env

def test_non_freight_trip_operates_without_loading_decision(tmp_path):
    env=make_env(tmp_path, trips=2, stops=6); seen=run_env(env)
    st=env.runtime_trip_states['test_trip_001']
    assert st.completed and st.visited_stop_indices == list(range(6))
    assert env.non_freight_trips_completed == 1
    assert env.bus_propulsion_energy_kwh > 0
    assert not any(o.get('event_type_detail')=='BUS_TERMINAL_DEPARTURE' and o['entity_id']=='test_trip_001' for o in seen)
