from tests.bus_event_chain_helpers import make_env, run_env

def test_runtime_trace_contains_required_stop_fields(tmp_path):
    env=make_env(tmp_path); run_env(env)
    row=env.bus_trace.as_dicts()[0]
    for key in ['physical_bus_id','trip_id','stop_index','stop_id','actual_arrival','actual_departure','soc_at_departure']:
        assert key in row
