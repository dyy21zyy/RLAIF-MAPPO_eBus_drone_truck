from envs.dynamics.bus_event_chain import BUS_ARRIVE_STOP
from tests.bus_event_chain_helpers import make_env

def test_duplicate_bus_event_guard_records_warning(tmp_path):
    env=make_env(tmp_path); env.reset()
    env._push_bus_event(1.0, BUS_ARRIVE_STOP, {'trip_id':'test_trip_000','physical_bus_id':'bus_000','stop_index':1})
    env._push_bus_event(1.0, BUS_ARRIVE_STOP, {'trip_id':'test_trip_000','physical_bus_id':'bus_000','stop_index':1})
    assert env.bus_invariant_warnings
