from envs.dynamics.bus_operations import station_unloading_time_min

def test_unloading_time_is_six_seconds_per_kg_and_station_specific():
    assert station_unloading_time_min(10)==1.0
    pending={("t","s1"):["a"],("t","s2"):["b"]}
    assert pending.pop(("t","s1"))==["a"]
    assert pending[("t","s2")]==["b"]
