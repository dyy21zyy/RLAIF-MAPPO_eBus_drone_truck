from envs.dynamics.bus_circulation import calculate_physical_fleet_size

def test_fleet_size_matches_cycle_formula():
    trips=[{"trip_id":"t0","start_time":0},{"trip_id":"t1","start_time":10}]
    sts=[{"trip_id":"t0","stop_sequence":0,"departure_time":0,"arrival_time":0},{"trip_id":"t0","stop_sequence":1,"departure_time":20,"arrival_time":20},{"trip_id":"t1","stop_sequence":0,"departure_time":10,"arrival_time":10},{"trip_id":"t1","stop_sequence":1,"departure_time":30,"arrival_time":30}]
    r=calculate_physical_fleet_size(trips,sts,10,5,2)
    assert r["nominal_cycle_time_min"] == 27
    assert r["physical_bus_count"] == 3
