from data_pipeline.generate_parcels import calculate_drone_feasible

CFG={"network":{},"drone":{"payload_capacity_kg":5,"service_radius_one_way_km":8,"speed_kmph":40,"maximum_round_trip_duration_min":120,"customer_service_time_min":0}}

def test_one_way_radius_not_round_trip_distance():
    assert calculate_drone_feasible(1,0,0,0,7.9/111.32,CFG)
    assert not calculate_drone_feasible(1,0,0,0,8.1/111.32,CFG)

def test_round_trip_duration_checked_independently():
    cfg={"network":{},"drone":{"payload_capacity_kg":5,"service_radius_one_way_km":8,"speed_kmph":4,"maximum_round_trip_duration_min":120,"customer_service_time_min":0}}
    assert not calculate_drone_feasible(1,0,0,0,7.9/111.32,cfg)
