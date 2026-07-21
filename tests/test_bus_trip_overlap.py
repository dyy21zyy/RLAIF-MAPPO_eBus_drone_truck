import pytest
from envs.dynamics.bus_circulation import build_trip_to_bus_mapping, assert_no_overlaps

def test_overlapping_generation_rejected():
    trips=[{"trip_id":"t0","start_time":0},{"trip_id":"t1","start_time":11}]
    sts=[{"trip_id":"t0","stop_sequence":0,"departure_time":0,"arrival_time":0},{"trip_id":"t0","stop_sequence":1,"departure_time":10,"arrival_time":10},{"trip_id":"t1","stop_sequence":0,"departure_time":11,"arrival_time":11},{"trip_id":"t1","stop_sequence":1,"departure_time":21,"arrival_time":21}]
    with pytest.raises(ValueError): build_trip_to_bus_mapping(trips,sts,1,5,2)

def test_overlap_validator_rejects_manual_mapping():
    with pytest.raises(ValueError): assert_no_overlaps([{"bus_id":"b","trip_id":"a","scheduled_start_min":0,"scheduled_end_min":10,"relocation_time_min":5,"minimum_layover_min":2},{"bus_id":"b","trip_id":"c","scheduled_start_min":11,"scheduled_end_min":20,"relocation_time_min":5,"minimum_layover_min":2}])
