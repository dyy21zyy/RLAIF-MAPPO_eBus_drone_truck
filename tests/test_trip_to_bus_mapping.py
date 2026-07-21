from envs.dynamics.bus_circulation import build_trip_to_bus_mapping

def _sts(ids):
    return [x for tid,start in ids for x in [{"trip_id":tid,"stop_sequence":0,"departure_time":start,"arrival_time":start},{"trip_id":tid,"stop_sequence":1,"departure_time":start+10,"arrival_time":start+10}]]

def test_every_trip_maps_once_and_is_deterministic():
    trips=[{"trip_id":f"t{i}","start_time":i*20} for i in range(3)]
    a=build_trip_to_bus_mapping(trips,_sts([(t["trip_id"],t["start_time"]) for t in trips]),1,5,2)
    b=build_trip_to_bus_mapping(trips,_sts([(t["trip_id"],t["start_time"]) for t in trips]),1,5,2)
    assert [r["trip_id"] for r in a] == ["t0","t1","t2"]
    assert len({r["trip_id"] for r in a}) == len(trips)
    assert a == b

def test_one_bus_serves_multiple_non_overlapping_trips():
    trips=[{"trip_id":"t0","start_time":0},{"trip_id":"t1","start_time":20}]
    rows=build_trip_to_bus_mapping(trips,_sts([("t0",0),("t1",20)]),1,5,2)
    assert {r["bus_id"] for r in rows} == {"bus_000"}
    assert rows[0]["next_trip_id"] == "t1"
