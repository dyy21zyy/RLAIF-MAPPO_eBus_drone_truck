from tests.real_readiness_helpers import run_pilot, load

def test_bus_trace_validation(tmp_path):
    r=load(run_pilot(tmp_path),"bus_trace_validation.json")
    assert r["passed"] and r["ordinary_stops_visited"]>0 and r["integrated_stations_visited"]>0 and r["causal_order"]
