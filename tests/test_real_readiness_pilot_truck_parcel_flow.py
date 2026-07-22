from tests.real_readiness_helpers import run_pilot, load

def test_truck_parcel_flow(tmp_path): assert load(run_pilot(tmp_path),"truck_parcel_flow_validation.json")["passed"]
