from tests.real_readiness_helpers import run_pilot, load

def test_passenger_reconciliation(tmp_path): assert load(run_pilot(tmp_path),"passenger_reconciliation.json")["passed"]
