from tests.real_readiness_helpers import run_pilot, load

def test_power_reconciliation(tmp_path): assert load(run_pilot(tmp_path),"station_power_reconciliation.json")["passed"]
