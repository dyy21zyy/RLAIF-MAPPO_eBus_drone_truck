from tests.real_readiness_helpers import run_pilot, load

def test_reward_reconciliation(tmp_path): assert load(run_pilot(tmp_path),"reward_reconciliation.json")["passed"]
