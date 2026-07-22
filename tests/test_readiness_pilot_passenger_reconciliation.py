from tests.readiness_test_utils import run_pilot, load
def test_passenger_reconciliation(tmp_path):
 out=run_pilot(tmp_path); r=load(out,'passenger_reconciliation.json'); assert r['passed']; assert r['waiting_passenger_minutes']==30; assert r['onboard_additional_delay_passenger_minutes']==35
