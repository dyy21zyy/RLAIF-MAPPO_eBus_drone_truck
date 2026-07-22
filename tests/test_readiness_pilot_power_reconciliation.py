from tests.readiness_test_utils import run_pilot, load
def test_power_reconciliation(tmp_path):
 out=run_pilot(tmp_path); r=load(out,'station_power_reconciliation.json'); assert r['passed']; assert r['peak_load_kw']==65; assert r['overload_kw_min']==150; assert abs(r['bus_charging_energy_kwh']-3.3333333333333335)<1e-9
