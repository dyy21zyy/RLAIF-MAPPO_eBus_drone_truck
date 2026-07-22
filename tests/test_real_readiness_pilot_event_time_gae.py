from tests.real_readiness_helpers import run_pilot, load

def test_event_time_gae_real_deltas(tmp_path):
    g=load(run_pilot(tmp_path),"event_time_gae_report.json")
    assert g["passed"] and g["uses_real_transition_times"]
    rows=g["rows"]; assert rows
    assert all(r["delta_time"]>=0 for r in rows)
    assert all(r["event_type_id"] is not None for r in rows)
