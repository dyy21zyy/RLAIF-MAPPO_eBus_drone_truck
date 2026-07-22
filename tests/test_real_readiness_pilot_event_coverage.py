from tests.real_readiness_helpers import run_pilot, load, jsonl

def test_required_real_event_coverage(tmp_path):
    out=run_pilot(tmp_path)
    cov=load(out,"readiness_event_coverage.json")
    assert cov["passed"]
    assert all(v>0 for v in cov["event_counts"].values())
    assert cov["automatic_events_create_no_transitions"]
    assert any(r.get("transition_id") for r in jsonl(out,"event_trace.jsonl"))
