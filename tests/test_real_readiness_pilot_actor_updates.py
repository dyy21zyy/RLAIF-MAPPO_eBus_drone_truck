from tests.real_readiness_helpers import run_pilot, load

def test_all_actors_update_from_real_transitions(tmp_path):
    r=load(run_pilot(tmp_path),"mappo_update_report.json")
    assert r["passed"] and r["advantages_finite"] and r["returns_finite"]
    for row in r["agents"].values():
        assert row["transition_count"]>0 and row["changed_parameter_count"]>0 and row["total_parameter_delta_norm"]>0 and row["gradient_norm"]>=0
