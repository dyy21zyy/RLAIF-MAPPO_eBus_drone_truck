from tests.real_readiness_helpers import run_pilot, load

def test_critic_update(tmp_path):
    c=load(run_pilot(tmp_path),"mappo_update_report.json")["critic"]
    assert c["passed"] and c["critic_transition_count"]>0 and c["changed_parameter_count"]>0 and c["parameter_delta_norm"]>0
