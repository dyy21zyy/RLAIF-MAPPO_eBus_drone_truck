from experiments.run_readiness_pilot import classify
from tests.real_readiness_helpers import run_pilot, load

def test_status_classification():
    assert classify({"a":False},"READY")=="NOT_READY"
    assert classify({"a":True},"BLOCKED_MISSING_FORMAL_REWARD_CHECKPOINTS")=="ENV_MAPPO_READY_RLAIF_BLOCKED"

def test_readiness_summary_not_formal_complete(tmp_path):
    s=load(run_pilot(tmp_path),"readiness_summary.json")
    assert s["overall_status"]=="ENV_MAPPO_READY_RLAIF_BLOCKED"
    assert s["overall_status"]!="FORMAL_EXPERIMENTS_COMPLETED"
