from tests.real_readiness_helpers import run_pilot, load

def test_rlaif_missing_formal_blocks_only_rlaif(tmp_path):
    out=run_pilot(tmp_path); r=load(out,"rlaif_artifact_report.json")
    assert r["status"]=="BLOCKED_MISSING_FORMAL_REWARD_CHECKPOINTS"
    assert r["smoke_checkpoints_rejected_as_formal"]
