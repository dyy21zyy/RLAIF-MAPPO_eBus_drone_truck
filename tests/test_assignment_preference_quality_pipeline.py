import json
from experiments.audit_assignment_preferences import audit_file

def record(pid, reason="Candidate A has lower expected lateness."):
    a={"action_name":"TD","features":{"expected_lateness":1},"feasible":True}
    b={"action_name":"TLD_1","features":{"expected_lateness":2},"feasible":True}
    return {"preference_id":pid,"agent_type":"assignment","event_type":"PARCEL_RELEASE","scenario_id":pid,"state_id":pid,
            "candidate_a_id":"a","candidate_b_id":"b","candidate_a_id_features":a,"candidate_b_id_features":b,
            "original_outcome":"candidate_a","criteria":"expected lateness","reason":reason,"dataset_split":"train",
            "prompt_version":"four_agent_consequence_v1","response_schema_version":"rlaif_preference_json_v1"}

def test_offline_audit_accepts_safe_and_quarantines_unsafe(tmp_path):
    path=tmp_path/"legacy.jsonl"; rows=[record("ok"),record("bad","Candidate A is more energy-efficient.")]
    path.write_text("\n".join(json.dumps(x) for x in rows)+"\n"); before=path.read_bytes()
    accepted,quarantined,report=audit_file(path,expected_count=2)
    assert len(accepted)==len(quarantined)==1 and report["external_api_call_count"]==0
    assert accepted[0]["original_outcome"]=="candidate_a" and accepted[0]["legacy_dataset_split"]=="train"
    assert path.read_bytes()==before and quarantined[0]["quality_status"]=="quarantined"
