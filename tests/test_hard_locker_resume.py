"""Fast, synthetic tests for content-addressed hard-locker resume machinery."""
import json
from pathlib import Path
import pytest

from experiments.run_hard_locker_reexperiment import (
    ExperimentContext, PhaseSpec, atomic_write_json, inspect_resume,
    invalidate_markers_from, marker_is_valid, write_phase_marker,
)


def test_marker_contract_detects_all_identity_and_content_changes(tmp_path):
    source=tmp_path/"input.txt"; output=tmp_path/"output.txt"
    source.write_text("input"); output.write_text("output")
    phase=PhaseSpec(2,"synthetic",(),(output,),(source,))
    marker=tmp_path/"phase_2.json"
    write_phase_marker(marker,phase,commit="abc",plan_digest="plan")
    payload=json.loads(marker.read_text())
    for field in ("phase","phase_name","repository_commit","plan_hash","selected_agent_scope","inputs","outputs","schema_versions","validation_status","completion_timestamp"):
        assert field in payload
    assert marker_is_valid(marker,phase,commit="abc",plan_digest="plan")
    assert not marker_is_valid(marker,phase,commit="other",plan_digest="plan")
    assert not marker_is_valid(marker,phase,commit="abc",plan_digest="other")
    assert not marker_is_valid(marker,phase,commit="abc",plan_digest="plan",selected_agents=("truck",))
    source.write_text("changed"); assert not marker_is_valid(marker,phase,commit="abc",plan_digest="plan")
    source.write_text("input"); output.unlink(); assert not marker_is_valid(marker,phase,commit="abc",plan_digest="plan")


def test_placeholder_config_invalidates_marker(tmp_path):
    config=tmp_path/"config.yaml"; output=tmp_path/"out"
    config.write_text("checkpoint: good\n"); output.write_text("x")
    phase=PhaseSpec(4,"gate",(),(output,),(config,)); marker=tmp_path/"marker"
    write_phase_marker(marker,phase,commit="c",plan_digest="p")
    config.write_text("checkpoint: REPLACE_WITH_REAL_HASH\n")
    assert not marker_is_valid(marker,phase,commit="c",plan_digest="p")


def test_restart_only_removes_markers_and_preserves_artifacts(tmp_path):
    markers=tmp_path/"phase_manifests"; markers.mkdir()
    for number in range(6): (markers/f"phase_{number}.json").write_text("{}")
    checkpoint=tmp_path/"mappo_env"/"seed_1"/"final.pt"; checkpoint.parent.mkdir(parents=True); checkpoint.write_bytes(b"expensive")
    assert invalidate_markers_from(tmp_path,3)==[3,4,5]
    assert all((markers/f"phase_{n}.json").exists() for n in range(3))
    assert checkpoint.read_bytes()==b"expensive"
    with pytest.raises(ValueError,match="invalid phase"): invalidate_markers_from(tmp_path,10)


def test_atomic_manifest_failure_keeps_previous_json(tmp_path,monkeypatch):
    path=tmp_path/"experiment_artifact_manifest.json"; atomic_write_json(path,{"old":True})
    import experiments.run_hard_locker_reexperiment as module
    monkeypatch.setattr(module.os,"replace",lambda *_: (_ for _ in ()).throw(OSError("synthetic")))
    with pytest.raises(OSError): atomic_write_json(path,{"new":True})
    assert json.loads(path.read_text())=={"old":True}


def test_inspection_reports_valid_chain_and_marker(tmp_path):
    artifact=tmp_path/"artifact"; artifact.write_text("ok")
    phase=PhaseSpec(0,"synthetic",(),(artifact,),())
    marker=tmp_path/"phase_manifests"/"phase_0.json"
    write_phase_marker(marker,phase,commit="c",plan_digest="p")
    report=inspect_resume(tmp_path,[phase],commit="c",digest="p")
    assert report["valid_completed_phases"]==[0]
    assert report["can_safely_resume"]


def test_typed_context_exposes_required_resume_fields(tmp_path):
    context=ExperimentContext("commit",tmp_path)
    assert context.repository_commit=="commit"
    assert context.selected_agents==("assignment",)
    assert context.scenario_banks=={} and context.policy_checkpoints=={}
    assert context.resolved_benchmark_config is None and context.final_freeze_artifact is None
