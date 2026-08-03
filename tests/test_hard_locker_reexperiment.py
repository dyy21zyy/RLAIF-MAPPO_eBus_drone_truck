import json
from pathlib import Path
import pytest
from experiments.run_hard_locker_reexperiment import (
    ExperimentContext,
    PhaseSpec,
    _scenario_banks,
    build_plan,
    plan_hash,
    restore_phase_0_context,
    validate_plan,
)


SPLITS = ("train", "validation", "test")


def _write_bank(root: Path, relative: str, split: str) -> Path:
    path = root / relative.format(split=split)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"split": split, "bank_hash": f"{split}-hash"}))
    return path


def _write_layout(root: Path, relative: str) -> dict[str, Path]:
    return {split: _write_bank(root, relative, split) for split in SPLITS}


def test_scenario_banks_select_canonical_layout(monkeypatch, tmp_path):
    expected = _write_layout(
        tmp_path, "results/formal/scenarios/{split}/scenario_bank_manifest.json"
    )
    monkeypatch.chdir(tmp_path)
    assert _scenario_banks() == {split: path.relative_to(tmp_path) for split, path in expected.items()}


def test_scenario_banks_prefer_canonical_manifest_over_aliases(monkeypatch, tmp_path):
    canonical = _write_layout(
        tmp_path, "results/formal/scenarios/{split}/scenario_bank_manifest.json"
    )
    _write_layout(tmp_path, "results/formal/scenarios/{split}/manifest.json")
    _write_layout(tmp_path, "results/formal/scenario_banks/{split}/manifest.json")
    monkeypatch.chdir(tmp_path)
    assert _scenario_banks() == {
        split: path.relative_to(tmp_path) for split, path in canonical.items()
    }


@pytest.mark.parametrize(
    "relative",
    (
        "results/formal/scenarios/{split}/manifest.json",
        "results/formal/scenario_banks/{split}/manifest.json",
    ),
)
def test_scenario_banks_accept_manifest_aliases(monkeypatch, tmp_path, relative):
    expected = _write_layout(tmp_path, relative)
    monkeypatch.chdir(tmp_path)
    assert _scenario_banks() == {split: path.relative_to(tmp_path) for split, path in expected.items()}


def test_scenario_banks_missing_split_lists_every_candidate(monkeypatch, tmp_path):
    for split in ("train", "validation"):
        _write_bank(
            tmp_path,
            "results/formal/scenarios/{split}/scenario_bank_manifest.json",
            split,
        )
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError) as caught:
        _scenario_banks()
    message = str(caught.value)
    assert "split 'test'" in message
    for candidate in (
        "results/formal/scenarios/test/scenario_bank_manifest.json",
        "results/formal/scenarios/test/manifest.json",
        "results/formal/scenario_banks/test/manifest.json",
        "results/formal/scenario_banks/test/scenario_bank_manifest.json",
        "data/scenarios/test/scenario_bank_manifest.json",
        "data/scenarios/test/manifest.json",
    ):
        assert candidate in message
    assert "must be generated or supplied" in message


def test_phase_zero_rejects_wrong_scenario_bank_split(tmp_path):
    paths = _write_layout(tmp_path, "banks/{split}/manifest.json")
    paths["test"].write_text(json.dumps({"split": "validation", "bank_hash": "hash"}))
    context = ExperimentContext("commit", tmp_path / "run")
    with pytest.raises(ValueError, match="expected 'test'"):
        restore_phase_0_context(context, paths)


def test_dry_run_plan_uses_canonical_formal_manifests(monkeypatch, tmp_path):
    _write_layout(tmp_path, "results/formal/scenarios/{split}/scenario_bank_manifest.json")
    monkeypatch.chdir(tmp_path)
    plan = build_plan(output_root=Path("run"), commit="deadbeef")
    canonical_test = Path("results/formal/scenarios/test/scenario_bank_manifest.json")
    assert canonical_test in plan[0].inputs
    assert canonical_test in plan[7].inputs
    assert canonical_test in plan[8].inputs


def test_plan_is_root_isolated_and_has_expected_work():
    root=Path("results/formal/hard_locker_rerun_deadbeef")
    plan=build_plan(output_root=root,commit="deadbeef")
    validate_plan(plan,root)
    assert sum(len(p.commands) for p in plan if p.phase in (5,6)) == 6
    assert any(p.name == "benchmark_600_rows" for p in plan)
    commands=json.dumps([p.commands for p in plan])
    assert "results/formal/benchmark" not in commands
    assert "results/formal/reward_scales/final_reward_reference_scales.json" not in commands


def test_plan_rejects_escape():
    root=Path("results/run")
    bad=[PhaseSpec(1,"bad",tuple(),(Path("../escape"),))]
    with pytest.raises(ValueError,match="escapes"): validate_plan(bad,root)


def test_plan_hash_changes_with_plan():
    root=Path("results/run")
    plan=build_plan(output_root=root,commit="a")
    assert plan_hash(plan) != plan_hash(plan[:-1])


def test_plan_uses_assignment_jsonl_and_paired_script():
 root=Path('results/formal/hard_locker_rerun_deadbeef')
 plan=build_plan(output_root=root,commit='deadbeef')
 phase2=' '.join(word for cmd in plan[2].commands for word in cmd)
 phase9=' '.join(word for cmd in plan[9].commands for word in cmd)
 assert '--agents assignment' in phase2
 assert 'preferences/preferences/preference_assignment.jsonl' in phase2
 assert 'analyze_hard_locker_paired_results' in phase9
 assert plan[8].outputs == (root/'benchmark'/'episode_results.jsonl',)
