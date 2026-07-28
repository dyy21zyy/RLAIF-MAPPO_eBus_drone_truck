import json
from pathlib import Path
import pytest
from experiments.run_hard_locker_reexperiment import PhaseSpec, build_plan, plan_hash, validate_plan


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
