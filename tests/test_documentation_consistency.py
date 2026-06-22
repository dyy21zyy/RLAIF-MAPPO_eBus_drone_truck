"""Lightweight regression checks for project status and RLAIF guardrails."""

from pathlib import Path

ROOT = Path(__file__).parents[1]


def _read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_readme_reports_consistent_stage_status_and_runtime_prerequisites() -> None:
    readme = _read("README.md")

    assert "PPO and MAPPO remain intentionally **not implemented**" not in readme
    for statement in (
        "Stage 1: foundation and documentation (complete)",
        "Stage 2: offline-capable Shanghai instance data pipeline (complete)",
        "Stage 3: event-driven MDP environment (implemented and hardened)",
        "Stage 4: RLAIF state/prompt collection and AI-label interface (complete)",
        "Stage 5: Code Gate complete; PyTorch Runtime Gate deferred",
        "Stage 6: Code Gate complete; runtime training deferred",
        "Stage 7: Code Gate complete; runtime training deferred",
        "Stage 8: experiment framework implemented; final experiments deferred",
    ):
        assert statement in readme
    for prerequisite in (
        "PyTorch environment",
        "Stage 5 Runtime Gate",
        "reward_model.pt",
        "assignment_ppo.pt",
        "mappo_async.pt",
        "benchmark/ablation/sensitivity runs",
    ):
        assert prerequisite in readme


def test_experiment_docs_do_not_describe_implemented_stages_as_planned() -> None:
    experiments = _read("docs/EXPERIMENTS.md")

    assert "Stage 2 planned" not in experiments
    assert "Stage 3 planned" not in experiments
    assert "Stage 6 Code Gate complete" in experiments
    assert "Stage 7 Code Gate complete" in experiments
    assert "Stage 8 experiment framework implemented" in experiments


def test_rlaif_docs_preserve_no_fake_label_rule() -> None:
    workflow = _read("docs/RLAIF_WORKFLOW.md").lower()

    assert "not labels" in workflow
    assert "rule-based" in workflow
    assert "fabricated" in workflow


def test_pitfalls_document_mvp_simplifications_and_hardened_behavior() -> None:
    pitfalls = _read("docs/PITFALLS.md")

    assert "Stage 3 MVP simplifications replaced by hardening" in pitfalls
    for behavior in (
        "hard-feasibility action masks",
        "17 + 10H",
        "piecewise-constant interval integration",
        "explicit per-truck state",
    ):
        assert behavior in pitfalls
