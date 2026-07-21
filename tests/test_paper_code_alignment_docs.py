from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).parents[1]


def _read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_paper_code_alignment_records_are_present_and_traceable() -> None:
    traceability = _read("docs/paper_code_alignment/requirements_traceability.md")
    validation = _read("docs/paper_code_alignment/validation_report.md")
    decision_log = _read("docs/paper_code_alignment/decision_log.md")
    plan = _read("docs/superpowers/plans/2026-07-11-paper-code-alignment.md")

    for requirement_id in ("REQ-DATA-001", "REQ-MDP-001", "REQ-RLAIF-001", "REQ-MAPPO-001", "REQ-EXP-001"):
        assert requirement_id in traceability

    assert "blocked-by-user-decision" in traceability
    assert "Do not fabricate real data, labels, rewards, checkpoints, or experiment results." in decision_log
    assert "PyTorch runtime unavailable" in validation
    assert "132 passed, 6 skipped" in validation
    assert "Use superpowers:subagent-driven-development" in plan
