"""Paper-code traceability checks for the four-agent Solution Method pass."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parents[1]


def _text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_four_agent_traceability_records_core_requirements() -> None:
    trace = _text("docs/paper_code_alignment/requirements_traceability.md")
    for requirement in (
        "REQ-MDP-FOUR-AGENT",
        "REQ-MDP-CANDIDATES",
        "REQ-MAPPO-ACTORS",
        "REQ-MAPPO-CANDIDATE-POLICY",
        "REQ-MAPPO-BUFFER",
        "REQ-RLAIF-SCOPE",
    ):
        assert requirement in trace
    assert "assignment, truck, bus, and station" in trace
    assert "not final experiment evidence" in trace


def test_decision_log_supersedes_two_agent_stage7_boundary() -> None:
    decision_log = _text("docs/paper_code_alignment/decision_log.md")
    assert "2026-07-20" in decision_log
    assert "supersedes the previous Stage 7 two-agent boundary" in decision_log
    assert "four-agent" in decision_log


def test_validation_report_keeps_no_fabrication_boundary() -> None:
    report = _text("docs/paper_code_alignment/validation_report.md")
    assert (
        "No preference labels, learned rewards, checkpoints, or final results "
        "are fabricated"
    ) in report
    assert "PyTorch runtime gates" in report
