from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "docs/paper_code_alignment/final_dynamic_contract.md"
TRACE = ROOT / "docs/paper_code_alignment/requirements_traceability_v2.md"


def test_contract_states_dynamic_release_assignment():
    text = CONTRACT.read_text(encoding="utf-8")
    assert "PARCEL_RELEASE" in text
    assert "TD" in text and "TBD_<station_id>" in text and "TLD_<station_id>" in text
    assert "must not select a specific truck" in text


def test_contract_states_multi_agent_rlaif():
    text = CONTRACT.read_text(encoding="utf-8")
    for agent in ["assignment", "truck", "bus", "station"]:
        assert agent in text
    assert "agent-aware preference data" in text


def test_later_phase_requirements_marked_specified_not_implemented():
    text = TRACE.read_text(encoding="utf-8")
    for requirement in ["truck batching", "physical-bus circulation", "passenger dynamics", "station battery decisions", "multi-agent RLAIF"]:
        assert f"| {requirement} | specified |" in text
        assert f"| {requirement} | implemented |" not in text
