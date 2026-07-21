from pathlib import Path

ROOT = Path(__file__).parents[1]
STALE = [
    "rlaif is assignment-only",
    "station charging is automatic",
    "truck uses one parcel per trip",
    "bus state is keyed by trip",
    "passenger delay equals charging time",
    "only assignment and bus actors are active",
    "same checkpoint represents rlaif and no-rlaif",
]


def test_documentation_contains_no_stale_claims():
    docs = "\n".join(p.read_text(encoding="utf-8").lower() for p in (ROOT / "docs").rglob("*.md"))
    for phrase in STALE:
        assert phrase not in docs
    assert "formal results not yet produced" in docs
