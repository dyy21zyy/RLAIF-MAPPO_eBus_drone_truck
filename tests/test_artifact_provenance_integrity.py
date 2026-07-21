import json
from pathlib import Path


def test_committed_scenario_artifacts_include_provenance():
    root = Path(__file__).parents[1] / "data/scenarios"
    for instance in root.glob("*/*/instance.json"):
        data = json.loads(instance.read_text(encoding="utf-8"))
        assert data.get("data_provenance") or data.get("artifacts", {}).get("data_provenance")
        assert data.get("config_snapshot")
