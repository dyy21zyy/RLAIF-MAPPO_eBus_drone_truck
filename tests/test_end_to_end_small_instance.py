from pathlib import Path
from experiments.smoke_test_environment import run_smoke_test


def test_end_to_end_small_instance_completes(tmp_path):
    result = run_smoke_test(Path(__file__).parents[1] / "configs/shanghai_small.yaml", output_root=tmp_path)
    assert result["steps"] > 0
    assert result["invariants"] == "passed"
    assert result["delivered_parcels"] + result["undelivered_parcels"] == result["total_parcels"]
