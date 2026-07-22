from pathlib import Path
from experiments.prepare_diagnostic_benchmark import build_fixture

def build_diagnostic_benchmark_fixture(root: Path, *, scenario_count: int = 3, seed: int = 1) -> Path:
    return build_fixture(root, force=True, scenario_count=scenario_count, seed=seed)
