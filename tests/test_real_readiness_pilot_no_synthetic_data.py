from pathlib import Path
import ast

def test_no_synthetic_random_or_fake_checkpoint():
    src=Path("experiments/run_readiness_pilot.py").read_text()
    assert "DynamicDeliveryEnv" in src and "collect_episode" in src and "update_mappo" in src
    assert "torch.randn" not in src and "write_bytes" not in src and "diagnostic checkpoint" not in src
    tree=ast.parse(src); assert tree
