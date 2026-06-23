"""Smoke-test RLAIF state/prompt generation for original-scale real-transit mode."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Sequence

from experiments.original_scale_smoke_common import smoke_config
from rlaif.collect_assignment_states import collect_assignment_states
from rlaif.prompt_builder import build_prompt_records


def run_smoke_test(config_path: str | Path) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="orig-scale-rlaif-") as directory:
        root = Path(directory)
        cfg = smoke_config(config_path, root)
        states_path = root / "assignment_states.jsonl"
        states = collect_assignment_states(cfg, episodes=1, output=states_path, fallback=False, seed=42)
        prompts = build_prompt_records(states)
        if not states:
            raise AssertionError("Expected assignment states")
        if not prompts:
            raise AssertionError("Expected at least one preference prompt")
        if "real_transit_data" not in prompts[0]["prompt_text"]:
            raise AssertionError("Prompt must include source-aware transit context")
        return {
            "states": len(states),
            "prompts": len(prompts),
            "preferences": 0,
            "labels_created": False,
        }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_smoke_test(args.config)
    print("Original-scale real-transit RLAIF smoke test passed.")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
