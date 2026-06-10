"""Offline Stage 4 gate for assignment states and AI preference prompts."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Sequence

from rlaif.ai_evaluator import run_offline, run_replay
from rlaif.collect_assignment_states import collect_assignment_states
from rlaif.preference_dataset import (read_jsonl, validate_assignment_state, validate_preference,
                                      validate_prompt, write_jsonl)
from rlaif.prompt_builder import build_prompt_records


def run_smoke_test(config_path: str | Path, fallback: bool = True,
                   replay_labels: str | Path | None = None) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="stage4-smoke-") as directory:
        root = Path(directory)
        states_path = root / "assignment_states.jsonl"
        prompts_path = root / "ai_preference_prompts.jsonl"
        preferences_path = root / "ai_preferences.jsonl"
        failed_path = root / "failed_preferences.jsonl"
        states = collect_assignment_states(config_path, 1, states_path, fallback=fallback)
        if len(states) < 20:
            raise AssertionError("Stage 4 smoke test collected fewer than 20 assignment states")
        loaded_states = read_jsonl(states_path)
        for state in loaded_states:
            validate_assignment_state(state)
        prompts = build_prompt_records(loaded_states)
        if len(prompts) < 10:
            raise AssertionError("Stage 4 smoke test generated fewer than 10 prompts")
        for prompt in prompts:
            validate_prompt(prompt)
        write_jsonl(prompts_path, prompts)
        offline = run_offline(prompts, preferences_path)
        if offline["preferences"] != 0 or preferences_path.exists():
            raise AssertionError("offline mode invented preference labels")
        replay = {"preferences": 0, "failed": 0}
        if replay_labels is not None:
            replay = run_replay(prompts, replay_labels, preferences_path, failed_path)
            for preference in read_jsonl(preferences_path):
                validate_preference(preference)
        return {
            "assignment_states_file": str(states_path), "assignment_states": len(states),
            "prompts_file": str(prompts_path), "prompts": len(prompts),
            "offline_preferences": offline["preferences"], "replay_preferences": replay["preferences"],
            "failed_preferences": replay["failed"], "external_api_used": False,
        }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True); parser.add_argument("--fallback", action="store_true")
    parser.add_argument("--replay-labels", type=Path)
    args = parser.parse_args(argv)
    result = run_smoke_test(args.config, args.fallback, args.replay_labels)
    print("Stage 4 RLAIF data smoke test passed."); print(json.dumps(result, indent=2)); return 0

if __name__ == "__main__": raise SystemExit(main())
