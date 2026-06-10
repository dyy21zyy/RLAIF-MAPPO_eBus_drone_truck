"""CLI for offline, API, or replay Stage 4 AI preference processing."""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Sequence
from rlaif.ai_evaluator import run_api, run_offline, run_replay
from rlaif.preference_dataset import read_jsonl, validate_prompt

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--mode", choices=("offline", "api", "replay"), required=True)
    parser.add_argument("--prompts", type=Path, required=True); parser.add_argument("--labels", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/preference/ai_preferences.jsonl"))
    parser.add_argument("--failed-output", type=Path, default=Path("data/preference/failed_preferences.jsonl"))
    parser.add_argument("--evaluator-model", default="external-llm"); parser.add_argument("--temperature", type=float, default=0.0); parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args(argv); prompts = read_jsonl(args.prompts)
    for prompt in prompts: validate_prompt(prompt)
    if args.mode == "offline": result = run_offline(prompts, args.output)
    elif args.mode == "replay":
        if args.labels is None: parser.error("--labels is required in replay mode")
        result = run_replay(prompts, args.labels, args.output, args.failed_output, args.evaluator_model, args.temperature)
    else: result = run_api(prompts, args.output, args.failed_output, args.evaluator_model, args.temperature, args.max_retries)
    print(result); return 0
if __name__ == "__main__": raise SystemExit(main())
