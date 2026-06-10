"""CLI for Stage 4 pairwise AI preference prompt generation."""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Sequence
from rlaif.preference_dataset import read_jsonl, validate_assignment_state, validate_prompt, write_jsonl
from rlaif.prompt_builder import build_prompt_records

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--states", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv); states = read_jsonl(args.states)
    for state in states: validate_assignment_state(state)
    prompts = build_prompt_records(states)
    for prompt in prompts: validate_prompt(prompt)
    write_jsonl(args.output, prompts); print(f"Wrote {len(prompts)} prompts to {args.output}"); return 0
if __name__ == "__main__": raise SystemExit(main())
