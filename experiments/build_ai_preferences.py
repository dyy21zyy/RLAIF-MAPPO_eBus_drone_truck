"""CLI for safe offline, configured API, or user-label replay processing."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from rlaif.ai_evaluator import (
    NoPreferenceLabelsError,
    load_api_settings,
    run_api,
    run_offline,
    run_replay,
)
from rlaif.preference_dataset import read_jsonl, validate_prompt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("offline", "api", "replay"), required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/preference/ai_preferences.jsonl"))
    parser.add_argument("--failed-output", type=Path, default=Path("data/preference/failed_preferences.jsonl"))
    parser.add_argument("--manual-template-output", type=Path, default=Path("data/preference/manual_labels_template.jsonl"))
    parser.add_argument("--small-template-output", type=Path, default=Path("data/preference/manual_labels_template_small.jsonl"))
    args = parser.parse_args(argv)

    try:
        prompts = read_jsonl(args.prompts)
        for prompt in prompts:
            validate_prompt(prompt)
        if args.mode == "offline":
            result = run_offline(
                prompts,
                args.output,
                args.manual_template_output,
                args.small_template_output,
            )
        elif args.mode == "replay":
            if args.labels is None:
                parser.error("--labels is required in replay mode")
            result = run_replay(prompts, args.labels, args.output, args.failed_output)
        else:
            result = run_api(
                prompts,
                args.output,
                args.failed_output,
                settings=load_api_settings(args.config),
            )
    except (NoPreferenceLabelsError, FileNotFoundError, ValueError) as exc:
        if args.mode in {"api", "replay"} and args.output.exists():
            args.output.unlink()
        print(f"Stage 4 labeling stopped: {exc}", file=sys.stderr)
        return 2
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
