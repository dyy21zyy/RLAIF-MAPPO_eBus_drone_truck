"""CLI for Stage 5 reward-model evaluation."""

from __future__ import annotations

import argparse
import json

from rlaif.preference_dataset import NoUsablePreferencesError
from rlaif.evaluate_reward_model import evaluate_from_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_reward_model.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", help="override preference JSONL path")
    args = parser.parse_args()
    try:
        result = evaluate_from_config(args.config, args.checkpoint, args.data)
    except NoUsablePreferencesError as exc:
        print(exc)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
