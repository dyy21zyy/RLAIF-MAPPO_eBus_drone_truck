"""CLI for Stage 5 reward-model training."""

from __future__ import annotations

import argparse

from rlaif.preference_dataset import NoUsablePreferencesError
from rlaif.train_reward_model import train_from_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_reward_model.yaml")
    parser.add_argument("--data", help="override preference JSONL path")
    args = parser.parse_args()
    try:
        result = train_from_config(args.config, args.data)
    except NoUsablePreferencesError as exc:
        print(exc)
        return 2
    checkpoint = result["checkpoint"]
    print(f"usable_preferences={checkpoint['usable_preference_count']}")
    print(f"split_sizes={checkpoint['split_sizes']}")
    print(f"checkpoint={checkpoint['config']['output']['checkpoint_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
