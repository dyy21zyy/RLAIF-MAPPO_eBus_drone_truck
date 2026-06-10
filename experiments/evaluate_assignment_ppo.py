"""Evaluate a Stage 6 assignment PPO checkpoint deterministically."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Sequence

from utils.config import load_config


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if importlib.util.find_spec("torch") is None:
        print("SKIP: Stage 6 assignment PPO evaluation requires PyTorch.")
        return 0
    from training.ppo_trainer import evaluate_assignment_ppo

    result = evaluate_assignment_ppo(load_config(args.config), args.checkpoint, episodes=args.episodes)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
