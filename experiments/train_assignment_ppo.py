"""Train the Stage 6 assignment-only PPO policy."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Sequence

from utils.config import load_config


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if importlib.util.find_spec("torch") is None:
        print("SKIP: Stage 6 assignment PPO training requires PyTorch.")
        return 0
    from training.ppo_trainer import train_assignment_ppo

    result = train_assignment_ppo(load_config(args.config))
    print(f"Saved Stage 6 assignment PPO checkpoint to {result['checkpoint_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
