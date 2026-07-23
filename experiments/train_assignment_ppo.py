"""Train the Stage 6 assignment-only PPO policy."""

from __future__ import annotations

import argparse
from pathlib import Path
import copy
import yaml
from typing import Sequence

from utils.config import load_config
from rlaif.torch_runtime import is_torch_runtime_available


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = copy.deepcopy(load_config(args.config))
    if args.seed is not None:
        config.setdefault("training", {})["seed"] = args.seed
    if args.output_root is not None:
        out = config.setdefault("output", {})
        out["output_root"] = str(args.output_root)
        seed = config.get("training", {}).get("seed", 1)
        out["checkpoint_path"] = str(args.output_root / out.get("checkpoint_name_template", "assignment_ppo_seed_{seed}.pt").format(seed=seed))
        out["training_log_path"] = str(args.output_root / out.get("training_log_name_template", "assignment_ppo_seed_{seed}.csv").format(seed=seed))
        out["eval_path"] = str(args.output_root / out.get("eval_name_template", "assignment_ppo_seed_{seed}_eval.json").format(seed=seed))
    config.setdefault("run_classification", "formal")
    if config["run_classification"] == "formal" and config.get("env", {}).get("fallback"):
        raise ValueError("formal Assignment PPO requires env.fallback=false")
    if args.validate_only:
        print(yaml.safe_dump(config, sort_keys=False))
        return 0
    if not is_torch_runtime_available():
        message = "Stage 6 assignment PPO training requires PyTorch."
        if config.get("run_classification") == "formal":
            raise RuntimeError(f"formal Assignment PPO cannot skip: {message}")
        print(f"SKIP: {message}")
        return 0
    from training.ppo_trainer import train_assignment_ppo

    result = train_assignment_ppo(config)
    if config.get("run_classification") == "formal":
        rows = result.get("rows", [])
        updates = sum(1 for row in rows if row.get("policy_loss") not in (None, ""))
        if updates <= 0:
            raise RuntimeError("formal Assignment PPO completed zero optimizer updates")
    print(f"Saved Stage 6 assignment PPO checkpoint to {result['checkpoint_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
