"""CLI for Stage 4 assignment-state collection."""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Sequence
from rlaif.collect_assignment_states import collect_assignment_states

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True); parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--fallback", action="store_true")
    args = parser.parse_args(argv)
    records = collect_assignment_states(args.config, args.episodes, args.output, args.fallback)
    print(f"Wrote {len(records)} assignment states to {args.output}")
    return 0
if __name__ == "__main__": raise SystemExit(main())
