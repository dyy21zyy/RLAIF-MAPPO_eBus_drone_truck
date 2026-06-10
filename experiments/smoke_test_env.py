"""Required Stage 3 gate command for the offline event-driven environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from experiments.smoke_test_environment import run_smoke_test


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--fallback", action="store_true", help="Use the required offline fallback instance builder.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.fallback:
        raise SystemExit("Stage 3 smoke testing requires --fallback")
    result = run_smoke_test(args.config, args.output_root)
    print("Stage 3 event-driven environment smoke test passed.")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
