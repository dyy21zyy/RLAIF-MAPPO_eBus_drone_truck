"""Read-only, exploratory comparison of aligned original/post-training results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"benchmark artifact does not exist: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _scenario_map(rows: list[dict[str, Any]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in rows:
        scenario_id, digest = str(row["scenario_id"]), str(row["scenario_content_hash"])
        if scenario_id in result and result[scenario_id] != digest:
            raise ValueError(f"inconsistent scenario hash for {scenario_id}")
        result[scenario_id] = digest
    return result


def compare(original: list[dict[str, Any]], post: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    """Return descriptive deltas only after exact scenario identity validation."""
    if _scenario_map(original) != _scenario_map(post):
        raise ValueError("scenario IDs and hashes must match exactly")

    def mean(method: str, rows: list[dict[str, Any]]) -> float:
        values = [float(row[metric]) for row in rows if row["method_id"] == method]
        if not values:
            raise ValueError(f"no rows for {method!r}")
        return sum(values) / len(values)

    original_env = mean("mappo_env", original)
    original_rlaif = mean("mappo_rlaif_assignment", original)
    continued = mean("mappo_env_post_continued", post)
    rlaif_post = mean("mappo_rlaif_assignment_post", post)
    return {
        "analysis_type": "exploratory_cross_experiment_not_confirmatory",
        "metric": metric,
        "deltas": {
            "original_rlaif_vs_original_mappo_env": original_rlaif - original_env,
            "rlaif_post_vs_environment_continuation": rlaif_post - continued,
            "rlaif_post_vs_original_rlaif": rlaif_post - original_rlaif,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path)
    parser.add_argument("--post-training", type=Path)
    parser.add_argument("--metric", default="episode_reward")
    args = parser.parse_args(argv)
    if args.original is None or args.post_training is None:
        print("optional artifacts absent; exploratory comparison not run")
        return 0
    print(json.dumps(compare(_read(args.original), _read(args.post_training), args.metric), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
