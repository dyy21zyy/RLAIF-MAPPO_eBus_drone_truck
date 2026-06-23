"""Smoke-test the original-scale real-transit data builder with tiny fixtures."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Sequence

from data_pipeline.build_instance import build_instance
from experiments.original_scale_smoke_common import smoke_config


def run_smoke_test(config_path: str | Path) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="orig-scale-data-") as directory:
        root = Path(directory)
        cfg = smoke_config(config_path, root)
        instance = build_instance(cfg, fallback=False, output_root=root / "processed")
        output_dir = Path(instance["output_directory"])
        required = ["instance.json", "data_provenance.json", "scale_match_report.json", "bus_timetable.json"]
        missing = [name for name in required if not (output_dir / name).is_file()]
        if missing:
            raise AssertionError(f"Missing original-scale data artifacts: {missing}")
        provenance = json.loads((output_dir / "data_provenance.json").read_text(encoding="utf-8"))
        scale = json.loads((output_dir / "scale_match_report.json").read_text(encoding="utf-8"))
        if not any(entry["field"] == "bus stop_times" for entry in provenance["entries"]):
            raise AssertionError("data_provenance.json must include bus stop_times")
        return {
            "output_directory": str(output_dir),
            "counts": instance["counts"],
            "scale_match_pass": scale["scale_match_pass"],
            "provenance_entries": len(provenance["entries"]),
        }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_smoke_test(args.config)
    print("Original-scale real-transit data smoke test passed.")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
