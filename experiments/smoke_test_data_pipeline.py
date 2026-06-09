"""Offline smoke test for the Stage 2 data pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from data_pipeline.build_instance import REQUIRED_FILENAMES, build_instance
from data_pipeline.common import npy_shape, read_csv
from utils.config import load_config


def run_smoke_test(config_path: str | Path, fallback: bool = True, output_root: str | Path | None = None) -> dict[str, Any]:
    if not fallback:
        raise ValueError("The Stage 2 smoke test must use offline fallback mode")
    instance = build_instance(config_path, fallback=True, output_root=output_root)
    output_dir = Path(instance["output_directory"])
    missing = [name for name in REQUIRED_FILENAMES if not (output_dir / name).is_file()]
    if missing:
        raise AssertionError(f"Missing Stage 2 files: {missing}")

    stops = read_csv(output_dir / "bus_stops.csv")
    sequences = [int(row["stop_sequence"]) for row in stops]
    assert sequences == sorted(sequences)
    stop_ids = {row["stop_id"] for row in stops}
    stations = read_csv(output_dir / "integrated_stations.csv")
    assert {row["stop_id"] for row in stations} <= stop_ids

    parcels = read_csv(output_dir / "parcels.csv")
    assert all(float(row["release_time"]) < float(row["deadline"]) for row in parcels)
    assert all(row["nearest_station_id"] for row in parcels)
    assert all(row["nearest_road_node"] for row in parcels)
    assert all(row["drone_feasible"].lower() in {"true", "false"} for row in parcels)

    truck_size = 2 + len(stations) + len(parcels)
    assert npy_shape(output_dir / "truck_distance_matrix.npy") == (truck_size, truck_size)
    assert npy_shape(output_dir / "truck_travel_time_matrix.npy") == (truck_size, truck_size)
    assert npy_shape(output_dir / "drone_distance_matrix.npy") == (len(stations), len(parcels))

    loaded_json = json.loads((output_dir / "instance.json").read_text(encoding="utf-8"))
    loaded_yaml = load_config(output_dir / "instance.yaml")
    assert loaded_json["city_name"] == loaded_yaml["city_name"] == instance["city_name"]
    assert instance["mode"] == "fallback"
    return {"output_directory": str(output_dir), "files": REQUIRED_FILENAMES,
            "counts": instance["counts"], "matrix_shapes": {
                "truck": [truck_size, truck_size], "drone": [len(stations), len(parcels)]}}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fallback", action="store_true", help="Required: verify the offline fallback path")
    parser.add_argument("--output-root", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.fallback:
        raise SystemExit("Pass --fallback: smoke tests must not access the internet")
    result = run_smoke_test(args.config, fallback=True, output_root=args.output_root)
    print("Stage 2 fallback data-pipeline smoke test passed.")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
