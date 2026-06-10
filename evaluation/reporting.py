"""Filesystem writers for evaluation records."""
from __future__ import annotations
import csv, json
from pathlib import Path
from evaluation.result_schema import RESULT_FIELDS

def write_episode(path, record):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True)+"\n", encoding="utf-8")

def write_records_csv(path, records):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer=csv.DictWriter(handle, fieldnames=RESULT_FIELDS, extrasaction="ignore"); writer.writeheader(); writer.writerows(records)
