"""Formal-candidate artifact inventory utilities for the pre-formal gate."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable
from evaluation.preformal_gate import sha256_file

FORBIDDEN_SUFFIXES = {'.npy','.npz','.pt','.pth','.pkl','.pickle','.bin','.onnx','.joblib'}

def inventory(paths: Iterable[str | Path]) -> list[dict[str, object]]:
    rows=[]
    for raw in paths:
        p=Path(raw)
        rows.append({'path':str(p),'exists':p.exists(),'is_file':p.is_file(),'suffix':p.suffix,'sha256':sha256_file(p),'binary_forbidden':p.suffix in FORBIDDEN_SUFFIXES})
    return rows

def write_inventory(paths: Iterable[str | Path], output: str | Path) -> list[dict[str, object]]:
    rows=inventory(paths); Path(output).parent.mkdir(parents=True, exist_ok=True); Path(output).write_text(json.dumps({'artifacts':rows},indent=2,sort_keys=True),encoding='utf-8'); return rows
