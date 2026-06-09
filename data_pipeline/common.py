"""Shared, dependency-light helpers for the Stage 2 data pipeline."""

from __future__ import annotations

import csv
import json
import math
import struct
from pathlib import Path
from typing import Any, Iterable, Sequence


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance in kilometres."""
    radius_km = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius_km * math.asin(math.sqrt(a))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_npy(path: Path, matrix: Sequence[Sequence[float]]) -> None:
    """Write a little-endian float64, C-order NumPy v1.0 array without NumPy."""
    rows = len(matrix)
    columns = len(matrix[0]) if rows else 0
    if any(len(row) != columns for row in matrix):
        raise ValueError("Matrix rows must have equal lengths")
    header = repr({"descr": "<f8", "fortran_order": False, "shape": (rows, columns)})
    header_bytes = header.encode("latin1")
    padding = (16 - ((10 + len(header_bytes) + 1) % 16)) % 16
    header_bytes += b" " * padding + b"\n"
    with path.open("wb") as handle:
        handle.write(b"\x93NUMPY\x01\x00")
        handle.write(struct.pack("<H", len(header_bytes)))
        handle.write(header_bytes)
        for row in matrix:
            handle.write(struct.pack(f"<{columns}d", *row))


def npy_shape(path: Path) -> tuple[int, ...]:
    """Read the shape from a NumPy v1/v2 array header without importing NumPy."""
    import ast

    with path.open("rb") as handle:
        if handle.read(6) != b"\x93NUMPY":
            raise ValueError(f"Not a NumPy array: {path}")
        major, _minor = handle.read(2)
        length_size = 2 if major == 1 else 4
        header_length = int.from_bytes(handle.read(length_size), "little")
        header = ast.literal_eval(handle.read(header_length).decode("latin1").strip())
    return tuple(header["shape"])


def minutes_from_time(value: str | int | float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    parts = value.strip().split(":")
    if len(parts) not in (2, 3):
        raise ValueError(f"Expected HH:MM or HH:MM:SS, got {value!r}")
    return int(parts[0]) * 60 + int(parts[1]) + (int(parts[2]) / 60 if len(parts) == 3 else 0)
