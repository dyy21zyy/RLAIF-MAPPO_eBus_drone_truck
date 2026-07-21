"""Canonical parcel status helpers."""
from __future__ import annotations

DELIVERED_STATUS = "DELIVERED"

def normalize_status(status: object) -> str:
    return str(status).strip().upper()

def is_delivered_status(status: object) -> bool:
    return normalize_status(status) == DELIVERED_STATUS
