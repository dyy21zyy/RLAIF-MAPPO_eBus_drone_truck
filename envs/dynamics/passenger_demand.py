"""Piecewise time-dependent passenger demand generation."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Any
import math
import numpy as np

from envs.dynamics.passenger_dynamics import PassengerArrivalEvent

@dataclass(frozen=True)
class PassengerTemporalBlock:
    block_id: str
    start_min: float
    end_min: float
    multiplier: float


def validate_temporal_profile(blocks: Iterable[PassengerTemporalBlock], *, horizon_min: float) -> tuple[PassengerTemporalBlock, ...]:
    if not math.isfinite(float(horizon_min)) or horizon_min <= 0:
        raise ValueError("horizon_min must be finite and positive")
    ordered = tuple(sorted(blocks, key=lambda b: (b.start_min, b.end_min, b.block_id)))
    if not ordered:
        raise ValueError("temporal profile must be nonempty")
    expected = 0.0
    for block in ordered:
        if not all(math.isfinite(float(x)) for x in (block.start_min, block.end_min, block.multiplier)):
            raise ValueError("temporal block bounds and multiplier must be finite")
        if block.start_min >= block.end_min:
            raise ValueError("temporal block start_min must be less than end_min")
        if block.multiplier <= 0:
            raise ValueError("temporal block multiplier must be positive")
        if abs(block.start_min - expected) > 1e-9:
            if block.start_min < expected:
                raise ValueError("temporal blocks must not overlap")
            raise ValueError("temporal blocks must cover horizon without gaps")
        expected = block.end_min
    if expected < float(horizon_min) - 1e-9:
        raise ValueError("temporal blocks must cover the required simulation horizon")
    if ordered[-1].end_min > float(horizon_min) + 1e-9:
        raise ValueError("temporal profile extends beyond horizon")
    return ordered


def sample_truncated_normal_rates(stop_ids: Iterable[str], *, seed: int, mean: float = 0.25, std: float = 0.10, min_rate: float = 0.05, max_rate: float = 0.60) -> dict[str, float]:
    ids = [str(s) for s in stop_ids]
    if len(ids) != len(set(ids)):
        raise ValueError("stop IDs must be unique")
    rng = np.random.default_rng(int(seed))
    rates: dict[str, float] = {}
    for sid in ids:
        for _ in range(10000):
            value = float(rng.normal(mean, std))
            if min_rate <= value <= max_rate:
                rates[sid] = value
                break
        else:
            rates[sid] = float(np.clip(rng.normal(mean, std), min_rate, max_rate))
    return rates


def effective_passenger_rate(baseline_rate: float, demand_intensity: float, temporal_multiplier: float) -> float:
    return float(baseline_rate) * float(demand_intensity) * float(temporal_multiplier)


def generate_time_dependent_arrivals(stop_ids: list[str], *, horizon_min: float, baseline_rates: dict[str, float], demand_intensity: float, temporal_blocks: tuple[PassengerTemporalBlock, ...], seed: int) -> list[PassengerArrivalEvent]:
    ids = [str(s) for s in stop_ids]
    if len(ids) != len(set(ids)):
        raise ValueError("stop IDs must be unique")
    if not math.isfinite(float(demand_intensity)) or demand_intensity <= 0:
        raise ValueError("demand_intensity must be finite and positive")
    blocks = validate_temporal_profile(temporal_blocks, horizon_min=horizon_min)
    rng = np.random.default_rng(int(seed))
    events: list[PassengerArrivalEvent] = []
    seq = 0
    for i, origin in enumerate(ids[:-1]):
        baseline = float(baseline_rates[origin])
        for block in blocks:
            rate = effective_passenger_rate(baseline, demand_intensity, block.multiplier)
            count = int(rng.poisson(rate * (block.end_min - block.start_min)))
            for _ in range(count):
                dest = ids[i + 1 + int(rng.integers(0, len(ids) - i - 1))]
                t = float(rng.uniform(block.start_min, block.end_min))
                events.append(PassengerArrivalEvent(f"pe_{seq:09d}", origin, dest, t, 1, block.block_id, baseline, demand_intensity, block.multiplier, rate, int(seed)))
                seq += 1
    events.sort(key=lambda e: (e.arrival_time_min, e.origin_stop_id, e.destination_stop_id, e.passenger_event_id))
    return events


def blocks_from_config(config: dict[str, Any], horizon_min: float) -> tuple[PassengerTemporalBlock, ...]:
    raw = config.get("temporal_profile") or [
        {"block_id":"early","start_min":0,"end_min":120,"multiplier":0.75},
        {"block_id":"peak","start_min":120,"end_min":240,"multiplier":1.25},
        {"block_id":"midday","start_min":240,"end_min":360,"multiplier":1.0},
        {"block_id":"late","start_min":360,"end_min":horizon_min,"multiplier":0.5},
    ]
    return validate_temporal_profile(tuple(PassengerTemporalBlock(str(b["block_id"]), float(b["start_min"]), float(b["end_min"]), float(b["multiplier"])) for b in raw), horizon_min=horizon_min)
