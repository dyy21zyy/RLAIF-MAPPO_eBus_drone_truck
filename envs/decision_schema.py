"""Shared decision and candidate-action schema for four-agent event control."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

BUS_LOADING_CANDIDATE_FEATURE_NAMES = (
    "parcel_count", "total_weight_kg", "freight_capacity_utilization",
    "loading_time_min", "maximum_single_station_unload_kg",
    "estimated_lateness_min", "estimated_passenger_impact_min", "idle_flag",
)
BUS_CHARGING_CANDIDATE_FEATURE_NAMES = (
    "duration_sec", "energy_added_kwh", "projected_soc_kwh",
    "projected_station_load_kw", "projected_overload_kw", "idle_flag",
)

TRUCK_BATCH_CANDIDATE_FEATURE_NAMES = (
    "batch_size", "weight_utilization", "volume_utilization", "distance",
    "travel_time", "estimated_lateness", "direct_customer_count",
    "terminal_delivery_count", "station_delivery_count", "route_stops", "idle_flag",
)


@dataclass(frozen=True)
class ActionCandidate:
    action_id: int
    action_type: str
    entity_id: str
    description: str
    features: dict[str, Any]
    feasible: bool
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if int(self.action_id) < 0:
            raise ValueError("action_id must be non-negative")
        normalized: dict[str, float] = {}
        for key, value in self.features.items():
            if isinstance(value, (list, tuple)):
                normalized[str(key)] = value
                continue
            numeric = float(value)
            if not isfinite(numeric):
                raise ValueError(f"Candidate feature {key!r} must be finite")
            normalized[str(key)] = numeric
        object.__setattr__(self, "action_id", int(self.action_id))
        object.__setattr__(self, "action_type", str(self.action_type))
        object.__setattr__(self, "entity_id", str(self.entity_id))
        object.__setattr__(self, "description", str(self.description))
        object.__setattr__(self, "features", normalized)
        object.__setattr__(self, "feasible", bool(self.feasible))
        object.__setattr__(self, "reasons", tuple(str(reason) for reason in self.reasons))

    def payload(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "action_type": self.action_type,
            "entity_id": self.entity_id,
            "description": self.description,
            "features": dict(self.features),
            "feasible": self.feasible,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class DecisionSurface:
    agent_id: str
    event_type: str
    entity_id: str
    features: list[float]
    feature_names: tuple[str, ...]
    candidates: list[ActionCandidate]

    def __post_init__(self) -> None:
        if len(self.features) != len(self.feature_names):
            raise ValueError("features and feature_names must have the same length")
        if not self.candidates:
            raise ValueError("Decision surface requires candidates")
        if not any(candidate.feasible for candidate in self.candidates):
            raise ValueError("Decision surface requires at least one feasible candidate")
        normalized_features: list[float] = []
        for value in self.features:
            numeric = float(value)
            if not isfinite(numeric):
                raise ValueError("Decision features must be finite")
            normalized_features.append(numeric)
        object.__setattr__(self, "agent_id", str(self.agent_id))
        object.__setattr__(self, "event_type", str(self.event_type))
        object.__setattr__(self, "entity_id", str(self.entity_id))
        object.__setattr__(self, "features", normalized_features)
        object.__setattr__(
            self, "feature_names", tuple(str(name) for name in self.feature_names)
        )
        object.__setattr__(self, "candidates", list(self.candidates))

    def action_mask(self) -> list[bool]:
        return [candidate.feasible for candidate in self.candidates]

    def candidate_feature_names(self) -> tuple[str, ...]:
        names: list[str] = []
        for candidate in self.candidates:
            for key in candidate.features:
                if key not in names:
                    names.append(key)
        return tuple(names)

    def candidate_feature_matrix(self) -> list[list[float]]:
        names = self.candidate_feature_names()
        return [
            [float(candidate.features.get(name, 0.0)) if not isinstance(candidate.features.get(name, 0.0), (list, tuple)) else 0.0 for name in names]
            for candidate in self.candidates
        ]

    def candidate_payloads(self) -> list[dict[str, Any]]:
        return [candidate.payload() for candidate in self.candidates]
