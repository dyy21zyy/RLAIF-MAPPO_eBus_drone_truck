"""Typed reward ledger for environment-reward-only asynchronous MAPPO.

The ledger records every reward component as raw, normalized, and weighted
amounts.  Costs are stored as positive raw/weighted amounts and contribute
negative reward through :meth:`RewardLedger.reward_sum`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

REWARD_LEDGER_SCHEMA_VERSION = 1

REWARD_COMPONENTS = {
    "passenger_waiting_delay",
    "onboard_additional_passenger_delay",
    "bus_operating_delay",
    "parcel_lateness",
    "undelivered_parcels",
    "truck_cost",
    "bus_energy",
    "drone_energy",
    "station_power_overload",
    "locker_overflow",
    "battery_shortage",
    "bus_battery_violation",
    "infeasible_action",
}

LEGACY_COMPONENT_MAP = {
    "passenger_delay": "passenger_waiting_delay",
    "bus_operating_delay": "bus_operating_delay",
    "parcel_lateness": "parcel_lateness",
    "energy_cost": "bus_energy",
    "power_overload": "station_power_overload",
    "bus_battery_violation": "bus_battery_violation",
    "locker_overflow": "locker_overflow",
    "truck_cost": "truck_cost",
    "undelivered": "undelivered_parcels",
    "battery_shortage": "battery_shortage",
    "infeasible_action": "infeasible_action",
}

@dataclass(frozen=True)
class RewardEntry:
    event_time: float
    component: str
    raw_amount: float
    normalized_amount: float
    weighted_amount: float
    entity_ids: tuple[str, ...] = ()
    parcel_ids: tuple[str, ...] = ()
    source_transition_ids: tuple[str, ...] = ()
    decision_chain_refs: tuple[str, ...] = ()
    provenance: str = "environment"

    def __post_init__(self) -> None:
        if self.component not in REWARD_COMPONENTS:
            raise ValueError(f"Unknown reward component: {self.component}")
        object.__setattr__(self, "event_time", float(self.event_time))
        object.__setattr__(self, "raw_amount", float(self.raw_amount))
        object.__setattr__(self, "normalized_amount", float(self.normalized_amount))
        object.__setattr__(self, "weighted_amount", float(self.weighted_amount))
        object.__setattr__(self, "entity_ids", tuple(map(str, self.entity_ids)))
        object.__setattr__(self, "parcel_ids", tuple(map(str, self.parcel_ids)))
        object.__setattr__(self, "source_transition_ids", tuple(map(str, self.source_transition_ids)))
        object.__setattr__(self, "decision_chain_refs", tuple(map(str, self.decision_chain_refs)))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass
class RewardLedger:
    schema_version: int = REWARD_LEDGER_SCHEMA_VERSION
    entries: list[RewardEntry] = field(default_factory=list)

    def add(self, entry: RewardEntry) -> None:
        self.entries.append(entry)

    def add_cost(self, *, event_time: float, component: str, raw_amount: float, weight: float,
                 reference_scale: float = 1.0, entity_ids=(), parcel_ids=(),
                 source_transition_ids=(), decision_chain_refs=(), provenance: str = "environment") -> float:
        canonical = LEGACY_COMPONENT_MAP.get(component, component)
        raw = max(0.0, float(raw_amount))
        scale = max(abs(float(reference_scale)), 1e-9)
        weighted = raw * float(weight)
        self.add(RewardEntry(event_time, canonical, raw, raw / scale, weighted,
                             tuple(entity_ids), tuple(parcel_ids), tuple(source_transition_ids),
                             tuple(decision_chain_refs), provenance))
        return -weighted

    def reward_sum(self) -> float:
        return -sum(entry.weighted_amount for entry in self.entries)

    def component_costs(self) -> dict[str, float]:
        costs: dict[str, float] = {}
        for entry in self.entries:
            costs[entry.component] = costs.get(entry.component, 0.0) + entry.weighted_amount
        return costs

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "entries": [e.to_dict() for e in self.entries]}
