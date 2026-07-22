"""Canonical MAPPO decision-event schema."""
from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum
from typing import Final

OBSERVATION_SCHEMA_VERSION: Final[int] = 3
CANDIDATE_SCHEMA_VERSION: Final[int] = 3
EVENT_SCHEMA_VERSION: Final[int] = 2
CHECKPOINT_SCHEMA_VERSION: Final[int] = 4
AGENT_TYPES: Final[tuple[str,...]] = ("assignment","truck","bus","station")

class DecisionEventId(IntEnum):
    PARCEL_RELEASE = 0
    TRUCK_AVAILABLE = 1
    BUS_TERMINAL_DEPARTURE = 2
    BUS_STATION_ARRIVAL = 3
    STATION_OPERATION = 4

@dataclass(frozen=True)
class DecisionEventSpec:
    name: str
    event_id: DecisionEventId
    agent_type: str

DECISION_EVENT_SPECS: Final[dict[str, DecisionEventSpec]] = {
    "PARCEL_RELEASE": DecisionEventSpec("PARCEL_RELEASE", DecisionEventId.PARCEL_RELEASE, "assignment"),
    "TRUCK_AVAILABLE": DecisionEventSpec("TRUCK_AVAILABLE", DecisionEventId.TRUCK_AVAILABLE, "truck"),
    "BUS_TERMINAL_DEPARTURE": DecisionEventSpec("BUS_TERMINAL_DEPARTURE", DecisionEventId.BUS_TERMINAL_DEPARTURE, "bus"),
    "BUS_STATION_ARRIVAL": DecisionEventSpec("BUS_STATION_ARRIVAL", DecisionEventId.BUS_STATION_ARRIVAL, "bus"),
    "STATION_OPERATION": DecisionEventSpec("STATION_OPERATION", DecisionEventId.STATION_OPERATION, "station"),
}
LEGACY_EVENT_ALIASES: Final[dict[str,str]] = {"BUS_DEPARTURE":"BUS_TERMINAL_DEPARTURE", "BUS_ARRIVAL":"BUS_STATION_ARRIVAL", "PARCEL_ARRIVAL":"PARCEL_RELEASE"}
AUTOMATIC_EVENT_TYPES: Final[frozenset[str]] = frozenset({"BUS_TRIP_START","BUS_ARRIVE_STOP","BUS_DEPART_STOP","BUS_TRIP_COMPLETE","BUS_RELOCATION_COMPLETE","TRUCK_ARRIVE","DRONE_RETURN","BATTERY_CHARGE_COMPLETE"})
EVENT_NAME_TO_ID: Final[dict[str,int]] = {name: int(spec.event_id) for name, spec in DECISION_EVENT_SPECS.items()}
REQUIRED_EVENT_COVERAGE: Final[dict[str,set[str]]] = {
    "assignment": {"PARCEL_RELEASE"}, "truck": {"TRUCK_AVAILABLE"}, "bus": {"BUS_TERMINAL_DEPARTURE","BUS_STATION_ARRIVAL"}, "station": {"STATION_OPERATION"},
}

def normalize_decision_event_type(value: object) -> str:
    name = value.name if isinstance(value, IntEnum) else str(value)
    name = LEGACY_EVENT_ALIASES.get(name, name)
    if name not in DECISION_EVENT_SPECS:
        raise ValueError(f"Unknown or non-decision MAPPO event type {value!r}; expected one of {sorted(DECISION_EVENT_SPECS)}")
    return name

def is_decision_event(event_type: object) -> bool:
    try:
        normalize_decision_event_type(event_type); return True
    except ValueError:
        return False

def decision_event_id(value: object) -> int:
    return int(DECISION_EVENT_SPECS[normalize_decision_event_type(value)].event_id)

def decision_event_agent(value: object) -> str:
    return DECISION_EVENT_SPECS[normalize_decision_event_type(value)].agent_type

def validate_agent_event(agent_type: str, event_type: object, event_type_id: int|None=None) -> str:
    name = normalize_decision_event_type(event_type)
    spec = DECISION_EVENT_SPECS[name]
    if str(agent_type) != spec.agent_type:
        raise ValueError(f"Decision event {name} belongs to {spec.agent_type}, not {agent_type}")
    if event_type_id is not None and int(event_type_id) != int(spec.event_id):
        raise ValueError(f"Decision event id mismatch for {name}: expected {int(spec.event_id)}, got {event_type_id}")
    return name
