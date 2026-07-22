"""Canonical deterministic seed tuples for generated scenarios."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import random
from typing import Any

@dataclass(frozen=True)
class ScenarioSeedTuple:
    base_seed: int
    parcel_seed: int
    passenger_seed: int
    timetable_seed: int
    physical_bus_seed: int
    station_load_seed: int
    initial_bus_energy_seed: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_any(cls, value: "ScenarioSeedTuple | dict[str, Any]") -> "ScenarioSeedTuple":
        if isinstance(value, cls):
            return value
        aliases = dict(value)
        if "base" in aliases and "base_seed" not in aliases: aliases["base_seed"] = aliases["base"]
        if "parcel" in aliases and "parcel_seed" not in aliases: aliases["parcel_seed"] = aliases["parcel"]
        if "passenger" in aliases and "passenger_seed" not in aliases: aliases["passenger_seed"] = aliases["passenger"]
        if "timetable" in aliases and "timetable_seed" not in aliases: aliases["timetable_seed"] = aliases["timetable"]
        if "physical_bus" in aliases and "physical_bus_seed" not in aliases: aliases["physical_bus_seed"] = aliases["physical_bus"]
        if "station_base_load_seed" in aliases and "station_load_seed" not in aliases: aliases["station_load_seed"] = aliases["station_base_load_seed"]
        return cls(**{f: int(aliases[f]) for f in cls.__dataclass_fields__})

def derive_scenario_seed_tuple(base_seed: int) -> ScenarioSeedTuple:
    """Derive component seeds with a stable SplitMix-like Random stream."""
    rng = random.Random(int(base_seed))
    vals = [rng.randrange(1, 2**31 - 1) for _ in range(6)]
    return ScenarioSeedTuple(int(base_seed), vals[0], vals[1], vals[2], vals[3], vals[4], vals[5])

def apply_seed_tuple(config: dict[str, Any], seed_tuple: ScenarioSeedTuple | dict[str, Any]) -> dict[str, Any]:
    t = ScenarioSeedTuple.from_any(seed_tuple)
    config.setdefault("project", {})["seed"] = t.base_seed
    seeds = config.setdefault("seeds", {})
    seeds.update({
        "network_seed": t.base_seed,
        "parcel_seed": t.parcel_seed,
        "passenger_seed": t.passenger_seed,
        "passenger_baseline_rate_seed": t.passenger_seed + 104729,
        "travel_time_seed": t.timetable_seed,
        "timetable_seed": t.timetable_seed,
        "physical_bus_seed": t.physical_bus_seed,
        "station_base_load_seed": t.station_load_seed,
        "station_load_seed": t.station_load_seed,
        "initial_bus_energy_seed": t.initial_bus_energy_seed,
    })
    config["scenario_seed_tuple"] = t.to_dict()
    return config
