from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any
import hashlib, json
MATCHED_SEED_FIELDS=("parcel_seed","passenger_seed","timetable_seed","physical_bus_seed","station_load_seed")
@dataclass(frozen=True)
class ScenarioFamilyMember:
    scenario_family_id: str; master_seed: int; split: str; baseline_config_hash: str; sensitivity_factor: str; sensitivity_value: Any; intended_override: dict[str,Any]; scenario_content_hash: str; instance_hash: str; artifact_hashes: dict[str,str]; seed_tuple: dict[str,int]
    def to_dict(self): return asdict(self)
def scenario_family_id(*, split: str, master_seed: int, sensitivity_factor: str, sensitivity_value: Any) -> str:
    return hashlib.sha256(json.dumps({"split":split,"master_seed":master_seed,"factor":sensitivity_factor,"value":sensitivity_value},sort_keys=True).encode()).hexdigest()[:16]
def validate_matched_master_seeds(members: list[ScenarioFamilyMember]) -> None:
    by={}
    for m in members: by.setdefault(m.master_seed,[]).append(m)
    for seed, rows in by.items():
        ref={k:rows[0].seed_tuple.get(k) for k in MATCHED_SEED_FIELDS}
        for r in rows[1:]:
            cur={k:r.seed_tuple.get(k) for k in MATCHED_SEED_FIELDS}
            if cur != ref: raise ValueError(f"unmatched scenario-family stochastic seeds for master_seed={seed}: {cur} != {ref}")
