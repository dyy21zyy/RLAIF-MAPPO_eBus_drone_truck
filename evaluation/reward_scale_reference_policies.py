"""Deterministic reference policies for reward-scale estimation rollouts."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class ReferencePolicy:
    name: str
    version: int = 1
    preference: str = "first"
    def select_action(self, observation: dict[str, Any]) -> int:
        mask = list(observation.get("action_mask", []))
        feasible = [i for i, ok in enumerate(mask) if bool(ok)]
        if not feasible: raise RuntimeError(f"{self.name} found no feasible action")
        candidates = observation.get("candidates") or observation.get("candidate_actions") or []
        if self.preference == "last": return feasible[-1]
        if self.preference == "coverage":
            scored=[]
            for i in feasible:
                cand = candidates[i] if isinstance(candidates, (list, tuple)) and i < len(candidates) else i
                text = str(cand).lower()
                score = sum(tok in text for tok in ("charge","dispatch","drone","locker","bus","station"))
                scored.append((-score, i))
            return sorted(scored)[0][1]
        return feasible[0]
    def metadata(self) -> dict[str, Any]: return {"name": self.name, "version": self.version}

def truck_direct_reference() -> ReferencePolicy: return ReferencePolicy("truck_direct_reference", 1, "first")
def integrated_rule_reference() -> ReferencePolicy: return ReferencePolicy("integrated_rule_reference", 1, "last")
def coverage_reference() -> ReferencePolicy: return ReferencePolicy("coverage_reference", 1, "coverage")

POLICY_FACTORIES = {f().name: f for f in (truck_direct_reference, integrated_rule_reference, coverage_reference)}
def get_reference_policy(name: str) -> ReferencePolicy: return POLICY_FACTORIES[name]()
def get_reference_policies(names=None): return [get_reference_policy(n) for n in (names or POLICY_FACTORIES.keys())]
