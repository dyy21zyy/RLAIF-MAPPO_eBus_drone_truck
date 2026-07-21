"""Informative pair selection and reproducible A/B display randomization."""
from __future__ import annotations
import random
from typing import Any

SIGNALS = ("objective", "entropy", "uncertainty", "disagreement", "urgent_deadline", "truck_capacity_pressure", "passenger_congestion", "low_bus_soc", "locker_congestion", "battery_scarcity", "station_power_risk")


def _score(row: list[float], names: list[str]) -> float:
    d = dict(zip(names, row))
    return float(d.get("estimated_lateness_norm", d.get("estimated_lateness", 0.0))) + float(d.get("estimated_time_norm", d.get("travel_time", 0.0))) - float(d.get("resource_margin_norm", d.get("power_margin", 0.0)))


def select_informative_pairs(state: dict[str, Any], *, max_pairs: int = 4) -> list[dict[str, Any]]:
    names = list(state["candidate_feature_names"]); rows = state["candidate_features"]
    feasible = [i for i, ok in enumerate(state["action_masks"]) if ok]
    pairs=[]
    for a_i, a in enumerate(feasible):
        for b in feasible[a_i+1:]:
            urgency = abs(_score(rows[a], names) - _score(rows[b], names))
            pairs.append((urgency, a, b))
    pairs.sort(key=lambda x: x[0])
    return [{"state_id": state["state_id"], "agent_type": state["agent_type"], "event_type": state["event_type"], "original_pair_order": [a,b], "selection_signals": list(SIGNALS[:4])} for _,a,b in pairs[:max_pairs]]


def randomize_display_order(pair: dict[str, Any], *, seed: int) -> dict[str, Any]:
    original = list(pair["original_pair_order"])
    display = original[:]
    random.Random(f"{seed}:{pair.get('state_id')}:{original}").shuffle(display)
    return {**pair, "display_order": display}


def resolve_original_winner(record: dict[str, Any]) -> dict[str, Any]:
    ans = str(record.get("evaluator_answer", "")).lower()
    if ans in {"tie", "abstain"}:
        return {**record, "resolved_original_winner": ans, "usable_for_training": False}
    if ans not in {"a", "b"}:
        raise ValueError("evaluator_answer must be A, B, tie, or abstain")
    display = list(record["display_order"]); winner = display[0 if ans == "a" else 1]
    return {**record, "resolved_original_winner": winner, "usable_for_training": True}
