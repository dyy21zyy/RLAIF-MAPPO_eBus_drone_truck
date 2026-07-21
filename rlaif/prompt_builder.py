"""Versioned pair selection and JSON-only prompts for AI preference evaluation."""

from __future__ import annotations

import json
from typing import Any

PROMPT_VERSION = "v1"


def _best(actions: list[dict[str, Any]], features: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    feasible = [action for action in actions if action["feasible"]]
    return min(feasible, key=lambda action: (
        features[action["action_name"]]["estimated_lateness_norm"],
        features[action["action_name"]]["estimated_delivery_time_norm"],
        features[action["action_name"]]["estimated_truck_distance_norm"],
        action["action_id"],
    ), default=None)


def select_action_pairs(state: dict[str, Any]) -> list[tuple[str, str]]:
    """Select the bounded, prescribed cross-mode comparisons."""
    actions = state["candidate_actions"]
    features = state["candidate_action_features"]
    td = next((action for action in actions if action["action_name"] == "TD"), None)
    tbd = [action for action in actions if action["action_name"].startswith("TBD_")]
    tld = [action for action in actions if action["action_name"].startswith("TLD_")]
    pairs: list[tuple[str, str]] = []
    nearest_tbd = min(
        (action for action in tbd if action["feasible"]),
        key=lambda action: features[action["action_name"]]["estimated_drone_time_norm"], default=None,
    )
    nearest_tld = min(
        (action for action in tld if action["feasible"]),
        key=lambda action: features[action["action_name"]]["estimated_drone_time_norm"], default=None,
    )
    if td and nearest_tbd:
        pairs.append((td["action_name"], nearest_tbd["action_name"]))
    if td and nearest_tld:
        pairs.append((td["action_name"], nearest_tld["action_name"]))
    best_tbd, best_tld = _best(tbd, features), _best(tld, features)
    if best_tbd and best_tld:
        pairs.append((best_tbd["action_name"], best_tld["action_name"]))
    return list(dict.fromkeys(pairs))


def build_prompt_text(state: dict[str, Any], action_a: str, action_b: str) -> str:
    context = {
        "parcel": state["parcel"],
        "system_state_summary": state["system_state_summary"],
        "data_sources": state.get("data_sources", {}),
        "action_a": {"name": action_a, **state["candidate_action_features"][action_a]},
        "action_b": {"name": action_b, **state["candidate_action_features"][action_b]},
    }
    return f"""You are evaluating two parcel-assignment actions. Choose the operationally preferred assignment.
Consider: (1) hard feasibility; (2) deadline reliability; (3) parcel weight and drone payload feasibility;
(4) truck travel burden; (5) quantified bus opportunity and passenger impact when available; (6) locker congestion risk;
(7) drone availability; (8) full battery availability; (9) station power stress; (10) urgent parcel service;
(11) remote customer service; and (12) avoiding excessive passenger-service impact.
Do not choose an infeasible action unless both options are infeasible.
If both actions are feasible, choose the one with the better operational trade-off.
Use the data_sources block to distinguish real transit observations from inherited or fallback settings.
Do not treat inherited or fallback fields as real-world observations.
Return only valid JSON. chosen must be either action_a or action_b. rejected must be the other action.
Use exactly this schema: {{"chosen":"ACTION_NAME","rejected":"ACTION_NAME","confidence":0.0,"reason":"..."}}
Do not include markdown or surrounding prose.
Operational context:
{json.dumps(context, sort_keys=True)}"""


def build_prompt_records(states: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for state in states:
        for pair_index, (action_a, action_b) in enumerate(select_action_pairs(state)):
            records.append({
                "prompt_id": f"{state['state_id']}:pair_{pair_index}",
                "state_id": state["state_id"],
                "action_a": action_a,
                "action_b": action_b,
                "prompt_version": PROMPT_VERSION,
                "prompt_text": build_prompt_text(state, action_a, action_b),
                "metadata": {"pair_index": pair_index, "feature_schema_version": state["feature_schema_version"]},
            })
    return records
