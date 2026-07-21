"""Agent-specific prompt builders for multi-agent RLAIF."""
from __future__ import annotations
import json
from typing import Any
PROMPT_VERSION_V2="v2"
FACTS={
"assignment":["parcel deadline","priority","mode","target station","truck availability","bus opportunity","drone reachability","locker occupancy","drone availability","full batteries","station power","estimated completion"],
"truck":["parcel IDs","priorities","deadlines","batch weight","batch volume","utilization","route stops","distance","route time","estimated lateness","terminal synchronization","station synchronization"],
"bus":["physical bus","trip","parcel batch","target stations","total freight","unloading profile","passenger load","schedule delay","SoC","future-trip feasibility","charging duration","energy added","waiting passengers","passenger delay","charger occupancy","station power","downstream energy","next-trip slack"],
"station":["drone-parcel matches","parcel urgency","mission duration","available drones","full batteries","depleted batteries","charging batteries","charging slots","new charging starts","locker occupancy","projected station power","future dispatch capacity"],
}

def build_agent_prompt(state:dict[str,Any], pair:dict[str,Any])->dict[str,Any]:
    agent=state["agent_type"]; facts=FACTS[agent]
    display=pair.get("display_order", pair["original_pair_order"])
    payload={"agent_type":agent,"event_type":state["event_type"],"event_time":state["event_time"],"facts_required":facts,"state_features":dict(zip(state["state_feature_names"],state["state_features"])),"action_a":_cand(state,display[0]),"action_b":_cand(state,display[1]),"provenance":state.get("data_provenance",{})}
    text=(f"Evaluate two {agent} decisions using only objective facts. Do not infer labels from heuristics, environment reward, feasibility, or rule fallbacks. "
          "Return JSON with answer A, B, tie, or abstain and confidence. Required facts: "+", ".join(facts)+".\n"+json.dumps(payload,sort_keys=True))
    return {"prompt_id":f"{state['state_id']}:{display[0]}:{display[1]}","state_id":state["state_id"],"agent_type":agent,"event_type":state["event_type"],"prompt_version":PROMPT_VERSION_V2,"action_a":display[0],"action_b":display[1],"prompt_text":text,"metadata":{"original_pair_order":pair["original_pair_order"],"display_order":display}}

def _cand(state:dict[str,Any], idx:int)->dict[str,Any]:
    return {"index":idx,"action":state["candidate_actions"][idx],"features":dict(zip(state["candidate_feature_names"],state["candidate_features"][idx])),"feasible":state["action_masks"][idx]}
