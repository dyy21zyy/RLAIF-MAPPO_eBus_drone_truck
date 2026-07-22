from __future__ import annotations
import argparse,json
from pathlib import Path
from datetime import datetime, timezone
from training.event_schema import OBSERVATION_SCHEMA_VERSION,CANDIDATE_SCHEMA_VERSION,EVENT_SCHEMA_VERSION,REQUIRED_EVENT_COVERAGE

def generate_rows(count_per_event:int=40):
    events={'assignment':['PARCEL_RELEASE'],'truck':['TRUCK_AVAILABLE'],'bus':['BUS_TERMINAL_DEPARTURE','BUS_STATION_ARRIVAL'],'station':['STATION_OPERATION']}
    i=0
    for agent, evs in events.items():
        for scenario in range(count_per_event):
            for event in evs:
                i+=1; base=float(i)
                yield {"preference_id":f"smoke-{agent}-{event}-{scenario}","scenario_id":f"scenario-{scenario}","episode_id":"episode-0","state_id":f"state-{scenario}-{event}","agent_type":agent,"event_type":event,"state_feature_names":["load","time"],"state_features":[base,base%7],"candidate_a_feature_names":["score","cost"],"candidate_a_features":[base+2,1.0],"candidate_b_feature_names":["score","cost"],"candidate_b_features":[base,2.0],"original_candidate_a_id":f"{agent}-a-{scenario}-{event}","original_candidate_b_id":f"{agent}-b-{scenario}-{event}","displayed_first_candidate_id":f"{agent}-b-{scenario}-{event}","displayed_second_candidate_id":f"{agent}-a-{scenario}-{event}","original_outcome":"candidate_a","label_source":"synthetic_smoke","evaluator_model":None,"evaluator_prompt_version":None,"observation_schema_version":OBSERVATION_SCHEMA_VERSION,"candidate_schema_version":CANDIDATE_SCHEMA_VERSION,"event_schema_version":EVENT_SCHEMA_VERSION,"created_at":"2026-07-22T00:00:00Z"}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',required=True); ns=ap.parse_args(); p=Path(ns.output); p.parent.mkdir(parents=True,exist_ok=True); p.write_text('\n'.join(json.dumps(r,sort_keys=True) for r in generate_rows())+'\n')
if __name__=='__main__': main()

# backwards-compatible name
def rows():
    return generate_rows(4)
