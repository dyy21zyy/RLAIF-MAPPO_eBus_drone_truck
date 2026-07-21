from __future__ import annotations
from experiments.run_ablation import main
SUPPORTED_ABLATIONS={"no_rlaif","assignment_only_rlaif","full_multi_agent_rlaif","no_event_time_discount","no_explicit_event_embedding","no_centralized_critic","no_learned_station","no_passenger_aware_charging","single_parcel_truck","truck_batching","greedy_bus_loading","learned_bus_loading","automatic_battery_charging","learned_battery_charging","preference_label_volume","per_agent_lambda"}
def validate_ablations(config):
    for a in config.get('ablations',[]):
        if a.get('name') not in SUPPORTED_ABLATIONS: raise ValueError(f"unsupported ablation: {a.get('name')}")
        if not a.get('config_switch') or not a.get('checkpoint'): raise ValueError(f"ablation {a.get('name')} requires config_switch and separate checkpoint")
    return True
