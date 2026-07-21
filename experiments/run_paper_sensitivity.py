from __future__ import annotations
from experiments.run_sensitivity import main, run_sensitivity
SENSITIVITY_DIMENSIONS={"parcel_count","parcel_arrival_intensity","urgent_ratio","passenger_demand","truck_count","truck_weight_capacity","truck_volume_capacity","headway","physical_bus_count","bus_freight_capacity","initial_bus_soc","drone_count","initial_full_batteries","charging_slots","locker_capacity","station_power","station_count","rlaif_lambda","preference_label_volume"}
MODES={"fixed_policy_robustness","retrained_policy_sensitivity"}
def validate_sensitivity(config):
    modes={m.get('mode') for m in config.get('experiments',[])}
    if modes and not modes<=MODES: raise ValueError(f'unsupported sensitivity modes: {modes-MODES}')
    if len(modes)>1 and any(m.get('combined_table') for m in config.get('experiments',[])): raise ValueError('fixed-policy and retrained-policy sensitivity must remain separate')
    dims={d.get('name') for e in config.get('experiments',[]) for d in e.get('dimensions',[])}
    bad=dims-SENSITIVITY_DIMENSIONS
    if bad: raise ValueError(f'unsupported sensitivity dimensions: {bad}')
    return True
