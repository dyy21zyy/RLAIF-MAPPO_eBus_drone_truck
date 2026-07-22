"""Versioned formal result schema; failed rows are retained explicitly."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Literal
RESULT_SCHEMA_VERSION='formal_result_v1'
ResultStatus=Literal['success','failed_checkpoint_validation','failed_scenario_validation','failed_metric_validation','failed_runtime','skipped_missing_checkpoint','skipped_unsupported','failed']
REQUIRED_RESULT_FIELDS=('result_schema_version','method_id','method_display_name','training_seed','scenario_id','scenario_split','instance_hash','scenario_manifest_hash','scenario_bank_hash','policy_checkpoint_path','policy_checkpoint_hash','policy_algorithm','policy_rlaif_scope','reward_checkpoint_hashes','code_commit','resolved_evaluation_config_hash','formal_metrics','rlaif_decomposition','runtime','status','failure_reason','reward_scale_artifact_path','reward_scale_artifact_hash','reward_scale_training_bank_hash')

@dataclass(frozen=True)
class FormalResultRow:
    result_schema_version: str
    method_id: str
    method_display_name: str
    training_seed: int|None
    evaluation_seed: int|None
    scenario_id: str
    scenario_split: str
    instance_hash: str
    scenario_manifest_hash: str
    scenario_bank_hash: str
    policy_checkpoint_path: str|None
    policy_checkpoint_hash: str|None
    policy_algorithm: str|None
    policy_rlaif_scope: str
    reward_checkpoint_hashes: dict[str,str]
    code_commit: str
    resolved_evaluation_config_hash: str
    formal_metrics: dict[str,Any]
    rlaif_decomposition: dict[str,Any]
    runtime: float
    status: str
    failure_reason: str=''
    reward_scale_artifact_path: str|None=None
    reward_scale_artifact_hash: str|None=None
    reward_scale_training_bank_hash: str|None=None
    artifact_hashes: dict[str,str]|None=None

def build_result_row(**kwargs)->dict[str,Any]:
    data={k:kwargs.get(k) for k in REQUIRED_RESULT_FIELDS}
    data['result_schema_version']=kwargs.get('result_schema_version',RESULT_SCHEMA_VERSION)
    data['evaluation_seed']=kwargs.get('evaluation_seed')
    data['artifact_hashes']=kwargs.get('artifact_hashes',{})
    data['reward_scale_artifact_path']=kwargs.get('reward_scale_artifact_path')
    data['reward_scale_artifact_hash']=kwargs.get('reward_scale_artifact_hash')
    data['reward_scale_training_bank_hash']=kwargs.get('reward_scale_training_bank_hash')
    missing=[k for k in REQUIRED_RESULT_FIELDS if k not in data or data[k] is None and k not in ('training_seed','policy_checkpoint_path','policy_checkpoint_hash','policy_algorithm')]
    if missing: raise ValueError(f'missing formal result fields: {missing}')
    return data


RESULT_FIELDS = (
    'experiment_id','method_name','seed','scenario_id','instance_name','config_hash','rlaif_enabled',
    'assignment_policy_name','bus_policy_name','episode_reward','total_env_reward','total_rlaif_reward',
    'delivered_parcels','undelivered_parcels','on_time_delivery_rate','urgent_on_time_rate',
    'total_parcel_lateness','average_parcel_lateness','late_delivery_count','truck_total_distance',
    'truck_operating_cost','truck_direct_count','truck_to_terminal_count','truck_to_locker_count',
    'bus_charging_count','bus_charging_energy','passenger_delay','bus_operating_delay','minimum_bus_soc',
    'drone_delivery_count','battery_shortage_count','locker_overflow_amount','locker_overflow_duration',
    'power_overload_amount','power_overload_duration','peak_station_load','infeasible_action_count',
    'fallback_feasibility_events','runtime_seconds','total_normalized_cost','environment_reward',
    'rlaif_reward_assignment','rlaif_reward_bus','rlaif_reward_truck','rlaif_reward_station',
    'fulfillment_rate','on_time_over_all_released','on_time_over_delivered','urgent_on_time_fulfillment',
    'average_lateness','maximum_lateness','truck_distance','truck_weight_utilization','truck_volume_utilization',
    'parcels_per_truck_route','bus_freight_utilization','bus_energy','battery_safety_violations',
    'waiting_passenger_minutes','onboard_additional_delay_passenger_minutes','drone_missions','drone_utilization',
    'full_batteries','depleted_batteries','charging_batteries','charging_slot_utilization','locker_occupancy',
    'locker_overflow','station_peak_power','overload_kw_min','infeasible_actions','runtime','reward_scale_artifact_path','reward_scale_artifact_hash','reward_scale_training_bank_hash','status','error_message')
NUMERIC_DEFAULTS = {field: 0.0 for field in RESULT_FIELDS if field not in {'experiment_id','method_name','scenario_id','instance_name','config_hash','assignment_policy_name','bus_policy_name','status','error_message','rlaif_enabled'}}

# Backward-compatible legacy normalization, but no longer hides missing formal metrics.
LEGACY_FIELDS=('experiment_id','method_name','seed','scenario_id','instance_name','config_hash','rlaif_enabled','status','error_message')
def normalize_result(record):
    result={field: NUMERIC_DEFAULTS.get(field, '') for field in RESULT_FIELDS}
    result.update(record)
    result['rlaif_enabled']=bool(result.get('rlaif_enabled', False))
    if 'seed' in result and result['seed']!='': result['seed']=int(result['seed'])
    return result
