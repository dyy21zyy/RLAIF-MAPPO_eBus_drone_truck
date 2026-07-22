from training.config_resolver import resolved_environment_parameter_report


def test_report_contains_required_values():
    r=resolved_environment_parameter_report({'run_classification':'formal','env':{'fallback':False,'config_path':'configs/paper/base_medium.yaml'},'reward':{'truck_cost':0,'scale_artifact':'x','scale_artifact_hash':'h'}})
    for k in ['bus_horizons','bus_charging_power_kw','drone_payload','drone_radius','drone_speed','maximum_drone_round_trip','battery_charging_power','battery_charging_duration','station_capacity','station_charging_slots','truck_cost_coefficients','reward_weights','reward_scale_artifact','reward_scale_artifact_hash']:
        assert k in r
