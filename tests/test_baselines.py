from types import SimpleNamespace
from baselines import (BatteryThresholdPolicy, BusDroneOnlyPolicy, NoChargePolicy, RandomFeasiblePolicy,
                       RuleBasedPolicy, TruckDronePolicy, TruckOnlyPolicy, UniformChargePolicy)

def test_truck_only_chooses_td(): assert TruckOnlyPolicy().select_action({"action_mask":[True,True]})==0

def test_random_feasible_only_samples_feasible():
    policy=RandomFeasiblePolicy(7); assert {policy.select_action({"action_mask":[False,True,False,True]}) for _ in range(50)} <= {1,3}

def test_integrated_baselines_fall_back_to_td():
    obs={"action_mask":[True,False,False,False,False]}
    assert BusDroneOnlyPolicy().select_action(obs)==0; assert TruckDronePolicy().select_action(obs)==0

def test_rule_based_returns_feasible_without_environment(): assert RuleBasedPolicy().select_action({"action_mask":[False,True,False]})==1

def test_charging_baselines_respect_masks():
    env=SimpleNamespace(config={"bus":{"charging_actions_sec":[0,30,60],"bus_battery_kwh":100}})
    observation={"action_mask":[False,True,False],"features":[0,10]}
    for policy in (NoChargePolicy(),UniformChargePolicy(60),BatteryThresholdPolicy()): assert policy.select_action(observation,env)==1
