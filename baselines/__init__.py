"""Stage 8 assignment and charging baselines."""
from baselines.battery_threshold import BatteryThresholdPolicy
from baselines.bus_drone_only import BusDroneOnlyPolicy
from baselines.no_charge import NoChargePolicy
from baselines.random_feasible import RandomFeasiblePolicy
from baselines.rule_based import RuleBasedPolicy
from baselines.truck_drone import TruckDronePolicy
from baselines.truck_only import TruckOnlyPolicy
from baselines.uniform_charge import UniformChargePolicy


def build_assignment_policy(name: str, seed: int = 0):
    policies = {"truck_only": TruckOnlyPolicy, "bus_drone_only": BusDroneOnlyPolicy, "truck_drone": TruckDronePolicy, "rule_based": RuleBasedPolicy}
    if name == "random_feasible": return RandomFeasiblePolicy(seed)
    if name not in policies: raise KeyError(f"Unknown baseline assignment policy: {name}")
    return policies[name]()


def build_bus_policy(name: str):
    if name == "no_charge": return NoChargePolicy()
    if name == "uniform_30": return UniformChargePolicy(30)
    if name == "uniform_60": return UniformChargePolicy(60)
    if name == "battery_threshold": return BatteryThresholdPolicy()
    raise KeyError(f"Unknown baseline bus policy: {name}")
