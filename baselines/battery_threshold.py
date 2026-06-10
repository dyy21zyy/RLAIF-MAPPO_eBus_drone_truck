"""Charge when bus state of charge falls below a configured threshold."""
from baselines.common import duration_action, fallback_action
class BatteryThresholdPolicy:
    name = "battery_threshold"
    def __init__(self, threshold_fraction=0.35, charge_seconds=60): self.threshold_fraction=float(threshold_fraction); self.charge_seconds=float(charge_seconds)
    def select_action(self, observation, env=None):
        if env is None or len(observation.get("features", [])) < 2: return fallback_action(observation["action_mask"])
        soc_fraction = float(observation["features"][1])
        if soc_fraction >= self.threshold_fraction: return fallback_action(observation["action_mask"], 0)
        return duration_action(observation["action_mask"], env.config["bus"]["charging_actions_sec"], self.charge_seconds)
