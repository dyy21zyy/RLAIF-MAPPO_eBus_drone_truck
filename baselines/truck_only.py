"""Direct-truck assignment baseline."""
from baselines.common import fallback_action

class TruckOnlyPolicy:
    name = "truck_only"
    def select_action(self, observation, env=None):
        return fallback_action(observation["action_mask"], 0)
