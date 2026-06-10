"""Never-charge bus baseline."""
from baselines.common import fallback_action
class NoChargePolicy:
    name = "no_charge"
    def select_action(self, observation, env=None): return fallback_action(observation["action_mask"], 0)
