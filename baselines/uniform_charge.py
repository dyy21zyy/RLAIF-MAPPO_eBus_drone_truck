"""Fixed-duration charging baselines."""
from baselines.common import duration_action
class UniformChargePolicy:
    def __init__(self, seconds=30): self.seconds=float(seconds); self.name=f"uniform_{int(seconds)}"
    def select_action(self, observation, env=None):
        durations = env.config["bus"]["charging_actions_sec"] if env is not None else list(range(len(observation["action_mask"])))
        return duration_action(observation["action_mask"], durations, self.seconds)
