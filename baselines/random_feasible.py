"""Uniform random feasible assignment baseline."""
from __future__ import annotations
import random
from baselines.common import feasible_indices

class RandomFeasiblePolicy:
    name = "random_feasible"
    def __init__(self, seed: int = 0): self.rng = random.Random(seed)
    def select_action(self, observation, env=None):
        feasible = feasible_indices(observation["action_mask"])
        return self.rng.choice(feasible) if feasible else 0
