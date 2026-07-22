from __future__ import annotations
import random
from typing import Sequence
class ScenarioSampler:
    def __init__(self, scenario_ids: Sequence[str], *, mode: str, seed: int) -> None:
        if not scenario_ids: raise ValueError('scenario_ids must be non-empty')
        if mode not in {'sequential','shuffled_cycle','uniform_random'}: raise ValueError(f'unknown scenario sampler mode: {mode}')
        self.scenario_ids=tuple(scenario_ids); self.mode=mode; self.rng=random.Random(int(seed)); self.position=0; self.cycle=0; self._order=[]; self._idx=0
    def _new_cycle(self):
        self._order=list(self.scenario_ids)
        if self.mode=='shuffled_cycle': self.rng.shuffle(self._order)
        self._idx=0; self.cycle += 1
    def next_scenario_id(self) -> str:
        if self.mode=='uniform_random':
            self.position += 1; return self.rng.choice(self.scenario_ids)
        if not self._order or self._idx>=len(self._order): self._new_cycle()
        sid=self._order[self._idx]; self._idx += 1; self.position += 1; return sid
