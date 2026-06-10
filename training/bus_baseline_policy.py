"""Fixed, mask-aware bus charging policies for Stage 6 assignment PPO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class BusBaselinePolicy:
    """Select a configured charging duration without learning bus behavior."""

    name: str = "uniform_30"
    battery_threshold_soc: float = 0.3
    max_charge_seconds: int = 120

    def __post_init__(self) -> None:
        if self.name not in {"no_charge", "uniform_30", "battery_threshold"}:
            raise ValueError(f"Unknown bus baseline policy: {self.name}")
        if not 0.0 <= self.battery_threshold_soc <= 1.0:
            raise ValueError("battery_threshold_soc must be in [0, 1]")
        if self.max_charge_seconds < 0:
            raise ValueError("max_charge_seconds must be non-negative")

    @staticmethod
    def _feasible_indices(action_mask: Sequence[bool]) -> list[int]:
        feasible = [index for index, allowed in enumerate(action_mask) if bool(allowed)]
        if not feasible:
            raise ValueError("Bus action mask contains no feasible action")
        return feasible

    def select_action(
        self,
        action_mask: Sequence[bool],
        charging_actions_sec: Sequence[int | float],
        *,
        bus_soc: float | None = None,
    ) -> int:
        """Return a feasible charging-action index under the selected baseline."""
        if len(action_mask) != len(charging_actions_sec) or not action_mask:
            raise ValueError("Bus action mask and charging durations must have equal non-zero length")
        feasible = self._feasible_indices(action_mask)

        def best_at_or_below(limit: float) -> int | None:
            candidates = [i for i in feasible if float(charging_actions_sec[i]) <= limit]
            return max(candidates, key=lambda i: (float(charging_actions_sec[i]), -i)) if candidates else None

        zero = next((i for i in feasible if float(charging_actions_sec[i]) == 0.0), None)
        if self.name == "no_charge":
            return zero if zero is not None else min(feasible, key=lambda i: float(charging_actions_sec[i]))
        if self.name == "uniform_30":
            selected = best_at_or_below(30.0)
            return selected if selected is not None else (zero if zero is not None else min(feasible))
        if bus_soc is None:
            raise ValueError("battery_threshold policy requires normalized bus_soc")
        if float(bus_soc) >= self.battery_threshold_soc:
            return zero if zero is not None else min(feasible, key=lambda i: float(charging_actions_sec[i]))
        selected = best_at_or_below(float(self.max_charge_seconds))
        return selected if selected is not None else (zero if zero is not None else min(feasible))


def build_bus_baseline_policy(config: dict[str, object]) -> BusBaselinePolicy:
    """Construct a baseline policy from the ``bus_baseline`` config section."""
    return BusBaselinePolicy(
        name=str(config.get("name", "uniform_30")),
        battery_threshold_soc=float(config.get("battery_threshold_soc", 0.3)),
        max_charge_seconds=int(config.get("max_charge_seconds", 120)),
    )
