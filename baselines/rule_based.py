"""Interpretable objective-feature baseline; never an RLAIF label source."""
from envs.state_builder import build_candidate_action_features
from baselines.common import fallback_action

class RuleBasedPolicy:
    name = "rule_based"
    def select_action(self, observation, env=None):
        mask = observation["action_mask"]
        feasible = [i for i, value in enumerate(mask) if value]
        if not feasible or env is None or env.current_decision is None:
            return fallback_action(mask)
        parcel = env.parcels[env.current_decision.event.payload["parcel_id"]]
        def score(action):
            f = build_candidate_action_features(env, parcel, action, True)
            occupancy = float(f["estimated_locker_load_after_assignment_norm"])
            power_stress = max(0.0, -float(f["estimated_station_power_margin_norm"]))
            return (float(f["estimated_lateness_norm"]), float(f["estimated_truck_distance_norm"]), occupancy, power_stress, action)
        return min(feasible, key=score)
