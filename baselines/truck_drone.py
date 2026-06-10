"""Nearest feasible truck-locker-drone assignment baseline."""
from baselines.common import fallback_action

class TruckDronePolicy:
    name = "truck_drone"
    def select_action(self, observation, env=None):
        mask = observation["action_mask"]
        station_count = (len(mask) - 1) // 2
        candidates = [i for i in range(1 + station_count, 1 + 2 * station_count) if mask[i]]
        if not candidates or env is None or env.current_decision is None:
            return fallback_action(mask, 0)
        parcel_id = env.current_decision.event.payload["parcel_id"]
        return min(candidates, key=lambda action: float(env.drone_distance_m[env.drone_row_index[env.station_ids[action-1-station_count]], env.drone_column_index[parcel_id]]))
