from training.mappo_networks import build_actor_registry
from training.event_schema import decision_event_id

def test_bus_events_share_one_actor_and_optimizer():
    actors=build_actor_registry({"assignment":(2,2),"truck":(2,2),"bus":(2,2),"station":(2,2)}, {"default":[4]}, event_embedding_dim=2)
    assert "bus" in actors and not any("bus_loading" in k or "bus_charging" in k for k in actors.keys())
    assert decision_event_id("BUS_TERMINAL_DEPARTURE") != decision_event_id("BUS_STATION_ARRIVAL")
