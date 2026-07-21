from training.mappo_async import validate_decision
from training.mappo_networks import event_embedding

def test_required_event_mapping_and_bus_embeddings_distinguish_load_charge():
    for agent,event in [("assignment","PARCEL_RELEASE"),("truck","TRUCK_AVAILABLE"),("bus","BUS_TERMINAL_DEPARTURE"),("bus","BUS_STATION_ARRIVAL"),("station","STATION_OPERATION")]:
        validate_decision(agent,event)
    assert event_embedding("BUS_TERMINAL_DEPARTURE") != event_embedding("BUS_STATION_ARRIVAL")
