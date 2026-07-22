from tests.real_readiness_helpers import run_pilot, load

def test_bus_embeddings_update(tmp_path):
    b=load(run_pilot(tmp_path),"mappo_update_report.json")["bus_event_embeddings"]
    assert set(b)=={"BUS_TERMINAL_DEPARTURE","BUS_STATION_ARRIVAL"}
    assert all(v["transition_count"]>0 and v["gradient_norm"]>0 and v["embedding_delta_norm"]>0 for v in b.values())
