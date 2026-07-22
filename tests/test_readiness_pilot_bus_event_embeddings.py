from tests.readiness_test_utils import run_pilot, load
def test_bus_event_embeddings_update(tmp_path):
 out=run_pilot(tmp_path); b=load(out,'mappo_update_report.json')['agents']['bus'];
 for e in ['BUS_TERMINAL_DEPARTURE','BUS_STATION_ARRIVAL']: assert b['event_counts'][e]>0 and b['event_embedding_gradient_norm'][e]>0 and b['event_embedding_delta_norm'][e]>0
