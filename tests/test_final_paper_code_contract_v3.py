from training.event_schema import DECISION_EVENT_SPECS, AUTOMATIC_EVENT_TYPES, REQUIRED_EVENT_COVERAGE
from training.mappo_networks import CandidateScoringActor
def test_final_paper_code_contract_v3():
 assert set(DECISION_EVENT_SPECS)=={'PARCEL_RELEASE','TRUCK_AVAILABLE','BUS_TERMINAL_DEPARTURE','BUS_STATION_ARRIVAL','STATION_OPERATION'}
 assert REQUIRED_EVENT_COVERAGE['bus']=={'BUS_TERMINAL_DEPARTURE','BUS_STATION_ARRIVAL'}
 assert 'BUS_ARRIVE_STOP' in AUTOMATIC_EVENT_TYPES
 a=CandidateScoringActor(3,2,(4,),event_embedding_dim=4); assert hasattr(a,'event_embedding')
 assert 'assignment' in REQUIRED_EVENT_COVERAGE and 'truck' in REQUIRED_EVENT_COVERAGE and 'station' in REQUIRED_EVENT_COVERAGE
