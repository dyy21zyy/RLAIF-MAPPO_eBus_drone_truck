import pytest
from utils.config import load_config
from data_pipeline.build_instance import normalize_dynamic_config
from data_pipeline.build_bus_circulation import build_bus_circulation

def test_medium_relocation_layover_resolve_to_paper_values():
    c=normalize_dynamic_config(load_config('configs/paper/base_medium.yaml'))
    assert c['bus']['non_service_relocation_time_min']==30
    assert c['bus']['minimum_layover_time_min']==10

def test_conflicting_relocation_aliases_fail():
    c=load_config('configs/paper/base_medium.yaml'); c.setdefault('bus',{})['non_service_relocation_time_min']=5
    with pytest.raises(ValueError): normalize_dynamic_config(c)

def test_circulation_artifact_contains_resolved_values(tmp_path):
    c=normalize_dynamic_config(load_config('configs/paper/base_medium.yaml'))
    trips=[{'trip_id':'t1'},{'trip_id':'t2'}]
    stop_times=[{'trip_id':'t1','stop_id':'s','stop_sequence':0,'arrival_time_min':0,'departure_time_min':1,'arrival_time':0,'departure_time':1},{'trip_id':'t2','stop_id':'s','stop_sequence':0,'arrival_time_min':40,'departure_time_min':41,'arrival_time':40,'departure_time':41}]
    r=build_bus_circulation(trips, stop_times, c, tmp_path)['bus_circulation']
    assert r['non_service_relocation_time_min']==30 and r['minimum_layover_time_min']==10
