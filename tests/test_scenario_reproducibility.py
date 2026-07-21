import copy
from pathlib import Path
from data_pipeline.build_instance import build_instance
from utils.config import load_config
import json

def test_same_seed_tuple_identical_hashes_and_parcel_seed_isolated(tmp_path):
    a=build_instance('configs/paper/base_medium.yaml', True, tmp_path/'a')
    b=build_instance('configs/paper/base_medium.yaml', True, tmp_path/'b')
    ma=a['scenario_manifest']['artifact_sha256']; mb=b['scenario_manifest']['artifact_sha256']
    assert ma['parcels']==mb['parcels'] and ma['bus_trips']==mb['bus_trips']
    cfg=copy.deepcopy(load_config('configs/paper/base_medium.yaml'))
    cfg['seeds']={"network_seed":0,"parcel_seed":999,"passenger_seed":2,"travel_time_seed":3,"initial_bus_energy_seed":4,"station_base_load_seed":5}
    p=tmp_path/'cfg.json'; p.write_text(json.dumps(cfg))
    c=build_instance(p, True, tmp_path/'c')
    mc=c['scenario_manifest']['artifact_sha256']
    assert mc['parcels'] != ma['parcels']
    assert mc['bus_trips'] == ma['bus_trips']
