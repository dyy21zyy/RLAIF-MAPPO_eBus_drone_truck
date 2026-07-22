from __future__ import annotations
import csv
from pathlib import Path
from data_pipeline.build_instance import build_instance
from envs import DynamicDeliveryEnv, first_feasible_policy

CONFIG = Path(__file__).parents[1] / "configs/shanghai_small.yaml"

def write_csv(path, rows, fields):
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

def make_env(tmp_path, *, trips=2, stops=6):
    inst=build_instance(CONFIG, fallback=True, output_root=tmp_path)
    d=Path(inst['output_directory'])
    stop_rows=list(csv.DictReader(open(d/'bus_stop_times.csv')))
    stop_ids=[]
    for r in stop_rows:
        if r['stop_id'] not in stop_ids: stop_ids.append(r['stop_id'])
        if len(stop_ids)>=stops: break
    trips_rows=[]; st=[]; mapping=[]
    for t in range(trips):
        tid=f'test_trip_{t:03d}'; start=t*120.0; freight=(t==0)
        trips_rows.append({'trip_id':tid,'route_id':'route_test','start_time':start,'freight_allowed':str(freight)})
        for i,sid in enumerate(stop_ids):
            tm=start+i*5.0
            st.append({'trip_id':tid,'stop_id':sid,'stop_sequence':i+1,'arrival_time':tm,'departure_time':tm,'freight_allowed':str(freight)})
        mapping.append({'trip_id':tid,'bus_id':'bus_000','sequence_index':t,'scheduled_start_min':start,'scheduled_end_min':start+(stops-1)*5,'previous_trip_id': '' if t==0 else f'test_trip_{t-1:03d}','next_trip_id':'' if t==trips-1 else f'test_trip_{t+1:03d}','relocation_time_min':2.0,'minimum_layover_min':3.0})
    write_csv(d/'bus_trips.csv', trips_rows, ['trip_id','route_id','start_time','freight_allowed'])
    write_csv(d/'bus_stop_times.csv', st, ['trip_id','stop_id','stop_sequence','arrival_time','departure_time','freight_allowed'])
    write_csv(d/'trip_to_bus.csv', mapping, ['trip_id','bus_id','sequence_index','scheduled_start_min','scheduled_end_min','previous_trip_id','next_trip_id','relocation_time_min','minimum_layover_min'])
    write_csv(d/'physical_buses.csv', [{'bus_id':'bus_000','initial_location_id':stop_ids[0],'initial_soc_kwh':150.0,'battery_capacity_kwh':160.0,'minimum_safe_energy_kwh':40.0}], ['bus_id','initial_location_id','initial_soc_kwh','battery_capacity_kwh','minimum_safe_energy_kwh'])
    # Move parcel releases outside the bus diagnostic window so focused tests exercise bus causality quickly.
    parcels=list(csv.DictReader(open(d/'parcels.csv')))
    for r in parcels:
        r['release_time_min']='470.0'; r['release_time']='470.0'
    if parcels:
        write_csv(d/'parcels.csv', parcels, list(parcels[0].keys()))
    # deterministic passenger at ordinary stop 1 to downstream ordinary stop 3
    write_csv(d/'passenger_arrivals.csv', [{'passenger_event_id':'pe_test','origin_stop_id':stop_ids[1],'destination_stop_id':stop_ids[3],'arrival_time_min':1.0,'passenger_count':3}], ['passenger_event_id','origin_stop_id','destination_stop_id','arrival_time_min','passenger_count'])
    return DynamicDeliveryEnv(d/'instance.json')

def run_env(env, limit=500, choose_charge=False):
    obs,_=env.reset(); seen=[]
    while obs['agent']!='terminal' and len(seen)<limit:
        seen.append(obs)
        if choose_charge and obs.get('event_type_detail')=='BUS_STATION_ARRIVAL' and len(obs['action_mask'])>1:
            action=1 if obs['action_mask'][1] else first_feasible_policy(obs)
        else:
            action=first_feasible_policy(obs)
        obs,*_=env.step(action)
    return seen
