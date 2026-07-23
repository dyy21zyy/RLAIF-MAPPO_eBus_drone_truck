import json, pytest
from pathlib import Path
from rlaif.ai_evaluator import APISettings
from experiments.generate_formal_multiagent_preferences import route_event, feasible_pairs, validate_structured_response, cache_key, generate


def test_four_agent_event_routing_and_bus_dual_events():
    assert route_event('PARCEL_RELEASE') == 'assignment'
    assert route_event('TRUCK_AVAILABLE') == 'truck'
    assert route_event('BUS_TERMINAL_DEPARTURE') == 'bus'
    assert route_event('BUS_STATION_ARRIVAL') == 'bus'
    assert route_event('STATION_OPERATION') == 'station'
    with pytest.raises(ValueError): route_event('BUS_TRIP_START')


def test_feasible_pair_only_construction():
    st={'candidate_actions':[{'candidate_id':'a','features':[1], 'feasible':True},{'candidate_id':'b','features':[2], 'feasible':False},{'candidate_id':'c','features':[3], 'feasible':True}], 'action_mask':[True,False,True]}
    pairs=feasible_pairs(st)
    assert len(pairs)==1 and {pairs[0][0]['candidate_id'], pairs[0][1]['candidate_id']} == {'a','c'}


def test_structured_response_validation_rejects_bad_values():
    assert validate_structured_response('{"preferred":"A","confidence":0.8,"criteria":{"timeliness":"A"},"reason":"better feasibility"}')['preferred']=='A'
    with pytest.raises(ValueError, match='non-JSON'): validate_structured_response('no')
    with pytest.raises(ValueError, match='unknown preferred'): validate_structured_response('{"preferred":"C","confidence":0.8,"reason":"x"}')
    with pytest.raises(ValueError, match='missing confidence'): validate_structured_response('{"preferred":"A","reason":"x"}')
    with pytest.raises(ValueError, match='hidden method'): validate_structured_response('{"preferred":"A","confidence":0.8,"reason":"PPO policy"}')


def test_evaluator_cache_identity_changes_on_event_and_pair():
    st={'agent_type':'bus','event_type':'BUS_TERMINAL_DEPARTURE','scenario_hash':'s','state_id':'x','state_features':[1]}
    cfg={'evaluator':{'prompt_version':'p','structured_output_schema_version':'r'}}; settings=APISettings('k','u','m',0.0,1)
    k1=cache_key(st, {'candidate_id':'a'}, {'candidate_id':'b'}, settings, cfg)
    st['event_type']='BUS_STATION_ARRIVAL'
    k2=cache_key(st, {'candidate_id':'a'}, {'candidate_id':'b'}, settings, cfg)
    assert k1 != k2


def test_missing_credential_failure_creates_no_dataset(tmp_path, monkeypatch):
    for n in ('OPENAI_API_KEY','OPENAI_BASE_URL','OPENAI_MODEL'): monkeypatch.delenv(n, raising=False)
    cfg=tmp_path/'cfg.yaml'; manifest=tmp_path/'bank.json'; manifest.write_text(json.dumps({'decision_states':[]}))
    cfg.write_text(f"""
run_classification: formal
scenario_bank: {{final_train_manifest: {manifest}}}
evaluator: {{api_key_env: OPENAI_API_KEY, base_url_env: OPENAI_BASE_URL, model_env: OPENAI_MODEL}}
preference_split: {{train_fraction: 0.7, validation_fraction: 0.15, test_fraction: 0.15, seed: 1}}
agents:
  assignment: {{target_valid_pair_count: 0, output_preferences: {tmp_path}/preference_assignment.jsonl}}
  truck: {{target_valid_pair_count: 0, output_preferences: {tmp_path}/preference_truck.jsonl}}
  bus: {{target_valid_pair_count: 0, output_preferences: {tmp_path}/preference_bus.jsonl}}
  station: {{target_valid_pair_count: 0, output_preferences: {tmp_path}/preference_station.jsonl}}
""")
    with pytest.raises(RuntimeError, match='missing API'):
        generate(cfg, tmp_path/'out')
    assert not (tmp_path/'out'/'preferences').exists()


def test_generation_collects_all_agents_splits_and_failure_recording(tmp_path, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY','k'); monkeypatch.setenv('OPENAI_BASE_URL','u'); monkeypatch.setenv('OPENAI_MODEL','m')
    states=[]
    for ev in ['PARCEL_RELEASE','TRUCK_AVAILABLE','BUS_TERMINAL_DEPARTURE','BUS_STATION_ARRIVAL','STATION_OPERATION']:
        for i in range(4):
            states.append({'event_type':ev,'scenario_id':f's{i}','scenario_hash':f'h{i}','state_id':f'{ev}{i}','state_features':[1.0], 'state_feature_names':['x'], 'candidate_actions':[{'candidate_id':'a','features':[1.0], 'feature_names':['c']},{'candidate_id':'b','features':[2.0], 'feature_names':['c']}], 'action_mask':[True,True]})
    manifest=tmp_path/'bank.json'; manifest.write_text(json.dumps({'decision_states':states}))
    cfg=tmp_path/'cfg.yaml'; cfg.write_text(f"""
run_classification: formal
scenario_bank: {{final_train_manifest: {manifest}}}
evaluator: {{api_key_env: OPENAI_API_KEY, base_url_env: OPENAI_BASE_URL, model_env: OPENAI_MODEL, max_retries: 2, temperature: 0.0}}
preference_split: {{train_fraction: 0.5, validation_fraction: 0.25, test_fraction: 0.25, seed: 1}}
agents:
  assignment: {{target_valid_pair_count: 1, output_preferences: {tmp_path}/pa.jsonl}}
  truck: {{target_valid_pair_count: 1, output_preferences: {tmp_path}/pt.jsonl}}
  bus: {{target_valid_pair_count: 2, output_preferences: {tmp_path}/pb.jsonl}}
  station: {{target_valid_pair_count: 1, output_preferences: {tmp_path}/ps.jsonl}}
""")
    calls={'n':0}
    def fake(prompt, settings):
        calls['n']+=1
        if calls['n']==1: raise RuntimeError('temporary')
        return json.dumps({'preferred':'A','confidence':0.9,'criteria':{'feasibility':'A'},'reason':'better feasible service'})
    m=generate(cfg,tmp_path/'out',api_call=fake)
    assert set(m['agents'])=={'assignment','truck','bus','station'}
    bus=[json.loads(l) for l in (tmp_path/'pb.jsonl').read_text().splitlines()]
    assert {r['event_type'] for r in bus} == {'BUS_TERMINAL_DEPARTURE','BUS_STATION_ARRIVAL'}
    assert (tmp_path/'out'/'failed'/'assignment_failed.jsonl').exists()
