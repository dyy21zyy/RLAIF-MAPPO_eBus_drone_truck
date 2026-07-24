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
scenario_bank: {{final_train_manifest: {manifest}, source_mode: injected_states}}
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
scenario_bank: {{final_train_manifest: {manifest}, source_mode: injected_states}}
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

def test_real_frozen_scenario_rollout_collects_four_agents(tmp_path, monkeypatch):
    from experiments.build_scenario_bank import build_bank
    from experiments.generate_formal_multiagent_preferences import collect_decision_states
    build_bank('configs/shanghai_small.yaml','train',2,123,tmp_path/'train',fallback=False,run_classification='diagnostic',force=True)
    cfg={'scenario_bank':{'final_train_manifest':str(tmp_path/'train'/'scenario_bank_manifest.json')},'collection':{'seed':7}}
    states=collect_decision_states(cfg, output_root=tmp_path/'out')
    assert {s['agent_type'] for s in states} == {'assignment','truck','bus','station'}
    assert {'BUS_TERMINAL_DEPARTURE','BUS_STATION_ARRIVAL'} <= {s['event_type'] for s in states}
    assert all(s['scenario_split']=='train' and s['scenario_bank_hash'] for s in states)
    assert any(feasible_pairs(s) for s in states)


def _state(ev, scenario, idx, cand=3):
    return {'event_type':ev,'scenario_id':scenario,'scenario_hash':scenario,'state_id':f'{ev}-{scenario}-{idx}','decision_id':f'{ev}-{scenario}-{idx}','episode_id':scenario,'state_features':[float(idx)], 'state_feature_names':['x'], 'candidate_actions':[{'candidate_id':f'c{j}','features':[float(j)], 'feature_names':['c']} for j in range(cand)], 'action_mask':[True]*cand}


def _formal_cfg(tmp_path, states, targets=None, budgets=None, seed=11):
    targets=targets or {'assignment':3,'truck':3,'bus':6,'station':3}; budgets=budgets or {'assignment':9,'truck':9,'bus':18,'station':9}
    manifest=tmp_path/'bank.json'; manifest.write_text(json.dumps({'decision_states':states}))
    cfg=tmp_path/'cfg.yaml'
    cfg.write_text(f"""
run_classification: formal
scenario_bank: {{final_train_manifest: {manifest}, source_mode: injected_states}}
evaluator: {{api_key_env: OPENAI_API_KEY, base_url_env: OPENAI_BASE_URL, model_env: OPENAI_MODEL, max_retries: 1, temperature: 0.0, cache_dir: {tmp_path}/cache}}
preference_split: {{train_fraction: 0.5, validation_fraction: 0.25, test_fraction: 0.25, seed: {seed}}}
agents:
  assignment: {{target_valid_pair_count: {targets['assignment']}, max_api_attempts: {budgets['assignment']}, supported_event_types: [PARCEL_RELEASE], output_preferences: {tmp_path}/pa.jsonl}}
  truck: {{target_valid_pair_count: {targets['truck']}, max_api_attempts: {budgets['truck']}, supported_event_types: [TRUCK_AVAILABLE], output_preferences: {tmp_path}/pt.jsonl}}
  bus: {{target_valid_pair_count: {targets['bus']}, max_api_attempts: {budgets['bus']}, supported_event_types: [BUS_TERMINAL_DEPARTURE, BUS_STATION_ARRIVAL], output_preferences: {tmp_path}/pb.jsonl}}
  station: {{target_valid_pair_count: {targets['station']}, max_api_attempts: {budgets['station']}, supported_event_types: [STATION_OPERATION], output_preferences: {tmp_path}/ps.jsonl}}
""")
    return cfg


def _coverage_states():
    states=[]
    for ev in ['PARCEL_RELEASE','TRUCK_AVAILABLE','BUS_TERMINAL_DEPARTURE','BUS_STATION_ARRIVAL','STATION_OPERATION']:
        for i in range(8): states.append(_state(ev, f's{i}', i))
    return states


def test_bounded_stops_deterministic_covers_splits_and_bus_events(tmp_path, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY','k'); monkeypatch.setenv('OPENAI_BASE_URL','u'); monkeypatch.setenv('OPENAI_MODEL','m')
    cfg=_formal_cfg(tmp_path, _coverage_states())
    calls=[]
    def fake(prompt, settings): calls.append(prompt); return json.dumps({'preferred':'A','confidence':0.8,'reason':'better service'})
    m1=generate(cfg,tmp_path/'out1',api_call=fake)
    assert all(a['external_api_call_count'] <= a['max_api_attempts'] for a in m1['agents'].values())
    assert m1['agents']['assignment']['valid_binary_label_count'] == 3
    assert {'BUS_TERMINAL_DEPARTURE','BUS_STATION_ARRIVAL'} == set(m1['agents']['bus']['events'])
    for a in m1['agents'].values(): assert all(a['counts_by_split'][s] > 0 for s in ('train','validation','test'))
    calls.clear(); m2=generate(cfg,tmp_path/'out2',api_call=fake)
    assert m1['agents']['bus']['selected_counts_by_event_split'] == m2['agents']['bus']['selected_counts_by_event_split']


def test_ties_failures_budget_exhaustion_and_cache_resume_dry_run(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('OPENAI_API_KEY','k'); monkeypatch.setenv('OPENAI_BASE_URL','u'); monkeypatch.setenv('OPENAI_MODEL','m')
    cfg=_formal_cfg(tmp_path, _coverage_states(), targets={'assignment':3,'truck':3,'bus':6,'station':3}, budgets={'assignment':9,'truck':9,'bus':18,'station':9})
    n={'c':0}
    def mixed(prompt, settings):
        n['c']+=1
        if n['c'] in (1,2): return json.dumps({'preferred':'equal','confidence':0.5,'reason':'similar'})
        return json.dumps({'preferred':'B','confidence':0.9,'reason':'better feasible option'})
    m=generate(cfg,tmp_path/'out',api_call=mixed)
    assert m['agents']['assignment']['tie_count'] == 2
    assert m['agents']['assignment']['valid_binary_label_count'] == 3
    (tmp_path/'pa.jsonl').unlink(); (tmp_path/'pt.jsonl').unlink(); (tmp_path/'pb.jsonl').unlink(); (tmp_path/'ps.jsonl').unlink()
    before=n['c']; m_resume=generate(cfg,tmp_path/'out_cached',resume=True,api_call=mixed)
    assert n['c'] == before
    assert m_resume['agents']['assignment']['cache_hit_count'] > 0
    dry=generate(cfg,tmp_path/'dry',dry_run=True,api_call=lambda *_: (_ for _ in ()).throw(AssertionError('api called')))
    out=capsys.readouterr().out
    assert 'assignment: pool=' in out and all(a['external_api_call_count'] == 0 for a in dry['agents'].values())
    (tmp_path/'bad').mkdir(exist_ok=True)
    bad=_formal_cfg(tmp_path/'bad', _coverage_states(), budgets={'assignment':1,'truck':1,'bus':1,'station':1})
    with pytest.raises(RuntimeError, match='usable-label target|budget exhausted'):
        generate(bad,tmp_path/'badout',api_call=lambda *_: json.dumps({'preferred':'A','confidence':0.9,'reason':'ok'}))
