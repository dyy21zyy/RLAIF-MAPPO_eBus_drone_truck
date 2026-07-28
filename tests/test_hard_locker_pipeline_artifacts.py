import json
from pathlib import Path
import pytest
from experiments.generate_formal_multiagent_preferences import parse_agent_scope
from experiments.run_hard_locker_reexperiment import (
 assignment_preference_path, assert_no_placeholders, find_placeholders,
 marker_is_valid, phase_artifact_state, PhaseSpec, write_phase_marker,
 validate_assignment_preferences,
)


def test_agent_scope_defaults_and_validates():
 assert parse_agent_scope(None) == ('assignment','truck','bus','station')
 assert parse_agent_scope('assignment') == ('assignment',)
 assert parse_agent_scope('bus, assignment') == ('assignment','bus')
 with pytest.raises(ValueError,match='empty'): parse_agent_scope('')
 with pytest.raises(ValueError,match='unknown'): parse_agent_scope('pilot')


def test_recursive_placeholder_detection():
 value={'runtime':{'hashes':['ok',{'checkpoint':'TODO_HASH_x'}]},'notes':'ordinary prose'}
 assert find_placeholders(value) == ['$.runtime.hashes[1].checkpoint']
 with pytest.raises(ValueError,match='unresolved'): assert_no_placeholders(value)
 assert_no_placeholders({'documentation':'Hash will be populated by the formal workflow.'})


def test_assignment_path_and_manifest_validation(tmp_path):
 root=tmp_path/'preference-output'; path=assignment_preference_path(root); path.parent.mkdir(parents=True)
 rows=[{'agent_type':'assignment','observation_schema_version':4,'candidate_schema_version':4,'dataset_split':s} for s in ('train','validation','test')]
 path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
 import hashlib
 digest=hashlib.sha256(path.read_bytes()).hexdigest()
 (root/'preference_manifest.json').write_text(json.dumps({'selected_agents':['assignment'],'agents':{'assignment':{'path':str(path),'hash':digest,'counts_by_split':{'train':1,'validation':1,'test':1}}}}))
 assert validate_assignment_preferences(root)['path']==str(path)
 rows[0]['agent_type']='truck'; path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
 with pytest.raises(ValueError,match='non-assignment'): validate_assignment_preferences(root)


def test_phase_marker_is_content_addressed_and_validation_precedes_write(tmp_path):
 inp=tmp_path/'in'; out=tmp_path/'out'; inp.write_text('a'); out.write_text('b')
 phase=PhaseSpec(2,'x',tuple(),(out,),(inp,))
 marker=tmp_path/'marker.json'
 with pytest.raises(RuntimeError): write_phase_marker(marker,phase,commit='c',plan_digest='p',validator=lambda:(_ for _ in ()).throw(RuntimeError('bad')))
 assert not marker.exists()
 write_phase_marker(marker,phase,commit='c',plan_digest='p')
 assert marker_is_valid(marker,phase,commit='c',plan_digest='p')
 inp.write_text('changed')
 assert not marker_is_valid(marker,phase,commit='c',plan_digest='p')
 assert not marker_is_valid(marker,phase,commit='different',plan_digest='p')
 assert not marker_is_valid(marker,phase,commit='c',plan_digest='different')
