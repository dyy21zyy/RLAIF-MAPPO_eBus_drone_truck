from rlaif.preference_schema_v3 import parse_preference_record, resolve_original_binary_target
from tests.preference_v3_fixtures import rec

def test_candidate_a_winner_resolves_to_1(): assert resolve_original_binary_target(parse_preference_record(rec(original_outcome='candidate_a')))==1
def test_candidate_b_winner_resolves_to_0(): assert resolve_original_binary_target(parse_preference_record(rec(original_outcome='candidate_b')))==0
def test_swapped_display_order_resolves_correctly(): assert resolve_original_binary_target(parse_preference_record(rec(displayed_first_candidate_id='b',displayed_second_candidate_id='a',original_outcome='candidate_a')))==1
def test_tie_returns_none(): assert resolve_original_binary_target(parse_preference_record(rec(original_outcome='tie'))) is None
def test_abstain_returns_none(): assert resolve_original_binary_target(parse_preference_record(rec(original_outcome='abstain'))) is None
