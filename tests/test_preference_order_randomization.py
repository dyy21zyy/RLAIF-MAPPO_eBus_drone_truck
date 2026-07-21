from rlaif.pair_selector import randomize_display_order, resolve_original_winner

def test_ab_order_randomizes_reproducibly_and_resolves():
    pair={"state_id":"s1","original_pair_order":[0,1]}
    a=randomize_display_order(pair,seed=7); b=randomize_display_order(pair,seed=7)
    assert a["display_order"]==b["display_order"]
    rec=resolve_original_winner({**a,"evaluator_answer":"A"})
    assert rec["resolved_original_winner"] in [0,1]

def test_tie_and_abstain_not_training():
    pair=randomize_display_order({"state_id":"s1","original_pair_order":[0,1]},seed=1)
    assert not resolve_original_winner({**pair,"evaluator_answer":"tie"})["usable_for_training"]
    assert not resolve_original_winner({**pair,"evaluator_answer":"abstain"})["usable_for_training"]
