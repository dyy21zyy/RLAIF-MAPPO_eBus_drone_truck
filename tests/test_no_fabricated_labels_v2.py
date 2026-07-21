from rlaif.pair_selector import resolve_original_winner

def test_no_rule_generated_labels_and_offline_zero_labels():
    assert resolve_original_winner({'display_order':[0,1],'original_pair_order':[0,1],'evaluator_answer':'abstain'})['usable_for_training'] is False
