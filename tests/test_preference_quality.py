import pytest

from rlaif.preference_quality import *

def candidate(mode, **features):
    return {"action_name": mode + ("_1" if mode != "TD" else ""), "features": features, "feasible": True}

@pytest.mark.parametrize("mode", ["TD", "TBD", "TLD"])
def test_action_type_extraction(mode): assert assignment_action_type(candidate(mode)) == mode

def test_unknown_action_fails_closed():
    with pytest.raises(ValueError): assignment_action_type({"action_name": "teleport"})

@pytest.mark.parametrize(("a","b","expected"), [("TD","TBD","TD-TBD"),("TD","TLD","TD-TLD"),("TBD","TBD","TBD-TBD"),("TBD","TLD","TBD-TLD"),("TLD","TLD","TLD-TLD")])
def test_pair_types(a,b,expected): assert assignment_pair_type(candidate(a),candidate(b)) == expected

@pytest.mark.parametrize(("alias","canonical"), [
    ("delivery time","delivery_time"),("delivery_time","delivery_time"),("estimated_delivery_time","delivery_time"),("estimated_delivery_time_norm","delivery_time"),
    ("lateness","expected_lateness"),("expected lateness","expected_lateness"),("estimated_lateness","expected_lateness"),("estimated_lateness_norm","expected_lateness"),
    ("locker load","locker_congestion"),("locker congestion","locker_congestion"),("estimated_locker_load_after_assignment_norm","locker_congestion"),
    ("power margin","station_power_margin"),("station power margin","station_power_margin"),("estimated_station_power_margin_norm","station_power_margin")])
def test_legacy_aliases(alias,canonical): assert canonicalize_criteria([alias],legacy=True)[0] == [canonical]

def test_criteria_contract_and_string_migration():
    assert canonicalize_criteria(["delivery_time"])[0] == ["delivery_time"]
    assert canonicalize_criteria("delivery time, lateness",legacy=True)[0] == ["delivery_time","expected_lateness"]
    for bad in ({"delivery_time":"A"}, [], ["energy"], ["delivery_time","delivery_time"]):
        with pytest.raises(ValueError): canonicalize_criteria(bad)

def test_direction_equality_and_applicability():
    a=candidate("TLD",expected_lateness=1,station_power_margin=4,drone_time=0)
    b=candidate("TLD",expected_lateness=2,station_power_margin=3,drone_time=5e-10)
    assert compare_metric(a,b,"expected_lateness") == "A"
    assert compare_metric(a,b,"station_power_margin") == "A"
    assert compare_metric(a,b,"drone_time") == "equal"
    with pytest.raises(ValueError): compare_metric(candidate("TD",drone_time=0),b,"drone_time")
    with pytest.raises(ValueError): compare_metric(candidate("TLD",bus_wait_time=0),candidate("TBD",bus_wait_time=2),"bus_wait_time")

def test_resources_and_forbidden_semantics():
    assert RESOURCE_USE["TD"] == {"truck"}; assert "bus" in RESOURCE_USE["TBD"]; assert "bus" not in RESOURCE_USE["TLD"]
    a,b=candidate("TLD",expected_lateness=1),candidate("TLD",expected_lateness=2)
    assert reason_semantic_errors("Candidate A uses bus freight",a,b)
    assert reason_semantic_errors("Candidate A is more energy-efficient and has lower emissions",a,b)
    assert reason_semantic_errors("Candidate A is potentially infeasible",a,b)
    td=candidate("TD",expected_lateness=1)
    assert reason_semantic_errors("Candidate A uses a drone and locker",td,b)

def test_grounded_v2_tradeoff_and_dominance():
    a=candidate("TLD",expected_lateness=1,truck_distance=5)
    b=candidate("TLD",expected_lateness=2,truck_distance=3)
    raw={"preferred":"A","confidence":.8,"criteria":["expected_lateness","truck_distance"],"evidence":[{"metric":"expected_lateness","better_candidate":"A"},{"metric":"truck_distance","better_candidate":"B"}],"reason":"Candidate A has lower expected lateness."}
    result=validate_assignment_v2(raw,a,b)
    assert result["raw_evaluator_reason"] == raw["reason"] and "trade-offs" in result["reason"]
    assert validate_assignment_v2(raw,a,b)["reason"] == result["reason"]
    dominated={**raw,"preferred":"B"}
    dominated["evidence"]=[{"metric":"expected_lateness","better_candidate":"A"},{"metric":"truck_distance","better_candidate":"B"}]
    # This remains a valid trade-off, not dominance.
    validate_assignment_v2(dominated,a,b)
    c=candidate("TLD",expected_lateness=3,truck_distance=6)
    assert is_strictly_dominated("B",a,c)

def test_reversed_evidence_rejected():
    with pytest.raises(ValueError,match="direction"):
        validate_evidence([{"metric":"expected_lateness","better_candidate":"B"}],candidate("TD",expected_lateness=1),candidate("TD",expected_lateness=2),"B")

def test_proportional_floors_and_absent_type():
    pool=[{"pair_type":"TD-TLD","id":i} for i in range(40)]+[{"pair_type":"TBD-TLD","id":i} for i in range(10)]
    selected=proportional_with_coverage_floors(pool,40,minimum=10)
    counts={k:sum(x["pair_type"]==k for x in selected) for k in {x["pair_type"] for x in pool}}
    assert counts["TBD-TLD"]==10 and coverage_pass({"TD-TLD":40,"TBD-TLD":10},counts,10)
    assert coverage_pass({"TD-TLD":40,"TBD-TBD":0},{"TD-TLD":10},10)
    assert not coverage_pass({"TD-TLD":40},{"TD-TLD":9},10)
