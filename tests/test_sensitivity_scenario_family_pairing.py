import pytest
from evaluation.scenario_family import ScenarioFamilyMember, validate_matched_master_seeds, scenario_family_id

def m(val, seeds): return ScenarioFamilyMember(scenario_family_id(split='test', master_seed=1, sensitivity_factor='f', sensitivity_value=val),1,'test','b','f',val,{'p':val},'c','i',{},seeds)
def test_matched_master_seeds_are_preserved(): validate_matched_master_seeds([m(1,{'parcel_seed':1,'passenger_seed':2,'timetable_seed':3,'physical_bus_seed':4,'station_load_seed':5}),m(2,{'parcel_seed':1,'passenger_seed':2,'timetable_seed':3,'physical_bus_seed':4,'station_load_seed':5})])
def test_unrelated_seed_changes_fail():
    with pytest.raises(ValueError): validate_matched_master_seeds([m(1,{'parcel_seed':1}),m(2,{'parcel_seed':9})])
def test_scenario_family_identity_differs_by_factor_value(): assert scenario_family_id(split='test',master_seed=1,sensitivity_factor='f',sensitivity_value=1)!=scenario_family_id(split='test',master_seed=1,sensitivity_factor='f',sensitivity_value=2)
def test_intended_sensitivity_override_recorded(): assert m(1,{}).intended_override=={'p':1}
