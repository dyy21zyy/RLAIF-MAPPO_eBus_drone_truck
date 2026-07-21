from types import SimpleNamespace as N
from envs.action_generators.bus_charging_actions import generate_bus_charging_candidates, energy_added_kwh

def env(soc=100, active=()):
    return N(now_min=0,trip_to_bus={"t":"b"},physical_buses={"b":N(physical_bus_id="b",soc_kwh=soc)},stations={"s":N(station_id="s",active_bus_charges=list(active),power_capacity_kw=1100)},config={"bus":{"charging_actions_sec":[0,15,30,45,60,75,90,105,120],"charging_power_kw":500,"charging_efficiency":0.95,"bus_battery_kwh":160}},_station_load_kw=lambda st,t:1200)

def test_no_charger_only_zero_feasible_and_power_soft():
    c=generate_bus_charging_candidates(env(active=[10,10]),N(payload={"trip_id":"t","station_id":"s"}))
    assert [x.feasible for x in c]==[True]+[False]*8
    assert c[0].projected_overload_kw>0

def test_overcharge_mask_and_energy_formula():
    c=generate_bus_charging_candidates(env(soc=159.9),N(payload={"trip_id":"t","station_id":"s"}))
    assert energy_added_kwh(120)==500*(120/3600)*0.95
    assert any((not x.feasible and "overcharge" in x.infeasibility_reasons) for x in c if x.duration_sec)
