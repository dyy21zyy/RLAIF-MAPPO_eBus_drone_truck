from tests.test_station_dispatch_matching import env

def test_same_time_station_triggers_coalesced():
    e=env(); e.event_sequence=0
    from envs.delivery_env import DynamicDeliveryEnv
    DynamicDeliveryEnv._push_station_operation(e,'s',5.0)
    DynamicDeliveryEnv._push_station_operation(e,'s',5.0)
    assert sum(1 for t,k,p in e.events if k=='station_operation')==1
