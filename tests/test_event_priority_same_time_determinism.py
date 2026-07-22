from envs.delivery_env import Event
from envs.event_types import EVENT_PRIORITY
import heapq


def test_same_time_priority_then_stable_sequence_order():
    q=[]
    heapq.heappush(q, Event(10.0, EVENT_PRIORITY['parcel_release'], 0, 'parcel_release', {}))
    heapq.heappush(q, Event(10.0, EVENT_PRIORITY['bus_trip_start'], 1, 'bus_trip_start', {}))
    assert [heapq.heappop(q).kind for _ in range(2)] == ['parcel_release','bus_trip_start']


def test_same_priority_uses_sequence_without_timestamp_mutation():
    q=[Event(5.0, EVENT_PRIORITY['parcel_release'], 2, 'parcel_release', {'id':2}), Event(5.0, EVENT_PRIORITY['parcel_release'], 1, 'parcel_release', {'id':1})]
    heapq.heapify(q)
    assert [heapq.heappop(q).payload['id'], heapq.heappop(q).payload['id']] == [1,2]
