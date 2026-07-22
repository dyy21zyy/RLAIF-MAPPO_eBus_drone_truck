from envs.delivery_env import Event
from envs.event_types import EVENT_PRIORITY
import heapq


def scheduled_bus_times(parcel_releases, departures):
    q=[]; seq=0
    for r in parcel_releases:
        heapq.heappush(q, Event(float(r), EVENT_PRIORITY['parcel_release'], seq, 'parcel_release', {})); seq+=1
    for d in departures:
        heapq.heappush(q, Event(float(d), EVENT_PRIORITY['bus_trip_start'], seq, 'bus_trip_start', {})); seq+=1
    return sorted(e.time_min for e in q if e.kind=='bus_trip_start')


def test_bus_at_zero_not_shifted_by_first_parcel_at_20():
    assert scheduled_bus_times([20], [0]) == [0.0]


def test_bus_and_parcel_same_timestamp_not_mutated():
    assert scheduled_bus_times([10], [10]) == [10.0]


def test_parcel_release_times_and_counts_do_not_change_trip_starts():
    assert scheduled_bus_times([0,20,30], [0,10]) == scheduled_bus_times([5], [0,10]) == [0.0,10.0]


def test_no_epsilon_timetable_mutation_literal_removed():
    from pathlib import Path
    src=Path('envs/delivery_env.py').read_text()
    assert 'first_parcel_release' not in src
