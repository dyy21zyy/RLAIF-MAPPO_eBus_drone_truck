"""Seeded passenger demand, queues, boarding, alighting, and delay accounting."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Iterable
import bisect
import numpy as np

DEFAULT_CAPACITY = 80

@dataclass(frozen=True)
class PassengerArrivalEvent:
    passenger_event_id: str
    origin_stop_id: str
    destination_stop_id: str
    arrival_time_min: float
    passenger_count: int = 1

@dataclass
class PassengerStopRuntimeState:
    stop_id: str
    waiting_by_destination: dict[str, int] = field(default_factory=dict)
    total_waiting: int = 0
    last_queue_update_time: float = 0.0
    cumulative_waiting_passenger_minutes: float = 0.0

    def integrate_waiting_until(self, time_min: float) -> float:
        if time_min < self.last_queue_update_time - 1e-9:
            raise ValueError("Passenger queue time cannot move backwards")
        elapsed = max(0.0, float(time_min) - self.last_queue_update_time)
        inc = self.total_waiting * elapsed
        self.cumulative_waiting_passenger_minutes += inc
        self.last_queue_update_time = float(time_min)
        return inc

    def add_arrival(self, destination_stop_id: str, count: int) -> None:
        count = int(count)
        if count < 0:
            raise ValueError("Passenger arrivals cannot be negative")
        self.waiting_by_destination[destination_stop_id] = self.waiting_by_destination.get(destination_stop_id, 0) + count
        self.total_waiting += count

@dataclass
class PassengerBusManifest:
    passenger_capacity: int = DEFAULT_CAPACITY
    onboard_passengers_by_destination: dict[str, int] = field(default_factory=dict)
    total_onboard_passengers: int = 0
    onboard_additional_delay_passenger_minutes: float = 0.0

    def alight(self, stop_id: str) -> int:
        count = self.onboard_passengers_by_destination.pop(stop_id, 0)
        self.total_onboard_passengers = max(0, self.total_onboard_passengers - count)
        return count

    def remaining_capacity(self) -> int:
        return max(0, self.passenger_capacity - self.total_onboard_passengers)

    def board_from_stop(self, stop: PassengerStopRuntimeState) -> int:
        can_board = self.remaining_capacity()
        boarded = 0
        for dest in sorted(list(stop.waiting_by_destination)):
            if can_board <= 0:
                break
            take = min(can_board, stop.waiting_by_destination.get(dest, 0))
            if take <= 0:
                continue
            self.onboard_passengers_by_destination[dest] = self.onboard_passengers_by_destination.get(dest, 0) + take
            stop.waiting_by_destination[dest] -= take
            if stop.waiting_by_destination[dest] == 0:
                del stop.waiting_by_destination[dest]
            stop.total_waiting -= take
            self.total_onboard_passengers += take
            can_board -= take
            boarded += take
        if stop.total_waiting < 0 or self.total_onboard_passengers < 0:
            raise AssertionError("Passenger counts became negative")
        return boarded

def sample_stop_rates(stop_ids: Iterable[str], seed: int, intensity: float = 1.0, mean: float = 0.25, std: float = 0.10, min_rate: float = 0.05, max_rate: float = 0.60) -> dict[str, float]:
    rng = np.random.default_rng(int(seed))
    rates = {}
    for stop_id in stop_ids:
        val = rng.normal(mean, std)
        while val < min_rate or val > max_rate:
            val = rng.normal(mean, std)
        rates[str(stop_id)] = float(min(max(val * float(intensity), max_rate), min_rate))
    return rates

def generate_arrival_events(stop_ids: list[str], horizon_min: float, seed: int, intensity: float = 1.0, rates: dict[str, float] | None = None) -> tuple[dict[str, float], list[PassengerArrivalEvent], dict[str, Any]]:
    rates = rates or sample_stop_rates(stop_ids, seed, intensity)
    rng = np.random.default_rng(int(seed) + 7919)
    events: list[PassengerArrivalEvent] = []
    for i, origin in enumerate(stop_ids[:-1]):
        count = rng.poisson(rates[origin] * float(horizon_min))
        for j in range(int(count)):
            dest = stop_ids[i + 1 + int(rng.integers(0, len(stop_ids) - i - 1))]
            t = float(rng.uniform(0.0, float(horizon_min)))
            events.append(PassengerArrivalEvent(f"pe_{origin}_{j:06d}", origin, dest, t, 1))
    events.sort(key=lambda e: (e.arrival_time_min, e.origin_stop_id, e.passenger_event_id))
    provenance = {"process":"time_dependent_poisson_pre_generated","seed":int(seed),"intensity":float(intensity),"rate_bounds_per_min":[0.05,0.60],"destination_rule":"downstream_uniform"}
    return rates, events, provenance

class PassengerArrivalIndex:
    def __init__(self, events: Iterable[PassengerArrivalEvent]):
        self.by_stop: dict[str, list[PassengerArrivalEvent]] = {}
        self.cursor: dict[str, int] = {}
        for event in sorted(events, key=lambda e: e.arrival_time_min):
            self.by_stop.setdefault(event.origin_stop_id, []).append(event)
        self.times = {s: [e.arrival_time_min for e in evs] for s, evs in self.by_stop.items()}

    def apply_until(self, stop: PassengerStopRuntimeState, time_min: float) -> int:
        stop.integrate_waiting_until(time_min)
        events = self.by_stop.get(stop.stop_id, [])
        cur = self.cursor.get(stop.stop_id, 0)
        end = bisect.bisect_right(self.times.get(stop.stop_id, []), float(time_min), lo=cur)
        added = 0
        for event in events[cur:end]:
            stop.add_arrival(event.destination_stop_id, event.passenger_count); added += event.passenger_count
        self.cursor[stop.stop_id] = end
        return added

@dataclass
class StopProcessingResult:
    waiting_before_arrival: int
    alighting_count: int
    boarding_count: int
    onboard_after_departure: int
    realized_dwell_min: float
    waiting_passenger_minutes: float
    onboard_additional_delay_passenger_minutes: float
    departure_time_min: float
    terminated_by_iteration_cap: bool = False

def process_bus_stop(stop: PassengerStopRuntimeState, bus: PassengerBusManifest, arrivals: PassengerArrivalIndex, arrival_time_min: float, *, baseline_dwell_min: float = 0.0, boarding_time_sec: float = 3.0, alighting_time_sec: float = 1.5, operational_dwell_min: float = 0.0, max_iterations: int = 1000) -> StopProcessingResult:
    before = stop.total_waiting
    wait_inc = arrivals.apply_until(stop, arrival_time_min)
    wait_minutes_inc = stop.cumulative_waiting_passenger_minutes
    alight = bus.alight(stop.stop_id)
    dwell = baseline_dwell_min + operational_dwell_min + alight * alighting_time_sec / 60.0
    boarded_total = 0
    iters = 0
    while True:
        iters += 1
        if iters > max_iterations:
            return StopProcessingResult(before, alight, boarded_total, bus.total_onboard_passengers, dwell, stop.cumulative_waiting_passenger_minutes, bus.onboard_additional_delay_passenger_minutes, arrival_time_min + dwell, True)
        boarded = bus.board_from_stop(stop)
        if boarded:
            boarded_total += boarded
            dwell += boarded * boarding_time_sec / 60.0
        departure = arrival_time_min + dwell
        added = arrivals.apply_until(stop, departure)
        if added <= 0 or bus.remaining_capacity() <= 0:
            break
    additional = max(0.0, dwell - baseline_dwell_min)
    bus.onboard_additional_delay_passenger_minutes += bus.total_onboard_passengers * additional
    return StopProcessingResult(before, alight, boarded_total, bus.total_onboard_passengers, dwell, stop.cumulative_waiting_passenger_minutes, bus.onboard_additional_delay_passenger_minutes, arrival_time_min + dwell, False)
