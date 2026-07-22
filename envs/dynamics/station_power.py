"""Time-varying station base-load profiles and exact soft-overload integration."""
from __future__ import annotations
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class StationBaseLoadInterval:
    station_id: str
    interval_id: str
    start_min: float
    end_min: float
    base_load_kw: float

@dataclass(frozen=True)
class ActivePowerInterval:
    station_id: str
    start_min: float
    end_min: float
    load_kw: float
    source: str

@dataclass(frozen=True)
class StationPowerSegment:
    station_id: str
    start_min: float
    end_min: float
    base_load_kw: float
    active_bus_chargers: int
    active_battery_chargers: int
    total_load_kw: float
    capacity_kw: float
    overload_kw: float
    overload_kw_min: float

@dataclass(frozen=True)
class StationPowerIntegration:
    station_id: str
    start_min: float
    end_min: float
    base_energy_kw_min: float
    bus_charging_energy_kw_min: float
    battery_charging_energy_kw_min: float
    overload_kw_min: float
    overload_duration_min: float
    peak_load_kw: float
    segments: tuple[StationPowerSegment, ...]

class StationBaseLoadProfile:
    def __init__(self, intervals):
        by = {}
        for it in intervals:
            by.setdefault(it.station_id, []).append(it)
        self.by_station = {sid: tuple(sorted(v, key=lambda x: x.start_min)) for sid, v in by.items()}
        for sid, vals in self.by_station.items():
            if not vals: raise ValueError("empty station load profile")
            expected = vals[0].start_min
            for it in vals:
                if it.start_min >= it.end_min or not math.isfinite(it.base_load_kw):
                    raise ValueError("invalid base-load interval")
                if abs(it.start_min - expected) > 1e-9:
                    raise ValueError("station base-load intervals must be contiguous and nonoverlapping")
                expected = it.end_min
    def load_at(self, station_id: str, time_min: float) -> float:
        vals = self.by_station[str(station_id)]
        for i, it in enumerate(vals):
            if it.start_min <= time_min < it.end_min or (i == len(vals)-1 and abs(time_min - it.end_min) <= 1e-9):
                return it.base_load_kw
        raise KeyError(f"no base-load interval for {station_id} at {time_min}")
    def next_boundary_after(self, station_id: str, time_min: float) -> float | None:
        candidates = [it.end_min for it in self.by_station[str(station_id)] if it.end_min > time_min + 1e-9]
        return min(candidates) if candidates else None

def integrate_station_power(station_id: str, start_min: float, end_min: float, *, profile: StationBaseLoadProfile, capacity_kw: float, bus_charges=(), battery_charges=(), bus_charging_power_kw: float = 500.0, battery_charging_power_kw: float = 2.0) -> StationPowerIntegration:
    start=float(start_min); end=float(end_min)
    if end <= start: return StationPowerIntegration(station_id,start,end,0,0,0,0,0,0,())
    boundaries={start,end}
    b=profile.next_boundary_after(station_id,start)
    while b is not None and b < end - 1e-9:
        boundaries.add(b); b=profile.next_boundary_after(station_id,b)
    for s,e in bus_charges:
        if start < s < end: boundaries.add(float(s))
        if start < e < end: boundaries.add(float(e))
    for s,e in battery_charges:
        if start < s < end: boundaries.add(float(s))
        if start < e < end: boundaries.add(float(e))
    segs=[]; base_e=bus_e=bat_e=over=dur=peak=0.0
    for a,b in zip(sorted(boundaries), sorted(boundaries)[1:]):
        dt=b-a; base=profile.load_at(station_id,a)
        bus_n=sum(1 for s,e in bus_charges if s <= a + 1e-9 and e > a + 1e-9)
        bat_n=sum(1 for s,e in battery_charges if s <= a + 1e-9 and e > a + 1e-9)
        bus_load=bus_n*bus_charging_power_kw; bat_load=bat_n*battery_charging_power_kw
        total=base+bus_load+bat_load; ov=max(0.0,total-capacity_kw)
        base_e += base*dt; bus_e += bus_load*dt; bat_e += bat_load*dt; over += ov*dt; peak=max(peak,total)
        if ov>1e-9: dur += dt
        segs.append(StationPowerSegment(station_id,a,b,base,bus_n,bat_n,total,capacity_kw,ov,ov*dt))
    return StationPowerIntegration(station_id,start,end,base_e,bus_e,bat_e,over,dur,peak,tuple(segs))
