from __future__ import annotations
from dataclasses import dataclass, asdict
@dataclass(frozen=True)
class PassengerTraceRow:
    event_sequence:int; time:float; bus_id:str; trip_id:str; stop_id:str; stop_type:str; waiting_before:int; arrivals_applied:int; alighted:int; boarded_before_extra_dwell:int; onboard_at_extra_dwell_start:int; extra_dwell_duration:float; arrivals_during_extra_dwell:int; boarded_after_extra_dwell:int; waiting_after_departure:int; normal_dwell:float; loading_dwell:float; unloading_dwell:float; charging_dwell:float; incremental_waiting_passenger_minutes:float; incremental_onboard_additional_delay_passenger_minutes:float
    def asdict(self): return asdict(self)
