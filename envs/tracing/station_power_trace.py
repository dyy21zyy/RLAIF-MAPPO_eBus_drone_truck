from __future__ import annotations
from dataclasses import dataclass, asdict
@dataclass(frozen=True)
class StationPowerTraceRow:
    station_id:str; interval_start:float; interval_end:float; base_load:float; active_bus_chargers:int; active_battery_chargers:int; total_load:float; capacity:float; overload:float; overload_kw_min:float
    def asdict(self): return asdict(self)
