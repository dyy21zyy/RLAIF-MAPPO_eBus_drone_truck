"""Generate seeded stop passenger rates and pre-generated passenger arrival events."""
from __future__ import annotations
import csv, json
from pathlib import Path
from typing import Any
from envs.dynamics.passenger_dynamics import generate_arrival_events

RATE_FIELDS = ["stop_id","baseline_rate_per_min","effective_rate_per_min","passenger_demand_intensity","source"]
EVENT_FIELDS = ["passenger_event_id","origin_stop_id","destination_stop_id","arrival_time_min","passenger_count"]

def generate_passenger_demand(stops: list[dict[str, Any]], config: dict[str, Any], output_dir: str | Path) -> dict[str, Path]:
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    stop_ids = [str(s.get("stop_id", s.get("id"))) for s in stops]
    pcfg = config.setdefault("passenger", {})
    seed = int(config.get("seeds", {}).get("passenger_seed", config.get("project", {}).get("seed", 0)))
    intensity = float(pcfg.get("passenger_demand_intensity", pcfg.get("demand_intensity", 1.0)))
    horizon = float(config.get("bus", {}).get("delivery_horizon_min", config.get("time", {}).get("delivery_evaluation_horizon_min", 480)))
    rates, events, provenance = generate_arrival_events(stop_ids, horizon, seed, intensity)
    rates_path = output / "passenger_stop_rates.csv"
    with rates_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=RATE_FIELDS); writer.writeheader()
        for sid, rate in rates.items():
            writer.writerow({"stop_id":sid,"baseline_rate_per_min":rate / max(intensity, 1e-12),"effective_rate_per_min":rate,"passenger_demand_intensity":intensity,"source":"truncated_normal_seeded"})
    events_path = output / "passenger_arrivals.csv"
    with events_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=EVENT_FIELDS); writer.writeheader()
        for e in events:
            writer.writerow({"passenger_event_id":e.passenger_event_id,"origin_stop_id":e.origin_stop_id,"destination_stop_id":e.destination_stop_id,"arrival_time_min":f"{e.arrival_time_min:.9f}","passenger_count":e.passenger_count})
    prov_path = output / "passenger_demand_provenance.json"
    prov_path.write_text(json.dumps({**provenance,"stop_count":len(stop_ids),"event_count":len(events)}, indent=2), encoding="utf-8")
    return {"passenger_stop_rates":rates_path,"passenger_arrivals":events_path,"passenger_demand_provenance":prov_path}
