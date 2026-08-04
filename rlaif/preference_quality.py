"""Pure, fail-closed quality gates for assignment preference labels."""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
import math
import re
from typing import Any, Iterable, Mapping, Sequence

QUALITY_GATE_VERSION = "assignment_preference_quality_v1"
ASSIGNMENT_PROMPT_VERSION = "assignment_grounded_preference_v2"
ASSIGNMENT_RESPONSE_SCHEMA_VERSION = "rlaif_preference_json_v2"
CONSEQUENCE_MODE = "estimated_candidate_attributes"
DEFAULT_TOLERANCE = 1.0e-9

CANONICAL_CRITERIA = frozenset({
    "delivery_feasibility", "delivery_time", "expected_lateness", "deadline_risk",
    "truck_distance", "truck_time", "truck_capacity", "bus_wait_time",
    "bus_linehaul_time", "bus_freight_capacity", "drone_time", "drone_feasibility",
    "locker_congestion", "station_power_margin", "downstream_congestion",
})
FORBIDDEN_CRITERIA = frozenset({"energy", "energy_efficiency", "emissions", "carbon_emissions",
                                "fuel_consumption", "electricity_consumption"})

def _aliases(canonical: str, *values: str) -> dict[str, str]:
    return {v: canonical for v in (canonical, *values)}

CRITERIA_ALIASES = {
    **_aliases(
        "delivery_time",
        "delivery time",
        "estimated delivery time",
        "estimated_delivery_time",
        "estimated_delivery_time_norm",
    ),
    **_aliases(
        "expected_lateness",
        "lateness",
        "expected lateness",
        "estimated_lateness",
        "estimated_lateness_norm",
    ),
    **_aliases(
        "truck_distance",
        "truck distance",
        "estimated_truck_distance",
        "estimated_truck_distance_norm",
    ),
    **_aliases(
        "truck_time",
        "truck time",
        "estimated_truck_time",
        "estimated_truck_time_norm",
    ),
    **_aliases(
        "drone_time",
        "drone time",
        "estimated_drone_time",
        "estimated_drone_time_norm",
    ),
    **_aliases(
        "locker_congestion",
        "locker load",
        "locker congestion",
        "locker_load",
        "estimated_locker_load",
        "estimated_locker_load_after_assignment",
        "estimated_locker_load_after_assignment_norm",
    ),
    **_aliases(
        "station_power_margin",
        "power margin",
        "station power margin",
        "station_power_margin_norm",
        "estimated_station_power_margin_norm",
    ),
    **{criterion: criterion for criterion in CANONICAL_CRITERIA},
}


@dataclass(frozen=True)
class Metric:
    direction: str
    modes: frozenset[str]
    aliases: tuple[str, ...] = ()

ALL_MODES = frozenset({"TD", "TBD", "TLD"})
DRONE_MODES = frozenset({"TBD", "TLD"})
METRIC_REGISTRY: dict[str, Metric] = {
    "delivery_feasibility": Metric("higher", ALL_MODES, ("feasible",)),
    "delivery_time": Metric("lower", ALL_MODES, ("estimated_delivery_time", "estimated_delivery_time_norm")),
    "expected_lateness": Metric("lower", ALL_MODES, ("estimated_lateness", "estimated_lateness_norm")),
    "truck_distance": Metric("lower", ALL_MODES, ("estimated_truck_distance", "estimated_truck_distance_norm")),
    "truck_time": Metric("lower", ALL_MODES, ("estimated_truck_time", "estimated_truck_time_norm")),
    "bus_wait_time": Metric("lower", frozenset({"TBD"}), ("estimated_bus_wait_time", "estimated_bus_wait_time_norm")),
    "bus_linehaul_time": Metric("lower", frozenset({"TBD"}), ("estimated_bus_linehaul_time", "estimated_bus_linehaul_time_norm")),
    "drone_time": Metric("lower", DRONE_MODES, ("estimated_drone_time", "estimated_drone_time_norm")),
    "locker_congestion": Metric("lower", DRONE_MODES, ("estimated_locker_load_after_assignment_norm", "locker_load")),
    "station_power_margin": Metric("higher", DRONE_MODES, ("estimated_station_power_margin_norm", "power_margin")),
}

RESOURCE_USE = {
    "TD": frozenset({"truck"}),
    "TBD": frozenset({"truck", "bus", "locker", "drone", "station", "station_power"}),
    "TLD": frozenset({"truck", "locker", "drone", "station", "station_power"}),
}

def assignment_action_type(candidate: Mapping[str, Any]) -> str:
    """Extract TD/TBD/TLD without guessing an ambiguous numeric action id."""
    values = [candidate.get(k) for k in ("action_type", "action_name", "mode", "delivery_mode", "candidate_id")]
    found = set()
    for value in values:
        if value is None:
            continue
        token = str(value).strip().upper().replace("-", "_")
        match = re.match(r"^(TBD|TLD|TD)(?:_|$)", token)
        if match:
            found.add(match.group(1))
    if len(found) != 1:
        raise ValueError("unknown or ambiguous assignment action type")
    return found.pop()

def assignment_pair_type(a: Mapping[str, Any], b: Mapping[str, Any]) -> str:
    ma, mb = assignment_action_type(a), assignment_action_type(b)
    order = {"TD": 0, "TBD": 1, "TLD": 2}
    left, right = sorted((ma, mb), key=order.__getitem__)
    if left == right == "TD":
        raise ValueError("TD-TD pairs are not part of the assignment comparison design")
    return f"{left}-{right}"

def canonicalize_criteria(value: Any, *, legacy: bool = False) -> tuple[list[str], list[str]]:
    actions: list[str] = []
    if isinstance(value, str) and legacy:
        value = [part.strip() for part in value.split(",") if part.strip()]
        actions.append("split_comma_separated_criteria")
    if not isinstance(value, list) or not value:
        raise ValueError("criteria must be a nonempty list")
    result = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError("criteria entries must be strings")
        key = re.sub(r"\s+", " ", item.strip().lower())
        canonical = CRITERIA_ALIASES.get(key)
        if canonical is None or canonical in FORBIDDEN_CRITERIA:
            raise ValueError(f"illegal or ambiguous criterion: {item}")
        if canonical in result:
            actions.append(f"deduplicated:{item}->{canonical}")
            continue
        result.append(canonical)
        if key != canonical:
            actions.append(f"canonicalized:{item}->{canonical}")
    return result, actions

def candidate_feature_mapping(candidate: Mapping[str, Any]) -> dict[str, float]:
    raw = candidate.get("features", candidate.get("candidate_features", {}))
    if isinstance(raw, Mapping):
        result = {str(k): float(v) for k, v in raw.items()}
    else:
        names = candidate.get("feature_names", candidate.get("candidate_feature_names", []))
        if len(names) != len(raw or []):
            raise ValueError("candidate feature names and values differ")
        result = {str(k): float(v) for k, v in zip(names, raw)}
    if "feasible" in candidate:
        result["feasible"] = float(bool(candidate["feasible"]))
    if not all(math.isfinite(v) for v in result.values()):
        raise ValueError("candidate features must be finite")
    return result

def metric_value(candidate: Mapping[str, Any], metric: str) -> float:
    spec = METRIC_REGISTRY.get(metric)
    if spec is None:
        raise ValueError(f"criterion has no deterministic candidate extractor: {metric}")
    values = candidate_feature_mapping(candidate)
    for name in (metric, *spec.aliases):
        if name in values:
            return values[name]
    raise ValueError(f"candidate metric unavailable: {metric}")

def metric_applicable(metric: str, mode: str) -> bool:
    return metric in METRIC_REGISTRY and mode in METRIC_REGISTRY[metric].modes

def compare_metric(a: Mapping[str, Any], b: Mapping[str, Any], metric: str,
                   tolerance: float = DEFAULT_TOLERANCE) -> str:
    ma, mb = assignment_action_type(a), assignment_action_type(b)
    if not (metric_applicable(metric, ma) and metric_applicable(metric, mb)):
        raise ValueError(f"metric {metric} is not applicable to both {ma} and {mb}")
    av, bv = metric_value(a, metric), metric_value(b, metric)
    if abs(av - bv) <= tolerance:
        return "equal"
    lower = METRIC_REGISTRY[metric].direction == "lower"
    return "A" if (av < bv) == lower else "B"

def validate_evidence(evidence: Any, a: Mapping[str, Any], b: Mapping[str, Any], preferred: str,
                      *, tolerance: float = DEFAULT_TOLERANCE) -> list[dict[str, str]]:
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("evidence must be a nonempty list")
    validated = []
    for item in evidence:
        if not isinstance(item, Mapping) or set(item) != {"metric", "better_candidate"}:
            raise ValueError("each evidence item must contain only metric and better_candidate")
        raw_metric = item["metric"]
        claimed = item["better_candidate"]

        if not isinstance(raw_metric, str):
            raise ValueError("invalid evidence metric or direction")

        metric_key = re.sub(
            r"\\s+",
            " ",
            raw_metric.strip().lower(),
        )
        metric = CRITERIA_ALIASES.get(metric_key)

        if (
            metric is None
            or metric in FORBIDDEN_CRITERIA
            or metric not in CANONICAL_CRITERIA
            or claimed not in {"A", "B", "equal"}
        ):
            raise ValueError("invalid evidence metric or direction")

        actual = compare_metric(a, b, metric, tolerance)
        if actual != claimed:
            raise ValueError(f"evidence direction error for {metric}: claimed {claimed}, actual {actual}")
        validated.append({"metric": metric, "better_candidate": claimed})
    if preferred in {"A", "B"} and not any(x["better_candidate"] == preferred for x in validated):
        raise ValueError("no grounded evidence supports preferred candidate")
    return validated

def reason_semantic_errors(reason: str, a: Mapping[str, Any], b: Mapping[str, Any]) -> list[str]:
    text = reason.lower()
    errors = []
    forbidden = ("energy efficient", "energy-efficient", "energy consumption", "emission", "carbon", "fuel use", "fuel consumption", "electricity consumption")
    if any(term in text for term in forbidden): errors.append("unsupported_energy_or_emissions_claim")
    if "simulated downstream consequence" in text: errors.append("fabricated_simulated_consequence_claim")
    for label, candidate in (("a", a), ("b", b)):
        mode = assignment_action_type(candidate)
        segment_patterns = [rf"candidate {label}[^.]*"]
        segments = " ".join(re.findall("|".join(segment_patterns), text))
        if mode == "TLD" and re.search(r"bus (?:transport|freight|wait|waiting|linehaul)", segments): errors.append(f"candidate_{label}_tld_bus_claim")
        if mode == "TD" and re.search(r"(?:uses?|via|through).*(?:bus|locker|drone|station)", segments): errors.append(f"candidate_{label}_td_resource_claim")
        if bool(candidate.get("feasible", True)) and re.search(r"(?:candidate )?"+label+r"[^.]*(?:potentially )?infeasible", text): errors.append(f"candidate_{label}_feasibility_contradiction")
    return sorted(set(errors))

def is_strictly_dominated(preferred: str, a: Mapping[str, Any], b: Mapping[str, Any],
                          *, tolerance: float = DEFAULT_TOLERANCE) -> bool:
    if preferred not in {"A", "B"}: return False
    comparisons = []
    for metric in METRIC_REGISTRY:
        try: comparisons.append(compare_metric(a, b, metric, tolerance))
        except ValueError: continue
    other = "B" if preferred == "A" else "A"
    return bool(comparisons) and all(x in {other, "equal"} for x in comparisons) and other in comparisons

def canonical_reason(preferred: str, evidence: Sequence[Mapping[str, str]], a: Mapping[str, Any], b: Mapping[str, Any]) -> str:
    selected = assignment_action_type(a if preferred == "A" else b)
    supporting = [x["metric"].replace("_", " ") for x in evidence if x["better_candidate"] == preferred]
    tradeoffs = [x["metric"].replace("_", " ") for x in evidence if x["better_candidate"] not in {preferred, "equal"}]
    reason = f"Candidate {preferred} ({selected}) is preferred based on " + ", ".join(supporting) + "."
    if tradeoffs: reason += " The validated trade-offs favor the alternative on " + ", ".join(tradeoffs) + "."
    return reason

def validate_assignment_v2(response: Mapping[str, Any], a: Mapping[str, Any], b: Mapping[str, Any], *, tolerance: float = DEFAULT_TOLERANCE) -> dict[str, Any]:
    data = deepcopy(dict(response))
    if data.get("preferred") not in {"A", "B", "equal"}: raise ValueError("unknown preferred value")
    confidence = data.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1: raise ValueError("confidence must be finite and in [0, 1]")
    criteria, _ = canonicalize_criteria(data.get("criteria"))
    raw_reason = data.get("reason")
    if not isinstance(raw_reason, str) or not raw_reason.strip(): raise ValueError("reason must be nonempty")
    errors = reason_semantic_errors(raw_reason, a, b)
    if errors: raise ValueError("; ".join(errors))
    evidence = validate_evidence(data.get("evidence"), a, b, data["preferred"], tolerance=tolerance)
    if is_strictly_dominated(data["preferred"], a, b, tolerance=tolerance): raise ValueError("preferred candidate is strictly dominated")
    data.update(criteria=criteria, validated_evidence=evidence, raw_evaluator_reason=raw_reason,
                reason=canonical_reason(data["preferred"], evidence, a, b), confidence=float(confidence),
                quality_gate_version=QUALITY_GATE_VERSION, consequence_mode=CONSEQUENCE_MODE)
    return data

def pair_identity(record: Mapping[str, Any]) -> str:
    return str(record.get("candidate_pair_hash") or "|".join((str(record.get("scenario_hash", record.get("scenario_id"))), str(record.get("state_id")), *sorted((str(record.get("candidate_a_id")), str(record.get("candidate_b_id")))))))

def audit_legacy_record(record: Mapping[str, Any], *, audit_timestamp: str) -> tuple[dict[str, Any], list[str]]:
    out = deepcopy(dict(record)); reasons = []
    a = out.get("candidate_a_id_features") or out.get("candidate_a") or {}
    b = out.get("candidate_b_id_features") or out.get("candidate_b") or {}
    try:
        if out.get("agent_type") != "assignment" or out.get("event_type") != "PARCEL_RELEASE": raise ValueError("record is outside assignment/PARCEL_RELEASE scope")
        assignment_pair_type(a, b)
        criteria, actions = canonicalize_criteria(out.get("criteria"), legacy=True)
        semantic = reason_semantic_errors(str(out.get("reason", "")), a, b)
        if semantic: raise ValueError("; ".join(semantic))
        preferred = {"candidate_a": "A", "candidate_b": "B"}.get(out.get("original_outcome"), out.get("preferred"))
        if preferred not in {"A", "B"}: raise ValueError("legacy record is not a binary preference")
        evidence = []
        for metric in criteria:
            try: evidence.append({"metric": metric, "better_candidate": compare_metric(a, b, metric)})
            except ValueError: continue
        evidence = validate_evidence(evidence, a, b, preferred)
        if is_strictly_dominated(preferred, a, b): raise ValueError("preferred candidate is strictly dominated")
        legacy_id = out.get("preference_id")
        out.update(criteria=criteria, validated_evidence=evidence, raw_evaluator_reason=out.get("reason"),
                   reason=canonical_reason(preferred, evidence, a, b), quality_gate_version=QUALITY_GATE_VERSION,
                   quality_status="accepted_legacy", legacy_preference_id=legacy_id,
                   legacy_prompt_version=out.get("prompt_version"), legacy_response_schema_version=out.get("response_schema_version"),
                   legacy_dataset_split=out.get("dataset_split"), canonicalization_actions=actions, audit_timestamp=audit_timestamp)
    except (TypeError, ValueError, KeyError) as exc:
        reasons.append(str(exc)); out.update(quality_status="quarantined", quarantine_reasons=reasons,
                                             legacy_preference_id=out.get("preference_id"), pair_identity=pair_identity(out),
                                             candidate_a=a, candidate_b=b)
    return out, reasons

def coverage_pass(pool: Mapping[str, int], accepted: Mapping[str, int], minimum: int) -> bool:
    return all(accepted.get(kind, 0) >= min(count, minimum) for kind, count in pool.items() if count)

def proportional_with_coverage_floors(pool: Sequence[Mapping[str, Any]], target: int, *, minimum: int = 30) -> list[Mapping[str, Any]]:
    by_type: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in pool: by_type[str(item["pair_type"])].append(item)
    if target > len(pool): raise ValueError("target exceeds unique candidate-pair pool")
    quotas = {k: min(len(v), minimum) for k, v in by_type.items()}
    if sum(quotas.values()) > target: raise ValueError("target cannot satisfy coverage floors")
    remaining = target - sum(quotas.values()); capacities = {k: len(v)-quotas[k] for k,v in by_type.items()}
    while remaining:
        eligible = [k for k in sorted(by_type) if capacities[k] > 0]
        if not eligible: break
        total = sum(capacities[k] for k in eligible)
        k = max(eligible, key=lambda x: (capacities[x]/total, x))
        quotas[k] += 1; capacities[k] -= 1; remaining -= 1
    return [item for k in sorted(by_type) for item in by_type[k][:quotas[k]]]

def aggregate_audit(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(rows); pairs = Counter(); statuses = Counter(); reasons = Counter()
    for row in rows:
        statuses[row.get("quality_status", "unknown")] += 1
        for reason in row.get("quarantine_reasons", []): reasons[reason] += 1
        try: pairs[assignment_pair_type(row.get("candidate_a_id_features") or row.get("candidate_a", {}), row.get("candidate_b_id_features") or row.get("candidate_b", {}))] += 1
        except ValueError: pairs["unknown"] += 1
    return {"total_count": len(rows), "accepted_count": statuses["accepted_legacy"], "quarantined_count": statuses["quarantined"],
            "quality_status_counts": dict(statuses), "quarantine_reason_distribution": dict(reasons),
            "pair_type_distribution": dict(pairs), "external_api_call_count": 0}
