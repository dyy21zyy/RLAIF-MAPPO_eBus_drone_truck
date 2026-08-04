"""Offline, read-only audit and Scheme A migration of legacy assignment labels."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

from rlaif.preference_dataset import write_jsonl
from rlaif.preference_quality import aggregate_audit, audit_legacy_record, pair_identity

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()

def audit_file(input_path: Path, *, expected_count: int | None = None,
               expected_sha256: str | None = None) -> tuple[list[dict], list[dict], dict]:
    actual_hash = sha256(input_path)
    if expected_sha256 and actual_hash.lower() != expected_sha256.lower(): raise ValueError("legacy input SHA256 mismatch")
    records = []
    for number, line in enumerate(input_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip(): continue
        try: row = json.loads(line)
        except json.JSONDecodeError as exc: raise ValueError(f"malformed JSON on line {number}: {exc.msg}") from exc
        if not isinstance(row, dict): raise ValueError(f"line {number} is not a JSON object")
        records.append(row)
    if expected_count is not None and len(records) != expected_count: raise ValueError(f"expected {expected_count} records, found {len(records)}")
    ids = [r.get("preference_id") for r in records]
    if any(x is None for x in ids) or len(ids) != len(set(ids)): raise ValueError("duplicate or missing preference identity")
    pairs = [pair_identity(r) for r in records]
    if len(pairs) != len(set(pairs)): raise ValueError("duplicate candidate-pair identity")
    timestamp = datetime.now(timezone.utc).isoformat()
    audited = [audit_legacy_record(r, audit_timestamp=timestamp)[0] for r in records]
    accepted = [r for r in audited if r["quality_status"] == "accepted_legacy"]
    quarantined = [r for r in audited if r["quality_status"] == "quarantined"]
    report = {**aggregate_audit(audited), "input_path": str(input_path), "input_sha256": actual_hash,
              "audit_timestamp": timestamp, "input_unchanged": sha256(input_path) == actual_hash}
    return accepted, quarantined, report

def markdown_report(report: dict) -> str:
    lines = ["# Legacy assignment preference quality audit", "", f"- Input SHA256: `{report['input_sha256']}`",
             f"- Total: {report['total_count']}", f"- Accepted: {report['accepted_count']}",
             f"- Quarantined: {report['quarantined_count']}", "- External evaluator API calls: 0", "",
             "## Quarantine reasons", ""]
    lines += [f"- `{key}`: {value}" for key, value in sorted(report["quarantine_reason_distribution"].items())] or ["- None"]
    lines += ["", "## Pair types", ""] + [f"- {key}: {value}" for key, value in sorted(report["pair_type_distribution"].items())]
    return "\n".join(lines) + "\n"

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Audit legacy assignment preferences offline (zero evaluator API calls).")
    parser.add_argument("--input", type=Path, required=True); parser.add_argument("--accepted-output", type=Path, required=True)
    parser.add_argument("--quarantine-output", type=Path, required=True); parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--audit-md", type=Path, required=True); parser.add_argument("--expected-count", type=int)
    parser.add_argument("--expected-sha256"); parser.add_argument("--config", type=Path)
    args = parser.parse_args(argv)
    try:
        accepted, quarantined, report = audit_file(args.input, expected_count=args.expected_count, expected_sha256=args.expected_sha256)
        for path in (args.accepted_output, args.quarantine_output, args.audit_json, args.audit_md): path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(args.accepted_output, accepted); write_jsonl(args.quarantine_output, quarantined)
        args.audit_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        args.audit_md.write_text(markdown_report(report), encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True)); return 0
    except Exception as exc:
        print(f"legacy assignment audit failed: {exc}", file=sys.stderr); return 2

if __name__ == "__main__": raise SystemExit(main())
