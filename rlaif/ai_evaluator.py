"""Offline, external-API, and replay interfaces for Stage 4 AI preferences."""

from __future__ import annotations

import json
import os
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from rlaif.preference_dataset import read_jsonl, validate_preference, write_jsonl


def _failure(prompt: dict[str, Any], raw_response: str, error: str) -> dict[str, Any]:
    return {"prompt_id": prompt["prompt_id"], "state_id": prompt["state_id"],
            "action_a": prompt["action_a"], "action_b": prompt["action_b"],
            "raw_response": raw_response, "parser_status": "error", "validation_status": "invalid", "error": error}


def parse_ai_response(prompt: dict[str, Any], raw_response: str, evaluator_model: str,
                      temperature: float, confidence_threshold: float = 0.6) -> dict[str, Any]:
    try:
        label = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc.msg}") from exc
    record = {
        "preference_id": f"preference:{prompt['prompt_id']}", "state_id": prompt["state_id"],
        "action_a": prompt["action_a"], "action_b": prompt["action_b"],
        "chosen": label.get("chosen"), "rejected": label.get("rejected"),
        "confidence": label.get("confidence"), "reason": str(label.get("reason", "")),
        "prompt_version": prompt["prompt_version"], "evaluator_model": evaluator_model,
        "temperature": float(temperature), "created_at": datetime.now(timezone.utc).isoformat(),
        "raw_response": raw_response,
    }
    return validate_preference(record, confidence_threshold)


def run_offline(prompts: list[dict[str, Any]], output: str | Path | None = None) -> dict[str, int]:
    """Intentionally create no labels; prompts are already persisted by the prior command."""
    if output is not None:
        path = Path(output)
        if path.exists():
            path.unlink()
    return {"prompts": len(prompts), "preferences": 0, "failed": 0}


def _default_api_call(prompt_text: str, model: str, temperature: float) -> str:
    url = os.environ.get("RLAIF_API_URL")
    key = os.environ.get("RLAIF_API_KEY")
    if not url or not key:
        raise RuntimeError("RLAIF_API_URL and RLAIF_API_KEY are required for api mode")
    body = json.dumps({"model": model, "temperature": temperature,
                       "messages": [{"role": "user", "content": prompt_text}]}).encode()
    request = urllib.request.Request(url, data=body, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 - explicitly configured endpoint
        payload = json.load(response)
    return str(payload["choices"][0]["message"]["content"])


def run_api(prompts: list[dict[str, Any]], output: str | Path, failed_output: str | Path,
            evaluator_model: str, temperature: float = 0.0, max_retries: int = 3,
            api_call: Callable[[str, str, float], str] | None = None) -> dict[str, int]:
    call = api_call or _default_api_call
    preferences, failures = [], []
    for prompt in prompts:
        raw, error = "", "API call failed"
        for attempt in range(max_retries):
            try:
                raw = call(prompt["prompt_text"], evaluator_model, temperature)
                preferences.append(parse_ai_response(prompt, raw, evaluator_model, temperature))
                error = ""
                break
            except (RuntimeError, ValueError, KeyError, OSError) as exc:
                error = str(exc)
                if attempt + 1 < max_retries:
                    time.sleep(0.1 * (attempt + 1))
        if error:
            failures.append(_failure(prompt, raw, error))
    write_jsonl(output, preferences)
    write_jsonl(failed_output, failures)
    return {"prompts": len(prompts), "preferences": len(preferences), "failed": len(failures)}


def run_replay(prompts: list[dict[str, Any]], labels_path: str | Path, output: str | Path,
               failed_output: str | Path, evaluator_model: str = "manual-replay",
               temperature: float = 0.0) -> dict[str, int]:
    labels = read_jsonl(labels_path)
    by_prompt = {prompt["prompt_id"]: prompt for prompt in prompts}
    preferences, failures = [], []
    for index, label in enumerate(labels):
        prompt = by_prompt.get(label.get("prompt_id"))
        if prompt is None and label.get("state_id"):
            prompt = next((item for item in prompts if item["state_id"] == label["state_id"] and
                           item["action_a"] == label.get("action_a") and item["action_b"] == label.get("action_b")), None)
        if prompt is None:
            failures.append({"label_index": index, "raw_response": json.dumps(label),
                             "parser_status": "ok", "validation_status": "invalid", "error": "no matching prompt"})
            continue
        raw = json.dumps({key: label.get(key) for key in ("chosen", "rejected", "confidence", "reason")})
        try:
            preferences.append(parse_ai_response(prompt, raw, label.get("evaluator_model", evaluator_model),
                                                 float(label.get("temperature", temperature))))
        except (ValueError, TypeError) as exc:
            failures.append(_failure(prompt, raw, str(exc)))
    write_jsonl(output, preferences)
    write_jsonl(failed_output, failures)
    return {"prompts": len(prompts), "preferences": len(preferences), "failed": len(failures)}
