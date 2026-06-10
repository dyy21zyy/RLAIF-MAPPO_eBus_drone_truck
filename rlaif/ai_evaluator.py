"""Offline, external-API, and replay interfaces for Stage 4 AI preferences.

Only a validated external API response or a user-supplied replay record can become
an AI preference. Objective candidate features are prompt context, never labels.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

from rlaif.preference_dataset import read_jsonl, validate_preference, write_jsonl

NO_LABELS_MESSAGE = (
    "No valid AI or human preference labels available. Refusing to create "
    "rule-based or fabricated labels."
)
MISSING_API_MESSAGE = (
    "API mode is not configured. Set RLAIF_API_KEY, RLAIF_API_BASE_URL, and "
    "RLAIF_MODEL_NAME (or configure the rlaif section), or use replay mode "
    "with a user-provided labeled JSONL file."
)


class NoPreferenceLabelsError(RuntimeError):
    """Raised when a mode has no provenance-valid labels to persist."""


@dataclass(frozen=True)
class APISettings:
    api_key: str
    api_base_url: str
    model_name: str
    temperature: float = 0.0
    max_retries: int = 3


def load_api_settings(config_path: str | Path | None = None) -> APISettings:
    """Load API settings without embedding credentials in the repository."""
    config: dict[str, Any] = {}
    if config_path is not None:
        with Path(config_path).open(encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        config = loaded.get("rlaif", {})
        if not isinstance(config, dict):
            raise ValueError("config field 'rlaif' must be a mapping")

    key_env = str(config.get("api_key_env") or "RLAIF_API_KEY")
    api_key = os.environ.get(key_env, "").strip()
    api_base_url = os.environ.get(
        "RLAIF_API_BASE_URL", str(config.get("api_base_url") or "")
    ).strip()
    model_name = os.environ.get(
        "RLAIF_MODEL_NAME", str(config.get("model_name") or "")
    ).strip()
    temperature = float(config.get("temperature", 0.0))
    max_retries = int(config.get("max_retries", 3))
    if not api_key or not api_base_url or not model_name:
        raise NoPreferenceLabelsError(MISSING_API_MESSAGE)
    if max_retries < 1:
        raise ValueError("rlaif.max_retries must be at least 1")
    return APISettings(api_key, api_base_url, model_name, temperature, max_retries)


def _failure(prompt: dict[str, Any], raw_response: str, error: str) -> dict[str, Any]:
    return {
        "prompt_id": prompt["prompt_id"],
        "state_id": prompt["state_id"],
        "action_a": prompt["action_a"],
        "action_b": prompt["action_b"],
        "raw_response": raw_response,
        "parser_status": "error",
        "validation_status": "invalid",
        "error": error,
    }


def parse_ai_response(
    prompt: dict[str, Any],
    raw_response: str,
    evaluator_model: str,
    temperature: float,
    confidence_threshold: float = 0.6,
) -> dict[str, Any]:
    try:
        label = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc.msg}") from exc
    record = {
        "preference_id": f"preference:{prompt['prompt_id']}",
        "state_id": prompt["state_id"],
        "action_a": prompt["action_a"],
        "action_b": prompt["action_b"],
        "chosen": label.get("chosen"),
        "rejected": label.get("rejected"),
        "confidence": label.get("confidence"),
        "reason": str(label.get("reason", "")),
        "prompt_version": prompt["prompt_version"],
        "evaluator_model": evaluator_model,
        "temperature": float(temperature),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "raw_response": raw_response,
        "label_source": "external_evaluator",
    }
    return validate_preference(record, confidence_threshold)


def _manual_template(prompt: dict[str, Any]) -> dict[str, Any]:
    return {
        "prompt_id": prompt["prompt_id"],
        "state_id": prompt["state_id"],
        "action_a": prompt["action_a"],
        "action_b": prompt["action_b"],
        "chosen": None,
        "rejected": None,
        "confidence": None,
        "reason": "",
        "evaluator_model": "",
        "temperature": 0.0,
    }


def run_offline(
    prompts: list[dict[str, Any]],
    output: str | Path | None = None,
    template_output: str | Path | None = None,
    small_template_output: str | Path | None = None,
    small_template_size: int = 10,
) -> dict[str, int]:
    """Write blank manual templates and categorically create no preferences."""
    if output is not None:
        path = Path(output)
        if path.exists():
            path.unlink()
    templates = [_manual_template(prompt) for prompt in prompts]
    if template_output is not None:
        write_jsonl(template_output, templates)
    if small_template_output is not None:
        write_jsonl(small_template_output, templates[:small_template_size])
    return {
        "prompts": len(prompts),
        "preferences": 0,
        "manual_templates": len(templates),
        "small_manual_templates": min(len(templates), small_template_size),
    }


def _api_url(base_url: str) -> str:
    stripped = base_url.rstrip("/")
    if stripped.endswith("/chat/completions"):
        return stripped
    return f"{stripped}/chat/completions"


def _default_api_call(
    prompt_text: str, settings: APISettings
) -> str:
    body = json.dumps(
        {
            "model": settings.model_name,
            "temperature": settings.temperature,
            "messages": [{"role": "user", "content": prompt_text}],
        }
    ).encode()
    request = urllib.request.Request(
        _api_url(settings.api_base_url),
        data=body,
        headers={
            "Authorization": f"Bearer {settings.api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        payload = json.load(response)
    return str(payload["choices"][0]["message"]["content"])


def _remove_preference_output(output: str | Path) -> None:
    path = Path(output)
    if path.exists():
        path.unlink()


def run_api(
    prompts: list[dict[str, Any]],
    output: str | Path,
    failed_output: str | Path,
    settings: APISettings | None = None,
    api_call: Callable[[str, APISettings], str] | None = None,
) -> dict[str, int]:
    """Persist only labels parsed from responses returned by a configured API."""
    if settings is None:
        _remove_preference_output(output)
        raise NoPreferenceLabelsError(MISSING_API_MESSAGE)
    call = api_call or _default_api_call
    preferences, failures = [], []
    for prompt in prompts:
        raw, error = "", "API call failed"
        for attempt in range(settings.max_retries):
            try:
                raw = call(prompt["prompt_text"], settings)
                preferences.append(
                    parse_ai_response(
                        prompt, raw, settings.model_name, settings.temperature
                    )
                )
                error = ""
                break
            except (RuntimeError, ValueError, KeyError, OSError) as exc:
                error = str(exc)
                if attempt + 1 < settings.max_retries:
                    time.sleep(0.1 * (attempt + 1))
        if error:
            failures.append(_failure(prompt, raw, error))
    write_jsonl(failed_output, failures)
    if not preferences:
        _remove_preference_output(output)
        raise NoPreferenceLabelsError(NO_LABELS_MESSAGE)
    write_jsonl(output, preferences)
    return {"prompts": len(prompts), "preferences": len(preferences), "failed": len(failures)}


def run_replay(
    prompts: list[dict[str, Any]],
    labels_path: str | Path,
    output: str | Path,
    failed_output: str | Path,
    evaluator_model: str = "user-provided-replay",
    temperature: float = 0.0,
) -> dict[str, int]:
    """Validate user-provided labels; never infer or generate a missing label."""
    labels = read_jsonl(labels_path)
    by_prompt = {prompt["prompt_id"]: prompt for prompt in prompts}
    preferences, failures = [], []
    for index, label in enumerate(labels):
        prompt = by_prompt.get(label.get("prompt_id"))
        if prompt is None and label.get("state_id"):
            prompt = next(
                (
                    item
                    for item in prompts
                    if item["state_id"] == label["state_id"]
                    and item["action_a"] == label.get("action_a")
                    and item["action_b"] == label.get("action_b")
                ),
                None,
            )
        if prompt is None:
            failures.append(
                {
                    "label_index": index,
                    "raw_response": json.dumps(label),
                    "parser_status": "ok",
                    "validation_status": "invalid",
                    "error": "no matching prompt",
                }
            )
            continue
        raw = json.dumps(
            {key: label.get(key) for key in ("chosen", "rejected", "confidence", "reason")}
        )
        try:
            record = parse_ai_response(
                prompt,
                raw,
                str(label.get("evaluator_model") or evaluator_model),
                float(label.get("temperature", temperature)),
            )
            record["label_source"] = "user_provided_replay"
            preferences.append(record)
        except (ValueError, TypeError) as exc:
            failures.append(_failure(prompt, raw, str(exc)))
    write_jsonl(failed_output, failures)
    if not preferences:
        _remove_preference_output(output)
        raise NoPreferenceLabelsError(NO_LABELS_MESSAGE)
    write_jsonl(output, preferences)
    return {"prompts": len(prompts), "preferences": len(preferences), "failed": len(failures)}
