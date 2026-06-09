"""Configuration loading and project-directory helpers."""

from __future__ import annotations

import ast
import importlib
import importlib.util
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNTIME_DIRECTORIES = (
    "data/raw",
    "data/processed",
    "logs",
    "outputs",
    "checkpoints",
)


class ConfigError(ValueError):
    """Raised when a configuration file is missing or malformed."""


def _parse_scalar(value: str) -> Any:
    """Parse the scalar/list subset used by Stage 1 YAML files."""

    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none", "~"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            inner = value[1:-1].strip()
            if not inner:
                return []
            return [_parse_scalar(item.strip()) for item in inner.split(",")]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value.strip("'\"")


def _load_simple_yaml(text: str) -> dict[str, Any]:
    """Load nested mappings from the dependency-free Stage 1 YAML subset.

    This fallback intentionally supports only indentation-based mappings and
    inline scalar lists. Install PyYAML for advanced YAML syntax in later stages.
    """

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        content = raw_line.split("#", 1)[0].rstrip()
        if not content.strip():
            continue
        indentation = len(content) - len(content.lstrip(" "))
        if indentation % 2:
            raise ConfigError(f"YAML indentation must use two-space levels at line {line_number}")
        stripped = content.strip()
        if ":" not in stripped:
            raise ConfigError(f"Expected 'key: value' at line {line_number}")
        key, raw_value = stripped.split(":", 1)
        key = key.strip()
        if not key:
            raise ConfigError(f"Empty YAML key at line {line_number}")

        while stack[-1][0] >= indentation:
            stack.pop()
        parent = stack[-1][1]
        value = raw_value.strip()
        if value:
            parent[key] = _parse_scalar(value)
        else:
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indentation, child))

    return root


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file as an independent dictionary.

    PyYAML is used when available. A dependency-free parser handles the simple
    nested mapping and inline-list syntax used by Stage 1 configurations, keeping
    the project smoke test runnable in a pre-provisioned offline environment.
    """

    path = Path(config_path).expanduser()
    if not path.is_file():
        raise ConfigError(f"Configuration file does not exist: {path}")
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ConfigError(f"Configuration file is empty: {path}")

    if text.lstrip().startswith(("{", "[")):
        try:
            loaded = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"Invalid JSON-compatible YAML in configuration file {path}: {exc}") from exc
    elif importlib.util.find_spec("yaml") is not None:
        yaml = importlib.import_module("yaml")
        try:
            loaded = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ConfigError(f"Invalid YAML in configuration file {path}: {exc}") from exc
    else:
        loaded = _load_simple_yaml(text)

    if loaded is None:
        raise ConfigError(f"Configuration file is empty: {path}")
    if not isinstance(loaded, Mapping):
        raise ConfigError(f"Configuration root must be a mapping: {path}")
    if not loaded:
        raise ConfigError(f"Configuration file has an empty root mapping: {path}")

    return deepcopy(dict(loaded))


def ensure_project_directories(
    project_root: str | Path | None = None,
    directories: tuple[str, ...] = DEFAULT_RUNTIME_DIRECTORIES,
) -> dict[str, Path]:
    """Create and return the standard writable project directories."""

    root = Path(project_root).expanduser().resolve() if project_root else PROJECT_ROOT
    created: dict[str, Path] = {}
    for relative_directory in directories:
        path = root / relative_directory
        path.mkdir(parents=True, exist_ok=True)
        created[relative_directory] = path
    return created
