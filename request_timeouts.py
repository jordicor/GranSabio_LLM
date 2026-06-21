"""
Central request-process timeout configuration.

This module intentionally keeps provider connection/admin timeouts out of the
main resolution path. The values here are for model calls, request phases,
stream reads, QA waits, SDK waits, and MCP waits that can otherwise cut a slow
but valid generation.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import json_utils as json

DEFAULT_PROCESS_TIMEOUT_SECONDS = 12000.0


def _repo_path(path: Optional[str], fallback: str) -> Path:
    raw_path = path or fallback
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return Path(__file__).resolve().parent / candidate


def _deep_merge(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_json_file(path: Path, *, required: bool) -> Dict[str, Any]:
    if not path.exists():
        if required:
            raise RuntimeError(f"Request timeout config file not found: {path}")
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        raise RuntimeError(f"Request timeout config file could not be parsed: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Request timeout config file must contain a JSON object: {path}")
    return payload


def load_request_timeout_settings(
    *,
    default_path: Optional[str] = None,
    override_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Load tracked defaults plus an optional local/user override file."""

    default_file = _repo_path(default_path, "request_timeouts.default.json")
    override_file = _repo_path(override_path, "request_timeouts.json")
    settings = _load_json_file(default_file, required=True)
    local_override = _load_json_file(override_file, required=False)
    return _deep_merge(settings, local_override) if local_override else settings


def coerce_timeout_seconds(value: Any) -> Optional[float]:
    """Return a positive timeout in seconds, or None when unset/invalid."""

    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric <= 0:
        return None
    return numeric


def coerce_timeout_retries(value: Any, default: int = 0) -> int:
    """Return a non-negative retry count."""

    if value is None or isinstance(value, bool):
        return max(0, int(default))
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return max(0, int(default))
    return max(0, parsed)


def get_timeout_config_value(
    settings: Optional[Mapping[str, Any]],
    path: Iterable[str],
    *,
    fallback: Optional[float] = None,
) -> Optional[float]:
    """Resolve a positive numeric timeout from a nested config path."""

    current: Any = settings or {}
    for key in path:
        if not isinstance(current, Mapping):
            return fallback
        current = current.get(key)
    return coerce_timeout_seconds(current) or fallback


def get_config_raw_value(
    settings: Optional[Mapping[str, Any]],
    path: Iterable[str],
    *,
    default: Any = None,
) -> Any:
    """Resolve a raw nested config value without numeric timeout coercion."""

    current: Any = settings or {}
    for key in path:
        if not isinstance(current, Mapping):
            return default
        current = current.get(key)
    return default if current is None else current


def _request_timeouts_payload(request_or_timeouts: Any) -> Dict[str, Any]:
    if request_or_timeouts is None:
        return {}
    payload = getattr(request_or_timeouts, "timeouts", request_or_timeouts)
    if payload is None:
        return {}
    if isinstance(payload, Mapping):
        return dict(payload)
    model_dump = getattr(payload, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(exclude_none=True)
        if not isinstance(dumped, Mapping):
            return {}
        return {
            key: value
            for key, value in dumped.items()
            if value is not None
        }
    return {}


def get_request_timeout_field(request_or_timeouts: Any, field_name: str) -> Optional[float]:
    """Read one positive field from a request's nested timeouts object."""

    return coerce_timeout_seconds(_request_timeouts_payload(request_or_timeouts).get(field_name))


def get_request_default_timeout(request: Any) -> Optional[float]:
    """Resolve the request-level generic process timeout, if supplied."""

    direct = coerce_timeout_seconds(getattr(request, "timeout_seconds", None))
    if direct is not None:
        return direct
    return get_request_timeout_field(request, "default_seconds")


def resolve_request_timeout(
    request: Any,
    field_name: str,
    *,
    settings: Optional[Mapping[str, Any]] = None,
    config_path: Optional[Iterable[str]] = None,
    fallback: Optional[float] = DEFAULT_PROCESS_TIMEOUT_SECONDS,
) -> float:
    """
    Resolve a process timeout for one request phase.

    Precedence:
    1. request.timeouts.<field_name>
    2. request.timeout_seconds
    3. request.timeouts.default_seconds
    4. dedicated config file path
    5. supplied fallback
    """

    specific = get_request_timeout_field(request, field_name)
    if specific is not None:
        return specific
    request_default = get_request_default_timeout(request)
    if request_default is not None:
        return request_default
    if config_path:
        configured = get_timeout_config_value(settings, config_path, fallback=None)
        if configured is not None:
            return configured
    return float(fallback if fallback is not None else DEFAULT_PROCESS_TIMEOUT_SECONDS)


def resolve_config_timeout(
    settings: Optional[Mapping[str, Any]],
    path: Iterable[str],
    *,
    fallback: float = DEFAULT_PROCESS_TIMEOUT_SECONDS,
) -> float:
    """Resolve a positive timeout from config settings with a high fallback."""

    return float(get_timeout_config_value(settings, path, fallback=fallback) or fallback)


def resolve_env_timeout(
    env_name: str,
    settings: Optional[Mapping[str, Any]],
    path: Iterable[str],
    *,
    fallback: float = DEFAULT_PROCESS_TIMEOUT_SECONDS,
) -> float:
    """Resolve env override first, then timeout config, then fallback."""

    env_value = os.getenv(env_name)
    coerced = coerce_timeout_seconds(env_value)
    if coerced is not None:
        return coerced
    return resolve_config_timeout(settings, path, fallback=fallback)
