"""Administrative local inference endpoint registry.

This module deliberately models API services, never machines or model lifecycle.
It is shared by discovery and transport so both use the exact same canonical URL.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import urlsplit, urlunsplit

import json_utils as json

SUPPORTED_ENDPOINT_ADAPTERS = frozenset({"ollama", "llamacpp", "openai_compatible"})
SUPPORTED_AVAILABILITY_POLICIES = frozenset({"loaded", "available"})
_ENDPOINT_CONFIG_FIELDS = frozenset(
    {"endpoint_id", "adapter", "base_url", "api_key_env", "max_concurrent_requests", "availability"}
)
_ENDPOINT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_ENVIRONMENT_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class LocalEndpointConfigurationError(ValueError):
    """Raised when administrative endpoint configuration is unsafe or ambiguous."""


@dataclass(frozen=True)
class LocalInferenceEndpoint:
    """One administrator-configured inference API service."""

    endpoint_id: str
    adapter: str
    base_url: str
    api_key_env: str | None = None
    max_concurrent_requests: int = 1
    availability: str = "loaded"
    api_key: str | None = field(default=None, repr=False)

    def authorization_header(self) -> dict[str, str]:
        """Return the configured bearer credential without storing secrets in JSON."""

        if self.api_key_env:
            api_key = os.getenv(self.api_key_env, "").strip()
            if not api_key:
                raise LocalEndpointConfigurationError(
                    f"Endpoint '{self.endpoint_id}' requires environment variable '{self.api_key_env}'."
                )
        else:
            api_key = str(self.api_key or "").strip()
        if not api_key:
            return {}
        return {"Authorization": f"Bearer {api_key}"}


@dataclass(frozen=True)
class LocalEndpointRegistry:
    """Immutable endpoint inventory read from administration configuration."""

    endpoints: tuple[LocalInferenceEndpoint, ...]
    source: str
    legacy: bool = False

    def get(self, endpoint_id: str) -> LocalInferenceEndpoint:
        normalized = str(endpoint_id or "").strip()
        for endpoint in self.endpoints:
            if endpoint.endpoint_id == normalized:
                return endpoint
        raise LocalEndpointConfigurationError(f"Unknown local endpoint '{normalized}'.")

    def __iter__(self):
        return iter(self.endpoints)


def normalize_endpoint_base_url(adapter: str, base_url: Any) -> str:
    """Validate and canonicalize an administrative HTTP endpoint URL.

    Ollama owns the ``/api`` namespace, while OpenAI-compatible adapters use a
    canonical ``/v1`` base.  Query strings, fragments and embedded credentials
    are deliberately rejected: they are neither endpoint identity nor safe
    administrative configuration.
    """

    normalized_adapter = str(adapter or "").strip().lower()
    if normalized_adapter not in SUPPORTED_ENDPOINT_ADAPTERS:
        raise LocalEndpointConfigurationError(f"Unsupported endpoint adapter '{adapter}'.")
    raw_url = str(base_url or "").strip()
    if not raw_url:
        raise LocalEndpointConfigurationError("Endpoint base_url is required.")
    if "://" not in raw_url:
        raw_url = f"http://{raw_url}"
    parsed = urlsplit(raw_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise LocalEndpointConfigurationError(f"Endpoint URL must be an absolute HTTP(S) URL: '{base_url}'.")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise LocalEndpointConfigurationError("Endpoint URL cannot contain credentials, query parameters or fragments.")
    host = parsed.hostname
    if host in {"0.0.0.0", "::"}:
        host = "127.0.0.1" if host == "0.0.0.0" else "[::1]"
    elif ":" in host and not host.startswith("["):
        host = f"[{host}]"
    try:
        port = parsed.port
    except ValueError as exc:
        raise LocalEndpointConfigurationError(f"Endpoint URL has an invalid port: '{base_url}'.") from exc
    netloc = host if port is None else f"{host}:{port}"
    path = parsed.path.rstrip("/")
    if normalized_adapter == "ollama":
        if path.endswith("/v1") or path.endswith("/api"):
            path = path.rsplit("/", 1)[0]
    elif not path.endswith("/v1"):
        path = f"{path}/v1"
    return urlunsplit((parsed.scheme, netloc, path, "", "")).rstrip("/")


def endpoint_url(endpoint: LocalInferenceEndpoint, suffix: str) -> str:
    """Build a read-only API URL from a canonical endpoint and path suffix."""

    return f"{endpoint.base_url.rstrip('/')}/{suffix.lstrip('/')}"


def load_local_endpoint_registry(path: str | os.PathLike[str]) -> LocalEndpointRegistry:
    """Load the JSON endpoint registry configured by ``LOCAL_ENDPOINTS_CONFIG_PATH``."""

    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise LocalEndpointConfigurationError(f"Local endpoint configuration file does not exist: {config_path}")
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise LocalEndpointConfigurationError(
            f"Could not read local endpoint configuration '{config_path}': {exc}"
        ) from exc
    endpoints = _parse_endpoints(payload)
    return LocalEndpointRegistry(tuple(endpoints), str(config_path), legacy=False)


def build_local_endpoint_registry(config_obj: Any) -> LocalEndpointRegistry:
    """Return JSON-configured endpoints or deterministic legacy singleton endpoints.

    A new registry and non-default legacy values would describe two competing
    inventories.  Reject that configuration rather than silently merge them.
    """

    configured_path = _text_setting(config_obj, "LOCAL_ENDPOINTS_CONFIG_PATH")
    if configured_path:
        legacy_values = _nondefault_legacy_values(config_obj)
        if legacy_values:
            raise LocalEndpointConfigurationError(
                "LOCAL_ENDPOINTS_CONFIG_PATH cannot be combined with legacy settings: "
                + ", ".join(legacy_values)
            )
        return load_local_endpoint_registry(configured_path)
    return LocalEndpointRegistry(tuple(_legacy_endpoints(config_obj)), "legacy environment", legacy=True)


def _parse_endpoints(payload: Any) -> list[LocalInferenceEndpoint]:
    raw_endpoints = payload.get("local_endpoints") if isinstance(payload, Mapping) else None
    if raw_endpoints is None:
        raise LocalEndpointConfigurationError("Endpoint JSON must contain 'local_endpoints'.")
    items: Iterable[tuple[Any, Any]]
    if isinstance(raw_endpoints, Mapping):
        items = raw_endpoints.items()
    elif isinstance(raw_endpoints, list):
        if not all(isinstance(item, Mapping) for item in raw_endpoints):
            raise LocalEndpointConfigurationError("Every local endpoint must be a JSON object.")
        items = ((item.get("endpoint_id"), item) for item in raw_endpoints)
    else:
        raise LocalEndpointConfigurationError("'local_endpoints' must be an object or array.")
    parsed: list[LocalInferenceEndpoint] = []
    seen: set[str] = set()
    for raw_id, raw_value in items:
        if not isinstance(raw_value, Mapping):
            raise LocalEndpointConfigurationError("Every local endpoint must be a JSON object.")
        endpoint_id = _validated_endpoint_id(raw_value.get("endpoint_id") or raw_id)
        if "api_key" in raw_value:
            raise LocalEndpointConfigurationError(
                f"Endpoint '{endpoint_id}' must use api_key_env; inline api_key is not allowed."
            )
        unknown_fields = sorted(set(raw_value) - _ENDPOINT_CONFIG_FIELDS)
        if unknown_fields:
            raise LocalEndpointConfigurationError(
                f"Endpoint '{endpoint_id}' has unknown field(s): {', '.join(unknown_fields)}. "
                f"Allowed fields: {', '.join(sorted(_ENDPOINT_CONFIG_FIELDS))}."
            )
        if endpoint_id in seen:
            raise LocalEndpointConfigurationError(f"Duplicate local endpoint id '{endpoint_id}'.")
        seen.add(endpoint_id)
        adapter = str(raw_value.get("adapter") or "").strip().lower()
        api_key_env = raw_value.get("api_key_env")
        if api_key_env is not None:
            api_key_env = str(api_key_env).strip()
            if not _ENVIRONMENT_NAME_RE.fullmatch(api_key_env):
                raise LocalEndpointConfigurationError(
                    f"Endpoint '{endpoint_id}' has invalid api_key_env."
                )
        availability = str(raw_value.get("availability") or "loaded").strip().lower()
        if availability not in SUPPORTED_AVAILABILITY_POLICIES:
            raise LocalEndpointConfigurationError(
                f"Endpoint '{endpoint_id}' availability must be 'loaded' or 'available'."
            )
        try:
            max_requests = int(raw_value.get("max_concurrent_requests", 1))
        except (TypeError, ValueError) as exc:
            raise LocalEndpointConfigurationError(
                f"Endpoint '{endpoint_id}' max_concurrent_requests must be a positive integer."
            ) from exc
        if max_requests < 1:
            raise LocalEndpointConfigurationError(
                f"Endpoint '{endpoint_id}' max_concurrent_requests must be a positive integer."
            )
        parsed.append(
            LocalInferenceEndpoint(
                endpoint_id=endpoint_id,
                adapter=adapter,
                base_url=normalize_endpoint_base_url(adapter, raw_value.get("base_url")),
                api_key_env=api_key_env or None,
                max_concurrent_requests=max_requests,
                availability=availability,
            )
        )
    if not parsed:
        raise LocalEndpointConfigurationError("Endpoint registry must contain at least one endpoint.")
    return parsed


def _validated_endpoint_id(value: Any) -> str:
    endpoint_id = str(value or "").strip()
    if not _ENDPOINT_ID_RE.fullmatch(endpoint_id):
        raise LocalEndpointConfigurationError(
            "endpoint_id must use letters, digits, '_' or '-' and start with a letter or digit."
        )
    return endpoint_id


def _legacy_endpoints(config_obj: Any) -> list[LocalInferenceEndpoint]:
    entries = (
        ("ollama", "OLLAMA_ENDPOINT_ALIAS", "OLLAMA_HOST", "OLLAMA_MAX_CONCURRENT_REQUESTS", None),
        (
            "llamacpp",
            "LLAMACPP_ENDPOINT_ALIAS",
            "LLAMACPP_BASE_URL",
            "LLAMACPP_MAX_CONCURRENT_REQUESTS",
            "LLAMACPP_API_KEY",
        ),
    )
    endpoints: list[LocalInferenceEndpoint] = []
    seen: set[str] = set()
    for adapter, alias_key, url_key, concurrency_key, api_key_key in entries:
        base_url = _text_setting(config_obj, url_key)
        if not base_url:
            continue
        endpoint_id = _validated_endpoint_id(_text_setting(config_obj, alias_key, adapter))
        if endpoint_id in seen:
            raise LocalEndpointConfigurationError(f"Duplicate local endpoint id '{endpoint_id}'.")
        seen.add(endpoint_id)
        endpoints.append(
            LocalInferenceEndpoint(
                endpoint_id=endpoint_id,
                adapter=adapter,
                base_url=normalize_endpoint_base_url(adapter, base_url),
                max_concurrent_requests=_positive_int_setting(config_obj, concurrency_key, 1),
                api_key=(
                    _text_setting(config_obj, api_key_key) or None
                    if api_key_key
                    else None
                ),
            )
        )
    return endpoints


def _nondefault_legacy_values(config_obj: Any) -> list[str]:
    values: list[str] = []
    ollama_host = _text_setting(config_obj, "OLLAMA_HOST")
    if ollama_host and normalize_endpoint_base_url("ollama", ollama_host) != "http://localhost:11434":
        values.append("OLLAMA_HOST")
    if _text_setting(config_obj, "LLAMACPP_BASE_URL"):
        values.append("LLAMACPP_BASE_URL")
    if _text_setting(config_obj, "OLLAMA_ENDPOINT_ALIAS", "ollama") != "ollama":
        values.append("OLLAMA_ENDPOINT_ALIAS")
    if _text_setting(config_obj, "LLAMACPP_ENDPOINT_ALIAS", "llamacpp") != "llamacpp":
        values.append("LLAMACPP_ENDPOINT_ALIAS")
    return values


def _text_setting(config_obj: Any, name: str | None, default: str = "") -> str:
    """Read a typed text setting without treating loose mock attributes as config."""

    if not name:
        return default
    value = getattr(config_obj, name, default)
    if not isinstance(value, str):
        return default
    normalized = value.strip()
    return normalized or default


def _positive_int_setting(config_obj: Any, name: str, default: int) -> int:
    """Read a positive integer setting, falling back for partial config objects."""

    value = getattr(config_obj, name, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return default
    return value
