"""Runtime health state for external LLM providers.

This module combines two signals:

- local observations from real provider calls; and
- cached official provider status checks.

It does not perform synthetic model calls.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from html.parser import HTMLParser
from threading import RLock
from typing import Any, Iterable
from xml.etree import ElementTree

import aiohttp

import json_utils as json
from provider_errors import ProviderErrorKind, ProviderFailure

PROVIDER_OPERATIONAL = "operational"
PROVIDER_SUSPECTED = "suspected"
PROVIDER_DEGRADED = "degraded"
PROVIDER_RECOVERING = "recovering"

OFFICIAL_SOURCE = "official_status"
LOCAL_SOURCE = "local"
LOCAL_ERRORS_SOURCE = "local_errors"
LOCAL_SUCCESSES_SOURCE = "local_successes"

REMOTE_STATUS_PROVIDERS = ("openai", "claude", "gemini", "openrouter", "xai")
DEFAULT_PROVIDERS = (*REMOTE_STATUS_PROVIDERS, "minimax", "moonshot", "ollama", "fake")

_EVENT_TTL_SECONDS = 15 * 60
_MAX_EVENTS_PER_PROVIDER = 300
_BASE_CHECK_INTERVAL_SECONDS = 5 * 60
_SUSPECTED_CHECK_INTERVAL_SECONDS = 90
_HTTP_TIMEOUT_SECONDS = 8
_USER_AGENT = "GranSabioLLM/1.0 provider-health"

_PROVIDER_DISPLAY_NAMES = {
    "openai": "OpenAI",
    "claude": "Claude",
    "gemini": "Gemini",
    "openrouter": "OpenRouter",
    "xai": "xAI",
    "minimax": "MiniMax",
    "moonshot": "Moonshot/Kimi",
    "ollama": "Ollama",
    "fake": "Fake AI",
}

_OFFICIAL_DEGRADED_INDICATORS = {"minor", "major", "critical", "maintenance"}

_SUSPICIOUS_KINDS = {
    ProviderErrorKind.TRANSIENT_NETWORK,
    ProviderErrorKind.TIMEOUT,
    ProviderErrorKind.PROVIDER_OVERLOADED,
    ProviderErrorKind.PROVIDER_DOWN,
    ProviderErrorKind.NO_CONTENT,
}


@dataclass
class ProviderHealthEvent:
    timestamp: float
    kind: str
    suspicious: bool = False
    reason_code: str = ""
    error_kind: str | None = None
    status_code: int | None = None
    model: str | None = None
    operation: str | None = None


@dataclass
class ProviderHealthState:
    provider: str
    status: str = PROVIDER_OPERATIONAL
    source: str = LOCAL_SOURCE
    message: str = ""
    official_indicator: str | None = None
    official_incident: str | None = None
    official_error: str | None = None
    last_checked_at: float | None = None
    last_activity_at: float | None = None
    last_status_change_at: float = field(default_factory=time.time)
    events: deque[ProviderHealthEvent] = field(default_factory=lambda: deque(maxlen=_MAX_EVENTS_PER_PROVIDER))


_state_lock = RLock()
_states: dict[str, ProviderHealthState] = {}
_check_tasks: dict[str, asyncio.Task] = {}


def normalize_provider_key(value: str | None) -> str:
    """Return the canonical runtime provider key."""

    provider = (value or "").strip().lower()
    aliases = {
        "anthropic": "claude",
        "claude": "claude",
        "google": "gemini",
        "gemini": "gemini",
        "gpt": "openai",
        "o1": "openai",
        "openai": "openai",
        "openai responses": "openai",
        "openrouter": "openrouter",
        "minimax": "minimax",
        "moonshot": "moonshot",
        "kimi": "moonshot",
        "moonshotai": "moonshot",
        "x-ai": "xai",
        "xai": "xai",
        "grok": "xai",
        "ollama": "ollama",
        "fake": "fake",
        "fake_ai": "fake",
    }
    return aliases.get(provider, provider)


def provider_display_name(provider: str | None) -> str:
    provider_key = normalize_provider_key(provider)
    return _PROVIDER_DISPLAY_NAMES.get(provider_key, provider_key or "AI provider")


def should_surface_provider_health(health: dict[str, Any] | None) -> bool:
    if not health:
        return False
    return health.get("status") in {PROVIDER_SUSPECTED, PROVIDER_DEGRADED, PROVIDER_RECOVERING}


async def record_provider_success(
    provider: str | None,
    *,
    model: str | None = None,
    operation: str | None = None,
) -> dict[str, Any]:
    """Record a successful real provider call."""

    provider_key = normalize_provider_key(provider)
    if not provider_key:
        return {}
    snapshot = _record_event(
        provider_key,
        ProviderHealthEvent(
            timestamp=time.time(),
            kind="success",
            model=model,
            operation=operation,
        ),
    )
    return snapshot


async def record_provider_failure(failure: ProviderFailure) -> dict[str, Any]:
    """Record a normalized provider failure."""

    return await record_provider_error(
        failure.provider,
        kind=failure.kind,
        status_code=failure.status_code,
        model=failure.model_id,
        operation=failure.operation,
    )


async def record_provider_error(
    provider: str | None,
    *,
    kind: ProviderErrorKind | str,
    status_code: int | None = None,
    model: str | None = None,
    operation: str | None = None,
) -> dict[str, Any]:
    """Record a failed real provider call."""

    provider_key = normalize_provider_key(provider)
    if not provider_key:
        return {}
    try:
        error_kind = kind if isinstance(kind, ProviderErrorKind) else ProviderErrorKind(str(kind))
    except ValueError:
        error_kind = ProviderErrorKind.UNKNOWN
    suspicious = error_kind in _SUSPICIOUS_KINDS
    snapshot = _record_event(
        provider_key,
        ProviderHealthEvent(
            timestamp=time.time(),
            kind="error",
            suspicious=suspicious,
            reason_code=error_kind.value,
            error_kind=error_kind.value,
            status_code=status_code,
            model=model,
            operation=operation,
        ),
    )
    maybe_schedule_provider_check(provider_key, force=should_surface_provider_health(snapshot))
    return snapshot


def get_provider_health(provider: str | None) -> dict[str, Any]:
    provider_key = normalize_provider_key(provider)
    if not provider_key:
        return {}
    now = time.time()
    with _state_lock:
        state = _get_state_locked(provider_key)
        _prune_events_locked(state, now)
        _recalculate_status_locked(state, now)
        snapshot = _public_snapshot_locked(state)
    maybe_schedule_provider_check(provider_key)
    return snapshot


def get_all_provider_health(providers: Iterable[str] | None = None) -> dict[str, Any]:
    provider_keys = [normalize_provider_key(provider) for provider in (providers or DEFAULT_PROVIDERS)]
    provider_keys = [provider for provider in provider_keys if provider]
    snapshots = {provider: get_provider_health(provider) for provider in provider_keys}
    overall_status = _overall_status(snapshots.values())
    return {
        "status": overall_status,
        "providers": snapshots,
        "generated_at": time.time(),
    }


def provider_health_for_failure_payload(failure: ProviderFailure) -> dict[str, Any]:
    health = get_provider_health(failure.provider)
    if not should_surface_provider_health(health):
        return {}
    return {
        "provider_health": health,
        "provider_status": health["status"],
        "provider_health_message": health["message"],
        "provider_health_source": health["source"],
    }


async def refresh_official_provider_health(providers: Iterable[str] | None = None) -> dict[str, Any]:
    """Refresh official status for remote providers and return a public snapshot."""

    provider_keys = [normalize_provider_key(provider) for provider in (providers or REMOTE_STATUS_PROVIDERS)]
    provider_keys = [provider for provider in provider_keys if provider in REMOTE_STATUS_PROVIDERS]
    await asyncio.gather(*(_run_official_check(provider) for provider in provider_keys))
    return get_all_provider_health(DEFAULT_PROVIDERS)


def maybe_schedule_provider_check(provider: str | None, *, force: bool = False) -> None:
    provider_key = normalize_provider_key(provider)
    if provider_key not in REMOTE_STATUS_PROVIDERS:
        return

    with _state_lock:
        state = _get_state_locked(provider_key)
        now = time.time()
        if not force and not _official_check_due_locked(state, now):
            return
        task = _check_tasks.get(provider_key)
        if task and not task.done():
            return

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return

    task = loop.create_task(_run_official_check(provider_key))
    with _state_lock:
        _check_tasks[provider_key] = task


def reset_provider_health_state() -> None:
    """Test helper: clear all in-memory provider health state."""

    with _state_lock:
        _states.clear()
        for task in _check_tasks.values():
            task.cancel()
        _check_tasks.clear()


def _record_event(provider: str, event: ProviderHealthEvent) -> dict[str, Any]:
    with _state_lock:
        state = _get_state_locked(provider)
        state.last_activity_at = event.timestamp
        state.events.append(event)
        _prune_events_locked(state, event.timestamp)
        _recalculate_status_locked(state, event.timestamp)
        return _public_snapshot_locked(state)


def _get_state_locked(provider: str) -> ProviderHealthState:
    state = _states.get(provider)
    if state is None:
        state = ProviderHealthState(provider=provider)
        _states[provider] = state
    return state


def _prune_events_locked(state: ProviderHealthState, now: float | None = None) -> None:
    cutoff = (now or time.time()) - _EVENT_TTL_SECONDS
    while state.events and state.events[0].timestamp < cutoff:
        state.events.popleft()


def _recalculate_status_locked(state: ProviderHealthState, now: float) -> None:
    recent_60 = [event for event in state.events if event.timestamp >= now - 60]
    recent_5m = [event for event in state.events if event.timestamp >= now - 300]
    suspicious_60 = [event for event in recent_60 if event.kind == "error" and event.suspicious]
    suspicious_5m = [event for event in recent_5m if event.kind == "error" and event.suspicious]
    successes_2m = [event for event in state.events if event.kind == "success" and event.timestamp >= now - 120]
    latest_suspicious_index = max(
        (
            index
            for index, event in enumerate(state.events)
            if event.kind == "error" and event.suspicious and event.timestamp >= now - 300
        ),
        default=-1,
    )
    successes_after_suspicion = [
        event
        for index, event in enumerate(state.events)
        if index > latest_suspicious_index and event.kind == "success" and event.timestamp >= now - 120
    ]

    if state.official_indicator in _OFFICIAL_DEGRADED_INDICATORS:
        _set_status_locked(state, PROVIDER_DEGRADED, OFFICIAL_SOURCE)
        state.message = _official_degraded_message(state)
        return

    if (
        successes_after_suspicion
        and (
            state.status in {PROVIDER_SUSPECTED, PROVIDER_DEGRADED, PROVIDER_RECOVERING}
            or state.source == LOCAL_SUCCESSES_SOURCE
        )
    ):
        if len(successes_after_suspicion) >= 3:
            _set_status_locked(state, PROVIDER_OPERATIONAL, LOCAL_SUCCESSES_SOURCE)
            state.message = ""
        else:
            _set_status_locked(state, PROVIDER_RECOVERING, LOCAL_SUCCESSES_SOURCE)
            state.message = (
                f"{provider_display_name(state.provider)} recently had provider or network errors, "
                "but successful responses are being seen again."
            )
        return

    if len(suspicious_5m) >= 5:
        _set_status_locked(state, PROVIDER_DEGRADED, LOCAL_ERRORS_SOURCE)
        state.message = _local_degraded_message(state)
        return

    if len(recent_5m) >= 10 and len(suspicious_5m) >= 3 and (len(suspicious_5m) / len(recent_5m)) >= 0.2:
        _set_status_locked(state, PROVIDER_DEGRADED, LOCAL_ERRORS_SOURCE)
        state.message = _local_degraded_message(state)
        return

    if len(suspicious_60) >= 3:
        _set_status_locked(state, PROVIDER_SUSPECTED, LOCAL_ERRORS_SOURCE)
        state.message = _local_suspected_message(state)
        return

    if state.status in {PROVIDER_SUSPECTED, PROVIDER_DEGRADED} and not suspicious_5m:
        _set_status_locked(state, PROVIDER_OPERATIONAL, LOCAL_SOURCE)
        state.message = ""
        return

    if state.status == PROVIDER_RECOVERING and not suspicious_5m and not successes_2m:
        _set_status_locked(state, PROVIDER_OPERATIONAL, LOCAL_SOURCE)
        state.message = ""
        return

    if state.status == PROVIDER_OPERATIONAL:
        state.source = LOCAL_SOURCE
        state.message = ""


def _set_status_locked(state: ProviderHealthState, status: str, source: str) -> None:
    if state.status != status:
        state.status = status
        state.last_status_change_at = time.time()
    state.source = source


def _official_check_due_locked(state: ProviderHealthState, now: float) -> bool:
    if not state.last_activity_at:
        return False
    interval = (
        _SUSPECTED_CHECK_INTERVAL_SECONDS
        if state.status in {PROVIDER_SUSPECTED, PROVIDER_DEGRADED, PROVIDER_RECOVERING}
        else _BASE_CHECK_INTERVAL_SECONDS
    )
    if state.last_checked_at is None:
        return True
    return (now - state.last_checked_at) >= interval


async def _run_official_check(provider: str) -> None:
    try:
        result = await _fetch_official_status(provider)
        with _state_lock:
            state = _get_state_locked(provider)
            state.last_checked_at = time.time()
            state.official_indicator = result.get("indicator")
            state.official_incident = result.get("incident")
            state.official_error = None
            if state.official_indicator in _OFFICIAL_DEGRADED_INDICATORS:
                _set_status_locked(state, PROVIDER_DEGRADED, OFFICIAL_SOURCE)
                state.message = _official_degraded_message(state)
            elif state.status == PROVIDER_DEGRADED and state.source == OFFICIAL_SOURCE:
                _set_status_locked(state, PROVIDER_RECOVERING, OFFICIAL_SOURCE)
                state.message = (
                    f"{provider_display_name(provider)} no longer reports an active incident. "
                    "We are watching for successful responses."
                )
            elif state.status == PROVIDER_OPERATIONAL:
                state.message = ""
    except Exception as exc:
        with _state_lock:
            state = _get_state_locked(provider)
            state.last_checked_at = time.time()
            state.official_error = f"{exc.__class__.__name__}: {exc}"
    finally:
        current_task = asyncio.current_task()
        with _state_lock:
            task = _check_tasks.get(provider)
            if task is current_task:
                _check_tasks.pop(provider, None)


async def _fetch_official_status(provider: str) -> dict[str, Any]:
    provider_key = normalize_provider_key(provider)
    if provider_key == "openai":
        return await _fetch_statuspage_json("https://status.openai.com/api/v2/status.json")
    if provider_key == "claude":
        return await _fetch_statuspage_json("https://status.claude.com/api/v2/status.json")
    if provider_key == "gemini":
        return await _fetch_google_cloud_incidents()
    if provider_key == "openrouter":
        return await _fetch_openrouter_status()
    if provider_key == "xai":
        return await _fetch_xai_status()
    return {"indicator": None, "incident": None, "source": "unsupported"}


async def _fetch_statuspage_json(url: str) -> dict[str, Any]:
    payload = await _fetch_json(url)
    status = payload.get("status") if isinstance(payload, dict) else {}
    status = status if isinstance(status, dict) else {}
    indicator = status.get("indicator")
    description = status.get("description")
    return {
        "indicator": "none" if indicator in (None, "", "none") else str(indicator),
        "incident": str(description) if description else None,
        "source": OFFICIAL_SOURCE,
    }


async def _fetch_google_cloud_incidents() -> dict[str, Any]:
    payload = await _fetch_json("https://status.cloud.google.com/incidents.json")
    active_incidents = []
    for incident in payload if isinstance(payload, list) else []:
        if not isinstance(incident, dict) or incident.get("end"):
            continue
        if _google_incident_mentions_ai_product(incident):
            active_incidents.append(incident)
    if active_incidents:
        incident = active_incidents[0]
        return {
            "indicator": "major",
            "incident": str(incident.get("external_desc") or incident.get("id") or "Google reports an active AI incident."),
            "source": OFFICIAL_SOURCE,
        }
    return {"indicator": "none", "incident": None, "source": OFFICIAL_SOURCE}


def _google_incident_mentions_ai_product(incident: dict[str, Any]) -> bool:
    product_values: list[str] = []
    for key in ("service_name", "external_desc"):
        value = incident.get(key)
        if value:
            product_values.append(str(value))
    for product in incident.get("affected_products") or []:
        if isinstance(product, dict):
            product_values.extend(str(value) for value in product.values() if value)
        elif product:
            product_values.append(str(product))

    normalized_values = [value.strip().lower() for value in product_values]
    ai_products = ("gemini", "vertex ai", "generative ai", "ai studio")
    return any(any(product in value for product in ai_products) for value in normalized_values)


async def _fetch_openrouter_status() -> dict[str, Any]:
    html = await _fetch_text("https://status.openrouter.ai/")
    text = _compact_html_text(html)
    lowered = text.lower()
    if "all systems operational" in lowered:
        return {"indicator": "none", "incident": None, "source": OFFICIAL_SOURCE}
    if "major outage" in lowered or "partial outage" in lowered or "degraded" in lowered:
        return {
            "indicator": "major",
            "incident": text[:200] or "OpenRouter status page reports an incident.",
            "source": OFFICIAL_SOURCE,
        }
    return {"indicator": None, "incident": None, "source": OFFICIAL_SOURCE}


async def _fetch_xai_status() -> dict[str, Any]:
    try:
        html = await _fetch_text("https://status.x.ai/")
        text = _compact_html_text(html)
        lowered = text.lower()
        if "no incidents declared" in lowered or "we are not actively mitigating any known incidents" in lowered:
            return {"indicator": "none", "incident": None, "source": OFFICIAL_SOURCE}
        if "outage" in lowered or "disruption" in lowered or "degraded" in lowered:
            return {
                "indicator": "major",
                "incident": text[:200] or "xAI status page reports an incident.",
                "source": OFFICIAL_SOURCE,
            }
    except Exception:
        feed = await _fetch_text("https://status.x.ai/feed.xml")
        root = ElementTree.fromstring(feed)
        channel = root.find("channel")
        items = channel.findall("item") if channel is not None else []
        if not items:
            return {"indicator": "none", "incident": None, "source": OFFICIAL_SOURCE}
        latest_title = items[0].findtext("title") or "xAI status feed has recent incident entries."
        return {"indicator": None, "incident": latest_title, "source": OFFICIAL_SOURCE}
    return {"indicator": None, "incident": None, "source": OFFICIAL_SOURCE}


async def _fetch_json(url: str) -> Any:
    text = await _fetch_text(url)
    return json.loads(text)


async def _fetch_text(url: str) -> str:
    timeout = aiohttp.ClientTimeout(total=_HTTP_TIMEOUT_SECONDS)
    headers = {"User-Agent": _USER_AGENT, "Accept": "application/json,text/html,application/rss+xml,*/*"}
    async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
        async with session.get(url) as response:
            body = await response.text()
            response.raise_for_status()
            return body


def _official_degraded_message(state: ProviderHealthState) -> str:
    provider_name = provider_display_name(state.provider)
    if state.official_incident:
        return f"{provider_name} reports an API incident. This model may fail temporarily or respond more slowly."
    return f"{provider_name} reports degraded service. This model may fail temporarily or respond more slowly."


def _local_suspected_message(state: ProviderHealthState) -> str:
    provider_name = provider_display_name(state.provider)
    return (
        f"We are detecting recent provider or network errors with {provider_name}. "
        "This model may fail temporarily or take longer than usual."
    )


def _local_degraded_message(state: ProviderHealthState) -> str:
    provider_name = provider_display_name(state.provider)
    return (
        f"We are detecting repeated recent provider or network errors with {provider_name}. "
        "This model may fail temporarily or take longer than usual."
    )


def _public_snapshot_locked(state: ProviderHealthState) -> dict[str, Any]:
    recent = list(state.events)
    suspicious = [event for event in recent if event.kind == "error" and event.suspicious]
    return {
        "provider": state.provider,
        "provider_name": provider_display_name(state.provider),
        "status": state.status,
        "source": state.source,
        "message": state.message,
        "official_indicator": state.official_indicator,
        "official_incident": state.official_incident,
        "official_error": state.official_error,
        "official_status": {
            "indicator": state.official_indicator,
            "incident": state.official_incident,
            "error": state.official_error,
            "last_checked_at": state.last_checked_at,
        },
        "local_errors": {
            "recent_event_count": len(recent),
            "recent_suspicious_error_count": len(suspicious),
        },
        "last_checked_at": state.last_checked_at,
        "last_activity_at": state.last_activity_at,
        "last_status_change_at": state.last_status_change_at,
        "recent_event_count": len(recent),
        "recent_suspicious_error_count": len(suspicious),
        "surface": state.status in {PROVIDER_SUSPECTED, PROVIDER_DEGRADED, PROVIDER_RECOVERING},
    }


def _overall_status(snapshots: Iterable[dict[str, Any]]) -> str:
    statuses = {snapshot.get("status") for snapshot in snapshots if snapshot}
    if PROVIDER_DEGRADED in statuses:
        return PROVIDER_DEGRADED
    if PROVIDER_SUSPECTED in statuses:
        return PROVIDER_SUSPECTED
    if PROVIDER_RECOVERING in statuses:
        return PROVIDER_RECOVERING
    return PROVIDER_OPERATIONAL


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if text:
            self.parts.append(text)


def _compact_html_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    return " ".join(parser.parts).strip()
