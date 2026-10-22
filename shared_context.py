"""User-level transport for an optional stable source prefix.

The prefix is data, never a system instruction. Cache reuse is provider-managed;
its presence does not promise a cache hit or change the variable task text.
"""

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Iterator, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from provider_capabilities import PromptCacheCapabilities

logger = logging.getLogger(__name__)


class PromptCachePolicy(BaseModel):
    """Request cache policy; offsets count Unicode code points in ``prompt``."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["auto", "off", "explicit", "adaptive"] = "auto"
    breakpoints: list[StrictInt] = Field(default_factory=list, max_length=4)

    @model_validator(mode="after")
    def validate_breakpoints(self) -> "PromptCachePolicy":
        """Accept ordered, positive endpoints in explicit or adaptive mode."""
        if self.breakpoints and self.mode not in {"explicit", "adaptive"}:
            raise ValueError("prompt_cache.breakpoints requires mode='explicit' or 'adaptive'")
        previous = 0
        for offset in self.breakpoints:
            if offset <= previous:
                raise ValueError("prompt_cache.breakpoints must be positive, sorted and unique")
            previous = offset
        return self


class PromptCachePolicyError(ValueError):
    """The selected provider cannot enforce the requested cache policy."""


_request_cache_mode: ContextVar[str] = ContextVar("request_prompt_cache_mode", default="auto")
_last_cache_report: ContextVar[Optional[dict[str, Any]]] = ContextVar("last_prompt_cache_report", default=None)


@contextmanager
def prompt_cache_scope(policy: Optional[PromptCachePolicy]) -> Iterator[None]:
    """Apply a policy to one request and its child tasks, then restore the caller.

    Only the mode is inherited. Character offsets belong to the generator's
    original prompt and are mapped by its builder, never applied to QA prompts.
    """
    token = _request_cache_mode.set(policy.mode if policy else "auto")
    report_token = _last_cache_report.set(None)
    try:
        yield
    finally:
        _request_cache_mode.reset(token)
        _last_cache_report.reset(report_token)


@dataclass(frozen=True)
class PromptCachePlan:
    """A pre-dispatch decision; failures never trigger a policy retry."""

    effective: str
    reason: str
    capabilities: PromptCacheCapabilities


def resolve_prompt_cache_policy(provider: str, model_id: str) -> PromptCachePlan:
    """Resolve controls for this route without changing models, text or routing."""
    from config import config

    caps = config.get_prompt_cache_capabilities(provider, model_id)
    mode = _request_cache_mode.get()
    if mode == "auto":
        return PromptCachePlan("auto", "legacy_behavior", caps)
    if mode == "adaptive":
        if caps.explicit_prefix.supported and caps.transport_available:
            if caps.explicit_only.supported:
                return PromptCachePlan("explicit", "explicit_controls_supported", caps)
            return PromptCachePlan("prefix", "implicit_cache_remains_provider_managed", caps)
        reason = ("prefix_transport_unavailable" if caps.explicit_prefix.supported
                  else f"explicit_prefix_{caps.explicit_prefix.support.value}")
        return PromptCachePlan("provider_default", reason, caps)
    supported = caps.off.supported if mode == "off" else caps.explicit_prefix.supported and caps.explicit_only.supported
    if not supported or not caps.transport_available:
        raise PromptCachePolicyError(
            f"prompt_cache mode='{mode}' cannot be enforced for {provider}/{model_id}. "
            "Inspect prompt_cache_capabilities or opt into mode='adaptive'."
        )
    return PromptCachePlan(mode, "strict_policy_supported", caps)


def _record_cache_plan(
    provider: str, model_id: str, plan: Optional[PromptCachePlan], count: int = 0,
    adaptations: tuple[str, ...] = (),
) -> None:
    """Record metadata only; never retain or log prompt/source text."""
    report = None
    if plan is not None:
        report = {
            "requested": _request_cache_mode.get(), "effective": plan.effective,
            "reason": plan.reason, "provider": provider, "model_id": model_id,
            "breakpoints_applied": count, "rule_version": plan.capabilities.rule_version,
            "boundary_selection": plan.capabilities.boundary_selection,
            "verified_at": plan.capabilities.explicit_prefix.verified_at,
            "adaptations": list(adaptations),
        }
        logger.info("Prompt cache policy: %s", report)
    _last_cache_report.set(report)


def consume_prompt_cache_report(provider: str, model_id: str) -> Optional[dict[str, Any]]:
    """Consume telemetry only for the matching call, using existing usage callbacks."""
    from model_capability_registry import normalize_provider

    report = _last_cache_report.get()
    _last_cache_report.set(None)
    if report and normalize_provider(report["provider"]) == normalize_provider(provider) and report["model_id"] == model_id:
        return report
    return None


def validate_prompt_cache_support(provider: str, model_id: str) -> None:
    """Reject unsupported controls before dispatch rather than silently ignore them."""
    mode = _request_cache_mode.get()
    if mode == "auto":
        return
    resolve_prompt_cache_policy(provider, model_id)


class CacheablePrompt(str):
    """Rendered prompt text with exact, internal cache-boundary offsets.

    Normal string consumers see all original text. Only compatible provider
    transports interpret the offsets; builders supply them without parsing.
    """

    __slots__ = ("_cache_breakpoints", "_explicit_cache_breakpoints")

    def __new__(cls, text: str, breakpoints: tuple[int, ...], *, explicit: bool = False):
        if not isinstance(text, str):
            raise TypeError("cacheable prompt text must be a string")
        if not isinstance(breakpoints, tuple) or any(type(offset) is not int for offset in breakpoints):
            raise TypeError("cache breakpoints must be a tuple of integer offsets")
        maximum = 4 if explicit else 3
        if len(breakpoints) > maximum:
            raise ValueError(f"a cacheable prompt supports at most {maximum} breakpoints")
        previous = 0
        for offset in breakpoints:
            if not previous < offset <= len(text):
                raise ValueError("cache breakpoints must be sorted, unique, positive offsets within the text")
            previous = offset
        instance = super().__new__(cls, text)
        instance._cache_breakpoints = breakpoints
        instance._explicit_cache_breakpoints = explicit
        return instance

    @property
    def cache_breakpoints(self) -> tuple[int, ...]:
        """Character endpoints of stable prefixes in the rendered text."""
        return self._cache_breakpoints

    def __getnewargs_ex__(self) -> tuple[tuple[str, tuple[int, ...]], dict[str, bool]]:
        """Preserve offsets when tool-loop transcripts copy string values."""
        return (str(self), self.cache_breakpoints), {"explicit": self._explicit_cache_breakpoints}


def _cache_marker(provider: str, model_id: str, plan: Optional[PromptCachePlan] = None) -> dict[str, Any]:
    """Return the provider's marker for a designated stable text boundary."""
    if _request_cache_mode.get() == "off":
        return {}
    if plan is not None:
        if plan.effective == "provider_default":
            return {}
        if plan.capabilities.marker == "prompt_cache_breakpoint":
            return {"prompt_cache_breakpoint": {"mode": "explicit"}}
        return {"cache_control": {"type": "ephemeral"}}
    # Preserve the old automatic wire contract, including generic router markers.
    if provider in {"claude", "anthropic", "openrouter"}:
        return {"cache_control": {"type": "ephemeral"}}
    if provider == "openai" and supports_openai_explicit_cache(model_id):
        return {"prompt_cache_breakpoint": {"mode": "explicit"}}
    return {}


def _cacheable_text_parts(text: CacheablePrompt, marker: dict[str, Any]) -> list[dict[str, Any]]:
    """Split at builder-supplied endpoints while retaining every character."""
    parts = []
    previous = 0
    for offset in text.cache_breakpoints:
        parts.append({"type": "text", "text": text[previous:offset], **marker})
        previous = offset
    if previous < len(text):
        parts.append({"type": "text", "text": text[previous:]})
    return parts


def _mark_user_prompt(message: dict[str, Any], marker: dict[str, Any]) -> dict[str, Any]:
    """Apply internal boundaries in place of user text, preserving other parts."""
    if message.get("role") != "user":
        return message
    content = message.get("content")
    if _uses_prompt_breakpoints(content):
        return {**message, "content": _cacheable_text_parts(content, marker)}
    if isinstance(content, list):
        parts = []
        changed = False
        for part in content:
            text = part.get("text") if isinstance(part, dict) else None
            if _uses_prompt_breakpoints(text) and part.get("type") in {"text", "input_text"}:
                parts.extend({**part, **segment, "type": part["type"]}
                             for segment in _cacheable_text_parts(text, marker))
                changed = True
            else:
                parts.append(part)
        if changed:
            return {**message, "content": parts}
    return message


def _uses_prompt_breakpoints(text: Any) -> bool:
    return (
        isinstance(text, CacheablePrompt) and bool(text.cache_breakpoints)
        and (_request_cache_mode.get() in {"auto", "adaptive"} or text._explicit_cache_breakpoints)
    )


def supports_openai_explicit_cache(model_id: str) -> bool:
    """Use curated controls; retain the historical model grammar only for auto."""
    if _request_cache_mode.get() != "auto":
        from config import config

        caps = config.get_prompt_cache_capabilities("openai", model_id.removeprefix("openai/"))
        return caps.explicit_prefix.supported and caps.transport_available
    model_id = model_id.lower().removeprefix("openai/")
    if not model_id.startswith("gpt-"):
        return False
    version = model_id[4:].split("-", 1)[0].split(".")
    try:
        major = int(version[0])
        minor = int(version[1]) if len(version) > 1 else 0
    except ValueError:
        return False
    return (major, minor) >= (5, 6)


def validate_shared_context(shared_context: Optional[str]) -> None:
    """Reject invalid direct-call values before a provider request is sent."""
    if shared_context is not None and not isinstance(shared_context, str):
        raise TypeError("shared_context must be a string or None")


def prepend_user_context(
    messages: list[dict[str, Any]],
    shared_context: Optional[str],
    *,
    provider: str = "",
    model_id: str = "",
) -> list[dict[str, Any]]:
    """Place source data after system messages and before the variable user task.

    Claude and OpenRouter use a stable content block with a cache boundary.
    OpenRouter translates its cache_control marker to the upstream protocol.
    Native OpenAI GPT-5.6+ uses the documented explicit breakpoint instead.
    """
    validate_shared_context(shared_context)
    mode = _request_cache_mode.get()
    plan = resolve_prompt_cache_policy(provider, model_id) if mode != "auto" else None
    marker = _cache_marker(provider, model_id, plan)
    adaptations = []
    result = messages
    if marker:
        marked = [_mark_user_prompt(message, marker) for message in messages]
        if any(original is not updated for original, updated in zip(messages, marked)):
            result = marked
    if shared_context:
        source_marker = marker if mode in {"auto", "adaptive"} else {}
        # Explicit caller endpoints take priority over our shared-source marker.
        if mode == "adaptive" and marker and plan.capabilities.max_breakpoints:
            existing = sum(
                next(iter(marker)) in part
                for message in result if isinstance(message.get("content"), list)
                for part in message["content"] if isinstance(part, dict)
            )
            if existing >= plan.capabilities.max_breakpoints:
                source_marker = {}
                adaptations.append("shared_context_boundary_omitted")
        content: Any = [{"type": "text", "text": shared_context, **source_marker}] if source_marker else shared_context
        insertion = next(
            (index for index, message in enumerate(result)
             if message.get("role") not in {"system", "developer"}),
            len(result),
        )
        result = [*result[:insertion], {"role": "user", "content": content}, *result[insertion:]]
    boundary_count = 0
    if marker:
        marker_name = next(iter(marker))
        if mode == "adaptive" and plan.capabilities.boundary_selection == "last":
            marked_parts = [
                part for message in result if isinstance(message.get("content"), list)
                for part in message["content"] if isinstance(part, dict) and marker_name in part
            ]
            if len(marked_parts) > 1:
                adaptations.append("last_breakpoint_only")
                result = [
                    {**message, "content": [
                        {k: v for k, v in part.items() if k != marker_name}
                        if isinstance(part, dict) and marker_name in part and part is not marked_parts[-1]
                        else part for part in message["content"]
                    ]} if isinstance(message.get("content"), list) else message
                    for message in result
                ]
        boundary_count = sum(
            marker_name in part
            for message in result
            if isinstance(message.get("content"), list)
            for part in message["content"]
            if isinstance(part, dict)
        )
        maximum = plan.capabilities.max_breakpoints if plan else 4
        if maximum is not None and boundary_count > maximum:
            limit_text = "four" if maximum == 4 else str(maximum)
            raise ValueError(f"a provider request supports at most {limit_text} cache breakpoints")
    _record_cache_plan(provider, model_id, plan, boundary_count, tuple(adaptations))
    return result


def with_openai_cache_options(params: dict[str, Any]) -> dict[str, Any]:
    """Enable explicit writing when an input block carries our breakpoint.

    The SDK may predate these protocol fields, so the request-level option is
    sent through extra_body. Ordinary calls return their original params.
    """
    messages = params.get("input", params.get("messages", []))
    if _request_cache_mode.get() == "adaptive":
        plan = resolve_prompt_cache_policy("openai", params.get("model", ""))
        if plan.effective in {"provider_default", "prefix"}:
            return params
    marked = isinstance(messages, list) and any(
        isinstance(message, dict)
        and isinstance(message.get("content"), list)
        and any(isinstance(part, dict) and part.get("prompt_cache_breakpoint") == {"mode": "explicit"}
                for part in message["content"])
        for message in messages
    )
    if not marked and _request_cache_mode.get() == "auto":
        return params
    return {
        **params,
        "extra_body": {
            **(params.get("extra_body") or {}),
            "prompt_cache_options": {"mode": "explicit", "ttl": "30m"},
        },
    }


def with_openrouter_cache_options(
    extra_body: Optional[dict[str, Any]], model_id: str,
) -> dict[str, Any]:
    """Disable implicit OpenAI writes through OpenRouter without changing routing."""
    result = dict(extra_body or {})
    if _request_cache_mode.get() != "auto":
        plan = resolve_prompt_cache_policy("openrouter", model_id)
        if plan.capabilities.marker == "prompt_cache_breakpoint" and plan.effective in {"explicit", "off"}:
            result["prompt_cache_options"] = {"mode": "explicit", "ttl": "30m"}
    return result


def prepend_gemini_context(
    contents: list[Any], shared_context: Optional[str], *, model_id: str = "gemini",
) -> list[Any]:
    """Prepend a user content item for Gemini without promoting source data."""
    validate_shared_context(shared_context)
    plan = resolve_prompt_cache_policy("gemini", model_id) if _request_cache_mode.get() != "auto" else None
    _record_cache_plan("gemini", model_id, plan)
    if not shared_context:
        return contents
    return [{"role": "user", "parts": [{"text": shared_context}]}, *contents]
