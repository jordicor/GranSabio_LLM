"""Pure capability-planning helpers for AI runtime calls."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from model_capability_registry import (
    model_supports as registry_model_supports,
    model_uses_responses_api,
    normalize_provider,
)
from provider_capabilities import CapabilitySupport


def normalize_tool_loop_provider(provider: str) -> str:
    """Normalize provider labels for tool-loop metadata and routing."""

    return normalize_provider(provider)


def supports_structured_outputs(provider: str, model_id: str, specs: Mapping[str, Any]) -> bool:
    """Return True when the provider/model can enforce JSON Schema natively."""

    state = registry_model_supports(
        specs=specs,
        provider=normalize_provider(provider),
        model_id=model_id,
        capability="json_schema",
    )
    return state.support == CapabilitySupport.SUPPORTED


def supports_json_object(provider: str, model_id: str, specs: Mapping[str, Any]) -> bool:
    """Return True when the provider/model supports provider-native JSON mode."""

    state = registry_model_supports(
        specs=specs,
        provider=normalize_provider(provider),
        model_id=model_id,
        capability="json_object",
    )
    return state.support == CapabilitySupport.SUPPORTED


def supports_tool_calling(provider: str, model_id: str, specs: Mapping[str, Any]) -> bool:
    """Return True when the provider/model advertises native tool calling."""

    state = registry_model_supports(
        specs=specs,
        provider=normalize_provider(provider),
        model_id=model_id,
        capability="tool_calling",
    )
    return state.support == CapabilitySupport.SUPPORTED


def openrouter_tool_streaming_supported(model_id: str, specs: Mapping[str, Any]) -> bool:
    """Return True when OpenRouter metadata allows observable streamed tool loops."""

    tool_state = registry_model_supports(
        specs=specs,
        provider="openrouter",
        model_id=model_id,
        capability="tool_calling",
    )
    streaming_state = registry_model_supports(
        specs=specs,
        provider="openrouter",
        model_id=model_id,
        capability="streaming",
    )
    return (
        tool_state.support == CapabilitySupport.SUPPORTED
        and streaming_state.support == CapabilitySupport.SUPPORTED
    )


def uses_native_structured_outputs(
    provider: str,
    model_id: str,
    json_schema: Optional[dict[str, Any]],
    specs: Mapping[str, Any],
) -> bool:
    """Return True when native structured outputs will enforce JSON."""

    return json_schema is not None and supports_structured_outputs(provider, model_id, specs)


def claude_supports_structured_outputs(model_lower: str, specs: Mapping[str, Any]) -> bool:
    """Return True when Claude supports JSON Schema for the model."""

    return supports_structured_outputs("claude", model_lower, specs)


def audit_model_supports_structured_outputs(
    provider_key: str,
    model_id: str,
    specs: Mapping[str, Any],
) -> bool:
    """Return True when the audit model supports native JSON-schema output."""

    return supports_structured_outputs(provider_key, model_id, specs)


def is_openai_responses_api_model(model_id: str, specs: Mapping[str, Any]) -> bool:
    """Return True when the given OpenAI model id uses the Responses API."""

    if not model_id:
        return False
    return model_uses_responses_api("openai", model_id, specs)
