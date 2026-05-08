"""Capability registry built from model specs, provider metadata, and docs rules."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from provider_capabilities import (
    CapabilitySource,
    CapabilityState,
    CapabilitySupport,
    ModelCapabilities,
    capability_state,
)


_CAPABILITY_ALIASES = {
    "structured_output": "json_schema",
    "structured_outputs": "json_schema",
    "json_schema": "json_schema",
    "json_object": "json_object",
    "tools": "tool_calling",
    "tool": "tool_calling",
    "tool_calling": "tool_calling",
    "function_calling": "tool_calling",
    "tool_use": "tool_calling",
    "tool_choice": "tool_choice",
    "parallel_tools": "parallel_tool_calls",
    "parallel_tool_calls": "parallel_tool_calls",
    "vision": "vision_input",
    "vision_input": "vision_input",
    "audio": "audio_input",
    "audio_input": "audio_input",
    "responses_api": "responses_api",
    "chat_completions_api": "chat_completions_api",
    "reasoning": "reasoning_effort",
    "reasoning_effort": "reasoning_effort",
    "thinking": "thinking_budget",
    "thinking_budget": "thinking_budget",
    "streaming": "streaming",
    "text": "text",
}


def normalize_provider(provider: str) -> str:
    provider_key = (provider or "").strip().lower()
    if provider_key == "anthropic":
        return "claude"
    if provider_key == "google":
        return "gemini"
    return provider_key


def normalize_capability_name(name: str) -> str:
    return _CAPABILITY_ALIASES.get((name or "").strip().lower(), (name or "").strip().lower())


def _provider_specs(specs: Mapping[str, Any], provider: str) -> Mapping[str, Any]:
    model_specs = specs.get("model_specifications", {}) if isinstance(specs, Mapping) else {}
    provider_key = normalize_provider(provider)
    if provider_key == "claude":
        return model_specs.get("anthropic", {}) or model_specs.get("claude", {}) or {}
    if provider_key == "gemini":
        return model_specs.get("google", {}) or model_specs.get("gemini", {}) or {}
    return model_specs.get(provider_key, {}) or {}


def find_model_spec(specs: Mapping[str, Any], provider: str, model_id: str) -> Mapping[str, Any]:
    target = (model_id or "").strip().lower()
    for model_key, model_data in _provider_specs(specs, provider).items():
        if not isinstance(model_data, Mapping):
            continue
        declared_id = str(model_data.get("model_id") or model_key).strip().lower()
        if target in {declared_id, str(model_key).strip().lower()}:
            return model_data
    return {}


def _state_from_provider_capabilities(
    model_data: Mapping[str, Any],
    capability: str,
) -> Optional[CapabilityState]:
    provider_caps = model_data.get("provider_capabilities") or {}
    if not isinstance(provider_caps, Mapping):
        return None
    raw = provider_caps.get(capability)
    if raw is None:
        return None
    if isinstance(raw, bool):
        return capability_state(raw, source=CapabilitySource.MODEL_SPECS)
    if isinstance(raw, Mapping):
        support = raw.get("support")
        if support is None and "supported" in raw:
            support = bool(raw.get("supported"))
        return capability_state(
            support,
            source=raw.get("source") or CapabilitySource.MODEL_SPECS,
            verified_at=raw.get("verified_at"),
            details={k: v for k, v in raw.items() if k not in {"support", "supported", "source", "verified_at"}},
        )
    if isinstance(raw, str):
        return capability_state(raw, source=CapabilitySource.MODEL_SPECS)
    return None


def _broad_capability_state(
    model_data: Mapping[str, Any],
    capability: str,
) -> CapabilityState:
    raw_values: list[str] = []
    for field in ("capabilities", "special_features"):
        values = model_data.get(field) or []
        if isinstance(values, list):
            raw_values.extend(str(value).lower() for value in values)
    normalized_values = {normalize_capability_name(value) for value in raw_values}

    if capability == "text":
        return capability_state(True, source=CapabilitySource.MODEL_SPECS)
    if capability in normalized_values:
        return capability_state(True, source=CapabilitySource.MODEL_SPECS)
    if capability == "reasoning_effort" and isinstance(model_data.get("reasoning_effort"), Mapping):
        return capability_state(bool(model_data["reasoning_effort"].get("supported")), source=CapabilitySource.MODEL_SPECS)
    if capability == "thinking_budget" and isinstance(model_data.get("thinking_budget"), Mapping):
        return capability_state(bool(model_data["thinking_budget"].get("supported")), source=CapabilitySource.MODEL_SPECS)
    return capability_state(CapabilitySupport.UNKNOWN, source=CapabilitySource.MODEL_SPECS)


def _openrouter_parameter_state(
    model_data: Mapping[str, Any],
    parameter_names: set[str],
    *,
    details_name: str,
) -> CapabilityState:
    params = model_data.get("supported_parameters")
    if params is None:
        params = (model_data.get("sync_metadata") or {}).get("supported_parameters")
    if not isinstance(params, list):
        return capability_state(CapabilitySupport.UNKNOWN, source=CapabilitySource.PROVIDER_API)
    normalized = {str(param).strip() for param in params}
    return capability_state(
        bool(parameter_names.intersection(normalized)),
        source=CapabilitySource.PROVIDER_API,
        details={details_name: sorted(normalized)},
    )


def _docs_rule_state(provider: str, model_id: str, capability: str) -> CapabilityState:
    provider_key = normalize_provider(provider)
    model_lower = (model_id or "").lower()

    if capability == "chat_completions_api":
        return capability_state(provider_key != "claude" and provider_key != "gemini", source=CapabilitySource.OFFICIAL_DOCS)
    if capability == "streaming":
        return capability_state(True, source=CapabilitySource.OFFICIAL_DOCS)
    if capability == "json_object":
        if provider_key in {"openai", "xai", "ollama"}:
            return capability_state(True, source=CapabilitySource.OFFICIAL_DOCS)
        if provider_key in {"claude", "gemini"}:
            return capability_state(False, source=CapabilitySource.OFFICIAL_DOCS)
    if capability == "json_schema":
        if provider_key == "openai":
            supported_prefixes = (
                "gpt-5",
                "gpt-4.1",
                "gpt-4o",
                "o3",
                "o4",
            )
            unsupported_prefixes = ("gpt-4-turbo", "gpt-3.5")
            if model_lower.startswith(unsupported_prefixes):
                return capability_state(False, source=CapabilitySource.OFFICIAL_DOCS)
            return capability_state(model_lower.startswith(supported_prefixes), source=CapabilitySource.OFFICIAL_DOCS)
        if provider_key == "claude":
            normalized_model = model_lower.replace(".", "-")
            supported_markers = (
                "claude-mythos",
                "claude-opus-4-7",
                "claude-opus-4-6",
                "claude-sonnet-4-6",
                "claude-sonnet-4-5",
                "claude-opus-4-5",
                "claude-haiku-4-5",
            )
            return capability_state(any(marker in normalized_model for marker in supported_markers), source=CapabilitySource.OFFICIAL_DOCS)
        if provider_key == "gemini":
            return capability_state("gemini" in model_lower, source=CapabilitySource.OFFICIAL_DOCS)
        if provider_key in {"xai", "ollama"}:
            return capability_state(True, source=CapabilitySource.OFFICIAL_DOCS)
    if capability == "tool_calling":
        if provider_key == "openai":
            supported_prefixes = (
                "gpt-3.5",
                "gpt-4",
                "gpt-4.1",
                "gpt-4o",
                "gpt-5",
                "o1",
                "o3",
                "o4",
            )
            return capability_state(model_lower.startswith(supported_prefixes), source=CapabilitySource.OFFICIAL_DOCS)
        if provider_key == "claude":
            return capability_state(model_lower.startswith("claude-"), source=CapabilitySource.OFFICIAL_DOCS)
        if provider_key == "gemini":
            return capability_state("gemini" in model_lower, source=CapabilitySource.OFFICIAL_DOCS)
        if provider_key == "xai":
            return capability_state("grok" in model_lower, source=CapabilitySource.OFFICIAL_DOCS)
    return capability_state(CapabilitySupport.UNKNOWN, source=CapabilitySource.OFFICIAL_DOCS)


def _capability_for(
    *,
    provider: str,
    model_id: str,
    model_data: Mapping[str, Any],
    capability: str,
) -> CapabilityState:
    explicit = _state_from_provider_capabilities(model_data, capability)
    if explicit is not None and explicit.support != CapabilitySupport.UNKNOWN:
        return explicit

    provider_key = normalize_provider(provider)
    if provider_key == "openrouter":
        if capability == "json_schema":
            return _openrouter_parameter_state(model_data, {"structured_outputs"}, details_name="supported_parameters")
        if capability == "json_object":
            return _openrouter_parameter_state(model_data, {"response_format"}, details_name="supported_parameters")
        if capability == "tool_calling":
            return _openrouter_parameter_state(model_data, {"tools"}, details_name="supported_parameters")
        if capability == "tool_choice":
            return _openrouter_parameter_state(model_data, {"tool_choice"}, details_name="supported_parameters")

    broad = _broad_capability_state(model_data, capability)
    docs = _docs_rule_state(provider_key, model_id, capability)

    if broad.support != CapabilitySupport.UNKNOWN:
        return broad
    if docs.support != CapabilitySupport.UNKNOWN:
        return docs
    if explicit is not None:
        return explicit
    return capability_state(CapabilitySupport.UNKNOWN, source=CapabilitySource.MODEL_SPECS)


def get_model_capabilities(
    *,
    specs: Mapping[str, Any],
    provider: str,
    model_id: str,
    model_data: Optional[Mapping[str, Any]] = None,
) -> ModelCapabilities:
    data = model_data or find_model_spec(specs, provider, model_id)
    return ModelCapabilities(
        provider=normalize_provider(provider),
        model_id=model_id,
        text=_capability_for(provider=provider, model_id=model_id, model_data=data, capability="text"),
        json_object=_capability_for(provider=provider, model_id=model_id, model_data=data, capability="json_object"),
        json_schema=_capability_for(provider=provider, model_id=model_id, model_data=data, capability="json_schema"),
        tool_calling=_capability_for(provider=provider, model_id=model_id, model_data=data, capability="tool_calling"),
        tool_choice=_capability_for(provider=provider, model_id=model_id, model_data=data, capability="tool_choice"),
        parallel_tool_calls=_capability_for(provider=provider, model_id=model_id, model_data=data, capability="parallel_tool_calls"),
        vision_input=_capability_for(provider=provider, model_id=model_id, model_data=data, capability="vision_input"),
        audio_input=_capability_for(provider=provider, model_id=model_id, model_data=data, capability="audio_input"),
        responses_api=_capability_for(provider=provider, model_id=model_id, model_data=data, capability="responses_api"),
        chat_completions_api=_capability_for(provider=provider, model_id=model_id, model_data=data, capability="chat_completions_api"),
        reasoning_effort=_capability_for(provider=provider, model_id=model_id, model_data=data, capability="reasoning_effort"),
        thinking_budget=_capability_for(provider=provider, model_id=model_id, model_data=data, capability="thinking_budget"),
        streaming=_capability_for(provider=provider, model_id=model_id, model_data=data, capability="streaming"),
    )


def model_supports(
    *,
    specs: Mapping[str, Any],
    provider: str,
    model_id: str,
    capability: str,
    model_data: Optional[Mapping[str, Any]] = None,
) -> CapabilityState:
    capabilities = get_model_capabilities(
        specs=specs,
        provider=provider,
        model_id=model_id,
        model_data=model_data,
    )
    return getattr(capabilities, normalize_capability_name(capability), capability_state(CapabilitySupport.UNKNOWN))
