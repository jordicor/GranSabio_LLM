"""Central model routing for all runtime AI calls.

This module owns model defaults and request overlays. Static model metadata
stays in ``model_specs.json``; platform settings stay in ``config.py``.
"""

from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import json_utils as json
from ai_runtime import parameters as runtime_parameters

DEFAULT_ROUTING_FILENAME = "llm_routing.default.json"


class LLMRoutingError(ValueError):
    """Raised when model routing cannot resolve a valid model for a call."""


@dataclass(frozen=True)
class LLMCallSpec:
    """Static runtime contract for one exact AI call site."""

    call_id: str
    multi_model: bool = False
    required_capabilities: tuple[str, ...] = ()
    kind: str = "chat"
    description: str = ""


@dataclass
class LLMRouteResolution:
    """Resolved model and parameters for one call."""

    call_id: str
    model: Optional[str] = None
    models: List[Dict[str, Any]] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)
    provider_options: Dict[str, Any] = field(default_factory=dict)
    source: str = "default"
    warnings: List[str] = field(default_factory=list)


LLM_CALL_REGISTRY: Dict[str, LLMCallSpec] = {
    "generation.main": LLMCallSpec("generation.main", description="Primary content generation"),
    "generation.direct_stream": LLMCallSpec("generation.direct_stream", description="Direct content streaming"),
    "generation.json_retry": LLMCallSpec("generation.json_retry", description="JSON repair regeneration"),
    "generation.smart_edit": LLMCallSpec("generation.smart_edit", description="Smart-edit generation fallback"),
    "qa.evaluate_layer": LLMCallSpec("qa.evaluate_layer", multi_model=True, description="QA layer evaluation"),
    "auto_qa.plan": LLMCallSpec("auto_qa.plan", description="Auto-QA contract planning"),
    "gransabio.review": LLMCallSpec("gransabio.review", description="Gran Sabio review and decision"),
    "gransabio.regenerate": LLMCallSpec("gransabio.regenerate", description="Gran Sabio content regeneration"),
    "gransabio.escalation": LLMCallSpec("gransabio.escalation", description="Gran Sabio deal-breaker escalation"),
    "arbiter.resolve": LLMCallSpec("arbiter.resolve", description="Arbiter conflict resolution"),
    "preflight.validate": LLMCallSpec("preflight.validate", description="Request feasibility validation"),
    "long_text.section_draft": LLMCallSpec("long_text.section_draft", description="Long Text section drafting"),
    "long_text.section_tools": LLMCallSpec("long_text.section_tools", description="Long Text section tool loop"),
    "long_text.semantic_eval": LLMCallSpec("long_text.semantic_eval", description="Long Text semantic evaluation"),
    "evidence.extract_claims": LLMCallSpec("evidence.extract_claims", description="Evidence claim extraction"),
    "evidence.score_logprobs": LLMCallSpec(
        "evidence.score_logprobs",
        required_capabilities=("logprobs",),
        description="Evidence budget scoring with logprobs",
    ),
    "feedback.analyze": LLMCallSpec("feedback.analyze", description="Feedback memory analysis"),
    "feedback.synthesize_rules": LLMCallSpec("feedback.synthesize_rules", description="Feedback rule synthesis"),
    "feedback.embed": LLMCallSpec(
        "feedback.embed",
        kind="embedding",
        required_capabilities=("embeddings",),
        description="Feedback memory embeddings",
    ),
    "feedback.embed_fallback": LLMCallSpec(
        "feedback.embed_fallback",
        kind="embedding",
        required_capabilities=("embeddings",),
        description="Feedback embedding fallback",
    ),
    "embedding.openai": LLMCallSpec(
        "embedding.openai",
        kind="embedding",
        required_capabilities=("embeddings",),
        description="OpenAI embedding call",
    ),
    "embedding.openai_fallback": LLMCallSpec(
        "embedding.openai_fallback",
        kind="embedding",
        required_capabilities=("embeddings",),
        description="OpenAI embedding fallback",
    ),
    "embedding.google": LLMCallSpec(
        "embedding.google",
        kind="embedding",
        required_capabilities=("embeddings",),
        description="Google embedding call",
    ),
    "accent.audit": LLMCallSpec("accent.audit", description="LLM accent audit"),
    "smart_edit.analyze": LLMCallSpec("smart_edit.analyze", description="Smart edit analysis"),
    "smart_edit.apply": LLMCallSpec("smart_edit.apply", description="Smart edit AI application"),
    "health.openai": LLMCallSpec("health.openai", description="OpenAI provider health check"),
    "health.claude": LLMCallSpec("health.claude", description="Claude provider health check"),
    "health.gemini": LLMCallSpec("health.gemini", description="Gemini provider health check"),
    "health.xai": LLMCallSpec("health.xai", description="xAI provider health check"),
    "health.minimax": LLMCallSpec("health.minimax", description="MiniMax provider health check"),
    "health.moonshot": LLMCallSpec("health.moonshot", description="Moonshot/Kimi provider health check"),
}


_SOFT_PARAMS = {
    "temperature",
    "top_p",
    "reasoning_effort",
    "thinking_budget_tokens",
    "max_tokens",
}


DEFAULT_TEMPERATURE_BY_CALL: Dict[str, float] = {
    "generation.json_retry": 0.3,
    "generation.smart_edit": 0.2,
    "qa.evaluate_layer": 0.3,
    "auto_qa.plan": 0.2,
    "gransabio.review": 0.4,
    "gransabio.regenerate": 0.7,
    "gransabio.escalation": 0.4,
    "preflight.validate": 0.2,
    "long_text.semantic_eval": 0.2,
    "evidence.extract_claims": 0.3,
    "evidence.score_logprobs": 0.1,
    "feedback.analyze": 0.2,
    "feedback.synthesize_rules": 0.2,
    "accent.audit": 0.2,
    "smart_edit.analyze": 0.3,
    "smart_edit.apply": 0.2,
}


def default_temperature_for_call(call_id: str, *, default: Optional[float] = None) -> float:
    """Return the central fallback temperature for a call id."""

    if call_id in DEFAULT_TEMPERATURE_BY_CALL:
        return DEFAULT_TEMPERATURE_BY_CALL[call_id]
    if default is not None:
        return float(default)
    raise LLMRoutingError(f"No temperature configured for LLM call '{call_id}'.")


def resolve_temperature(route: LLMRouteResolution, *, default: Optional[float] = None) -> float:
    """Return the effective temperature for a resolved call.

    Route params remain the primary source. The central fallback avoids spreading
    internal sampling defaults across runtime call-sites, and also covers routes
    whose default model drops temperature during parameter sanitization.
    """

    routed_value = route.params.get("temperature")
    if routed_value is not None:
        return float(routed_value)
    return default_temperature_for_call(route.call_id, default=default)


def _repo_default_path() -> str:
    return os.path.join(os.path.dirname(__file__), DEFAULT_ROUTING_FILENAME)


def _configured_default_path() -> str:
    env_path = os.getenv("LLM_ROUTING_DEFAULT_PATH")
    if env_path:
        return env_path
    try:
        from config import config

        configured = getattr(config, "LLM_ROUTING_DEFAULT_PATH", None)
        if configured:
            configured_path = str(configured)
            if not os.path.isabs(configured_path):
                configured_path = os.path.join(os.path.dirname(__file__), configured_path)
            return configured_path
    except Exception:
        pass
    return _repo_default_path()


def load_routing_document(path: Optional[str] = None) -> Dict[str, Any]:
    """Load a routing JSON document."""

    routing_path = path or _configured_default_path()
    with open(routing_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise LLMRoutingError(f"LLM routing document '{routing_path}' must be a JSON object.")
    return payload


def get_default_routing() -> Dict[str, Any]:
    """Return a fresh copy of the default routing document."""

    return deepcopy(load_routing_document())


def _deep_merge(base: Dict[str, Any], overlay: Mapping[str, Any]) -> Dict[str, Any]:
    result = deepcopy(base)
    for key, value in overlay.items():
        if (
            isinstance(value, Mapping)
            and isinstance(result.get(key), dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def merge_routing_documents(
    default_routing: Mapping[str, Any],
    request_routing: Optional[Mapping[str, Any]],
    legacy_overlay: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Merge default routing, request overlay, and legacy request fields."""

    merged = deepcopy(dict(default_routing))
    if request_routing:
        global_model = request_routing.get("global_override")
        global_cfg = request_routing.get("global")
        if not global_model and isinstance(global_cfg, Mapping):
            global_model = global_cfg.get("model")
        if global_model:
            global_calls: Dict[str, Any] = {}
            for call_id, spec in LLM_CALL_REGISTRY.items():
                if spec.multi_model:
                    global_calls[call_id] = {"models": [{"model": str(global_model)}]}
                else:
                    global_calls[call_id] = {"model": str(global_model)}
            merged = _deep_merge(merged, {"calls": global_calls})
        merged = _deep_merge(merged, request_routing)
    if legacy_overlay:
        merged = _deep_merge(merged, legacy_overlay)
    _validate_known_call_ids(merged)
    return merged


def _validate_known_call_ids(routing: Mapping[str, Any]) -> None:
    calls = routing.get("calls", {})
    if not isinstance(calls, Mapping):
        raise LLMRoutingError("llm_routing.calls must be an object keyed by registered call id.")
    unknown = sorted(str(call_id) for call_id in calls if call_id not in LLM_CALL_REGISTRY)
    if unknown:
        raise LLMRoutingError(
            "Unknown llm_routing call id(s): "
            + ", ".join(unknown)
            + ". Use one of: "
            + ", ".join(sorted(LLM_CALL_REGISTRY))
        )


def _request_fields_set(request: Any) -> set[str]:
    fields = getattr(request, "model_fields_set", None)
    if fields is None:
        fields = getattr(request, "__fields_set__", set())
    return {str(field) for field in fields or set()}


def _model_entry(model: Any, params: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    entry: Dict[str, Any] = {"model": str(model)}
    if params:
        entry["params"] = dict(params)
    return entry


def build_legacy_request_overlay(request: Any, fields_set: Optional[set[str]] = None) -> Dict[str, Any]:
    """Translate explicit legacy model fields into a routing overlay."""

    explicit = fields_set if fields_set is not None else _request_fields_set(request)
    calls: Dict[str, Any] = {}

    if {"generator_model", "model"} & explicit:
        generator_model = getattr(request, "generator_model", None)
        if generator_model:
            calls["generation.main"] = {"model": str(generator_model)}
            calls["generation.direct_stream"] = {"model": str(generator_model)}
            calls["generation.json_retry"] = {"model": str(generator_model)}
            calls["long_text.section_draft"] = {"model": str(generator_model)}
            calls["long_text.section_tools"] = {"model": str(generator_model)}

    if "qa_models" in explicit:
        qa_models = getattr(request, "qa_models", None) or []
        entries: List[Dict[str, Any]] = []
        for item in qa_models:
            if isinstance(item, str):
                entries.append(_model_entry(item))
            elif hasattr(item, "model"):
                params = {
                    key: getattr(item, key)
                    for key in ("max_tokens", "reasoning_effort", "thinking_budget_tokens", "temperature")
                    if getattr(item, key, None) is not None
                }
                entries.append(_model_entry(getattr(item, "model"), params))
            elif isinstance(item, Mapping) and item.get("model"):
                entries.append(dict(item))
        calls["qa.evaluate_layer"] = {"models": entries}

    if "gran_sabio_model" in explicit:
        gran_sabio_model = getattr(request, "gran_sabio_model", None)
        if gran_sabio_model:
            calls["gransabio.review"] = {"model": str(gran_sabio_model)}
            calls["gransabio.regenerate"] = {"model": str(gran_sabio_model)}
            calls["gransabio.escalation"] = {"model": str(gran_sabio_model)}
            calls["auto_qa.plan"] = {"model": str(gran_sabio_model)}

    if "arbiter_model" in explicit:
        arbiter_model = getattr(request, "arbiter_model", None)
        if arbiter_model:
            calls["arbiter.resolve"] = {"model": str(arbiter_model)}

    return {"calls": calls} if calls else {}


def attach_request_llm_routing(request: Any, fields_set: Optional[set[str]] = None) -> Dict[str, Any]:
    """Attach merged routing to a request object and return it."""

    request_routing = getattr(request, "llm_routing", None) or {}
    if request_routing and not isinstance(request_routing, Mapping):
        raise LLMRoutingError("llm_routing must be a JSON object.")
    merged = merge_routing_documents(
        get_default_routing(),
        request_routing,
        build_legacy_request_overlay(request, fields_set),
    )
    setattr(request, "_llm_routing", merged)
    setattr(request, "_llm_routing_warnings", [])
    return merged


def _routing_for_request(request: Any = None) -> Dict[str, Any]:
    if request is not None:
        existing = getattr(request, "_llm_routing", None)
        if isinstance(existing, Mapping):
            return deepcopy(dict(existing))
        return attach_request_llm_routing(request)
    return get_default_routing()


def _call_config(routing: Mapping[str, Any], call_id: str) -> tuple[Dict[str, Any], str]:
    if call_id not in LLM_CALL_REGISTRY:
        raise LLMRoutingError(f"Unknown LLM call id '{call_id}'.")

    calls = routing.get("calls", {}) or {}
    call_cfg = calls.get(call_id)
    source = "call"
    if not isinstance(call_cfg, Mapping):
        call_cfg = {}
        source = "global"

    global_cfg = routing.get("global", {}) or {}
    merged: Dict[str, Any] = {}
    if isinstance(global_cfg, Mapping):
        merged = _deep_merge(merged, global_cfg)
    merged = _deep_merge(merged, call_cfg)
    return merged, source


def _normalize_models(value: Any, inherited_params: Mapping[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise LLMRoutingError("Multi-model routing entries must use a 'models' array.")
    entries: List[Dict[str, Any]] = []
    for item in value:
        if isinstance(item, str):
            entries.append({"model": item, "params": dict(inherited_params)})
            continue
        if not isinstance(item, Mapping) or not item.get("model"):
            raise LLMRoutingError("Each multi-model routing entry must include a model.")
        params = _deep_merge(dict(inherited_params), item.get("params", {}) or {})
        entry = dict(item)
        entry["model"] = str(item["model"])
        if params:
            entry["params"] = params
        entries.append(entry)
    return entries


def _provider_key(model_info: Mapping[str, Any]) -> str:
    provider = str(model_info.get("provider") or model_info.get("catalog_provider") or "").lower()
    if provider == "anthropic":
        return "claude"
    if provider == "google":
        return "gemini"
    return provider


def _validate_known_model(model: str, call_id: str, config_obj: Any) -> Dict[str, Any]:
    try:
        return config_obj.get_model_info(model)
    except Exception as exc:
        model_lower = model.lower()
        if "embedding" in model_lower:
            provider = "google" if model_lower.startswith("models/") else "openai"
            return {"model_id": model, "provider": provider, "capabilities": ["embeddings"]}
        raise LLMRoutingError(
            f"LLM routing for '{call_id}' references unknown model '{model}': {exc}"
        ) from exc


def _supports_logprobs(model: str, model_info: Mapping[str, Any]) -> bool:
    provider = _provider_key(model_info)
    model_id = str(model_info.get("model_id") or model).lower()
    if provider == "openai":
        return not (
            model_id.startswith("o1")
            or model_id.startswith("o3")
            or "/o1" in model_id
            or "/o3" in model_id
            or "gpt-5-pro" in model_id
            or "o3-pro" in model_id
        )
    if provider == "xai":
        return "non-reasoning" in model_id
    return False


def _supports_embeddings(model: str, model_info: Mapping[str, Any]) -> bool:
    provider = _provider_key(model_info)
    model_id = str(model_info.get("model_id") or model).lower()
    return provider in {"openai", "gemini", "google"} and "embedding" in model_id


def _validate_required_capabilities(
    *,
    model: str,
    model_info: Mapping[str, Any],
    call_id: str,
    required: Iterable[str],
) -> None:
    missing: List[str] = []
    for capability in required:
        normalized = str(capability).strip().lower()
        if normalized == "logprobs" and not _supports_logprobs(model, model_info):
            missing.append("logprobs")
        elif normalized == "embeddings" and not _supports_embeddings(model, model_info):
            missing.append("embeddings")
    if missing:
        raise LLMRoutingError(
            f"LLM routing for '{call_id}' uses model '{model}', which does not support required "
            f"capabilities: {', '.join(missing)}."
        )


def _param_supported(model_info: Mapping[str, Any], param_name: str) -> bool:
    provider = _provider_key(model_info)
    model_id = str(model_info.get("model_id") or "").lower()
    if not runtime_parameters.accepts_parameter(provider, model_id, param_name):
        return False
    if param_name == "temperature":
        if "gpt-5" in model_id or model_id.startswith(("o1", "o3", "o4")):
            return False
        return True
    if param_name == "reasoning_effort":
        return "gpt-5" in model_id or model_id.startswith(("o1", "o3", "o4"))
    if param_name == "thinking_budget_tokens":
        return provider in {"claude", "gemini", "google"} or "claude" in model_id or "gemini" in model_id
    return True


def _sanitize_params(
    *,
    call_id: str,
    model: str,
    model_info: Mapping[str, Any],
    params: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> tuple[Dict[str, Any], List[str]]:
    clean: Dict[str, Any] = {}
    warnings: List[str] = []
    unsupported_policy = str(policy.get("unsupported_optional_params") or "warn_and_drop")
    for key, value in (params or {}).items():
        if value is None:
            continue
        if key in _SOFT_PARAMS and not _param_supported(model_info, key):
            message = (
                f"LLM routing param '{key}' for '{call_id}' is not supported by '{model}' "
                "and was ignored."
            )
            if unsupported_policy == "fail_fast":
                raise LLMRoutingError(message)
            warnings.append(message)
            continue
        clean[str(key)] = value
    return clean, warnings


def resolve_call(
    call_id: str,
    *,
    request: Any = None,
    routing: Optional[Mapping[str, Any]] = None,
    required_capabilities: Optional[Iterable[str]] = None,
) -> LLMRouteResolution:
    """Resolve a single-model call."""

    active_routing = deepcopy(dict(routing)) if routing is not None else _routing_for_request(request)
    call_spec = LLM_CALL_REGISTRY.get(call_id)
    if call_spec is None:
        raise LLMRoutingError(f"Unknown LLM call id '{call_id}'.")
    cfg, source = _call_config(active_routing, call_id)
    model = cfg.get("model")
    if not model:
        missing_policy = ((active_routing.get("policy") or {}).get("missing_required_call") or "fail_fast")
        if missing_policy == "fail_fast":
            raise LLMRoutingError(f"No model configured for LLM call '{call_id}'.")
        return LLMRouteResolution(call_id=call_id, source=source)

    try:
        from config import config
    except Exception as exc:
        raise LLMRoutingError("Unable to load runtime config for model validation.") from exc

    model_name = str(model)
    model_info = _validate_known_model(model_name, call_id, config)
    required = tuple(call_spec.required_capabilities) + tuple(required_capabilities or ())
    explicit_required = tuple(cfg.get("requires") or ())
    required = required + explicit_required
    _validate_required_capabilities(
        model=model_name,
        model_info=model_info,
        call_id=call_id,
        required=required,
    )
    params, warnings = _sanitize_params(
        call_id=call_id,
        model=model_name,
        model_info=model_info,
        params=cfg.get("params", {}) or {},
        policy=active_routing.get("policy", {}) or {},
    )
    if request is not None and warnings:
        getattr(request, "_llm_routing_warnings", []).extend(warnings)
    return LLMRouteResolution(
        call_id=call_id,
        model=model_name,
        params=params,
        provider_options=dict(cfg.get("provider_options", {}) or {}),
        source=source,
        warnings=warnings,
    )


def resolve_call_models(
    call_id: str,
    *,
    request: Any = None,
    routing: Optional[Mapping[str, Any]] = None,
) -> LLMRouteResolution:
    """Resolve a multi-model call."""

    active_routing = deepcopy(dict(routing)) if routing is not None else _routing_for_request(request)
    call_spec = LLM_CALL_REGISTRY.get(call_id)
    if call_spec is None:
        raise LLMRoutingError(f"Unknown LLM call id '{call_id}'.")
    cfg, source = _call_config(active_routing, call_id)
    inherited_params = cfg.get("params", {}) or {}
    raw_models = cfg.get("models")
    if raw_models is None:
        model = cfg.get("model")
        raw_models = [model] if model else []
    models = _normalize_models(raw_models, inherited_params)
    if not models:
        raise LLMRoutingError(f"No models configured for LLM call '{call_id}'.")

    try:
        from config import config
    except Exception as exc:
        raise LLMRoutingError("Unable to load runtime config for model validation.") from exc

    warnings: List[str] = []
    validated: List[Dict[str, Any]] = []
    for entry in models:
        model_name = str(entry["model"])
        model_info = _validate_known_model(model_name, call_id, config)
        required = tuple(call_spec.required_capabilities) + tuple(entry.get("requires") or ())
        _validate_required_capabilities(
            model=model_name,
            model_info=model_info,
            call_id=call_id,
            required=required,
        )
        params, entry_warnings = _sanitize_params(
            call_id=call_id,
            model=model_name,
            model_info=model_info,
            params=entry.get("params", {}) or {},
            policy=active_routing.get("policy", {}) or {},
        )
        clean_entry = dict(entry)
        if params:
            clean_entry["params"] = params
        else:
            clean_entry.pop("params", None)
        validated.append(clean_entry)
        warnings.extend(entry_warnings)

    if request is not None and warnings:
        getattr(request, "_llm_routing_warnings", []).extend(warnings)
    return LLMRouteResolution(
        call_id=call_id,
        models=validated,
        source=source,
        warnings=warnings,
    )


def legacy_default_models_view() -> Dict[str, Any]:
    """Compatibility view for old helpers without defining defaults in code."""

    routing = get_default_routing()
    return {
        "generator": resolve_call("generation.main", routing=routing).model,
        "gran_sabio": resolve_call("gransabio.review", routing=routing).model,
        "gran_sabio_fallback": resolve_call("gransabio.regenerate", routing=routing).model,
        "arbiter": resolve_call("arbiter.resolve", routing=routing).model,
        "qa": [entry["model"] for entry in resolve_call_models("qa.evaluate_layer", routing=routing).models],
    }
