"""Provider/model request-parameter policy helpers."""

from __future__ import annotations

from typing import Any, Mapping, MutableMapping

from model_capability_registry import find_model_spec, normalize_provider

KIMI_FIXED_SAMPLING_PARAMETER_POLICY: dict[str, Any] = {
    "temperature": {"mode": "omit", "fixed_value": 1.0},
    "top_p": {"mode": "omit", "fixed_value": 0.95},
    "n": {"mode": "omit", "fixed_value": 1},
    "presence_penalty": {"mode": "omit", "fixed_value": 0.0},
    "frequency_penalty": {"mode": "omit", "fixed_value": 0.0},
}


def normalize_parameter_model_id(model_id: str) -> str:
    """Return a comparable model id while preserving provider prefixes."""

    normalized = (model_id or "").strip().lower()
    if ":" in normalized:
        normalized = normalized.split(":", 1)[0]
    return normalized.replace(".", "-")


def is_kimi_fixed_sampling_model(provider: str, model_id: str) -> bool:
    """Return True for Kimi K2 models whose sampling params are fixed by provider."""

    provider_key = normalize_provider(provider)
    normalized = normalize_parameter_model_id(model_id)
    if provider_key == "moonshot":
        return normalized.startswith(("kimi-k2-5", "kimi-k2-6", "kimi-k2-7"))
    if provider_key == "openrouter":
        return normalized.startswith(
            (
                "moonshotai/kimi-k2-5",
                "moonshotai/kimi-k2-6",
                "moonshotai/kimi-k2-7",
            )
        )
    return False


def _parameter_constraints_from_specs(
    *,
    specs: Mapping[str, Any],
    provider: str,
    model_id: str,
) -> Mapping[str, Any]:
    model_spec = find_model_spec(specs, provider, model_id)
    constraints = model_spec.get("parameter_constraints") if isinstance(model_spec, Mapping) else None
    return constraints if isinstance(constraints, Mapping) else {}


def parameter_policy(
    *,
    provider: str,
    model_id: str,
    specs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return request-parameter policy for a provider/model pair."""

    provider_key = normalize_provider(provider)
    policy: dict[str, Any] = {}
    constraints = _parameter_constraints_from_specs(
        specs=specs or {},
        provider=provider_key,
        model_id=model_id,
    )
    for name, constraint in constraints.items():
        if isinstance(constraint, Mapping):
            policy[str(name)] = dict(constraint)

    if is_kimi_fixed_sampling_model(provider_key, model_id):
        policy.update(KIMI_FIXED_SAMPLING_PARAMETER_POLICY)
    return policy


def accepts_parameter(
    provider: str,
    model_id: str,
    parameter_name: str,
    *,
    specs: Mapping[str, Any] | None = None,
) -> bool:
    """Return whether the runtime should forward a parameter to the provider."""

    policy = parameter_policy(provider=provider, model_id=model_id, specs=specs)
    parameter = policy.get(parameter_name)
    if isinstance(parameter, Mapping) and parameter.get("mode") == "omit":
        return False
    return True


def add_parameter_if_allowed(
    params: MutableMapping[str, Any],
    parameter_name: str,
    value: Any,
    *,
    provider: str,
    model_id: str,
    specs: Mapping[str, Any] | None = None,
) -> bool:
    """Attach a request parameter only if provider/model policy allows it."""

    if not accepts_parameter(provider, model_id, parameter_name, specs=specs):
        params.pop(parameter_name, None)
        return False
    params[parameter_name] = value
    return True


def openai_compatible_token_parameter(
    provider: str,
    model_id: str,
    *,
    specs: Mapping[str, Any] | None = None,
) -> str:
    """Return the token-budget parameter for OpenAI-compatible chat calls."""

    provider_key = normalize_provider(provider)
    policy = parameter_policy(provider=provider_key, model_id=model_id, specs=specs)
    token_policy = policy.get("max_tokens")
    if isinstance(token_policy, Mapping):
        replacement = token_policy.get("use_parameter")
        if isinstance(replacement, str) and replacement.strip():
            return replacement.strip()
    if provider_key == "moonshot" and is_kimi_fixed_sampling_model(provider_key, model_id):
        return "max_completion_tokens"
    return "max_tokens"
