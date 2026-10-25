"""Tests for provider/model request parameter policies."""

import pytest

from ai_runtime import parameters as runtime_parameters


@pytest.mark.parametrize("parameter", ["temperature", "top_p", "top_k", "candidate_count"])
def test_gemini_38_omits_sampling_controls(parameter: str) -> None:
    assert not runtime_parameters.accepts_parameter("google", "gemini-3.8-flash", parameter)
    assert runtime_parameters.accepts_parameter("google", "gemini-2.5-flash", parameter)


def test_kimi_fixed_sampling_policy_overrides_stale_catalog() -> None:
    specs = {
        "model_specifications": {
            "moonshot": {
                "kimi-k2.7-code": {
                    "parameter_constraints": {
                        "temperature": {"mode": "allow"},
                    },
                },
            },
        },
    }

    assert (
        runtime_parameters.accepts_parameter(
            "moonshot",
            "kimi-k2.7-code",
            "temperature",
            specs=specs,
        )
        is False
    )


@pytest.mark.parametrize(
    "provider,model_id,allowed",
    [
        ("openai", "gpt-4o-mini-2024-07-18", True),
        ("openai", "ft:gpt-4o-2024-08-06:example:custom:id", True),
        ("openai", "gpt-4-1106-preview", True),
        ("openai", "o3", False),
        ("openai", "gpt-6-luna", False),
        ("openai", "gpt-future", False),
        ("openrouter", "openai/gpt-4.1:online", True),
        ("openrouter", "openai/gpt-6-sol", False),
        ("claude", "claude-3-5-sonnet-20241022", True),
        ("anthropic", "claude-opus-4-6", True),
        ("claude", "claude-opus-5-5", False),
        ("claude", "claude-future", False),
        ("openrouter", "anthropic/claude-3.5-sonnet", True),
        ("openrouter", "anthropic/claude-opus-5.5:thinking", False),
        ("openrouter", "anthropic/claude-sonnet-4.6", False),
        ("openrouter", "google/gemini-3.1-pro-preview", True),
    ],
)
def test_temperature_requires_known_compatible_model(provider, model_id, allowed):
    params = {"temperature": 0.9, "max_tokens": 64}
    assert runtime_parameters.add_parameter_if_allowed(
        params, "temperature", 0.4, provider=provider, model_id=model_id,
    ) is allowed
    assert params == ({"temperature": 0.4, "max_tokens": 64} if allowed else {"max_tokens": 64})


@pytest.mark.parametrize(
    "provider,model_id,mode",
    [
        ("openai", "gpt-future", "allow"),
        ("claude", "claude-opus-5-5", "allow"),
        ("openrouter", "openai/gpt-future", "allow"),
        ("openrouter", "anthropic/claude-future", "allow"),
        ("openai", "gpt-4o", "omit"),
        ("claude", "claude-opus-4-6", "omit"),
    ],
)
def test_catalog_can_restrict_but_not_widen_temperature_support(provider, model_id, mode):
    specs = {"model_specifications": {provider: {
        model_id: {"parameter_constraints": {"temperature": {"mode": mode}}},
    }}}
    assert not runtime_parameters.accepts_parameter(
        provider, model_id, "temperature", specs=specs,
    )
