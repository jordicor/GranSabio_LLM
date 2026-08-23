from __future__ import annotations

from types import SimpleNamespace

import pytest

from llm_routing import (
    LLM_CALL_REGISTRY,
    LLMRoutingError,
    attach_request_llm_routing,
    get_default_routing,
    merge_routing_documents,
    resolve_call,
    resolve_call_models,
)
from local_endpoint_registry import LocalEndpointRegistry, LocalInferenceEndpoint


@pytest.fixture(autouse=True)
def stub_model_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    import config as config_module

    def fake_model_info(_self: object, model_name: str) -> dict[str, object]:
        model_lower = model_name.lower()
        if model_lower.startswith("claude-"):
            provider = "anthropic"
        elif model_lower.startswith("gemini-") or model_lower.startswith("models/"):
            provider = "google"
        elif model_lower.startswith("grok-"):
            provider = "xai"
        elif model_lower.startswith("kimi-"):
            provider = "moonshot"
        elif model_lower.startswith("moonshotai/"):
            provider = "openrouter"
        else:
            provider = "openai"
        capabilities = ["text"]
        if "embedding" in model_lower:
            capabilities.append("embeddings")
        return {
            "model_id": model_name,
            "provider": provider,
            "capabilities": capabilities,
        }

    monkeypatch.setattr(config_module.Config, "get_model_info", fake_model_info)


def test_default_routing_resolves_every_registered_call() -> None:
    routing = get_default_routing()
    assert "default_models" not in routing

    for call_id, spec in LLM_CALL_REGISTRY.items():
        if spec.multi_model:
            resolution = resolve_call_models(call_id, routing=routing)
            assert resolution.models
        else:
            resolution = resolve_call(call_id, routing=routing)
            assert resolution.model


def test_request_overlay_can_override_exact_call_without_grouping() -> None:
    request = SimpleNamespace(
        llm_routing={
            "calls": {
                "generation.main": {"model": "gpt-4o-mini"},
                "gransabio.review": {"model": "claude-haiku-4-5"},
            }
        }
    )

    attach_request_llm_routing(request, set())

    assert resolve_call("generation.main", request=request).model == "gpt-4o-mini"
    assert resolve_call("gransabio.review", request=request).model == "claude-haiku-4-5"
    assert resolve_call("gransabio.regenerate", request=request).model != "claude-haiku-4-5"


def test_global_override_applies_then_call_override_wins() -> None:
    routing = merge_routing_documents(
        get_default_routing(),
        {
            "global_override": "gpt-4o-mini",
            "calls": {
                "generation.main": {"model": "gpt-4o"},
            },
        },
    )

    assert resolve_call("generation.main", routing=routing).model == "gpt-4o"
    assert resolve_call("gransabio.review", routing=routing).model == "gpt-4o-mini"
    qa_models = resolve_call_models("qa.evaluate_layer", routing=routing).models
    assert [entry["model"] for entry in qa_models] == ["gpt-4o-mini"]


def test_endpoint_id_is_preserved_for_simple_and_multi_model_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import llm_routing

    monkeypatch.setattr(
        llm_routing,
        "build_local_endpoint_registry",
        lambda _config: LocalEndpointRegistry(
            endpoints=(
                LocalInferenceEndpoint("local-default", "ollama", "http://ollama:11434"),
                LocalInferenceEndpoint("local-qa-b", "llamacpp", "http://llamacpp:8080/v1"),
            ),
            source="test",
        ),
    )
    routing = merge_routing_documents(
        get_default_routing(),
        {
            "global": {"endpoint_id": "  local-default  "},
            "calls": {
                "generation.main": {"model": "gpt-4o-mini"},
                "qa.evaluate_layer": {
                    "models": [
                        {"model": "gpt-4o-mini"},
                        {"model": "claude-haiku-4-5", "endpoint_id": "local-qa-b"},
                    ]
                },
            },
        },
    )

    assert resolve_call("generation.main", routing=routing).endpoint_id == "local-default"
    assert [entry.get("endpoint_id") for entry in resolve_call_models(
        "qa.evaluate_layer", routing=routing
    ).models] == ["local-default", "local-qa-b"]


def test_explicit_endpoint_validates_an_unqualified_colliding_local_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import config as config_module
    import llm_routing

    registry = LocalEndpointRegistry(
        endpoints=(
            LocalInferenceEndpoint("ollama-a", "ollama", "http://ollama-a:11434"),
            LocalInferenceEndpoint("llamacpp-a", "llamacpp", "http://llamacpp-a:8080/v1"),
        ),
        source="test",
    )
    monkeypatch.setattr(llm_routing, "build_local_endpoint_registry", lambda _config: registry)

    def qualified_model_info(_self: object, model_name: str, **_kwargs: object) -> dict[str, object]:
        provider, _, model_id = model_name.partition("::")
        if provider not in {"ollama", "llamacpp"} or model_id != "shared":
            raise KeyError(model_name)
        return {"model_id": model_id, "provider": provider, "capabilities": ["text"]}

    monkeypatch.setattr(config_module.Config, "get_model_info", qualified_model_info)
    routing = {
        "calls": {
            "generation.main": {"model": "shared", "endpoint_id": "ollama-a"},
            "qa.evaluate_layer": {
                "models": [{"model": "shared", "endpoint_id": "llamacpp-a"}]
            },
        }
    }

    assert resolve_call("generation.main", routing=routing).model == "ollama::shared"
    assert (
        resolve_call_models("qa.evaluate_layer", routing=routing).models[0]["model"]
        == "llamacpp::shared"
    )
    assert (
        resolve_call("generation.main", routing=routing, validate_model=False).model
        == "shared"
    )
    assert (
        resolve_call_models(
            "qa.evaluate_layer", routing=routing, validate_model=False
        ).models[0]["model"]
        == "shared"
    )


def test_endpoint_scoped_catalog_identity_reaches_generation_token_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A normal endpoint route must stay qualified before AIService dispatch."""

    import config as config_module
    import llm_routing
    from core.generation_processor import _resolve_generation_reasoning_telemetry

    registry = LocalEndpointRegistry(
        endpoints=(
            LocalInferenceEndpoint("ollama-a", "ollama", "http://ollama-a:11434"),
        ),
        source="test",
    )
    monkeypatch.setattr(llm_routing, "build_local_endpoint_registry", lambda _config: registry)

    def qualified_model_info(_self: object, model_name: str, **_kwargs: object) -> dict[str, object]:
        if model_name != "ollama::shared":
            raise RuntimeError(f"global collision for {model_name}")
        return {"model_id": "shared", "provider": "ollama", "capabilities": ["text"]}

    validated_models: list[str] = []

    def validate_token_limits(_self: object, model_name: str, *_args: object, **_kwargs: object) -> dict[str, object]:
        validated_models.append(model_name)
        if model_name != "ollama::shared":
            raise RuntimeError(f"global collision for {model_name}")
        return {
            "adjusted_reasoning_effort": None,
            "adjusted_thinking_budget_tokens": None,
        }

    monkeypatch.setattr(config_module.Config, "get_model_info", qualified_model_info)
    monkeypatch.setattr(config_module.Config, "validate_token_limits", validate_token_limits)
    route = resolve_call(
        "generation.main",
        routing={
            "calls": {
                "generation.main": {
                    "model": "shared",
                    "endpoint_id": "ollama-a",
                }
            }
        },
    )

    telemetry = _resolve_generation_reasoning_telemetry(
        SimpleNamespace(
            generator_model=route.model,
            max_tokens=64,
            reasoning_effort=None,
            thinking_budget_tokens=None,
        )
    )

    assert validated_models == ["ollama::shared"]
    assert telemetry["is_thinking_model"] is False


@pytest.mark.parametrize("endpoint_id", ["", "   ", 42])
def test_endpoint_id_must_be_a_non_empty_string(endpoint_id: object) -> None:
    routing = merge_routing_documents(
        get_default_routing(),
        {"calls": {"generation.main": {"model": "gpt-4o-mini", "endpoint_id": endpoint_id}}},
    )

    with pytest.raises(LLMRoutingError, match="endpoint_id"):
        resolve_call("generation.main", routing=routing)


def test_global_openrouter_provider_options_require_a_global_openrouter_model() -> None:
    zero_cost_options = {
        "max_price": {"prompt": 0, "completion": 0},
        "allow_fallbacks": False,
    }
    unsafe_routing = merge_routing_documents(
        get_default_routing(),
        {"global": {"provider_options": zero_cost_options}},
    )
    with pytest.raises(LLMRoutingError, match="requires an OpenRouter model"):
        resolve_call("generation.main", routing=unsafe_routing)

    routing = merge_routing_documents(
        get_default_routing(),
        {
            "global": {
                "model": "moonshotai/kimi-k2.7-code",
                "provider_options": zero_cost_options,
            }
        },
    )
    for call_id in (
        "generation.main",
        "preflight.validate",
        "long_text.section_draft",
    ):
        route = resolve_call(call_id, routing=routing)
        assert route.model == "moonshotai/kimi-k2.7-code"
        assert route.provider_options == {
            "max_price": {"prompt": 0.0, "completion": 0.0},
            "allow_fallbacks": False,
        }


def test_unknown_call_id_fails_fast() -> None:
    with pytest.raises(LLMRoutingError, match="Unknown llm_routing call id"):
        merge_routing_documents(
            get_default_routing(),
            {"calls": {"not.a.real.call": {"model": "gpt-4o"}}},
        )


def test_required_logprobs_capability_fails_fast() -> None:
    routing = merge_routing_documents(
        get_default_routing(),
        {"calls": {"evidence.score_logprobs": {"model": "claude-haiku-4-5"}}},
    )

    with pytest.raises(LLMRoutingError, match="logprobs"):
        resolve_call("evidence.score_logprobs", routing=routing)


def test_unsupported_optional_param_warns_and_drops_by_default() -> None:
    routing = merge_routing_documents(
        get_default_routing(),
        {
            "calls": {
                "generation.main": {
                    "model": "gpt-5-mini",
                    "params": {"temperature": 0.7, "reasoning_effort": "low"},
                }
            }
        },
    )

    resolution = resolve_call("generation.main", routing=routing)

    assert resolution.params == {"reasoning_effort": "low"}
    assert resolution.warnings
    assert "temperature" in resolution.warnings[0]


def test_explicit_llamacpp_reasoning_metadata_preserves_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import config as config_module

    monkeypatch.setattr(
        config_module.Config,
        "get_model_info",
        lambda _self, _model: {
            "model_id": "server-model-q4",
            "provider": "llamacpp",
            "capabilities": ["text", "reasoning"],
            "supported_parameters": ["reasoning_effort"],
        },
    )
    routing = merge_routing_documents(
        get_default_routing(),
        {
            "calls": {
                "generation.main": {
                    "model": "llamacpp/server-model-q4",
                    "params": {"reasoning_effort": "none"},
                }
            }
        },
    )

    resolution = resolve_call("generation.main", routing=routing)

    assert resolution.params == {"reasoning_effort": "none"}
    assert resolution.warnings == []


def test_llamacpp_without_reasoning_metadata_drops_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import config as config_module

    monkeypatch.setattr(
        config_module.Config,
        "get_model_info",
        lambda _self, _model: {
            "model_id": "server-model-q4",
            "provider": "llamacpp",
            "capabilities": ["text"],
            "supported_parameters": [],
        },
    )
    routing = merge_routing_documents(
        get_default_routing(),
        {
            "calls": {
                "generation.main": {
                    "model": "llamacpp/server-model-q4",
                    "params": {"reasoning_effort": "none"},
                }
            }
        },
    )

    resolution = resolve_call("generation.main", routing=routing)

    assert resolution.params == {}
    assert len(resolution.warnings) == 1
    assert "reasoning_effort" in resolution.warnings[0]


def test_kimi_27_temperature_warns_and_drops_by_default() -> None:
    routing = merge_routing_documents(
        get_default_routing(),
        {
            "calls": {
                "generation.main": {
                    "model": "kimi-k2.7-code",
                    "params": {"temperature": 0.2},
                }
            }
        },
    )

    resolution = resolve_call("generation.main", routing=routing)

    assert resolution.params == {}
    assert resolution.warnings
    assert "temperature" in resolution.warnings[0]


def test_kimi_27_top_p_warns_and_drops_by_default() -> None:
    routing = merge_routing_documents(
        get_default_routing(),
        {
            "calls": {
                "generation.main": {
                    "model": "kimi-k2.7-code",
                    "params": {"top_p": 0.5},
                }
            }
        },
    )

    resolution = resolve_call("generation.main", routing=routing)

    assert resolution.params == {}
    assert resolution.warnings
    assert any("top_p" in warning for warning in resolution.warnings)


def test_openrouter_kimi_27_temperature_warns_and_drops_by_default() -> None:
    routing = merge_routing_documents(
        get_default_routing(),
        {
            "calls": {
                "qa.evaluate_layer": {
                    "models": [
                        {
                            "model": "moonshotai/kimi-k2.7-code",
                            "params": {"temperature": 0.3},
                        }
                    ]
                }
            }
        },
    )

    resolution = resolve_call_models("qa.evaluate_layer", routing=routing)

    assert resolution.models == [{"model": "moonshotai/kimi-k2.7-code"}]
    assert resolution.warnings
    assert "temperature" in resolution.warnings[0]
