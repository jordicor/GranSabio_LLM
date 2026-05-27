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
