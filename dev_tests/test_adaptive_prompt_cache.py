"""Cache capability and mixed-route policy checks. All provider IO is mocked."""

import asyncio
import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import openai
import pytest

from ai_service import AIService
from config import config
from core.analytics_routes import get_model_info
from core.generation_routes import _generation_request_fingerprint
from dev_tests.test_internal_cache_transport import service as service
from dev_tests.test_prompt_cache_policy import END, PROMPT, flattened, render
from dev_tests.test_shared_context_transport import FakeResponsesStream, FakeStream
from model_capability_registry import get_prompt_cache_capabilities, validate_prompt_cache_rules
from models import ContentRequest
from provider_capabilities import CapabilitySupport
from shared_context import (
    CacheablePrompt,
    PromptCachePolicy,
    PromptCachePolicyError,
    consume_prompt_cache_report,
    prepend_gemini_context,
    prepend_user_context,
    prompt_cache_scope,
    with_openai_cache_options,
    with_openrouter_cache_options,
)


def capabilities(provider, model, *, specs=None, rules=None):
    return get_prompt_cache_capabilities(
        specs=specs or {}, rules=config.MODEL_CACHE_CAPABILITIES if rules is None else rules,
        provider=provider, model_id=model,
    )


@pytest.mark.parametrize("provider,model,automatic,prefix,exclusive,off", [
    ("openai", "gpt-6-luna", "supported", "supported", "supported", "supported"),
    ("openrouter", "openai/gpt-6.1-sol", "supported", "supported", "supported", "supported"),
    ("anthropic", "claude-sonnet-4-6", "supported", "supported", "supported", "supported"),
    ("openrouter", "anthropic/claude-haiku-4.5", "supported", "supported", "supported", "supported"),
    ("openai", "gpt-4o", "supported", "unsupported", "unsupported", "unsupported"),
    ("gemini", "gemini-2.5-pro", "supported", "unsupported", "unsupported", "unknown"),
    ("openrouter", "google/gemini-2.5-pro", "supported", "supported", "unknown", "unknown"),
    ("openrouter", "qwen/qwen3-max", "unknown", "conditional", "unknown", "unknown"),
    ("openrouter", "qwen/qwen3.5-plus-02-15", "unknown", "unsupported", "unsupported", "unknown"),
    ("openrouter", "moonshotai/kimi-k3", "supported", "unknown", "unknown", "unknown"),
    ("openai", "gpt-7", "unknown", "unknown", "unknown", "unknown"),
    ("openrouter", "openai/gpt-6.2-sol", "unknown", "unknown", "unknown", "unknown"),
    ("ollama", "qwen3", "unknown", "unknown", "unknown", "unknown"),
])
def test_route_contracts_are_independent(provider, model, automatic, prefix, exclusive, off):
    caps = capabilities(provider, model)
    assert [caps.automatic.support, caps.explicit_prefix.support, caps.explicit_only.support, caps.off.support] == [
        automatic, prefix, exclusive, off,
    ]


def test_metadata_absence_and_prices_are_not_negative_or_positive_evidence():
    specs = {"model_specifications": {"openrouter": {"new/model": {
        "supported_parameters": [], "pricing": {"cached_input_per_million": 0.01},
    }}}}
    assert capabilities("openrouter", "new/model", specs=specs).explicit_prefix.support == CapabilitySupport.UNKNOWN


def test_curated_contract_overrides_metadata_without_relabelling_its_date():
    specs = {"model_specifications": {"openai": {"gpt-6-luna": {
        "provider_capabilities": {"prompt_cache": {
            "explicit_prefix": "unsupported", "verified_at": "2099-01-01",
        }}, "sync_metadata": {"last_synced_at": "2099-01-01"},
    }}}}
    caps = capabilities("openai", "gpt-6-luna", specs=specs)
    assert caps.explicit_prefix.supported
    assert caps.explicit_prefix.verified_at == "2026-10-22"
    assert caps.explicit_prefix.source == "official_docs"


@pytest.mark.parametrize("update", [
    {"marker": "invented"}, {"max_breakpoints": True}, {"min_prefix_tokens": -2},
    {"explicit_prefix": "yes"}, {"ttl": "12h"}, {"boundary_selection": "random"},
])
def test_invalid_curated_contract_fails_at_load(update):
    rule = {"provider": "openai", "source": "official", "verified_at": "2026-10-22", **update}
    with pytest.raises(ValueError):
        validate_prompt_cache_rules({"version": 1, "rules": [rule]})


@pytest.mark.asyncio
async def test_existing_model_endpoint_exposes_resolved_contract():
    result = await get_model_info("openrouter::openai/gpt-6-luna")
    assert result["prompt_cache_capabilities"]["off"]["support"] == "supported"
    assert result["prompt_cache_capabilities"]["max_breakpoints"] == 4


@pytest.mark.parametrize("provider,model", [
    ("openai", "gpt-4o"), ("openai", "gpt-7"), ("openrouter", "moonshotai/kimi-k3"),
    ("openrouter", "qwen/qwen3-max"), ("openrouter", "unknown/model"), ("ollama", "local"),
])
def test_adaptive_falls_back_before_dispatch_without_controls(provider, model):
    policy = PromptCachePolicy(mode="adaptive", breakpoints=[END])
    original = [{"role": "user", "content": render(policy)}]
    with prompt_cache_scope(policy):
        result = prepend_user_context(original, "source", provider=provider, model_id=model)
        assert flattened(result) == [("user", ["source"]), ("user", [str(render(policy))])]
        assert all(isinstance(m["content"], str) for m in result)
        if provider == "openai":
            params = {"model": model, "messages": result}
            assert with_openai_cache_options(params) is params
        if provider == "openrouter":
            routing = {"provider": {"allow_fallbacks": False}}
            assert with_openrouter_cache_options(routing, model) == routing
        report = consume_prompt_cache_report(provider, model)
        assert report["effective"] == "provider_default"
        assert report["breakpoints_applied"] == 0
        assert consume_prompt_cache_report(provider, model) is None


@pytest.mark.parametrize("mode", ["explicit", "off"])
@pytest.mark.parametrize("model", ["google/gemini-2.5-pro", "qwen/qwen3-max", "unknown/model"])
def test_prefix_support_does_not_silently_weaken_strict_modes(mode, model):
    with prompt_cache_scope(PromptCachePolicy(mode=mode)), pytest.raises(PromptCachePolicyError):
        prepend_user_context([], None, provider="openrouter", model_id=model)


def test_adaptive_uses_each_phase_own_prefix_and_effective_model():
    policy = PromptCachePolicy(mode="adaptive", breakpoints=[END])
    with prompt_cache_scope(policy):
        generator = prepend_user_context([{"role": "user", "content": render(policy)}], None,
                                         provider="openrouter", model_id="openai/gpt-6-luna")
        assert generator[0]["content"][0]["text"].endswith(PROMPT[2:END])
        for phase in ("preflight", "auto_qa", "qa"):
            prompt = CacheablePrompt("own source. " + phase, (12,))
            internal = prepend_user_context([{"role": "user", "content": prompt}], None,
                                            provider="claude", model_id="claude-haiku-4-5")
            assert internal[0]["content"][0]["text"] == "own source. "
            assert "cache_control" in internal[0]["content"][0]
            automatic = prepend_user_context([{"role": "user", "content": prompt}], None,
                                             provider="openrouter", model_id="moonshotai/kimi-k3")
            assert automatic[0]["content"] == prompt
        native = prepend_gemini_context([], "shared", model_id="gemini-2.5-pro")
        assert native == [{"role": "user", "parts": [{"text": "shared"}]}]
        assert consume_prompt_cache_report("gemini", "gemini-2.5-pro")["effective"] == "provider_default"


def test_adaptive_shared_context_and_four_caller_endpoints_respect_limit():
    policy = PromptCachePolicy(mode="adaptive", breakpoints=[4, 6, 8, 10])
    with prompt_cache_scope(policy):
        result = prepend_user_context([{"role": "user", "content": render(policy)}], "shared",
                                      provider="openai", model_id="gpt-6-luna")
        assert result[0]["content"] == "shared"
        assert sum("prompt_cache_breakpoint" in p for p in result[1]["content"]) == 4
        report = consume_prompt_cache_report("openai", "gpt-6-luna")
        assert report["adaptations"] == ["shared_context_boundary_omitted"]


def test_openrouter_gemini_uses_last_boundary_without_mutating_inputs():
    policy = PromptCachePolicy(mode="adaptive", breakpoints=[4, 6, 8, 10])
    original = [{"role": "user", "content": render(policy)}]
    before = copy.deepcopy(original)
    with prompt_cache_scope(policy):
        result = prepend_user_context(original, "shared", provider="openrouter", model_id="google/gemini-2.5-pro")
        parts = result[1]["content"]
        assert sum("cache_control" in p for p in parts) == 1
        assert "cache_control" in parts[-2] and "cache_control" not in parts[-1]
        assert flattened(result)[1:] == flattened(original)
        assert original == before
        assert with_openrouter_cache_options({}, "google/gemini-2.5-pro") == {}
        report = consume_prompt_cache_report("openrouter", "google/gemini-2.5-pro")
        assert report["effective"] == "prefix"
        assert report["reason"] == "implicit_cache_remains_provider_managed"
        assert "last_breakpoint_only" in report["adaptations"]


@pytest.mark.asyncio
async def test_concurrent_policies_and_routes_do_not_share_decisions():
    async def dispatch(mode, model):
        with prompt_cache_scope(PromptCachePolicy(mode=mode)):
            prepend_user_context([], None, provider="openrouter", model_id=model)
            await asyncio.sleep(0)
            return with_openrouter_cache_options({}, model), consume_prompt_cache_report("openrouter", model)

    adaptive, legacy, off = await asyncio.gather(
        dispatch("adaptive", "moonshotai/kimi-k3"), dispatch("auto", "openai/gpt-6-luna"),
        dispatch("off", "openai/gpt-6-luna"),
    )
    assert adaptive[0] == {} and adaptive[1]["effective"] == "provider_default"
    assert legacy == ({}, None)
    assert off[0]["prompt_cache_options"]["mode"] == "explicit" and off[1]["effective"] == "off"
    assert consume_prompt_cache_report("openrouter", "openai/gpt-6-luna") is None


def test_adaptive_identity_distinguishes_intent_and_endpoints():
    original = ContentRequest(prompt=PROMPT)
    adaptive = ContentRequest(prompt=PROMPT, prompt_cache={"mode": "adaptive", "breakpoints": [END]})
    other = adaptive.model_copy(update={"prompt_cache": PromptCachePolicy(mode="adaptive")})
    assert len({_generation_request_fingerprint(r) for r in (original, adaptive, other)}) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("provider,model", [("openai", "gpt-6-luna"), ("openrouter", "openai/gpt-6-luna"),
                                          ("openrouter", "moonshotai/kimi-k3")])
async def test_adaptive_actual_sdk_wire_and_usage(service, provider, model):
    sent, usage = [], []

    def wire(request):
        import json_utils as json
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "offline", "object": "chat.completion", "created": 1, "model": model,
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
        })

    sdk = openai.AsyncOpenAI(api_key="offline", base_url="https://offline.invalid/v1", max_retries=0,
                            http_client=httpx.AsyncClient(transport=httpx.MockTransport(wire)))
    setattr(service, f"{provider}_client", sdk)
    service._openrouter_client_for_attempt = Mock(return_value=sdk)
    service._is_openai_responses_api_model = Mock(return_value=False)
    service._emit_usage = AIService._emit_usage.__get__(service)
    policy = PromptCachePolicy(mode="adaptive", breakpoints=[END])
    try:
        with prompt_cache_scope(policy):
            kwargs = {"provider_options": {"allow_fallbacks": False}} if provider == "openrouter" else {}
            _, raw_usage = await getattr(service, f"_generate_{provider}")(
                render(policy), model, 0.2, 32, "system", shared_context="shared",
                **kwargs,
            )
            service._emit_usage(usage.append, model, provider, raw_usage)
    finally:
        await sdk.close()
    assert len(sent) == 1
    body = sent[0]
    assert body["model"] == model
    assert flattened(body["messages"]) == [("system", ["system"]), ("user", ["shared"]), ("user", [str(render(policy))])]
    if provider == "openrouter":
        assert body["provider"] == {"allow_fallbacks": False}
    if "openai" == provider or model.startswith("openai/"):
        assert body["prompt_cache_options"] == {"mode": "explicit", "ttl": "30m"}
    else:
        assert "prompt_cache_options" not in body
    assert usage[-1]["prompt_cache"]["requested"] == "adaptive"


@pytest.mark.asyncio
async def test_provider_failure_does_not_retry_with_another_policy(service):
    create = AsyncMock(side_effect=RuntimeError("transport failed"))
    service.openrouter_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    service._openrouter_client_for_attempt = Mock(return_value=service.openrouter_client)
    policy = PromptCachePolicy(mode="adaptive", breakpoints=[END])
    with prompt_cache_scope(policy), pytest.raises(RuntimeError, match="transport failed"):
        await service._generate_openrouter(render(policy), "openai/gpt-6-luna", 0.2, 32, "system")
    assert create.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming,responses", [(False, True), (True, False), (True, True)])
async def test_adaptive_native_openai_responses_and_streaming(service, streaming, responses):
    response = SimpleNamespace(output_text="ok", usage=None,
                               choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])
    create = AsyncMock(return_value=(FakeResponsesStream() if responses else FakeStream())
                       if streaming else response)
    service.openai_client = SimpleNamespace(responses=SimpleNamespace(create=create),
                                           chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    service._is_openai_responses_api_model = Mock(return_value=responses)
    policy = PromptCachePolicy(mode="adaptive", breakpoints=[END])
    kwargs = dict(prompt=render(policy), model_id="gpt-6-luna", temperature=0.2,
                  max_tokens=32, system_prompt="system")
    with prompt_cache_scope(policy):
        if streaming:
            _ = [chunk async for chunk in service._stream_openai(**kwargs, extra_verbose=False)]
        else:
            await service._generate_openai(**kwargs)
    params = create.await_args.kwargs
    assert create.await_count == 1
    assert params["extra_body"]["prompt_cache_options"] == {"mode": "explicit", "ttl": "30m"}
    transcript = params["input" if responses else "messages"]
    assert flattened(transcript)[-1] == ("user", [str(render(policy))])
    assert sum("prompt_cache_breakpoint" in p for p in transcript[-1]["content"]) == 1


@pytest.mark.asyncio
async def test_adaptive_openrouter_streaming_preserves_routing_and_text(service):
    create = AsyncMock(return_value=FakeStream())
    service.openrouter_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    service._openrouter_client_for_attempt = Mock(return_value=service.openrouter_client)
    policy = PromptCachePolicy(mode="adaptive", breakpoints=[END])
    with prompt_cache_scope(policy):
        _ = [chunk async for chunk in service._stream_openrouter(
            render(policy), "openai/gpt-6-luna", 0.2, 32, "system",
            provider_options={"allow_fallbacks": False},
        )]
    params = create.await_args.kwargs
    assert create.await_count == 1
    assert params["extra_body"]["provider"] == {"allow_fallbacks": False}
    assert params["extra_body"]["prompt_cache_options"] == {"mode": "explicit", "ttl": "30m"}
    assert flattened(params["messages"])[-1] == ("user", [str(render(policy))])
