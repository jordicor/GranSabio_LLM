"""Request cache-policy regressions. All provider clients are mocked; no paid calls."""

import asyncio
import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import openai
import pytest
from fastapi import HTTPException
from pydantic import ValidationError

import core.generation_routes as routes
import json_utils as json
from client import AsyncGranSabioClient, GranSabioClient
from core.generation_processor import _build_generation_prompt
from dev_tests.test_internal_cache_transport import service as service
from dev_tests.test_shared_context_transport import FakeStream
from models import ContentRequest
from shared_context import (
    CacheablePrompt,
    PromptCachePolicy,
    PromptCachePolicyError,
    prepend_user_context,
    prompt_cache_scope,
    with_openai_cache_options,
)

PROMPT = "  Source: caf\u00e9 \U0001f680.\nQuestion about source.  "
END = len("  Source: caf\u00e9 \U0001f680.\n")


def render(policy=None, *, suffix="Feedback", context="Attachment reference"):
    return _build_generation_prompt(
        context, [PROMPT, "Fixed policy"], [suffix],
        cache_retry_prefix=True, prompt_cache=policy,
    )


def flattened(messages):
    """Compare roles, text and non-text ordering independent of text-block splits."""
    result = []
    for message in messages:
        content = message["content"]
        parts = [{"type": "text", "text": content}] if isinstance(content, str) else content
        combined = []
        for part in parts:
            if part.get("type") in {"text", "input_text"}:
                if combined and isinstance(combined[-1], str):
                    combined[-1] += part["text"]
                else:
                    combined.append(str(part["text"]))
            else:
                combined.append(part)
        result.append((message["role"], combined))
    return result


@pytest.mark.parametrize("policy", [
    {"mode": "unknown"}, {"mode": "off", "breakpoints": [2]},
    {"mode": "auto", "breakpoints": [2]}, {"mode": "explicit", "breakpoints": [True]},
    {"mode": "explicit", "breakpoints": [1.5]}, {"mode": "explicit", "breakpoints": ["2"]},
    {"mode": "explicit", "breakpoints": [0]}, {"mode": "explicit", "breakpoints": [4, 3]},
    {"mode": "explicit", "breakpoints": [3, 3]},
    {"mode": "explicit", "breakpoints": [3, 4, 5, 6, 7]},
    {"mode": "explicit", "breakpoints": [len(PROMPT) + 1]},
    {"mode": "explicit", "breakpoints": [1]}, {"mode": "off", "ttl": "12h"},
])
def test_invalid_policy_rejected_at_request_boundary(policy):
    with pytest.raises(ValidationError):
        ContentRequest(prompt=PROMPT, prompt_cache=policy)


def test_unicode_offsets_and_builder_preserve_existing_text_on_retries():
    policy = PromptCachePolicy(mode="explicit", breakpoints=[END, len(PROMPT)])
    first = render(policy)
    retry = render(policy, suffix="Different feedback")
    assert str(first) == str(render())
    assert first[:first.cache_breakpoints[0]] == "Attachment reference\n\n" + PROMPT[2:END]
    assert first.cache_breakpoints == retry.cache_breakpoints
    assert first[:first.cache_breakpoints[-1]] == retry[:retry.cache_breakpoints[-1]]
    assert "Fixed policy" not in first[:first.cache_breakpoints[-1]]
    assert copy.deepcopy(first).cache_breakpoints == first.cache_breakpoints
    assert type(render(PromptCachePolicy(mode="off"))) is str


@pytest.mark.parametrize("provider,model", [
    ("openai", "gpt-6-luna"), ("claude", "claude-haiku-5-5"),
    ("openrouter", "openai/gpt-6-luna"), ("openrouter", "anthropic/claude-haiku-5.5"),
])
@pytest.mark.parametrize("mode", ["auto", "off", "explicit"])
def test_policy_preserves_all_roles_text_images_and_order(provider, model, mode):
    policy = PromptCachePolicy(mode=mode, breakpoints=[END] if mode == "explicit" else [])
    picture = {"type": "image_url", "image_url": {"url": "image"}}

    def messages(prompt):
        return [{"role": "system", "content": "system"},
                {"role": "user", "content": [picture, {"type": "text", "text": prompt}, picture]}]

    baseline = prepend_user_context(messages(render()), "shared", provider=provider, model_id=model)
    with prompt_cache_scope(policy):
        result = prepend_user_context(messages(copy.deepcopy(render(policy))), "shared",
                                      provider=provider, model_id=model)
    assert flattened(result) == flattened(baseline)
    markers = [p for m in result if isinstance(m["content"], list) for p in m["content"]
               if "cache_control" in p or "prompt_cache_breakpoint" in p]
    assert len(markers) == {"auto": 3, "off": 0, "explicit": 1}[mode]


def test_explicit_ignores_internal_phase_and_shared_context_markers():
    with prompt_cache_scope(PromptCachePolicy(mode="explicit", breakpoints=[END])):
        result = prepend_user_context(
            [{"role": "user", "content": CacheablePrompt("preflight source and task", (16,))}],
            "shared evidence", provider="openai", model_id="gpt-6-luna",
        )
        params = with_openai_cache_options({"model": "gpt-6-luna", "messages": result})
    assert [m["content"] for m in result] == ["shared evidence", "preflight source and task"]
    assert params["extra_body"]["prompt_cache_options"]["mode"] == "explicit"


def test_four_explicit_endpoints_do_not_add_automatic_shared_boundary():
    policy = PromptCachePolicy(mode="explicit", breakpoints=[4, 6, 8, 10])
    with prompt_cache_scope(policy):
        result = prepend_user_context([{"role": "user", "content": render(policy)}],
                                      "shared", provider="openai", model_id="gpt-6-luna")
    assert isinstance(result[0]["content"], str)
    assert sum("prompt_cache_breakpoint" in p for p in result[1]["content"]) == 4


@pytest.mark.parametrize("provider,model", [("gemini", "gemini-2.5-pro"),
                                           ("openai", "gpt-4o"), ("openrouter", "openrouter/auto")])
def test_unsupported_route_does_not_silently_ignore_policy(provider, model):
    with prompt_cache_scope(PromptCachePolicy(mode="off")), pytest.raises(PromptCachePolicyError):
        prepend_user_context([{"role": "user", "content": PROMPT}], None, provider=provider, model_id=model)


@pytest.mark.asyncio
async def test_request_scope_survives_background_tasks_without_leaking(monkeypatch):
    release = asyncio.Event()
    pending = []

    def options():
        return with_openai_cache_options({"input": "plain input"}).get("extra_body")

    async def admission(request):
        initial = options()

        async def background():
            await release.wait()
            return options()

        pending.append(asyncio.create_task(background()))
        await asyncio.sleep(0)
        return initial

    monkeypatch.setattr(routes, "_generate_content_once_with_cache_policy", admission)
    off, auto = await asyncio.gather(
        routes._generate_content_once(ContentRequest(prompt=PROMPT, prompt_cache={"mode": "off"})),
        routes._generate_content_once(ContentRequest(prompt=PROMPT)),
    )
    assert off == {"prompt_cache_options": {"mode": "explicit", "ttl": "30m"}}
    assert auto is None and options() is None
    release.set()
    assert await asyncio.gather(*pending) == [off, auto]
    with pytest.raises(RuntimeError), prompt_cache_scope(PromptCachePolicy(mode="off")):
        raise RuntimeError("cancelled request")
    assert options() is None


def test_request_identity_keeps_legacy_auto_but_distinguishes_policy():
    original = ContentRequest(prompt=PROMPT)
    auto = ContentRequest(prompt=PROMPT, prompt_cache={"mode": "auto"})
    off = ContentRequest(prompt=PROMPT, prompt_cache={"mode": "off"})
    assert routes._generation_request_fingerprint(original) == routes._generation_request_fingerprint(auto)
    assert routes._generation_request_fingerprint(original) != routes._generation_request_fingerprint(off)


@pytest.mark.asyncio
async def test_unsupported_generator_rejected_before_input_resolution(monkeypatch):
    monkeypatch.setattr(routes.config, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(routes, "_apply_request_model_routing", Mock())
    resolve_inputs = AsyncMock(side_effect=AssertionError("Admission must stop before resolving inputs"))
    monkeypatch.setattr(routes, "resolve_generation_inputs", resolve_inputs)
    request = ContentRequest(prompt=PROMPT, generator_model="gpt-4o", max_tokens=32,
                             gran_sabio_fallback=False, long_text_mode="off", prompt_cache={"mode": "off"})
    with pytest.raises(HTTPException) as error:
        await routes._generate_content_once(request)
    assert error.value.status_code == 400
    assert "prompt_cache" in error.value.detail and "gpt-4o" in error.value.detail
    resolve_inputs.assert_not_called()


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("responses", [False, True])
@pytest.mark.parametrize("mode", ["off", "explicit"])
@pytest.mark.asyncio
async def test_native_openai_wire_disables_implicit_suffix(service, streaming, responses, mode):
    response = SimpleNamespace(output_text="ok", usage=None,
                               choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])
    create = AsyncMock(return_value=FakeStream() if streaming and not responses else response)
    service.openai_client = SimpleNamespace(responses=SimpleNamespace(create=create),
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    service._is_openai_responses_api_model = Mock(return_value=responses)
    policy = PromptCachePolicy(mode=mode, breakpoints=[END] if mode == "explicit" else [])
    kwargs = dict(prompt=render(policy), model_id="gpt-6-luna", temperature=0.2,
                  max_tokens=32, system_prompt="system")
    with prompt_cache_scope(policy):
        if streaming:
            _ = [chunk async for chunk in service._stream_openai(**kwargs, extra_verbose=False)]
        else:
            await service._generate_openai(**kwargs)
    params = create.await_args.kwargs
    assert params["model"] == "gpt-6-luna"
    assert params["extra_body"]["prompt_cache_options"] == {"mode": "explicit", "ttl": "30m"}
    transcript = params["input" if responses else "messages"]
    if isinstance(transcript, str):
        assert mode == "off" and transcript == render()
    else:
        assert flattened(transcript)[-1] == ("user", [str(render())])
        content = transcript[-1]["content"]
        marks = [p for p in content if "prompt_cache_breakpoint" in p] if isinstance(content, list) else []
        assert len(marks) == (1 if mode == "explicit" else 0)


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("mode", ["off", "explicit"])
@pytest.mark.asyncio
async def test_openrouter_openai_wire_disables_implicit_suffix(service, streaming, mode):
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))], usage=None)
    create = AsyncMock(return_value=FakeStream() if streaming else response)
    service.openrouter_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    service._openrouter_client_for_attempt = Mock(return_value=service.openrouter_client)
    policy = PromptCachePolicy(mode=mode, breakpoints=[END] if mode == "explicit" else [])
    kwargs = dict(prompt=render(policy), model_id="openai/gpt-6-luna", temperature=0.2,
                  max_tokens=32, system_prompt="system", provider_options={"allow_fallbacks": False})
    with prompt_cache_scope(policy):
        if streaming:
            _ = [chunk async for chunk in service._stream_openrouter(**kwargs)]
        else:
            await service._generate_openrouter(**kwargs)
    params = create.await_args.kwargs
    assert params["model"] == "openai/gpt-6-luna"
    assert params["extra_body"]["prompt_cache_options"] == {"mode": "explicit", "ttl": "30m"}
    assert params["extra_body"]["provider"] == {"allow_fallbacks": False}
    assert flattened(params["messages"]) == [("system", ["system"]), ("user", [str(render())])]
    content = params["messages"][-1]["content"]
    marks = [p for p in content if "prompt_cache_breakpoint" in p] if isinstance(content, list) else []
    assert len(marks) == (1 if mode == "explicit" else 0)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["openai", "openrouter"])
async def test_actual_sdk_http_json_preserves_model_text_roles_and_routing(service, provider):
    sent = []
    model = "openai/gpt-6-luna" if provider == "openrouter" else "gpt-6-luna"

    def wire(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "offline", "object": "chat.completion", "created": 1, "model": model,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "ok"}}],
        })

    sdk = openai.AsyncOpenAI(api_key="offline", base_url="https://offline.invalid/v1",
                            http_client=httpx.AsyncClient(transport=httpx.MockTransport(wire)))
    setattr(service, f"{provider}_client", sdk)
    service._openrouter_client_for_attempt = Mock(return_value=sdk)
    service._is_openai_responses_api_model = Mock(return_value=False)
    policies = [None, PromptCachePolicy(), PromptCachePolicy(mode="off"),
                PromptCachePolicy(mode="explicit", breakpoints=[END])]
    try:
        for policy in policies:
            with prompt_cache_scope(policy):
                kwargs = {"provider_options": {"allow_fallbacks": False}} if provider == "openrouter" else {}
                await getattr(service, f"_generate_{provider}")(
                    render(policy), model, 0.2, 32, "system", shared_context="existing context", **kwargs,
                )
    finally:
        await sdk.close()
    assert sent[0] == sent[1]  # Omitted and explicit auto have identical wire JSON.
    baseline = flattened(sent[0]["messages"])
    for body in sent:
        assert body["model"] == model
        assert flattened(body["messages"]) == baseline
        assert body["max_tokens"] == 32
        if provider == "openrouter":
            assert body["provider"] == {"allow_fallbacks": False}
    for body in sent[2:]:
        assert body["prompt_cache_options"] == {"mode": "explicit", "ttl": "30m"}
    assert "prompt_cache_breakpoint" not in json.dumps(sent[2])
    assert "cache_control" not in json.dumps(sent[2])
    assert json.dumps(sent[3]).count('"prompt_cache_breakpoint"') == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("policy", [None, {"mode": "off"}, {"mode": "explicit", "breakpoints": [END]},
                                    {"mode": "adaptive", "breakpoints": [END]}])
async def test_sdk_forwards_policy_to_generate_http_without_changing_prompt(asynchronous, streaming, policy):
    class AtHttpBoundary(Exception):
        pass

    client = (AsyncGranSabioClient if asynchronous else GranSabioClient)(base_url="http://offline.invalid")
    dispatch = AsyncMock(side_effect=AtHttpBoundary) if asynchronous else Mock(side_effect=AtHttpBoundary)
    client._request = dispatch
    method = client.generate_streaming if streaming else client.generate
    with pytest.raises(AtHttpBoundary):
        result = method(PROMPT, generator_model="openrouter::openai/gpt-6-luna", max_tokens=32,
                        prompt_cache=policy)
        if asynchronous:
            await result
    payload = dispatch.call_args.kwargs["json_data"]
    assert payload["prompt"] == PROMPT
    assert payload["generator_model"] == "openrouter::openai/gpt-6-luna"
    assert payload.get("prompt_cache") == policy
    assert ("prompt_cache" in payload) == (policy is not None)
