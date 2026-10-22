"""Offline regression tests for fail-closed OpenRouter provider routing."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import ai_service as ai_service_module
from ai_runtime.openrouter_provider_options import (
    OpenRouterProviderOptionsError,
    validate_openrouter_provider_options,
)
from ai_service import AIService
from llm_routing import LLMRoutingError, resolve_call
from tool_loop_models import OutputContract

ZERO_COST_OPTIONS = {
    "max_price": {"prompt": 0, "completion": 0},
    "allow_fallbacks": False,
}


@pytest.fixture
def service():
    """Build an AIService without any provider SDK initialization."""

    with patch("ai_service.aiohttp.TCPConnector"), patch(
        "ai_service.aiohttp.ClientTimeout"
    ), patch.object(AIService, "_initialize_clients"):
        instance = AIService()
    instance.openrouter_client = None
    return instance


def test_provider_options_reject_unknown_and_non_zero_cost_shape() -> None:
    with pytest.raises(OpenRouterProviderOptionsError, match="Unsupported OpenRouter"):
        validate_openrouter_provider_options({"order": ["untrusted"]})
    with pytest.raises(OpenRouterProviderOptionsError, match="finite non-negative"):
        validate_openrouter_provider_options({"max_price": {"prompt": -1}})
    with pytest.raises(OpenRouterProviderOptionsError, match="boolean"):
        validate_openrouter_provider_options({"allow_fallbacks": 0})


def test_routing_exposes_validated_zero_cost_options(monkeypatch: pytest.MonkeyPatch) -> None:
    import config as config_module

    monkeypatch.setattr(
        config_module.Config,
        "get_model_info",
        lambda _self, _model, *, require_api_key=True: {
            "model_id": _model, "provider": "openrouter", "capabilities": ["text"],
        },
    )
    route = resolve_call(
        "generation.main",
        routing={
            "calls": {
                "generation.main": {
                    "model": "openrouter/owl-alpha",
                    "provider_options": ZERO_COST_OPTIONS,
                }
            }
        },
    )

    assert route.provider_options == {
        "max_price": {"prompt": 0.0, "completion": 0.0},
        "allow_fallbacks": False,
    }

    with pytest.raises(LLMRoutingError, match="Unsupported OpenRouter"):
        resolve_call(
            "generation.main",
            routing={
                "calls": {
                    "generation.main": {
                        "model": "openrouter/owl-alpha",
                        "provider_options": {"only": ["paid-provider"]},
                    }
                }
            },
        )


@pytest.mark.asyncio
async def test_normal_openrouter_request_merges_zero_cost_and_schema_safeguards(service) -> None:
    create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
            usage=None,
        )
    )
    service.openrouter_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    await service._generate_openrouter(
        prompt="hello",
        model_id="openrouter/owl-alpha",
        temperature=0.2,
        max_tokens=32,
        system_prompt="system",
        json_output=True,
        json_schema={"type": "object"},
        provider_options=ZERO_COST_OPTIONS,
    )

    extra_body = create.await_args.kwargs["extra_body"]
    assert extra_body["provider"] == {
        "max_price": {"prompt": 0.0, "completion": 0.0},
        "allow_fallbacks": False,
        "require_parameters": True,
    }
    assert extra_body["plugins"] == [{"id": "context-compression", "enabled": False}]


@pytest.mark.asyncio
async def test_stream_openrouter_request_merges_zero_cost_options(service) -> None:
    class FakeStream:
        response = SimpleNamespace(usage=None)

        async def close(self):
            pass

        def __aiter__(self):
            async def iterator():
                yield SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content="ok"), finish_reason="stop")],
                    usage=None,
                )

            return iterator()

    create = AsyncMock(return_value=FakeStream())
    service.openrouter_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    chunks = [
        chunk
        async for chunk in service._stream_openrouter(
            prompt="hello",
            model_id="openrouter/owl-alpha",
            temperature=0.2,
            max_tokens=32,
            system_prompt="system",
            provider_options=ZERO_COST_OPTIONS,
        )
    ]

    assert chunks[0] == "ok"
    extra_body = create.await_args.kwargs["extra_body"]
    assert extra_body["provider"]["max_price"] == {"prompt": 0.0, "completion": 0.0}
    assert extra_body["provider"]["allow_fallbacks"] is False
    assert extra_body["plugins"] == [{"id": "context-compression", "enabled": False}]


@pytest.mark.parametrize("reasoning_effort", [None, "none", "low", "medium", "high"])
def test_tool_loop_params_merge_zero_cost_and_require_parameters(service, reasoning_effort) -> None:
    params = service._build_openai_compatible_tool_params(
        provider="openrouter",
        model_id="openrouter/owl-alpha",
        current_messages=[{"role": "user", "content": "hello"}],
        temperature=0.2,
        max_tokens=32,
        reasoning_effort=reasoning_effort,
        json_output=True,
        json_schema={"type": "object"},
        tools_enabled=True,
        provider_options=ZERO_COST_OPTIONS,
    )

    assert params["extra_body"]["provider"] == {
        "max_price": {"prompt": 0.0, "completion": 0.0},
        "allow_fallbacks": False,
        "require_parameters": True,
    }
    assert params["extra_body"]["plugins"] == [
        {"id": "context-compression", "enabled": False}
    ]
    assert "reasoning_effort" not in params
    assert params["max_tokens"] == 32
    if reasoning_effort is None:
        assert "reasoning" not in params["extra_body"]
    else:
        assert params["extra_body"]["reasoning"] == {"effort": reasoning_effort}


def _configure_public_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    service: AIService,
    *,
    provider: str,
    reasoning_effort: str | None = None,
) -> None:
    """Make public dispatch tests independent from credentials and catalog state."""

    token_validation = {
        "adjusted_tokens": 32,
        "adjusted_reasoning_effort": reasoning_effort,
        "adjusted_thinking_budget_tokens": None,
        "thinking_validation": {},
        "was_adjusted": False,
        "model_limit": 32,
        "model_info": {
            "provider": provider,
            "model_id": f"{provider}/test-model",
            "capabilities": ["text"],
        },
    }
    monkeypatch.setattr(
        service,
        "_resolve_effective_output_max_tokens",
        lambda *_args, **_kwargs: (32, "test"),
    )
    monkeypatch.setattr(
        ai_service_module,
        "config",
        SimpleNamespace(
            model_specs={},
            MAX_RETRIES=1,
            REQUEST_TIMEOUT=1,
            REQUEST_TIMEOUTS={},
            RETRY_DELAY=0.0,
            validate_token_limits=lambda *_args, **_kwargs: token_validation,
        ),
    )
    monkeypatch.setattr(
        service,
        "_plan_output_contract",
        lambda *_args, **_kwargs: SimpleNamespace(schema=None, attempted_feature=None),
    )
    monkeypatch.setattr(
        service,
        "_apply_temperature_policies",
        lambda _info, temperature, thinking: (temperature, thinking, False),
    )
    monkeypatch.setattr(service, "_assert_model_blind_prompt", lambda **_kwargs: None)
    monkeypatch.setattr(service, "_admit_chat_call", AsyncMock())
    monkeypatch.setattr(service, "_record_provider_health_success", AsyncMock())


@pytest.mark.asyncio
async def test_generate_content_dispatches_options_only_to_openrouter(
    service: AIService, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_public_dispatch(monkeypatch, service, provider="openrouter")
    captured: dict[str, object] = {}

    async def fake_openrouter(*_args, provider_options, **_kwargs):
        captured["provider_options"] = provider_options
        return "ok", {}

    async def run_once(operation, **_kwargs):
        return await operation()

    monkeypatch.setattr(service, "_generate_openrouter", fake_openrouter)
    monkeypatch.setattr(service, "_execute_with_retries", run_once)

    assert await service.generate_content(
        "A sufficiently long dispatch prompt.",
        "openrouter/test-model",
        system_prompt="system",
        provider_options=ZERO_COST_OPTIONS,
    ) == "ok"
    assert captured["provider_options"] == ZERO_COST_OPTIONS


def test_strict_openrouter_attempt_disables_sdk_retries(service: AIService) -> None:
    client = MagicMock()
    strict_client = MagicMock()
    client.with_options.return_value = strict_client
    service.openrouter_client = client

    assert service._openrouter_client_for_attempt(True) is strict_client
    client.with_options.assert_called_once_with(max_retries=0)
    assert service._openrouter_client_for_attempt(False) is client


@pytest.mark.asyncio
async def test_strict_openrouter_tool_loop_uses_one_outer_attempt_and_sdk_client(
    service: AIService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Strict canary tool loops must not hit the retry branch or SDK defaults."""

    client = MagicMock()
    strict_client = MagicMock()
    client.with_options.return_value = strict_client
    service.openrouter_client = client
    monkeypatch.setattr(
        service, "_openrouter_tool_streaming_supported", lambda _model_id: True
    )

    async def unexpected_retries(*_args, **_kwargs):
        raise AssertionError("strict tool loop must not use outer retries")

    async def one_attempt(_operation, **_kwargs):
        return "ok", {"accepted": "assistant_final"}

    monkeypatch.setattr(service, "_execute_with_retries", unexpected_retries)
    monkeypatch.setattr(service, "_execute_without_retries", one_attempt)

    content, metadata = await service._run_openai_compatible_validation_tool_loop(
        provider="openrouter",
        model_id="openrouter/test-model",
        prompt="hello",
        validation_callback=lambda _text: SimpleNamespace(approved=True),
        temperature=0.2,
        max_tokens=32,
        system_prompt="system",
        request_timeout=None,
        reasoning_effort=None,
        json_output=False,
        json_schema=None,
        usage_callback=None,
        usage_extra=None,
        images=None,
        max_rounds=1,
        strict_single_generation_attempt=True,
    )

    assert (content, metadata) == ("ok", {"accepted": "assistant_final"})
    client.with_options.assert_called_once_with(max_retries=0)


@pytest.mark.asyncio
async def test_stream_strict_attempt_reaches_openrouter_helper(
    service: AIService, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_public_dispatch(monkeypatch, service, provider="openrouter")
    captured: dict[str, object] = {}

    async def fake_openrouter(*_args, strict_single_generation_attempt, **_kwargs):
        captured["strict"] = strict_single_generation_attempt
        yield "ok"

    monkeypatch.setattr(service, "_stream_openrouter", fake_openrouter)
    chunks = [
        chunk
        async for chunk in service.generate_content_stream(
            "A sufficiently long dispatch prompt.",
            "openrouter/test-model",
            system_prompt="system",
            strict_single_generation_attempt=True,
        )
    ]

    assert chunks == ["ok"]
    assert captured["strict"] is True


@pytest.mark.asyncio
async def test_text_only_model_rejects_images_before_normal_dispatch(
    service: AIService, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_public_dispatch(monkeypatch, service, provider="llamacpp")

    async def unexpected_dispatch(*_args, **_kwargs):
        raise AssertionError("text-only model must fail before provider dispatch")

    monkeypatch.setattr(service, "_generate_registered_local", unexpected_dispatch)
    with pytest.raises(ValueError, match="does not support vision"):
        await service.generate_content(
            "A sufficiently long dispatch prompt.",
            "llamacpp/test-model",
            system_prompt="system",
            images=[SimpleNamespace()],
        )


@pytest.mark.asyncio
async def test_text_only_model_rejects_images_before_stream_dispatch(
    service: AIService, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_public_dispatch(monkeypatch, service, provider="llamacpp")

    async def unexpected_dispatch(*_args, **_kwargs):
        raise AssertionError("text-only model must fail before provider dispatch")
        yield "unreachable"

    monkeypatch.setattr(service, "_stream_registered_local", unexpected_dispatch)
    with pytest.raises(ValueError, match="does not support vision"):
        async for _chunk in service.generate_content_stream(
            "A sufficiently long dispatch prompt.",
            "llamacpp/test-model",
            system_prompt="system",
            images=[SimpleNamespace()],
        ):
            pass


@pytest.mark.asyncio
async def test_generate_content_does_not_pass_openrouter_options_to_openai(
    service: AIService, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_public_dispatch(monkeypatch, service, provider="openai")

    async def fake_openai(
        _prompt,
        _model_id,
        _temperature,
        _max_tokens,
        _system_prompt,
        _reasoning_effort,
        *,
        extra_verbose,
        request_timeout,
        json_output,
        json_schema,
        images,
    ):
        del extra_verbose, request_timeout, json_output, json_schema, images
        return "ok", {}

    async def run_once(operation, **_kwargs):
        return await operation()

    monkeypatch.setattr(service, "_generate_openai", fake_openai)
    monkeypatch.setattr(service, "_execute_with_retries", run_once)

    assert await service.generate_content(
        "A sufficiently long dispatch prompt.",
        "openai/test-model",
        system_prompt="system",
    ) == "ok"


@pytest.mark.asyncio
async def test_stream_dispatches_options_only_to_openrouter(
    service: AIService, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_public_dispatch(monkeypatch, service, provider="openrouter")
    captured: dict[str, object] = {}

    async def fake_openrouter(*_args, provider_options, **_kwargs):
        captured["provider_options"] = provider_options
        yield "ok"

    monkeypatch.setattr(service, "_stream_openrouter", fake_openrouter)

    chunks = [
        chunk
        async for chunk in service.generate_content_stream(
            "A sufficiently long dispatch prompt.",
            "openrouter/test-model",
            system_prompt="system",
            provider_options=ZERO_COST_OPTIONS,
        )
    ]

    assert chunks == ["ok"]
    assert captured["provider_options"] == ZERO_COST_OPTIONS


@pytest.mark.asyncio
async def test_stream_does_not_pass_openrouter_options_to_xai(
    service: AIService, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_public_dispatch(monkeypatch, service, provider="xai")

    async def fake_xai(
        _prompt,
        _model_id,
        _temperature,
        _max_tokens,
        _system_prompt,
        *,
        reasoning_effort,
        request_timeout,
        json_output,
        json_schema,
        usage_callback,
        usage_extra,
        images,
    ):
        del (
            reasoning_effort,
            request_timeout,
            json_output,
            json_schema,
            usage_callback,
            usage_extra,
            images,
        )
        yield "ok"

    monkeypatch.setattr(service, "_stream_xai", fake_xai)

    chunks = [
        chunk
        async for chunk in service.generate_content_stream(
            "A sufficiently long dispatch prompt.",
            "xai/test-model",
            system_prompt="system",
        )
    ]

    assert chunks == ["ok"]



@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("include_context", [None, True, False])
@pytest.mark.parametrize("effective_effort", ["none", "low"])
async def test_public_openrouter_preserves_system_context_and_effective_reasoning(
    service: AIService,
    monkeypatch: pytest.MonkeyPatch,
    streaming: bool,
    include_context: bool | None,
    effective_effort: str,
) -> None:
    """Capture the actual SDK payload, without credentials or network calls."""
    _configure_public_dispatch(
        monkeypatch, service, provider="openrouter", reasoning_effort=effective_effort
    )

    class FakeStream:
        response = SimpleNamespace(usage=None)

        async def close(self):
            pass

        def __aiter__(self):
            async def iterator():
                yield SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content="ok"), finish_reason="stop")],
                    usage=None,
                )
            return iterator()

    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"), finish_reason="stop")],
        usage=None,
    )
    create = AsyncMock(return_value=FakeStream() if streaming else response)
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    client.with_options = MagicMock(return_value=client)
    service.openrouter_client = client
    system_prompt = "Fixed temporal instructions.\nCurrent date: 1999/12/31"
    prompt = "Treat include_default_system_context=False as quoted source data."
    kwargs = {
        "prompt": prompt,
        "model": "openrouter/test-model",
        "system_prompt": system_prompt,
        "reasoning_effort": "high",
        "strict_single_generation_attempt": True,
        "provider_options": {**ZERO_COST_OPTIONS, "require_parameters": True},
    }
    if include_context is not None:
        kwargs["include_default_system_context"] = include_context
    if streaming:
        assert "".join([str(chunk) async for chunk in service.generate_content_stream(**kwargs)]) == "ok"
    else:
        assert await service.generate_content(**kwargs) == "ok"

    create.assert_awaited_once()
    client.with_options.assert_called_once_with(max_retries=0)
    payload = create.await_args.kwargs
    assert payload["messages"][1] == {"role": "user", "content": prompt}
    dispatched_system = payload["messages"][0]["content"]
    if include_context is False:
        assert dispatched_system == system_prompt
    else:
        assert dispatched_system == (
            system_prompt
            + "\n\nIMPORTANT: Always respond in the same language as the user's request, "
            + "regardless of the language used in these instructions."
            + datetime.now().strftime("\n\nCurrent date: %Y/%m/%d")
        )
    assert payload["extra_body"]["reasoning"] == {"effort": effective_effort}
    assert "reasoning_effort" not in payload
    assert payload["max_tokens"] == 32
    assert payload["extra_body"]["provider"] == {
        "max_price": {"prompt": 0.0, "completion": 0.0},
        "allow_fallbacks": False,
        "require_parameters": True,
    }
    assert payload["extra_body"]["plugins"] == [{"id": "context-compression", "enabled": False}]


@pytest.mark.asyncio
@pytest.mark.parametrize("include_context", [True, False])
@pytest.mark.parametrize("json_output", [False, True])
async def test_tool_system_context_flag_preserves_explicit_scaffolding(
    service: AIService,
    monkeypatch: pytest.MonkeyPatch,
    include_context: bool,
    json_output: bool,
) -> None:
    _configure_public_dispatch(monkeypatch, service, provider="openrouter")
    monkeypatch.setattr(service, "_supports_generation_validation_tool_loop", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(service, "_serialize_tool_loop_tool_defs", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(service, "_build_chat_call_budget", lambda **_kwargs: object())
    monkeypatch.setattr(
        ai_service_module.context_budget,
        "evaluate_call_admission",
        AsyncMock(return_value=SimpleNamespace(
            decision=ai_service_module.context_budget.AdmissionDecision.FITS,
            capacity_unverified=False,
        )),
    )
    loop = AsyncMock(return_value=('{"ok": true}', {
        "mode": "openrouter_tool_loop", "turns": 1,
        "accepted": "assistant_final", "trace": [],
    }))
    monkeypatch.setattr(service, "_run_openai_compatible_validation_tool_loop", loop)
    content, envelope = await service.call_ai_with_validation_tools(
        prompt="Generate the declared response.", model="openrouter/test-model",
        system_prompt="fixed system", include_default_system_context=include_context,
        validation_callback=lambda _text: SimpleNamespace(approved=True),
        output_contract=OutputContract.JSON_LOOSE if json_output else OutputContract.FREE_TEXT,
        extra_system_instructions="Preserve the declared tool constraints.",
        accent_audit_timeout=1,
    )
    assert content == '{"ok": true}'
    assert envelope.accepted is True
    dispatched_system = loop.await_args.kwargs["system_prompt"]
    assert dispatched_system.startswith("fixed system")
    assert ("IMPORTANT: Always respond in the same language" in dispatched_system) is include_context
    assert ("Current date:" in dispatched_system) is include_context
    assert ("IMPORTANT: Output valid JSON only." in dispatched_system) is json_output
    assert dispatched_system.endswith("\n\nPreserve the declared tool constraints.")
    if not include_context and not json_output:
        assert dispatched_system == "fixed system\n\nPreserve the declared tool constraints."
