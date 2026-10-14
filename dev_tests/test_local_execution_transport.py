"""Focused offline coverage for the local-only AIService transport guard."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import context_budget
from ai_service import (
    AIRequestError,
    AIService,
    _local_openai_async_http_client,
)
from llm_routing import get_default_routing, merge_routing_documents, resolve_call
from local_endpoint_registry import LocalEndpointRegistry, LocalInferenceEndpoint
from local_execution_policy import (
    LocalExecutionManifest,
    LocalExecutionPolicyError,
    LocalExecutionRuntime,
    ManifestRoute,
    build_local_execution_attestation,
)
from models import ImageData
from provider_errors import ProviderErrorKind, ProviderFailure
from usage_tracking import UsageTracker


def _runtime(
    *,
    provider: str = "ollama",
    model_id: str = "writer-local",
    endpoint_id: str = "local_primary",
) -> LocalExecutionRuntime:
    manifest = LocalExecutionManifest(
        routes=(
            ManifestRoute(
                call_id="generation.main",
                route_ordinal=0,
                provider=provider,
                endpoint_id=endpoint_id,
                model_id=model_id,
            ),
        ),
        manifest_hash="manifest-test",
    )
    return LocalExecutionRuntime.create(session_id="session-test", manifest=manifest)


def _service(monkeypatch: pytest.MonkeyPatch, create: AsyncMock) -> AIService:
    service = AIService.__new__(AIService)
    endpoint = LocalInferenceEndpoint("local_primary", "ollama", "http://local:11434")
    service.local_endpoint_registry = LocalEndpointRegistry((endpoint,), "test", legacy=True)
    service._local_endpoints = {endpoint.endpoint_id: endpoint}
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    semaphore = asyncio.Semaphore(1)
    service._local_endpoint_clients = {endpoint.endpoint_id: client}
    service._local_endpoint_semaphores = {endpoint.endpoint_id: semaphore}
    service.ollama_client = client
    service.llamacpp_client = None
    service._ollama_request_semaphore = semaphore
    monkeypatch.setattr(service, "_resolve_effective_output_max_tokens", lambda *_args, **_kwargs: (32, {}))
    monkeypatch.setattr(
        service,
        "_plan_output_contract",
        lambda *_args, **_kwargs: SimpleNamespace(
            schema=None,
            attempted_feature=None,
            inject_json_instruction=False,
        ),
    )
    monkeypatch.setattr(service, "_apply_temperature_policies", lambda *_args: (0.7, None, False))
    monkeypatch.setattr(service, "_assert_model_blind_prompt", lambda **_kwargs: None)
    monkeypatch.setattr(service, "_admit_chat_call", AsyncMock())
    monkeypatch.setattr(service, "_record_provider_health_success", AsyncMock())
    monkeypatch.setattr(service, "_record_provider_health_failure", AsyncMock())
    monkeypatch.setattr(service, "_raise_if_generation_stopped_unusably", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(service, "_max_retry_attempts", lambda: 2)
    monkeypatch.setattr(service, "_calculate_retry_delay", lambda _attempt: 0.0)

    @asynccontextmanager
    async def _scope(*_args, **_kwargs):
        yield None

    monkeypatch.setattr(service, "_provider_call_scope", _scope)
    return service


def _install_config(
    monkeypatch: pytest.MonkeyPatch,
    *,
    provider: str = "ollama",
    model_id: str = "writer-local",
    capabilities: tuple[str, ...] = ("text",),
) -> None:
    monkeypatch.setattr(
        "ai_service.config",
        SimpleNamespace(
            GENERATOR_SYSTEM_PROMPT="system",
            GENERATOR_SYSTEM_PROMPT_RAW="raw system",
            REQUEST_TIMEOUT=30,
            model_specs={},
            validate_token_limits=lambda *_args, **_kwargs: {
                "adjusted_tokens": 32,
                "adjusted_reasoning_effort": None,
                "adjusted_thinking_budget_tokens": None,
                "thinking_validation": {},
                "was_adjusted": False,
                "model_limit": 32,
                "model_info": {
                    "provider": provider,
                    "model_id": model_id,
                    "capabilities": list(capabilities),
                },
            },
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["ollama", "llamacpp", "openai_compatible"])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("vision", [False, True])
async def test_registered_local_dispatch_preserves_images_and_vision_guard(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    streaming: bool,
    vision: bool,
) -> None:
    """Check admitted images at the SDK edge through both public dispatches."""

    class OfflineStream:
        def __aiter__(self):
            return self._iterate()

        async def _iterate(self):
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="image described"),
                        finish_reason="stop",
                    )
                ],
                usage=None,
            )

    response = SimpleNamespace(
        choices=[
            SimpleNamespace(message=SimpleNamespace(content="image described"), finish_reason="stop")
        ],
        usage=None,
    )
    create = AsyncMock(return_value=OfflineStream() if streaming else response)
    endpoint = LocalInferenceEndpoint("vision_local", provider, "http://local:8080/v1")
    service = _service(monkeypatch, AsyncMock())
    service.local_endpoint_registry = LocalEndpointRegistry((endpoint,), "test")
    service._local_endpoints = {endpoint.endpoint_id: endpoint}
    service._local_endpoint_clients = {
        endpoint.endpoint_id: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
    }
    service._local_endpoint_semaphores = {endpoint.endpoint_id: asyncio.Semaphore(1)}
    _install_config(
        monkeypatch,
        provider=provider,
        capabilities=("text", "vision") if vision else ("text",),
    )
    runtime = _runtime(provider=provider, endpoint_id=endpoint.endpoint_id)
    image = ImageData(
        base64_data="aW1hZ2U=",
        mime_type="image/png",
        original_filename="image.png",
        size_bytes=5,
        detail="low",
    )

    async def dispatch():
        kwargs = {
            "prompt": "Describe this image.",
            "model": "writer-local",
            "images": [image],
            "local_execution": runtime,
            "dispatch_call_id": "generation.main",
            "endpoint_id": endpoint.endpoint_id,
        }
        if streaming:
            return "".join([str(chunk) async for chunk in service.generate_content_stream(**kwargs)])
        return await service.generate_content(**kwargs)

    if not vision:
        with pytest.raises(ValueError, match="does not support vision input"):
            await dispatch()
        create.assert_not_awaited()
        assert runtime.ledger.entries == []
        return

    assert await dispatch() == "image described"
    create.assert_awaited_once()
    assert create.await_args.kwargs["messages"][-1] == {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,aW1hZ2U=", "detail": "low"},
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
    assert [(entry.allowed, entry.endpoint_id) for entry in runtime.ledger.entries] == [
        (True, endpoint.endpoint_id)
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["ollama", "llamacpp", "openai_compatible"])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("measured", [False, True])
async def test_registered_local_usage_reaches_tracker_without_inventing_counts(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    streaming: bool,
    measured: bool,
) -> None:
    """Request streaming usage and retain missing data through exposed totals."""
    from config import Config

    monkeypatch.setattr(Config, "get_model_info", lambda *_args, **_kwargs: {
        "pricing": {"input_per_million": 0, "output_per_million": 0}
    })
    sdk_usage = SimpleNamespace(prompt_tokens=9, completion_tokens=5, total_tokens=14) if measured else None

    class OfflineStream:
        # A wrapper with no response-level usage must retain the final chunk's
        # measurement, including when that chunk has no choices.
        response = SimpleNamespace(usage=None)

        def __aiter__(self):
            return self._iterate()

        async def _iterate(self):
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="ok"), finish_reason="stop")],
                usage=None,
            )
            if measured:
                yield SimpleNamespace(choices=[], usage=sdk_usage)

    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"), finish_reason="stop")],
        usage=sdk_usage,
    )
    create = AsyncMock(return_value=OfflineStream() if streaming else response)
    endpoint = LocalInferenceEndpoint("usage_local", provider, "http://local:8080/v1")
    service = _service(monkeypatch, AsyncMock())
    service.local_endpoint_registry = LocalEndpointRegistry((endpoint,), "test")
    service._local_endpoints = {endpoint.endpoint_id: endpoint}
    service._local_endpoint_clients = {
        endpoint.endpoint_id: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
    }
    service._local_endpoint_semaphores = {endpoint.endpoint_id: asyncio.Semaphore(1)}
    _install_config(monkeypatch, provider=provider)
    runtime = _runtime(provider=provider, endpoint_id=endpoint.endpoint_id)
    tracker = UsageTracker(detail_level=3)
    kwargs = {
        "prompt": "offline",
        "model": "writer-local",
        "local_execution": runtime,
        "dispatch_call_id": "generation.main",
        "endpoint_id": endpoint.endpoint_id,
        "usage_callback": tracker.create_callback(phase="generation", role="generator"),
    }
    if streaming:
        assert "".join([str(chunk) async for chunk in service.generate_content_stream(**kwargs)]) == "ok"
        assert create.await_args.kwargs["stream_options"] == {"include_usage": True}
    else:
        assert await service.generate_content(**kwargs) == "ok"
        assert "stream_options" not in create.await_args.kwargs

    stats = tracker.build_query_stats(3)
    costs = tracker.build_summary(2)
    assert len(stats["calls"]) == 1
    call = stats["calls"][0]
    assert call["endpoint_id"] == endpoint.endpoint_id
    assert call["finish_reason"] == "stop"
    expected = (9, 5, 14) if measured else (None, None, None)
    for totals in (call, stats["totals"], costs["grand_totals"]):
        assert tuple(totals[field] for field in ("input_tokens", "output_tokens", "total_tokens")) == expected
    assert call["estimated_cost_usd"] == (0.0 if measured else None)
    assert costs["grand_totals"]["cost"] == (0.0 if measured else None)


def _install_endpoint_discovery(
    monkeypatch: pytest.MonkeyPatch,
    discovered_by_endpoint: dict[str, list[dict[str, object]]],
) -> list[str]:
    """Install deterministic read-only endpoint discovery for transport tests."""

    calls: list[str] = []

    class FakeModelSyncService:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        async def discover_local_endpoint(
            self, endpoint: LocalInferenceEndpoint
        ) -> list[dict[str, object]]:
            calls.append(endpoint.endpoint_id)
            return list(discovered_by_endpoint.get(endpoint.endpoint_id, []))

    monkeypatch.setattr("services.model_sync.ModelSyncService", FakeModelSyncService)
    return calls


def _install_colliding_local_config(
    monkeypatch: pytest.MonkeyPatch,
) -> list[str]:
    """Provide two adapter namespaces for the same raw model ID."""

    validated_models: list[str] = []

    def validate_token_limits(model: str, *_args: object, **_kwargs: object) -> dict[str, object]:
        validated_models.append(model)
        provider, separator, model_id = model.partition("::")
        assert separator
        assert provider in {"ollama", "llamacpp"}
        assert model_id == "shared"
        return {
            "adjusted_tokens": 32,
            "adjusted_reasoning_effort": None,
            "adjusted_thinking_budget_tokens": None,
            "thinking_validation": {},
            "was_adjusted": False,
            "model_limit": 32,
            "model_info": {
                "provider": provider,
                "model_id": model_id,
                "capabilities": ["text"],
            },
        }

    monkeypatch.setattr(
        "ai_service.config",
        SimpleNamespace(
            GENERATOR_SYSTEM_PROMPT="system",
            GENERATOR_SYSTEM_PROMPT_RAW="raw system",
            REQUEST_TIMEOUT=30,
            AI_ACCENT_AUDIT_TIMEOUT_SECONDS=30,
            TOOL_LOOP_MAX_PROMPT_CHARS=200_000,
            model_specs={},
            validate_token_limits=validate_token_limits,
        ),
    )
    return validated_models


@pytest.mark.asyncio
async def test_local_guard_authorizes_before_adapter_and_reauthorizes_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime()
    attempts = 0

    async def adapter(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        assert len(runtime.ledger.entries) == attempts
        assert runtime.ledger.entries[-1].allowed
        if attempts == 1:
            raise TimeoutError("offline retry")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"), finish_reason="stop")],
            usage=None,
        )

    service = _service(monkeypatch, AsyncMock(side_effect=adapter))
    _install_config(monkeypatch)
    await service.generate_content(
        prompt="offline",
        model="writer-local",
        local_execution=runtime,
        dispatch_call_id="generation.main",
    )

    assert attempts == 2
    assert [entry.allowed for entry in runtime.ledger.entries] == [True, True]


def test_local_openai_transport_disables_redirects() -> None:
    client = _local_openai_async_http_client()
    try:
        assert client.follow_redirects is False
    finally:
        asyncio.run(client.aclose())


def test_local_openai_clients_disable_sdk_retries_and_use_local_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint = LocalInferenceEndpoint("ollama_a", "ollama", "http://a:11434")
    service = AIService.__new__(AIService)
    service._local_endpoint_clients = {}
    service._local_endpoint_semaphores = {}
    local_transport = object()
    clients: list[dict[str, object]] = []

    monkeypatch.setattr(
        "ai_service.build_local_endpoint_registry",
        lambda _config: LocalEndpointRegistry((endpoint,), "test"),
    )
    monkeypatch.setattr(
        "ai_service._local_openai_async_http_client",
        lambda: local_transport,
    )
    monkeypatch.setattr(
        "ai_service.openai.AsyncOpenAI",
        lambda **kwargs: clients.append(kwargs) or object(),
    )
    monkeypatch.setattr(
        "ai_service.config",
        SimpleNamespace(
            SDK_OLLAMA_TIMEOUT_SECONDS=12,
            SDK_LLAMACPP_TIMEOUT_SECONDS=12,
        ),
    )

    service._initialize_local_endpoint_clients()

    assert clients == [
        {
            "api_key": "ollama",
            "base_url": "http://a:11434/v1",
            "timeout": 12.0,
            "max_retries": 0,
            "http_client": local_transport,
        }
    ]


def test_missing_local_route_records_one_blocked_ledger_entry_before_catalog_resolution() -> None:
    service = AIService.__new__(AIService)
    runtime = _runtime()

    with pytest.raises(LocalExecutionPolicyError, match="local_route_not_admitted"):
        service._resolve_endpoint_scoped_catalog_model(
            "writer-local",
            endpoint_id=None,
            local_execution=runtime,
            dispatch_call_id="generation.missing",
            dispatch_route_ordinal=0,
        )

    assert [(entry.allowed, entry.reason) for entry in runtime.ledger.entries] == [
        (False, "local_route_not_admitted")
    ]
    attestation = build_local_execution_attestation(runtime.manifest, runtime.ledger)
    assert attestation.local_only_verified is False
    assert attestation.blocked_dispatch_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("availability", "state"),
    [("loaded", "loaded"), ("available", "unloaded")],
)
async def test_normal_local_endpoint_resolves_colliding_model_and_availability(
    monkeypatch: pytest.MonkeyPatch,
    availability: str,
    state: str,
) -> None:
    endpoint_a = LocalInferenceEndpoint("ollama_a", "ollama", "http://a:11434")
    endpoint_b = LocalInferenceEndpoint(
        "llamacpp_b",
        "llamacpp",
        "http://b:8080/v1",
        availability=availability,
    )
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="served by llama.cpp b"),
                finish_reason="stop",
            )
        ],
        usage=None,
    )
    endpoint_a_create = AsyncMock()
    endpoint_b_create = AsyncMock(return_value=response)
    service = _service(monkeypatch, AsyncMock())
    service.local_endpoint_registry = LocalEndpointRegistry(
        (endpoint_a, endpoint_b), "test"
    )
    service._local_endpoints = {
        endpoint_a.endpoint_id: endpoint_a,
        endpoint_b.endpoint_id: endpoint_b,
    }
    service._local_endpoint_clients = {
        endpoint_a.endpoint_id: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=endpoint_a_create))
        ),
        endpoint_b.endpoint_id: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=endpoint_b_create))
        ),
    }
    service._local_endpoint_semaphores = {
        endpoint_a.endpoint_id: asyncio.Semaphore(1),
        endpoint_b.endpoint_id: asyncio.Semaphore(1),
    }
    service.ollama_client = None
    service.llamacpp_client = None
    validated_models = _install_colliding_local_config(monkeypatch)
    discovery_calls = _install_endpoint_discovery(
        monkeypatch,
        {"llamacpp_b": [{"model_id": "shared", "state": state}]},
    )

    content = await service.generate_content(
        prompt="offline",
        model="shared",
        endpoint_id="llamacpp_b",
    )

    assert content == "served by llama.cpp b"
    assert validated_models == ["llamacpp::shared"]
    assert discovery_calls == ["llamacpp_b"]
    endpoint_a_create.assert_not_awaited()
    endpoint_b_create.assert_awaited_once()
    assert endpoint_b_create.await_args.kwargs["model"] == "shared"


@pytest.mark.asyncio
async def test_local_only_manifest_resolves_colliding_model_without_normal_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint_a = LocalInferenceEndpoint("ollama_a", "ollama", "http://a:11434")
    endpoint_b = LocalInferenceEndpoint("llamacpp_b", "llamacpp", "http://b:8080/v1")
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="served by manifest endpoint"),
                finish_reason="stop",
            )
        ],
        usage=None,
    )
    endpoint_a_create = AsyncMock()
    endpoint_b_create = AsyncMock(return_value=response)
    service = _service(monkeypatch, AsyncMock())
    service.local_endpoint_registry = LocalEndpointRegistry(
        (endpoint_a, endpoint_b), "test"
    )
    service._local_endpoints = {
        endpoint_a.endpoint_id: endpoint_a,
        endpoint_b.endpoint_id: endpoint_b,
    }
    service._local_endpoint_clients = {
        endpoint_a.endpoint_id: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=endpoint_a_create))
        ),
        endpoint_b.endpoint_id: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=endpoint_b_create))
        ),
    }
    service._local_endpoint_semaphores = {
        endpoint_a.endpoint_id: asyncio.Semaphore(1),
        endpoint_b.endpoint_id: asyncio.Semaphore(1),
    }
    service.ollama_client = None
    service.llamacpp_client = None
    validated_models = _install_colliding_local_config(monkeypatch)
    discovery_calls = _install_endpoint_discovery(monkeypatch, {})
    runtime = _runtime(
        provider="llamacpp",
        model_id="shared",
        endpoint_id="llamacpp_b",
    )

    content = await service.generate_content(
        prompt="offline",
        model="shared",
        local_execution=runtime,
        dispatch_call_id="generation.main",
    )

    assert content == "served by manifest endpoint"
    assert validated_models == ["llamacpp::shared"]
    assert discovery_calls == []
    endpoint_a_create.assert_not_awaited()
    endpoint_b_create.assert_awaited_once()
    assert endpoint_b_create.await_args.kwargs["model"] == "shared"
    assert [(entry.allowed, entry.endpoint_id) for entry in runtime.ledger.entries] == [
        (True, "llamacpp_b")
    ]


@pytest.mark.asyncio
async def test_normal_local_tool_loop_discovers_once_for_multiple_turns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint = LocalInferenceEndpoint("ollama_b", "ollama", "http://b:11434")
    service = _service(monkeypatch, AsyncMock())
    service.local_endpoint_registry = LocalEndpointRegistry((endpoint,), "test")
    service._local_endpoints = {endpoint.endpoint_id: endpoint}
    service._local_endpoint_clients = {endpoint.endpoint_id: object()}
    service._local_endpoint_semaphores = {endpoint.endpoint_id: asyncio.Semaphore(1)}
    service.ollama_client = None
    service.llamacpp_client = None
    validated_models = _install_colliding_local_config(monkeypatch)
    discovery_calls = _install_endpoint_discovery(
        monkeypatch,
        {endpoint.endpoint_id: [{"model_id": "shared", "state": "loaded"}]},
    )
    loop = AsyncMock(
        return_value=(
            "tool-loop result",
            {
                "mode": "ollama_tool_loop",
                "turns": 2,
                "accepted": "assistant_final",
                "trace": [],
            },
        )
    )
    monkeypatch.setattr(service, "_normalize_tool_loop_provider", lambda _provider: "ollama")
    monkeypatch.setattr(service, "_supports_generation_validation_tool_loop", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(service, "_serialize_tool_loop_tool_defs", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(service, "_build_chat_call_budget", lambda **_kwargs: object())
    monkeypatch.setattr(service, "_run_openai_compatible_validation_tool_loop", loop)
    monkeypatch.setattr(
        context_budget,
        "evaluate_call_admission",
        AsyncMock(
            return_value=SimpleNamespace(
                decision=context_budget.AdmissionDecision.FITS,
                capacity_unverified=False,
            )
        ),
    )

    content, envelope = await service.call_ai_with_validation_tools(
        prompt="offline",
        model="shared",
        validation_callback=lambda _text: SimpleNamespace(),
        endpoint_id=endpoint.endpoint_id,
        max_tool_rounds=2,
    )

    assert content == "tool-loop result"
    assert envelope.turns == 2
    assert validated_models == ["ollama::shared"]
    assert discovery_calls == [endpoint.endpoint_id]
    loop.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "discovered_models",
    [[], [{"model_id": "writer-local", "state": "unloaded"}]],
)
async def test_normal_loaded_endpoint_blocks_absent_or_unloaded_model(
    monkeypatch: pytest.MonkeyPatch,
    discovered_models: list[dict[str, object]],
) -> None:
    endpoint = LocalInferenceEndpoint("ollama_b", "ollama", "http://b:11434")
    client_create = AsyncMock()
    service = _service(monkeypatch, AsyncMock())
    service.local_endpoint_registry = LocalEndpointRegistry((endpoint,), "test")
    service._local_endpoints = {endpoint.endpoint_id: endpoint}
    service._local_endpoint_clients = {
        endpoint.endpoint_id: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=client_create))
        )
    }
    service._local_endpoint_semaphores = {endpoint.endpoint_id: asyncio.Semaphore(1)}
    service.ollama_client = None
    service.llamacpp_client = None
    _install_config(monkeypatch)
    discovery_calls = _install_endpoint_discovery(
        monkeypatch,
        {endpoint.endpoint_id: discovered_models},
    )

    with pytest.raises(LocalExecutionPolicyError) as exc_info:
        await service.generate_content(
            prompt="offline",
            model="writer-local",
            endpoint_id=endpoint.endpoint_id,
        )

    assert exc_info.value.code == "local_model_not_announced"
    assert discovery_calls == [endpoint.endpoint_id]
    client_create.assert_not_awaited()


@pytest.mark.asyncio
async def test_registered_ollama_retry_stays_on_selected_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint_a = LocalInferenceEndpoint("ollama_a", "ollama", "http://a:11434")
    endpoint_b = LocalInferenceEndpoint("ollama_b", "ollama", "http://b:11434")
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="served by endpoint b"),
                finish_reason="stop",
            )
        ],
        usage=None,
    )
    endpoint_a_create = AsyncMock()
    endpoint_b_create = AsyncMock(side_effect=[TimeoutError("offline"), response])
    service = _service(monkeypatch, AsyncMock())
    service.local_endpoint_registry = LocalEndpointRegistry(
        (endpoint_a, endpoint_b), "test"
    )
    service._local_endpoints = {
        endpoint_a.endpoint_id: endpoint_a,
        endpoint_b.endpoint_id: endpoint_b,
    }
    service._local_endpoint_clients = {
        endpoint_a.endpoint_id: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=endpoint_a_create))
        ),
        endpoint_b.endpoint_id: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=endpoint_b_create))
        ),
    }
    service._local_endpoint_semaphores = {
        endpoint_a.endpoint_id: asyncio.Semaphore(1),
        endpoint_b.endpoint_id: asyncio.Semaphore(1),
    }
    service.ollama_client = None
    service.llamacpp_client = None
    _install_config(monkeypatch)
    runtime = _runtime(endpoint_id="ollama_b")

    content = await service.generate_content(
        prompt="offline",
        model="writer-local",
        local_execution=runtime,
        dispatch_call_id="generation.main",
        endpoint_id="ollama_b",
    )

    assert content == "served by endpoint b"
    endpoint_a_create.assert_not_awaited()
    assert endpoint_b_create.await_count == 2
    assert [(entry.allowed, entry.endpoint_id) for entry in runtime.ledger.entries] == [
        (True, "ollama_b"),
        (True, "ollama_b"),
    ]


@pytest.mark.asyncio
async def test_registered_ollama_stream_uses_selected_endpoint_and_attributes_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class OfflineStream:
        def __init__(self) -> None:
            self.usage = None

        def __aiter__(self):
            return self._iterate()

        async def _iterate(self):
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="streamed by endpoint b"),
                        finish_reason=None,
                    )
                ],
                usage=None,
            )
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content=None),
                        finish_reason="stop",
                    )
                ],
                usage=None,
            )

    endpoint_a = LocalInferenceEndpoint("ollama_a", "ollama", "http://a:11434")
    endpoint_b = LocalInferenceEndpoint("ollama_b", "ollama", "http://b:11434")
    endpoint_a_create = AsyncMock()
    endpoint_b_create = AsyncMock(return_value=OfflineStream())
    service = _service(monkeypatch, AsyncMock())
    service.local_endpoint_registry = LocalEndpointRegistry(
        (endpoint_a, endpoint_b), "test"
    )
    service._local_endpoints = {
        endpoint_a.endpoint_id: endpoint_a,
        endpoint_b.endpoint_id: endpoint_b,
    }
    service._local_endpoint_clients = {
        endpoint_a.endpoint_id: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=endpoint_a_create))
        ),
        endpoint_b.endpoint_id: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=endpoint_b_create))
        ),
    }
    service._local_endpoint_semaphores = {
        endpoint_a.endpoint_id: asyncio.Semaphore(1),
        endpoint_b.endpoint_id: asyncio.Semaphore(1),
    }
    service.ollama_client = None
    service.llamacpp_client = None
    _install_config(monkeypatch)
    runtime = _runtime(endpoint_id="ollama_b")
    usage_events: list[dict[str, object]] = []

    chunks = [
        chunk
        async for chunk in service.generate_content_stream(
            prompt="offline",
            model="writer-local",
            local_execution=runtime,
            dispatch_call_id="generation.main",
            endpoint_id="ollama_b",
            usage_callback=usage_events.append,
        )
    ]

    assert chunks[0] == "streamed by endpoint b"
    endpoint_a_create.assert_not_awaited()
    endpoint_b_create.assert_awaited_once()
    assert [(entry.allowed, entry.endpoint_id) for entry in runtime.ledger.entries] == [
        (True, "ollama_b")
    ]
    assert len(usage_events) == 1
    assert usage_events[0]["endpoint_id"] == "ollama_b"


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("structured", [False, True])
@pytest.mark.parametrize("provider", ["ollama", "llamacpp"])
async def test_registered_local_serializes_explicit_reasoning_effort(
    monkeypatch: pytest.MonkeyPatch,
    streaming: bool,
    structured: bool,
    provider: str,
) -> None:
    """Registered local transports retain explicit effort and output format."""

    class OfflineStream:
        usage = None

        def __aiter__(self):
            return self._iterate()

        async def _iterate(self):
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="ok"),
                        finish_reason="stop",
                    )
                ],
                usage=None,
            )

    response = SimpleNamespace(choices=[], usage=None)
    create = AsyncMock(return_value=OfflineStream() if streaming else response)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    service = AIService.__new__(AIService)
    endpoint = LocalInferenceEndpoint("local_reasoning", provider, "http://local:8080/v1")
    service.local_endpoint_registry = LocalEndpointRegistry((endpoint,), "test")
    service._local_endpoints = {endpoint.endpoint_id: endpoint}
    service._local_endpoint_clients = {endpoint.endpoint_id: client}
    service._local_endpoint_semaphores = {endpoint.endpoint_id: asyncio.Semaphore(1)}
    import config as config_module

    monkeypatch.setattr(
        config_module.Config,
        "get_model_info",
        lambda _self, _model: {
            "model_id": "qwen3.8:27b",
            "provider": provider,
            "capabilities": ["text", "reasoning"],
            "supported_parameters": ["reasoning_effort"],
        },
    )
    route = resolve_call(
        "generation.main",
        routing=merge_routing_documents(
            get_default_routing(),
            {
                "calls": {
                    "generation.main": {
                        "model": f"{provider}/qwen3.8:27b",
                        "params": {"reasoning_effort": "none"},
                    }
                }
            },
        ),
    )
    monkeypatch.setattr(service, "_should_inject_json_prompt", lambda *_args: False)
    monkeypatch.setattr(service, "_supports_structured_outputs", lambda *_args: structured)
    monkeypatch.setattr(service, "_supports_json_object", lambda *_args: False)
    monkeypatch.setattr(service, "_chat_completion_content_and_usage", lambda *_args, **_kwargs: ("ok", {}))
    monkeypatch.setattr(service, "_usage_with_finish_metadata", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(service, "_emit_usage", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        service,
        "_openai_compatible_token_parameter",
        lambda *_args: "max_tokens",
    )
    monkeypatch.setattr(
        service,
        "_add_chat_parameter_if_allowed",
        lambda params, **_kwargs: params.update(temperature=0.2) or True,
    )
    schema = {"type": "object", "properties": {"result": {"type": "string"}}}

    if streaming:
        chunks = [
            chunk
            async for chunk in service._stream_registered_local(
                provider=provider,
                endpoint_id=endpoint.endpoint_id,
                prompt="offline",
                model_id="local-model",
                temperature=0.2,
                max_tokens=32,
                system_prompt="system",
                json_output=structured,
                json_schema=schema if structured else None,
                reasoning_effort=route.params["reasoning_effort"],
            )
        ]
        assert chunks
    else:
        await service._generate_registered_local(
            provider=provider,
            endpoint_id=endpoint.endpoint_id,
            prompt="offline",
            model_id="local-model",
            temperature=0.2,
            max_tokens=32,
            system_prompt="system",
            json_output=structured,
            json_schema=schema if structured else None,
            reasoning_effort=route.params["reasoning_effort"],
        )

    payload = create.await_args.kwargs
    assert payload["model"] == "local-model"
    assert payload["reasoning_effort"] == "none"
    assert ("response_format" in payload) is structured


@pytest.mark.asyncio
async def test_local_guard_blocks_provider_or_model_change_before_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(model_id="writer-local")
    adapter = AsyncMock()
    service = _service(monkeypatch, adapter)
    _install_config(monkeypatch, model_id="other-local")

    with pytest.raises(LocalExecutionPolicyError, match="local_dispatch_not_authorized"):
        await service.generate_content(
            prompt="offline",
            model="other-local",
            local_execution=runtime,
            dispatch_call_id="generation.main",
        )

    adapter.assert_not_awaited()
    assert runtime.ledger.entries[-1].allowed is False


@pytest.mark.asyncio
async def test_normal_generation_does_not_require_local_dispatch_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = AsyncMock(return_value=SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"), finish_reason="stop")],
        usage=None,
    ))
    service = _service(monkeypatch, adapter)
    _install_config(monkeypatch)

    assert await service.generate_content(prompt="offline", model="writer-local") == "ok"
    adapter.assert_awaited_once()


@pytest.mark.asyncio
async def test_local_admission_does_not_receive_native_counter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = AIService.__new__(AIService)
    service._native_token_count = AsyncMock()
    observed_native_counters: list[object] = []

    async def _admit(_budget: object, *, native_counter: object = None) -> object:
        observed_native_counters.append(native_counter)
        return object()

    monkeypatch.setattr(context_budget, "admit_call", _admit)

    await service._admit_budget(object(), local_execution=object())

    assert observed_native_counters == [None]
    service._native_token_count.assert_not_awaited()


@pytest.mark.asyncio
async def test_local_tool_loop_ledgers_only_the_sdk_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = AIService.__new__(AIService)
    endpoint = LocalInferenceEndpoint("ollama_b", "ollama", "http://b:11434")
    service.local_endpoint_registry = LocalEndpointRegistry((endpoint,), "test")
    service._local_endpoints = {endpoint.endpoint_id: endpoint}
    service._local_endpoint_clients = {endpoint.endpoint_id: object()}
    service._local_endpoint_semaphores = {endpoint.endpoint_id: asyncio.Semaphore(1)}
    service.ollama_client = None
    service.llamacpp_client = None
    service._serialize_tool_loop_tool_defs = lambda *_args, **_kwargs: ""
    service._build_openai_compatible_tool_params = lambda *_args, **_kwargs: {}
    service._get_tool_loop_call_budget = lambda *_args, **_kwargs: 4
    service._build_trace_metrics = lambda *_args, **_kwargs: {}
    service._record_provider_health_success = AsyncMock()
    service._record_provider_health_failure = AsyncMock()

    @asynccontextmanager
    async def _scope(*_args, **_kwargs):
        yield None

    async def _turn(*_args, **_kwargs):
        return SimpleNamespace(content="approved response", tool_calls=[]), None, "stop", None

    monkeypatch.setattr(service, "_provider_call_scope", _scope)
    monkeypatch.setattr(service, "_stream_openai_compatible_tool_turn", _turn)
    monkeypatch.setattr(
        service,
        "_invoke_validation_callback",
        lambda *_args, **_kwargs: SimpleNamespace(
            approved=True,
            score=10.0,
            word_count=2,
            hard_failed=False,
            feedback="",
            issues=[],
        ),
    )
    runtime = _runtime(endpoint_id="ollama_b")

    content, _metadata = await service._run_openai_compatible_validation_tool_loop(
        provider="ollama",
        model_id="writer-local",
        prompt="offline",
        validation_callback=lambda _text: None,
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
        retries_enabled=False,
        local_execution=runtime,
        dispatch_call_id="generation.main",
    )

    assert content == "approved response"
    assert [(entry.allowed, entry.endpoint_id) for entry in runtime.ledger.entries] == [
        (True, "ollama_b")
    ]


@pytest.mark.asyncio
async def test_logprobs_error_exposes_endpoint_stamped_provider_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = AIService.__new__(AIService)
    endpoint = LocalInferenceEndpoint("llamacpp_b", "llamacpp", "http://b:8080/v1")
    create = AsyncMock(side_effect=TimeoutError("offline"))
    service.local_endpoint_registry = LocalEndpointRegistry((endpoint,), "test")
    service._local_endpoints = {endpoint.endpoint_id: endpoint}
    service._local_endpoint_clients = {
        endpoint.endpoint_id: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
    }
    service._local_endpoint_semaphores = {endpoint.endpoint_id: asyncio.Semaphore(1)}
    service.ollama_client = None
    service.llamacpp_client = None
    service._admit_chat_call = AsyncMock()
    service._record_provider_health_failure = AsyncMock()

    @asynccontextmanager
    async def _scope(*_args, **_kwargs):
        yield None

    failure = ProviderFailure(
        provider="llamacpp",
        model_id="qa-local",
        operation="generate_with_logprobs",
        kind=ProviderErrorKind.TIMEOUT,
        message="offline",
        retryable=True,
    )
    monkeypatch.setattr(service, "_provider_call_scope", _scope)
    monkeypatch.setattr(service, "_classify_provider_failure", lambda *_args, **_kwargs: failure)
    monkeypatch.setattr(
        "ai_service.config",
        SimpleNamespace(
            model_specs={},
            get_model_info=lambda _model: {"provider": "llamacpp", "model_id": "qa-local"},
        ),
    )
    monkeypatch.setattr(
        "model_capability_registry.model_supports",
        lambda **_kwargs: SimpleNamespace(supported=True),
    )
    _install_endpoint_discovery(
        monkeypatch,
        {endpoint.endpoint_id: [{"model_id": "qa-local", "state": "loaded"}]},
    )

    with pytest.raises(AIRequestError) as exc_info:
        await service.generate_with_logprobs(
            prompt="offline",
            model="qa-local",
            endpoint_id="llamacpp_b",
        )

    assert exc_info.value.provider_failure.endpoint_id == "llamacpp_b"
    assert service._record_provider_health_failure.await_args.args[0].endpoint_id == "llamacpp_b"


def test_multi_endpoint_transport_requires_and_selects_exact_endpoint() -> None:
    service = AIService.__new__(AIService)
    first = object()
    second = object()
    endpoint_a = LocalInferenceEndpoint("ollama_a", "ollama", "http://a:11434")
    endpoint_b = LocalInferenceEndpoint("ollama_b", "ollama", "http://b:11434")
    service.local_endpoint_registry = LocalEndpointRegistry((endpoint_a, endpoint_b), "test")
    service._local_endpoints = {"ollama_a": endpoint_a, "ollama_b": endpoint_b}
    service._local_endpoint_clients = {"ollama_a": first, "ollama_b": second}
    service._local_endpoint_semaphores = {
        "ollama_a": asyncio.Semaphore(1),
        "ollama_b": asyncio.Semaphore(1),
    }
    service.ollama_client = None
    service.llamacpp_client = None

    assert service._local_endpoint_transport("ollama", "ollama_b")[0] is second
    with pytest.raises(LocalExecutionPolicyError, match="local_endpoint_required"):
        service._local_endpoint_transport("ollama", None)
    with pytest.raises(LocalExecutionPolicyError, match="local_endpoint_adapter_mismatch"):
        service._local_endpoint_transport("openai_compatible", "ollama_a")
