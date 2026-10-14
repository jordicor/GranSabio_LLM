from __future__ import annotations

from types import SimpleNamespace

import pytest

from local_endpoint_registry import LocalEndpointRegistry, LocalInferenceEndpoint
from local_execution_policy import (
    LocalExecutionManifest,
    LocalExecutionPolicyError,
    LocalExecutionRuntime,
    LocalRouteRequirement,
    ManifestRoute,
    SessionDispatchLedger,
    admit_local_only,
    authorize_local_dispatch,
    build_local_endpoint_inventory,
    build_local_execution_attestation,
    create_dispatch_context,
    resolve_local_route_requirement,
)
from models import ContentRequest, GenerationInitResponse


class _FakeSyncService:
    def __init__(self, config_obj: object, _logger: object):
        self.config_obj = config_obj

    async def discover_local_endpoint(self, endpoint: LocalInferenceEndpoint) -> list[dict[str, object]]:
        return list(
            self.config_obj.discovered_models.get(
                endpoint.endpoint_id,
                self.config_obj.discovered_models.get(endpoint.adapter, []),
            )
        )


def _config() -> SimpleNamespace:
    model_infos = {
        "writer-local": {
            "provider": "ollama",
            "model_id": "writer-local",
            "capabilities": ["text", "structured_outputs"],
        },
        "qa-local": {
            "provider": "llamacpp",
            "model_id": "qa-local",
            "capabilities": ["text", "logprobs"],
        },
        "cloud-model": {
            "provider": "openai",
            "model_id": "cloud-model",
            "capabilities": ["text"],
        },
    }
    config_obj = SimpleNamespace(
        OLLAMA_HOST="http://127.0.0.1:11434",
        OLLAMA_ENDPOINT_ALIAS="local_primary",
        LLAMACPP_BASE_URL="http://192.0.2.10:11435/v1",
        LLAMACPP_ENDPOINT_ALIAS="local_secondary",
        discovered_models={
            "ollama": [{"model_id": "writer-local"}],
            "llamacpp": [{"model_id": "qa-local"}],
        },
    )
    def _model_info(model: str, require_api_key: bool = False) -> dict[str, object]:
        del require_api_key
        qualified_provider, separator, unqualified_model = model.partition("::")
        lookup_model = unqualified_model if separator else model
        result = dict(model_infos[lookup_model])
        if separator:
            result["provider"] = qualified_provider
        return result

    config_obj.get_model_info = _model_info
    return config_obj


def test_content_request_defaults_execution_policy() -> None:
    request = ContentRequest(prompt="A prompt long enough for validation.")
    response = GenerationInitResponse(
        status="preflight_rejected",
        local_only_attestation={"local_only_verified": True},
    )

    assert request.execution_policy.scope == "default"
    assert response.local_only_attestation == {"local_only_verified": True}


@pytest.mark.asyncio
async def test_local_admission_builds_stable_multiroute_manifest() -> None:
    config_obj = _config()
    inventory = build_local_endpoint_inventory(config_obj)
    routes = [
        LocalRouteRequirement("generation.main", 0, "writer-local", frozenset()),
        LocalRouteRequirement("qa.evaluate_layer", 0, "qa-local", frozenset({"logprobs"})),
    ]

    manifest = await admit_local_only(
        routes,
        config_obj=config_obj,
        inventory=inventory,
        model_sync_factory=_FakeSyncService,
    )
    repeated = await admit_local_only(
        list(reversed(routes)),
        config_obj=config_obj,
        inventory=inventory,
        model_sync_factory=_FakeSyncService,
    )

    assert manifest.manifest_hash == repeated.manifest_hash
    assert [(route.call_id, route.endpoint_id) for route in manifest.routes] == [
        ("generation.main", "local_primary"),
        ("qa.evaluate_layer", "local_secondary"),
    ]


@pytest.mark.asyncio
async def test_local_admission_rejects_cloud_before_discovery() -> None:
    config_obj = _config()

    with pytest.raises(LocalExecutionPolicyError) as exc_info:
        await admit_local_only(
            [LocalRouteRequirement("generation.main", 0, "cloud-model")],
            config_obj=config_obj,
            model_sync_factory=_FakeSyncService,
        )

    assert exc_info.value.code == "local_provider_required"


@pytest.mark.asyncio
async def test_local_admission_rejects_stale_ollama_catalog_entry() -> None:
    config_obj = _config()
    config_obj.discovered_models["ollama"] = [
        {
            "model_id": "writer-local",
            "status": "stale",
            "local_only": True,
        }
    ]

    with pytest.raises(LocalExecutionPolicyError) as exc_info:
        await admit_local_only(
            [LocalRouteRequirement("generation.main", 0, "writer-local")],
            config_obj=config_obj,
            model_sync_factory=_FakeSyncService,
        )

    assert exc_info.value.code == "local_model_not_announced"


@pytest.mark.asyncio
async def test_dispatch_authorization_rejects_route_change_and_attests() -> None:
    config_obj = _config()
    manifest = await admit_local_only(
        [LocalRouteRequirement("generation.main", 0, "writer-local")],
        config_obj=config_obj,
        model_sync_factory=_FakeSyncService,
    )
    context = create_dispatch_context(
        manifest,
        session_id="session-1",
        call_id="generation.main",
    )
    ledger = SessionDispatchLedger("session-1")

    authorize_local_dispatch(
        context,
        manifest,
        ledger,
        provider="ollama",
        model_id="writer-local",
    )
    with pytest.raises(LocalExecutionPolicyError) as exc_info:
        authorize_local_dispatch(
            context,
            manifest,
            ledger,
            provider="openai",
            model_id="writer-local",
        )

    attestation = build_local_execution_attestation(manifest, ledger)
    assert exc_info.value.code == "local_dispatch_not_authorized"
    assert attestation.external_provider_calls == 0
    assert attestation.dispatch_count == 1
    assert attestation.blocked_dispatch_count == 1
    assert not attestation.local_only_verified


@pytest.mark.asyncio
async def test_session_runtime_authorizes_and_owns_its_ledger() -> None:
    config_obj = _config()
    manifest = await admit_local_only(
        [LocalRouteRequirement("generation.main", 0, "writer-local")],
        config_obj=config_obj,
        model_sync_factory=_FakeSyncService,
    )
    runtime = LocalExecutionRuntime.create(session_id="session-1", manifest=manifest)

    route = runtime.authorize("generation.main", 0, "ollama", "writer-local")

    assert route.endpoint_id == "local_primary"
    assert runtime.ledger.session_id == "session-1"
    assert len(runtime.ledger.entries) == 1


def _two_ollama_inventory() -> LocalEndpointRegistry:
    return LocalEndpointRegistry(
        endpoints=(
            LocalInferenceEndpoint("spark_a", "ollama", "http://spark-a:11434"),
            LocalInferenceEndpoint("spark_b", "ollama", "http://spark-b:11434"),
        ),
        source="test",
    )


@pytest.mark.asyncio
async def test_local_admission_uses_explicit_endpoint_between_same_adapter_hosts() -> None:
    config_obj = _config()
    config_obj.discovered_models = {
        "spark_a": [{"model_id": "writer-local"}],
        "spark_b": [{"model_id": "writer-local"}],
    }

    manifest = await admit_local_only(
        [LocalRouteRequirement("generation.main", 0, "writer-local", endpoint_id="spark_b")],
        config_obj=config_obj,
        inventory=_two_ollama_inventory(),
        model_sync_factory=_FakeSyncService,
    )

    assert manifest.routes[0].endpoint_id == "spark_b"
    assert manifest.routes[0].adapter == "ollama"


@pytest.mark.asyncio
async def test_explicit_endpoint_selects_its_catalog_adapter_for_shared_model_id() -> None:
    model_infos = {
        "ollama::shared": {"provider": "ollama", "model_id": "shared", "capabilities": ["text"]},
        "llamacpp::shared": {"provider": "llamacpp", "model_id": "shared", "capabilities": ["text"]},
    }
    config_obj = SimpleNamespace(
        discovered_models={
            "ollama-a": [{"model_id": "shared"}],
            "llamacpp-a": [{"model_id": "shared"}],
        },
    )
    config_obj.get_model_info = lambda model, require_api_key=False: model_infos[model]
    inventory = LocalEndpointRegistry(
        endpoints=(
            LocalInferenceEndpoint("ollama-a", "ollama", "http://ollama-a:11434"),
            LocalInferenceEndpoint("llamacpp-a", "llamacpp", "http://llamacpp-a:8080/v1"),
        ),
        source="test",
    )

    manifest = await admit_local_only(
        [LocalRouteRequirement("generation.main", 0, "shared", endpoint_id="ollama-a")],
        config_obj=config_obj,
        inventory=inventory,
        model_sync_factory=_FakeSyncService,
    )

    assert manifest.routes[0].provider == "ollama"
    assert manifest.routes[0].endpoint_id == "ollama-a"


@pytest.mark.asyncio
async def test_route_requirement_resolves_colliding_model_by_explicit_endpoint_before_global_catalog() -> None:
    model_infos = {
        "ollama::shared": {"provider": "ollama", "model_id": "shared", "capabilities": ["text"]},
        "llamacpp::shared": {"provider": "llamacpp", "model_id": "shared", "capabilities": ["text"]},
    }
    config_obj = SimpleNamespace(
        discovered_models={"ollama-a": [{"model_id": "shared"}]},
        get_model_info=lambda model, require_api_key=False: model_infos[model],
    )
    inventory = LocalEndpointRegistry(
        endpoints=(
            LocalInferenceEndpoint("ollama-a", "ollama", "http://ollama-a:11434"),
            LocalInferenceEndpoint("llamacpp-a", "llamacpp", "http://llamacpp-a:8080/v1"),
        ),
        source="test",
    )
    routing = {
        "calls": {
            "generation.main": {"model": "shared", "endpoint_id": "ollama-a"},
        }
    }

    requirement = resolve_local_route_requirement("generation.main", routing=routing)
    manifest = await admit_local_only(
        [requirement],
        config_obj=config_obj,
        inventory=inventory,
        model_sync_factory=_FakeSyncService,
    )

    assert requirement.endpoint_id == "ollama-a"
    assert manifest.routes[0].provider == "ollama"


@pytest.mark.asyncio
async def test_local_admission_rejects_omitted_endpoint_when_adapter_is_ambiguous() -> None:
    config_obj = _config()

    with pytest.raises(LocalExecutionPolicyError) as exc_info:
        await admit_local_only(
            [LocalRouteRequirement("generation.main", 0, "writer-local")],
            config_obj=config_obj,
            inventory=_two_ollama_inventory(),
            model_sync_factory=_FakeSyncService,
        )

    assert exc_info.value.code == "local_endpoint_ambiguous"


@pytest.mark.asyncio
async def test_local_admission_requires_model_from_the_exact_endpoint() -> None:
    config_obj = _config()
    config_obj.discovered_models = {
        "spark_a": [{"model_id": "writer-local"}],
        "spark_b": [],
    }

    with pytest.raises(LocalExecutionPolicyError) as exc_info:
        await admit_local_only(
            [LocalRouteRequirement("generation.main", 0, "writer-local", endpoint_id="spark_b")],
            config_obj=config_obj,
            inventory=_two_ollama_inventory(),
            model_sync_factory=_FakeSyncService,
        )

    assert exc_info.value.code == "local_model_not_announced"


@pytest.mark.asyncio
async def test_local_admission_caches_discovery_by_endpoint_id() -> None:
    config_obj = _config()
    config_obj.discovered_models = {
        "spark_a": [{"model_id": "writer-local"}],
        "spark_b": [{"model_id": "writer-local"}],
    }
    calls: list[str] = []

    class _CountingSync(_FakeSyncService):
        async def discover_local_endpoint(self, endpoint: LocalInferenceEndpoint) -> list[dict[str, object]]:
            calls.append(endpoint.endpoint_id)
            return await super().discover_local_endpoint(endpoint)

    await admit_local_only(
        [
            LocalRouteRequirement("generation.main", 0, "writer-local", endpoint_id="spark_a"),
            LocalRouteRequirement("generation.json_retry", 0, "writer-local", endpoint_id="spark_a"),
            LocalRouteRequirement("generation.smart_edit", 0, "writer-local", endpoint_id="spark_b"),
        ],
        config_obj=config_obj,
        inventory=_two_ollama_inventory(),
        model_sync_factory=_CountingSync,
    )

    assert calls == ["spark_a", "spark_b"]


@pytest.mark.asyncio
async def test_legacy_singleton_configuration_remains_an_unambiguous_endpoint() -> None:
    config_obj = _config()

    manifest = await admit_local_only(
        [LocalRouteRequirement("generation.main", 0, "writer-local")],
        config_obj=config_obj,
        model_sync_factory=_FakeSyncService,
    )

    assert manifest.routes[0].endpoint_id == "local_primary"
    assert manifest.routes[0].adapter == "ollama"


def test_dispatch_authorization_rejects_different_effective_endpoint() -> None:
    manifest = LocalExecutionManifest(
        routes=(
            ManifestRoute(
                call_id="generation.main",
                route_ordinal=0,
                provider="ollama",
                endpoint_id="spark_a",
                model_id="writer-local",
                adapter="ollama",
            ),
        ),
        manifest_hash="manifest-test",
    )
    runtime = LocalExecutionRuntime.create(session_id="session-1", manifest=manifest)

    with pytest.raises(LocalExecutionPolicyError) as exc_info:
        runtime.authorize("generation.main", 0, "ollama", "writer-local", endpoint_id="spark_b")

    assert exc_info.value.code == "local_dispatch_not_authorized"
    assert runtime.ledger.entries[-1].endpoint_id == "spark_b"


@pytest.mark.asyncio
async def test_session_runtime_ledgers_unadmitted_route_rejection() -> None:
    config_obj = _config()
    manifest = await admit_local_only(
        [LocalRouteRequirement("generation.main", 0, "writer-local")],
        config_obj=config_obj,
        model_sync_factory=_FakeSyncService,
    )
    runtime = LocalExecutionRuntime.create(session_id="session-1", manifest=manifest)

    with pytest.raises(LocalExecutionPolicyError, match="local_route_not_admitted"):
        runtime.authorize("qa.evaluate_layer", 0, "llamacpp", "qa-local")

    attestation = build_local_execution_attestation(manifest, runtime.ledger)
    assert attestation.blocked_dispatch_count == 1
    assert not attestation.local_only_verified
