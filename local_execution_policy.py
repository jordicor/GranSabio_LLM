"""Fail-closed local execution policy primitives.

This module deliberately owns policy state but not provider clients.  The
generation entrypoint builds a manifest after request routing is known, and
the transport boundary authorizes each concrete dispatch against it.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from llm_routing import LLM_CALL_REGISTRY, resolve_call, resolve_call_models
from local_endpoint_registry import (
    LocalEndpointConfigurationError,
    LocalEndpointRegistry,
    LocalInferenceEndpoint,
    build_local_endpoint_registry,
)

logger = logging.getLogger(__name__)

LOCAL_ONLY_SCOPE = "local_only"
LOCAL_PROVIDERS = frozenset({"ollama", "llamacpp", "openai_compatible"})


class LocalExecutionPolicyError(RuntimeError):
    """A deterministic rejection raised before an unauthorized dispatch."""

    def __init__(self, code: str, detail: str):
        self.code = str(code)
        self.detail = str(detail)
        super().__init__(f"{self.code}: {self.detail}")

    def to_payload(self) -> dict[str, str]:
        """Return the stable error payload suitable for an API response."""

        return {"code": self.code, "detail": self.detail}


# Kept as public aliases for Track A consumers.  URL normalization and
# administrative validation live exclusively in local_endpoint_registry.
LocalEndpoint = LocalInferenceEndpoint
LocalEndpointInventory = LocalEndpointRegistry


@dataclass(frozen=True)
class LocalRouteRequirement:
    """A routed call that must be admitted before it may dispatch."""

    call_id: str
    route_ordinal: int
    model: str
    required_capabilities: frozenset[str] = frozenset()
    endpoint_id: Optional[str] = None


@dataclass(frozen=True)
class ManifestRoute:
    """One immutable, fully resolved local dispatch authorization."""

    call_id: str
    route_ordinal: int
    provider: str
    endpoint_id: str
    model_id: str
    capabilities_verified: frozenset[str] = frozenset()
    adapter: str = ""

    def canonical_payload(self) -> dict[str, Any]:
        """Return the deterministic representation used in manifest hashing."""

        return {
            "call_id": self.call_id,
            "route_ordinal": self.route_ordinal,
            "provider": self.provider,
            "endpoint_id": self.endpoint_id,
            "model_id": self.model_id,
            "adapter": self.adapter,
            "capabilities_verified": sorted(self.capabilities_verified),
        }


@dataclass(frozen=True)
class LocalExecutionManifest:
    """Immutable authorization set for one local-only execution."""

    routes: tuple[ManifestRoute, ...]
    manifest_hash: str
    scope: str = LOCAL_ONLY_SCOPE

    def route_for(self, call_id: str, route_ordinal: int = 0) -> ManifestRoute:
        """Return the exact role/ordinal authorization or fail closed."""

        matches = tuple(
            route
            for route in self.routes
            if route.call_id == call_id and route.route_ordinal == route_ordinal
        )
        if len(matches) != 1:
            raise LocalExecutionPolicyError(
                "local_route_not_admitted",
                f"No local-only manifest route exists for '{call_id}' ordinal {route_ordinal}.",
            )
        return matches[0]


@dataclass(frozen=True)
class LocalDispatchContext:
    """Caller-supplied immutable identity for one intended provider dispatch."""

    session_id: str
    manifest_hash: str
    call_id: str
    route_ordinal: int = 0


@dataclass(frozen=True)
class DispatchLedgerEntry:
    """One authorization verdict recorded before a transport call."""

    session_id: str
    manifest_hash: str
    call_id: str
    route_ordinal: int
    provider: str
    endpoint_id: Optional[str]
    model_id: str
    allowed: bool
    reason: Optional[str] = None


@dataclass
class SessionDispatchLedger:
    """In-memory ledger owned by exactly one generation session."""

    session_id: str
    entries: list[DispatchLedgerEntry] = field(default_factory=list)

    def record(self, entry: DispatchLedgerEntry) -> None:
        """Append a verdict while preserving the session boundary."""

        if entry.session_id != self.session_id:
            raise LocalExecutionPolicyError(
                "local_ledger_session_mismatch",
                "Dispatch ledger entry belongs to a different session.",
            )
        self.entries.append(entry)


@dataclass(frozen=True)
class LocalExecutionRuntime:
    """Session-scoped local-only state carried to every transport boundary."""

    session_id: str
    manifest: LocalExecutionManifest
    ledger: SessionDispatchLedger

    @classmethod
    def create(
        cls,
        *,
        session_id: str,
        manifest: LocalExecutionManifest,
    ) -> "LocalExecutionRuntime":
        """Create the mutable ledger owned by one immutable execution runtime."""

        normalized_session_id = str(session_id or "").strip()
        if not normalized_session_id:
            raise LocalExecutionPolicyError(
                "local_session_id_missing",
                "A local-only runtime requires a session id.",
            )
        return cls(
            session_id=normalized_session_id,
            manifest=manifest,
            ledger=SessionDispatchLedger(normalized_session_id),
        )

    def authorize(
        self,
        call_id: str,
        route_ordinal: int,
        provider: str,
        model_id: str,
        endpoint_id: Optional[str] = None,
    ) -> ManifestRoute:
        """Authorize and ledger one exact dispatch before its transport starts."""

        context = create_dispatch_context(
            self.manifest,
            session_id=self.session_id,
            call_id=call_id,
            route_ordinal=route_ordinal,
        )
        return authorize_local_dispatch(
            context,
            self.manifest,
            self.ledger,
            provider=provider,
            model_id=model_id,
            endpoint_id=endpoint_id,
        )

    def record_blocked(
        self,
        *,
        call_id: str,
        route_ordinal: int,
        provider: str,
        model_id: str,
        reason: str,
    ) -> None:
        """Record a rejection raised before a dispatch context can be built."""

        self.ledger.record(
            DispatchLedgerEntry(
                session_id=self.session_id,
                manifest_hash=self.manifest.manifest_hash,
                call_id=str(call_id or ""),
                route_ordinal=route_ordinal,
                provider=_normalize_provider(provider),
                endpoint_id=None,
                model_id=str(model_id or "").strip(),
                allowed=False,
                reason=reason,
            )
        )


@dataclass(frozen=True)
class LocalExecutionAttestation:
    """Summary derived solely from a manifest and its session ledger."""

    local_only_verified: bool
    external_provider_calls: int
    providers_used: tuple[str, ...]
    endpoints_used: tuple[str, ...]
    routing_manifest_hash: str
    dispatch_count: int
    blocked_dispatch_count: int

    def to_payload(self) -> dict[str, Any]:
        """Return the public audit representation."""

        return {
            "local_only_verified": self.local_only_verified,
            "external_provider_calls": self.external_provider_calls,
            "providers_used": list(self.providers_used),
            "endpoints_used": list(self.endpoints_used),
            "routing_manifest_hash": self.routing_manifest_hash,
            "dispatch_count": self.dispatch_count,
            "blocked_dispatch_count": self.blocked_dispatch_count,
        }


def build_local_endpoint_inventory(config_obj: Any) -> LocalEndpointInventory:
    """Return the sole administrative endpoint registry for local admission."""

    try:
        return build_local_endpoint_registry(config_obj)
    except LocalEndpointConfigurationError as exc:
        raise LocalExecutionPolicyError("local_endpoint_configuration_invalid", str(exc)) from exc


def resolve_local_route_requirement(
    call_id: str,
    *,
    request: Any = None,
    routing: Optional[Mapping[str, Any]] = None,
    route_ordinal: int = 0,
) -> LocalRouteRequirement:
    """Resolve one exact ``call_id`` and ordinal through canonical LLM routing."""

    spec = LLM_CALL_REGISTRY.get(call_id)
    if spec is None:
        raise LocalExecutionPolicyError("local_unknown_call", f"Unknown LLM call '{call_id}'.")
    if route_ordinal < 0:
        raise LocalExecutionPolicyError("local_route_ordinal_invalid", "route_ordinal must be non-negative.")
    if spec.multi_model:
        resolution = resolve_call_models(
            call_id,
            request=request,
            routing=routing,
            validate_model=False,
        )
        try:
            entry = resolution.models[route_ordinal]
        except IndexError as exc:
            raise LocalExecutionPolicyError(
                "local_route_ordinal_invalid",
                f"LLM call '{call_id}' has no model at ordinal {route_ordinal}.",
            ) from exc
        model = str(entry.get("model") or "").strip()
        extra_required = entry.get("requires") or ()
        endpoint_id = entry.get("endpoint_id")
    else:
        if route_ordinal != 0:
            raise LocalExecutionPolicyError(
                "local_route_ordinal_invalid",
                f"Single-model LLM call '{call_id}' only accepts ordinal 0.",
            )
        resolution = resolve_call(
            call_id,
            request=request,
            routing=routing,
            validate_model=False,
        )
        model = str(resolution.model or "").strip()
        extra_required = ()
        endpoint_id = getattr(resolution, "endpoint_id", None) or resolution.params.get("endpoint_id")
    if not model:
        raise LocalExecutionPolicyError(
            "local_route_model_missing", f"LLM call '{call_id}' resolved without a model."
        )
    required = frozenset(
        _normalize_capability(item) for item in tuple(spec.required_capabilities) + tuple(extra_required)
    )
    return LocalRouteRequirement(
        call_id,
        route_ordinal,
        model,
        required,
        str(endpoint_id).strip() or None if endpoint_id is not None else None,
    )


async def admit_local_only(
    route_requirements: Sequence[LocalRouteRequirement],
    *,
    config_obj: Any,
    inventory: Optional[LocalEndpointInventory] = None,
    model_sync_factory: Optional[Callable[[Any, Any], Any]] = None,
) -> LocalExecutionManifest:
    """Read-only admit every known route and return its immutable manifest.

    ``model_sync_factory`` exists for deterministic tests and must construct an
    object exposing ``discover_local_endpoint(endpoint)``. Production uses
    ``ModelSyncService`` and never invokes its write/sync methods.
    """

    if not route_requirements:
        raise LocalExecutionPolicyError(
            "local_manifest_empty", "local_only admission requires at least one reachable route."
        )
    effective_inventory = inventory or build_local_endpoint_inventory(config_obj)
    discovered_by_endpoint: dict[str, tuple[Mapping[str, Any], ...]] = {}
    manifest_routes: list[ManifestRoute] = []
    seen_route_keys: set[tuple[str, int]] = set()

    for requirement in route_requirements:
        call_spec = LLM_CALL_REGISTRY.get(requirement.call_id)
        if call_spec is None:
            raise LocalExecutionPolicyError(
                "local_unknown_call",
                f"Unknown LLM call '{requirement.call_id}'.",
            )
        if requirement.route_ordinal < 0 or (
            not call_spec.multi_model and requirement.route_ordinal != 0
        ):
            raise LocalExecutionPolicyError(
                "local_route_ordinal_invalid",
                f"Invalid ordinal {requirement.route_ordinal} for '{requirement.call_id}'.",
            )
        route_key = (requirement.call_id, requirement.route_ordinal)
        if route_key in seen_route_keys:
            raise LocalExecutionPolicyError(
                "local_manifest_duplicate_route",
                f"Manifest contains duplicate route '{requirement.call_id}' ordinal {requirement.route_ordinal}.",
            )
        seen_route_keys.add(route_key)
        model_info, endpoint = _resolve_model_and_endpoint(
            config_obj,
            effective_inventory,
            requirement,
        )
        provider = _normalize_provider(model_info.get("provider"))
        if endpoint.endpoint_id not in discovered_by_endpoint:
            discovered_by_endpoint[endpoint.endpoint_id] = await _discover_models(
                endpoint,
                config_obj=config_obj,
                model_sync_factory=model_sync_factory,
            )
        model_id = str(model_info.get("model_id") or requirement.model).strip()
        discovered_model = _find_announced_model(discovered_by_endpoint[endpoint.endpoint_id], model_id)
        if discovered_model is None:
            raise LocalExecutionPolicyError(
                "local_model_not_announced",
                f"Model '{model_id}' is not announced by endpoint '{endpoint.endpoint_id}'.",
            )
        verified_capabilities = _verified_capabilities(model_info, discovered_model)
        missing_capabilities = requirement.required_capabilities - verified_capabilities
        if missing_capabilities:
            raise LocalExecutionPolicyError(
                "local_capability_missing",
                f"Model '{model_id}' at endpoint '{endpoint.endpoint_id}' lacks required capabilities: "
                + ", ".join(sorted(missing_capabilities)),
            )
        manifest_routes.append(
            ManifestRoute(
                call_id=requirement.call_id,
                route_ordinal=requirement.route_ordinal,
                provider=provider,
                endpoint_id=endpoint.endpoint_id,
                model_id=model_id,
                capabilities_verified=verified_capabilities,
                adapter=endpoint.adapter,
            )
        )

    ordered_routes = tuple(sorted(manifest_routes, key=lambda route: (route.call_id, route.route_ordinal)))
    return LocalExecutionManifest(ordered_routes, _manifest_hash(ordered_routes))


def create_dispatch_context(
    manifest: LocalExecutionManifest,
    *,
    session_id: str,
    call_id: str,
    route_ordinal: int = 0,
) -> LocalDispatchContext:
    """Create context for a dispatch that ``authorize_local_dispatch`` validates."""

    if not str(session_id or "").strip():
        raise LocalExecutionPolicyError("local_session_id_missing", "A local-only dispatch requires a session id.")
    return LocalDispatchContext(str(session_id), manifest.manifest_hash, call_id, route_ordinal)


def authorize_local_dispatch(
    context: LocalDispatchContext,
    manifest: LocalExecutionManifest,
    ledger: SessionDispatchLedger,
    *,
    provider: str,
    model_id: str,
    endpoint_id: Optional[str] = None,
) -> ManifestRoute:
    """Authorize and ledger a transport dispatch before opening a connection."""

    normalized_provider = _normalize_provider(provider)
    normalized_model_id = str(model_id or "").strip()
    effective_endpoint_id = str(endpoint_id or "").strip() or None
    ledger_endpoint_id: Optional[str] = effective_endpoint_id
    try:
        if context.session_id != ledger.session_id:
            raise LocalExecutionPolicyError(
                "local_dispatch_session_mismatch", "Dispatch context belongs to a different session."
            )
        if context.manifest_hash != manifest.manifest_hash:
            raise LocalExecutionPolicyError(
                "local_manifest_hash_mismatch", "Dispatch context does not match the admitted manifest."
            )
        route = manifest.route_for(context.call_id, context.route_ordinal)
        if ledger_endpoint_id is None:
            ledger_endpoint_id = route.endpoint_id
        if (
            route.provider != normalized_provider
            or route.model_id != normalized_model_id
            or (effective_endpoint_id is not None and route.endpoint_id != effective_endpoint_id)
        ):
            raise LocalExecutionPolicyError(
                "local_dispatch_not_authorized",
                f"Dispatch for '{context.call_id}' does not match its admitted provider/model route.",
            )
    except LocalExecutionPolicyError as exc:
        ledger.record(
            DispatchLedgerEntry(
                session_id=context.session_id,
                manifest_hash=context.manifest_hash,
                call_id=context.call_id,
                route_ordinal=context.route_ordinal,
                provider=normalized_provider,
                endpoint_id=ledger_endpoint_id,
                model_id=normalized_model_id,
                allowed=False,
                reason=exc.code,
            )
        )
        raise

    ledger.record(
        DispatchLedgerEntry(
            session_id=context.session_id,
            manifest_hash=context.manifest_hash,
            call_id=context.call_id,
            route_ordinal=context.route_ordinal,
            provider=normalized_provider,
            endpoint_id=ledger_endpoint_id,
            model_id=normalized_model_id,
            allowed=True,
        )
    )
    return route


def build_local_execution_attestation(
    manifest: LocalExecutionManifest,
    ledger: SessionDispatchLedger,
) -> LocalExecutionAttestation:
    """Derive a local-only attestation from the complete session ledger."""

    manifest_entries = tuple(
        entry for entry in ledger.entries if entry.manifest_hash == manifest.manifest_hash
    )
    allowed_entries = tuple(entry for entry in manifest_entries if entry.allowed)
    blocked_entries = tuple(entry for entry in manifest_entries if not entry.allowed)
    external_calls = sum(1 for entry in allowed_entries if entry.provider not in LOCAL_PROVIDERS)
    providers = tuple(sorted({entry.provider for entry in allowed_entries}))
    endpoints = tuple(sorted({entry.endpoint_id for entry in allowed_entries if entry.endpoint_id}))
    return LocalExecutionAttestation(
        local_only_verified=external_calls == 0 and not blocked_entries,
        external_provider_calls=external_calls,
        providers_used=providers,
        endpoints_used=endpoints,
        routing_manifest_hash=manifest.manifest_hash,
        dispatch_count=len(allowed_entries),
        blocked_dispatch_count=len(blocked_entries),
    )


async def _discover_models(
    endpoint: LocalEndpoint,
    *,
    config_obj: Any,
    model_sync_factory: Optional[Callable[[Any, Any], Any]],
) -> tuple[Mapping[str, Any], ...]:
    """Read one endpoint's models through ModelSyncService without catalog sync."""
    if model_sync_factory is None:
        from services.model_sync import ModelSyncService

        model_sync_factory = ModelSyncService
    try:
        service = model_sync_factory(config_obj, logger)
        models = await service.discover_local_endpoint(endpoint)
    except LocalExecutionPolicyError:
        raise
    except Exception as exc:
        raise LocalExecutionPolicyError(
            "local_endpoint_unavailable",
            f"Could not discover local endpoint '{endpoint.endpoint_id}': {exc}",
        ) from exc
    if not isinstance(models, Sequence) or isinstance(models, (str, bytes, bytearray)):
        raise LocalExecutionPolicyError(
            "local_discovery_invalid", f"Local discovery for endpoint '{endpoint.endpoint_id}' returned no models list."
        )
    return tuple(item for item in models if isinstance(item, Mapping))


def _find_announced_model(
    discovered_models: Iterable[Mapping[str, Any]], model_id: str
) -> Optional[Mapping[str, Any]]:
    for item in discovered_models:
        # ``fetch_remote_models`` returns a UI-oriented merge.  For full-sync
        # providers it appends catalog-only entries as ``status=stale`` and
        # ``local_only=True``.  They were not returned by the live endpoint
        # and therefore must never satisfy a local_only admission.  Missing
        # markers remain valid for minimal test doubles and older payloads.
        status = str(item.get("status") or "").strip().lower()
        if status == "stale" or item.get("local_only") is True:
            continue
        announced = {
            str(item.get(key) or "").strip()
            for key in ("model_id", "key", "id")
        }
        if model_id in announced:
            return item
    return None


def _verified_capabilities(
    model_info: Mapping[str, Any], discovered_model: Mapping[str, Any]
) -> frozenset[str]:
    """Return catalog-declared capabilities for a model proved live-present.

    Discovery proves identity and liveness.  Local endpoint APIs do not expose
    a uniform capability protocol, so the administrator-maintained model
    catalog remains the capability authority for this first Track A slice.
    """

    catalog_capabilities = model_info.get("capabilities") or []
    if not isinstance(catalog_capabilities, Sequence) or isinstance(
        catalog_capabilities, (str, bytes, bytearray)
    ):
        return frozenset()
    return frozenset(_normalize_capability(item) for item in catalog_capabilities)


def _manifest_hash(routes: Sequence[ManifestRoute]) -> str:
    canonical = {
        "scope": LOCAL_ONLY_SCOPE,
        "routes": [route.canonical_payload() for route in routes],
    }
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _normalize_provider(provider: Any) -> str:
    normalized = str(provider or "").strip().lower()
    if normalized == "llama.cpp":
        return "llamacpp"
    return normalized


def _endpoint_supports_provider(endpoint: LocalEndpoint, provider: str) -> bool:
    """Return whether the explicit adapter is intentionally usable for provider."""

    compatible_adapters = {
        "ollama": frozenset({"ollama"}),
        "llamacpp": frozenset({"llamacpp", "openai_compatible"}),
        "openai_compatible": frozenset({"openai_compatible"}),
    }
    return endpoint.adapter in compatible_adapters.get(provider, frozenset())


def _resolve_model_and_endpoint(
    config_obj: Any,
    inventory: LocalEndpointInventory,
    requirement: LocalRouteRequirement,
) -> tuple[Mapping[str, Any], LocalEndpoint]:
    """Resolve catalog identity without letting a colliding model choose an endpoint.

    An explicit endpoint is an administrative identity boundary. Its adapter
    therefore selects the catalog namespace before an unqualified model name is
    looked up globally. This lets two local adapters intentionally serve the
    same model ID while retaining fail-closed routing.
    """

    if requirement.endpoint_id:
        endpoint = _configured_endpoint(inventory, requirement.endpoint_id)
        model_info = _model_info_for_endpoint(config_obj, requirement.model, endpoint)
        provider = _normalize_provider(model_info.get("provider"))
        _validate_local_provider(provider, requirement, endpoint)
        return model_info, endpoint

    try:
        model_info = config_obj.get_model_info(requirement.model, require_api_key=False)
    except Exception as global_error:
        candidates: list[tuple[Mapping[str, Any], LocalEndpoint]] = []
        for endpoint in inventory:
            try:
                candidate_info = _model_info_for_endpoint(config_obj, requirement.model, endpoint)
            except Exception:
                continue
            provider = _normalize_provider(candidate_info.get("provider"))
            if provider in LOCAL_PROVIDERS and _endpoint_supports_provider(endpoint, provider):
                candidates.append((candidate_info, endpoint))
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise LocalExecutionPolicyError(
                "local_endpoint_ambiguous",
                f"Model '{requirement.model}' is available through more than one local endpoint; routing must specify endpoint_id.",
            ) from global_error
        raise LocalExecutionPolicyError(
            "local_model_unknown",
            f"Could not resolve local model '{requirement.model}' for '{requirement.call_id}': {global_error}",
        ) from global_error

    provider = _normalize_provider(model_info.get("provider"))
    if provider not in LOCAL_PROVIDERS:
        raise LocalExecutionPolicyError(
            "local_provider_required",
            f"Route '{requirement.call_id}' resolves to non-local provider '{provider or 'unknown'}'.",
        )
    return model_info, _resolve_endpoint_for_requirement(inventory, provider=provider, endpoint_id=None)


def _configured_endpoint(inventory: LocalEndpointInventory, endpoint_id: str) -> LocalEndpoint:
    """Get an administrator-registered endpoint with a policy error envelope."""

    try:
        return inventory.get(endpoint_id)
    except LocalEndpointConfigurationError as exc:
        raise LocalExecutionPolicyError("local_endpoint_unconfigured", str(exc)) from exc


def _model_info_for_endpoint(
    config_obj: Any,
    model_name: str,
    endpoint: LocalEndpoint,
) -> Mapping[str, Any]:
    """Resolve a model using an endpoint's intentional adapter namespace first."""

    if _model_name_is_provider_qualified(model_name):
        return config_obj.get_model_info(model_name, require_api_key=False)

    provider_hints = {
        "ollama": ("ollama",),
        "llamacpp": ("llamacpp",),
        # A generic OpenAI-compatible endpoint has its own catalog category.
        # The llamacpp fallback is retained only for this explicit endpoint
        # compatibility path and only after the native category did not match.
        "openai_compatible": ("openai_compatible", "llamacpp"),
    }.get(endpoint.adapter, ())
    last_error: Optional[Exception] = None
    for provider_hint in provider_hints:
        try:
            return config_obj.get_model_info(
                f"{provider_hint}::{model_name}",
                require_api_key=False,
            )
        except Exception as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise LocalExecutionPolicyError(
        "local_endpoint_adapter_mismatch",
        f"Endpoint '{endpoint.endpoint_id}' has unsupported adapter '{endpoint.adapter}'.",
    )


def _model_name_is_provider_qualified(model_name: str) -> bool:
    """Recognize only the mechanical local-provider qualification syntaxes."""

    value = str(model_name or "").strip()
    if "::" in value:
        return True
    return value.startswith(("ollama/", "llamacpp/", "openai_compatible/"))


def _validate_local_provider(
    provider: str,
    requirement: LocalRouteRequirement,
    endpoint: LocalEndpoint,
) -> None:
    """Ensure a catalog result remains local and adapter-compatible."""

    if provider not in LOCAL_PROVIDERS:
        raise LocalExecutionPolicyError(
            "local_provider_required",
            f"Route '{requirement.call_id}' resolves to non-local provider '{provider or 'unknown'}'.",
        )
    if not _endpoint_supports_provider(endpoint, provider):
        raise LocalExecutionPolicyError(
            "local_endpoint_adapter_mismatch",
            f"Endpoint '{endpoint.endpoint_id}' adapter '{endpoint.adapter}' is not compatible with provider '{provider}'.",
        )


def _resolve_endpoint_for_requirement(
    inventory: LocalEndpointInventory,
    *,
    provider: str,
    endpoint_id: Optional[str],
) -> LocalEndpoint:
    """Resolve one routed local model to an explicit or unambiguous endpoint."""

    if endpoint_id:
        endpoint = _configured_endpoint(inventory, endpoint_id)
        if not _endpoint_supports_provider(endpoint, provider):
            raise LocalExecutionPolicyError(
                "local_endpoint_adapter_mismatch",
                f"Endpoint '{endpoint.endpoint_id}' adapter '{endpoint.adapter}' is not compatible with provider '{provider}'.",
            )
        return endpoint

    candidates = tuple(endpoint for endpoint in inventory if _endpoint_supports_provider(endpoint, provider))
    if not candidates:
        raise LocalExecutionPolicyError(
            "local_endpoint_unconfigured",
            f"No trusted local endpoint is configured for provider '{provider}'.",
        )
    if len(candidates) != 1:
        raise LocalExecutionPolicyError(
            "local_endpoint_ambiguous",
            f"Provider '{provider}' has more than one compatible trusted endpoint; routing must specify endpoint_id.",
        )
    return candidates[0]


def _normalize_capability(value: Any) -> str:
    return str(value or "").strip().lower()
