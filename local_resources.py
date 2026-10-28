"""Opt-in, durable ownership of physical GPUs shared by local inference services.

One coordinator process owns each state file. Reservations never expire. A
crashed or canceled inference leaves admission closed for operator recovery.
No lifecycle request is issued merely by importing or constructing this module.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit
from uuid import UUID

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

import json_utils as json
from context_budget import ContextAdmissionError
from local_endpoint_registry import LocalEndpointRegistry
from local_execution_policy import LocalExecutionPolicyError
from local_resource_backend import LocalResourceBackend, LocalResourceError


class LocalResourceSpec(BaseModel):
    """An operator's explicit physical-resource and service mapping."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    gpu_uuid: str = Field(min_length=1, max_length=128)
    endpoint_ids: tuple[str, ...] = Field(min_length=1)
    external_service_url: str | None = Field(
        default=None, validation_alias=AliasChoices("external_service_url", "comfyui_url")
    )

    @field_validator("external_service_url")
    @classmethod
    def validate_url(cls, value: str | None) -> str | None:
        """Reject credentials and arbitrary query/fragment URL components."""
        if value is None:
            return None
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("External service URL must be absolute HTTP(S) without credentials, query or fragment")
        _ = parsed.port
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))


def load_resource_specs(path: str, registry: LocalEndpointRegistry) -> tuple[LocalResourceSpec, ...]:
    """Validate the complete resource map before creating any runtime state."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data) != {"resources"} or not isinstance(data["resources"], list):
        raise ValueError("Local resource configuration must contain only a resources array")
    specs = tuple(LocalResourceSpec.model_validate(item) for item in data["resources"])
    seen_ids, seen_gpus, seen_endpoints, seen_urls = set(), set(), set(), set()
    for spec in specs:
        if spec.id in seen_ids or spec.gpu_uuid in seen_gpus:
            raise ValueError("A physical GPU must have exactly one resource definition")
        seen_ids.add(spec.id)
        seen_gpus.add(spec.gpu_uuid)
        if spec.external_service_url and spec.external_service_url in seen_urls:
            raise ValueError("An external service cannot belong to multiple resources")
        seen_urls.add(spec.external_service_url)
        for endpoint_id in spec.endpoint_ids:
            registry.get(endpoint_id)
            if endpoint_id in seen_endpoints:
                raise ValueError("A local endpoint cannot belong to multiple resources")
            seen_endpoints.add(endpoint_id)
    return specs


class _ProcessLock:
    """Hold an OS lock so a second engine cannot bypass this coordinator."""

    def __init__(self, path: Path):
        self.file = path.open("a+b")
        self.file.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.file.close()
            raise ValueError("Local resources require a single coordinator process for this state file") from exc

    def close(self) -> None:
        self.file.close()


class LocalResourceManager:
    """Serialize physical-resource admission and idempotent lifecycle operations."""

    def __init__(
        self, specs: tuple[LocalResourceSpec, ...], registry: LocalEndpointRegistry,
        state_path: str | Path, backend: LocalResourceBackend | None = None,
    ):
        self.specs = {item.id: item for item in specs}
        self.registry = registry
        self.endpoint_resources = {endpoint: item.id for item in specs for endpoint in item.endpoint_ids}
        self.locks = {item.id: asyncio.Lock() for item in specs}
        self.closed = False
        path = Path(state_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.process_lock = _ProcessLock(path.with_suffix(path.suffix + ".lock"))
        try:
            self.db = sqlite3.connect(path)
            self.db.execute("CREATE TABLE IF NOT EXISTS resources (id TEXT PRIMARY KEY, config_hash TEXT NOT NULL, data TEXT NOT NULL)")
            self.db.execute("CREATE TABLE IF NOT EXISTS operations (id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, result TEXT)")
            for spec in specs:
                # Include endpoint URLs/adapters, not credentials: changing a
                # target must not silently carry an old reservation elsewhere.
                identity = {"resource": spec.model_dump(), "endpoints": [
                    {"id": eid, "url": registry.get(eid).base_url, "adapter": registry.get(eid).adapter}
                    for eid in spec.endpoint_ids
                ]}
                digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
                row = self.db.execute("SELECT config_hash, data FROM resources WHERE id=?", (spec.id,)).fetchone()
                if row:
                    if row[0] != digest:
                        raise ValueError(f"Resource {spec.id} changed identity; reconcile its existing state before reconfiguration")
                    state = json.loads(row[1])
                    state.setdefault("requires_reconciliation", False)
                    if state["active_dispatches"]:
                        state["state"] = "uncertain"
                        state["requires_reconciliation"] = True
                        state["last_error"] = "Inference was active at process exit; operator reconciliation is required."
                    elif state["state"] == "idle":
                        state["state"] = "unverified"
                    elif state["state"] in {"unloading", "releasing"}:
                        state["state"] = "uncertain"
                        state["last_error"] = "Lifecycle operation interrupted; retry its original request_id to verify it."
                    self._save(spec.id, state)
                else:
                    state = {"state": "unverified", "active_dispatches": 0, "operation": None,
                             "reservation": None, "last_error": None, "requires_reconciliation": False}
                    self.db.execute("INSERT INTO resources VALUES (?, ?, ?)", (spec.id, digest, json.dumps(state)))
            for resource_id, raw in self.db.execute("SELECT id, data FROM resources"):
                if resource_id not in self.specs and json.loads(raw)["state"] not in {"idle", "unverified"}:
                    raise ValueError(f"Resource {resource_id} was removed while it still needs reconciliation")
            self.db.commit()
        except BaseException:
            if hasattr(self, "db"):
                self.db.close()
            self.process_lock.close()
            raise
        self.backend = backend or LocalResourceBackend()

    def _state(self, resource_id: str) -> dict:
        if self.closed:
            raise LocalResourceError("resource_coordinator_closed", "The resource coordinator is closed.", 503)
        if resource_id not in self.specs:
            raise LocalResourceError("resource_unknown", "Unknown configured resource.", 404)
        return json.loads(self.db.execute("SELECT data FROM resources WHERE id=?", (resource_id,)).fetchone()[0])

    def _save(self, resource_id: str, state: dict) -> None:
        self.db.execute("UPDATE resources SET data=? WHERE id=?", (json.dumps(state), resource_id))

    def _persist(self, resource_id: str, state: dict) -> None:
        with self.db:
            self._save(resource_id, state)

    async def _verify_initial_idle(self, spec: LocalResourceSpec, state: dict) -> None:
        if state["state"] != "unverified":
            return
        if spec.external_service_url:
            status = await self.backend.barrier_status(spec.external_service_url, spec.gpu_uuid)
            if not status["idle"] or not status["memory_released"]:
                raise LocalResourceError("resource_memory_not_released", "The external service must confirm idle and released memory before local inference.")
        state["state"] = "idle"
        self._persist(spec.id, state)

    @asynccontextmanager
    async def admission(self, endpoint_id: str):
        """Hold a durable admission for the entire local transport call/stream."""
        resource_id = self.endpoint_resources.get(endpoint_id)
        if resource_id is None:
            yield
            return
        if self.locks[resource_id].locked():
            raise LocalResourceError("resource_busy", "A GPU lifecycle operation is in progress.")
        async with self.locks[resource_id]:
            state = self._state(resource_id)
            await self._verify_initial_idle(self.specs[resource_id], state)
            if state["state"] != "idle":
                raise LocalResourceError("resource_busy", f"The GPU is {state['state']}; local dispatch is closed.")
            state["active_dispatches"] += 1
            self._persist(resource_id, state)
        try:
            yield
        except (ContextAdmissionError, LocalExecutionPolicyError):
            # Deterministic admission refusals never reached the model backend.
            state = self._state(resource_id)
            state["active_dispatches"] -= 1
            self._persist(resource_id, state)
            raise
        except BaseException:
            state = self._state(resource_id)
            state["active_dispatches"] -= 1
            state["state"] = "uncertain"
            state["requires_reconciliation"] = True
            state["last_error"] = "Local transport did not finish normally; remote completion must be reconciled by the operator."
            self._persist(resource_id, state)
            raise
        else:
            state = self._state(resource_id)
            state["active_dispatches"] -= 1
            self._persist(resource_id, state)

    def _prior_result(self, request_id: str, fingerprint: str) -> dict | None:
        row = self.db.execute("SELECT fingerprint, result FROM operations WHERE id=?", (request_id,)).fetchone()
        if not row:
            return None
        if row[0] != fingerprint:
            raise LocalResourceError("idempotency_conflict", "request_id was already used for different input.")
        return json.loads(row[1]) if row[1] else None

    @staticmethod
    def _public_reservation(state: dict) -> dict | None:
        """Expose the neutral identity while retaining the first consumer's field."""
        reservation = state["reservation"]
        if reservation is None:
            return None
        return {**reservation, "service_instance_id": reservation["comfy_instance_id"]}

    @staticmethod
    def _response(resource_id: str, state: dict, request_id: str, *, complete: bool) -> dict:
        return {"resource_id": resource_id, "request_id": request_id, "complete": complete,
                "state": state["state"], "reservation": LocalResourceManager._public_reservation(state),
                "active_dispatches": state["active_dispatches"], "operation": state["operation"]}

    async def operate(
        self, resource_id: str, kind: Literal["reserve", "release", "unload"], request_id: str, **payload: Any,
    ) -> dict:
        """Advance one bounded operation; clients retry the same ID while pending."""
        request_id = str(UUID(request_id))
        state = self._state(resource_id)
        spec = self.specs[resource_id]
        fingerprint = json.dumps({"resource": resource_id, "kind": kind, "payload": payload}, sort_keys=True)
        prior = self._prior_result(request_id, fingerprint)
        if prior:
            if kind == "reserve":
                if not state["reservation"] or state["reservation"]["request_id"] != request_id:
                    return {**prior, "state": "canceled" if prior["state"] == "canceled" else "released", "reservation": None}
                if state["state"] != "reserved":
                    return self._response(resource_id, state, request_id, complete=False)
            return prior
        if self.locks[resource_id].locked():
            if state["operation"] and state["operation"]["request_id"] == request_id:
                return self._response(resource_id, state, request_id, complete=False)
            raise LocalResourceError("resource_busy", "A resource operation is in progress.")
        async with self.locks[resource_id]:
            state = self._state(resource_id)
            if state["requires_reconciliation"]:
                raise LocalResourceError("resource_reconciliation_required", "Prior local inference has uncertain completion; lifecycle operations remain closed.", 503)
            operation = state["operation"]
            canceling = (
                kind == "release" and operation and operation["kind"] == "reserve"
                and state["reservation"] and not state["reservation"]["ready"]
                and state["reservation"]["reservation_id"] == payload["reservation_id"]
            )
            if canceling:
                canceled_id = operation["request_id"]
                replacement = {
                    "request_id": request_id, "kind": "release", "payload": payload,
                    "previous_state": operation["previous_state"], "comfy_instance_id": None,
                    "cancel_pending": True, "unload_started": operation.get("unload_started", False),
                }
                canceled_result = {**self._response(resource_id, state, canceled_id, complete=True),
                                   "state": "canceled", "reservation": None, "operation": None}
                state["operation"] = replacement
                state["state"] = "releasing"
                with self.db:
                    self.db.execute("UPDATE operations SET result=? WHERE id=?", (json.dumps(canceled_result), canceled_id))
                    self.db.execute("INSERT INTO operations VALUES (?, ?, NULL)", (request_id, fingerprint))
                    self._save(resource_id, state)
                operation = replacement
            if operation and operation["request_id"] != request_id:
                raise LocalResourceError("resource_busy", "Another operation owns this resource.")
            if not operation:
                if kind == "release":
                    if state["state"] != "reserved" or not state["reservation"] or state["reservation"]["reservation_id"] != payload["reservation_id"]:
                        raise LocalResourceError("reservation_mismatch", "The release does not own this reservation.")
                elif state["state"] not in {"idle", "unverified"}:
                    raise LocalResourceError("resource_busy", f"The GPU is {state['state']}.")
                if kind == "unload" and state["active_dispatches"]:
                    raise LocalResourceError("resource_busy", "Local inference is active; unloading is refused.")
                if kind == "reserve":
                    if not spec.external_service_url:
                        raise LocalResourceError("resource_service_not_configured", "This resource has no configured external service.")
                    if any(self.registry.get(eid).adapter != "ollama" for eid in spec.endpoint_ids):
                        raise LocalResourceError("unload_adapter_unsupported", "External reservation requires unload support for every linked endpoint.")
                if kind == "unload":
                    if payload["target"] == "ollama":
                        if payload.get("endpoint_id") not in spec.endpoint_ids:
                            raise LocalResourceError("endpoint_not_linked", "The endpoint is not linked to this GPU.", 422)
                        if self.registry.get(payload["endpoint_id"]).adapter != "ollama":
                            raise LocalResourceError("unload_adapter_unsupported", "Model lifecycle requires an Ollama endpoint.")
                    elif not spec.external_service_url:
                        raise LocalResourceError("resource_service_not_configured", "This resource has no configured external service.")
                    if spec.external_service_url and not await self.backend.barrier_idle(spec.external_service_url, spec.gpu_uuid):
                        raise LocalResourceError("resource_busy", "The external service has running or queued work; unloading is refused.")
                operation = {"request_id": request_id, "kind": kind, "payload": payload,
                             "previous_state": state["state"], "comfy_instance_id": None, "unload_started": False}
                if kind == "reserve":
                    # The acquire UUID is also its stable reservation identity,
                    # allowing explicit cleanup after a lost/pending response.
                    state["reservation"] = {"reservation_id": request_id, "request_id": request_id,
                                            "purpose": payload["purpose"], "comfy_instance_id": None, "ready": False}
                state["operation"] = operation
                state["state"] = {"reserve": "draining", "release": "releasing", "unload": "unloading"}[kind]
                state["last_error"] = None
                with self.db:
                    self.db.execute("INSERT INTO operations VALUES (?, ?, NULL)", (request_id, fingerprint))
                    self._save(resource_id, state)
            if state["active_dispatches"] and not (operation.get("cancel_pending") and not operation["unload_started"]):
                return self._response(resource_id, state, request_id, complete=False)
            try:
                done = await self._advance(spec, state, operation)
            except BaseException as exc:
                state["state"] = "uncertain"
                state["last_error"] = exc.code if isinstance(exc, LocalResourceError) else "resource_operation_interrupted"
                self._persist(resource_id, state)
                raise
            if not done:
                self._persist(resource_id, state)
                return self._response(resource_id, state, request_id, complete=False)
            state["operation"] = None
            state["last_error"] = None
            result = self._response(resource_id, state, request_id, complete=True)
            with self.db:
                self._save(resource_id, state)
                self.db.execute("UPDATE operations SET result=? WHERE id=?", (json.dumps(result), request_id))
            return result

    async def _advance(self, spec: LocalResourceSpec, state: dict, operation: dict) -> bool:
        kind, payload = operation["kind"], operation["payload"]
        if operation.get("cancel_pending"):
            # Draining alone never touched either backend: cancel immediately
            # while the already-admitted inference continues undisturbed.
            if operation["unload_started"]:
                for endpoint_id in spec.endpoint_ids:
                    if not await self.backend.unload_ollama(self.registry.get(endpoint_id)):
                        return False
            state["reservation"] = None
            state["state"] = operation["previous_state"]
            return True
        if kind == "reserve":
            status = await self.backend.barrier_status(spec.external_service_url, spec.gpu_uuid)
            if not status["idle"]:
                raise LocalResourceError("resource_busy", "The external service has running or queued work.")
            operation["comfy_instance_id"] = status["instance_id"]
            operation["unload_started"] = True
            state["state"] = "unloading"
            self._persist(spec.id, state)
            for endpoint_id in spec.endpoint_ids:
                if not await self.backend.unload_ollama(self.registry.get(endpoint_id)):
                    return False
            state["reservation"]["comfy_instance_id"] = status["instance_id"]
            state["reservation"]["ready"] = True
            state["state"] = "reserved"
            return True
        if kind == "release" or payload["target"] in {"external", "comfyui"}:
            if kind == "release":
                instance_id = state["reservation"]["comfy_instance_id"]
            else:
                if not operation["comfy_instance_id"]:
                    status = await self.backend.barrier_status(spec.external_service_url, spec.gpu_uuid)
                    operation["comfy_instance_id"] = status["instance_id"]
                    self._persist(spec.id, state)
                instance_id = operation["comfy_instance_id"]
            if not await self.backend.release_barrier(spec.external_service_url, spec.gpu_uuid, operation["request_id"], instance_id):
                return False
            state["reservation"] = None
            state["state"] = "idle"
            return True
        if spec.external_service_url and not await self.backend.barrier_idle(spec.external_service_url, spec.gpu_uuid):
            raise LocalResourceError("resource_busy", "The external service has running or queued work; unloading is refused.")
        if not await self.backend.unload_ollama(self.registry.get(payload["endpoint_id"]), payload.get("model")):
            return False
        state["state"] = operation["previous_state"]
        return True

    async def status(self) -> list[dict]:
        """Expose configured identities and uncertainty without mutating services."""
        results = []
        for spec in self.specs.values():
            item = {**spec.model_dump(), **self._state(spec.id), "loaded_models": [], "backend_errors": []}
            # Existing ComfyUI consumers can keep their inventory lookup unchanged.
            item["comfyui_url"] = spec.external_service_url
            item["reservation"] = self._public_reservation(item)
            for endpoint_id in spec.endpoint_ids:
                endpoint = self.registry.get(endpoint_id)
                if endpoint.adapter != "ollama":
                    continue
                try:
                    item["loaded_models"].extend({**model, "endpoint_id": endpoint_id} for model in await self.backend.loaded_models(endpoint))
                except LocalResourceError as exc:
                    item["backend_errors"].append({"endpoint_id": endpoint_id, **exc.to_payload()})
            if spec.external_service_url:
                try:
                    item["external_service"] = await self.backend.barrier_status(spec.external_service_url, spec.gpu_uuid)
                except LocalResourceError as exc:
                    item["external_service"] = None
                    item["backend_errors"].append(exc.to_payload())
                item["comfyui"] = item["external_service"]
            results.append(item)
        return results

    async def close(self) -> None:
        """Close resources without erasing any reservation or uncertain work."""
        if not self.closed:
            self.closed = True
            try:
                await self.backend.close()
            finally:
                self.db.close()
                self.process_lock.close()


_manager: LocalResourceManager | None = None
_manager_identity: tuple | None = None


def get_local_resource_manager(config: Any, registry: LocalEndpointRegistry) -> LocalResourceManager | None:
    """Share one opt-in coordinator across all AIService instances in the process."""
    global _manager, _manager_identity
    config_path = str(getattr(config, "LOCAL_RESOURCES_CONFIG_PATH", "") or "").strip()
    if not config_path:
        if _manager is not None and not _manager.closed:
            raise ValueError("Local resource configuration cannot be disabled in a running coordinator")
        return None
    state_path = str(getattr(config, "LOCAL_RESOURCES_STATE_PATH", "data/local_resources.sqlite3"))
    specs = load_resource_specs(config_path, registry)
    identity = (str(Path(state_path).resolve()), tuple((spec.model_dump_json()) for spec in specs),
                tuple((ep.endpoint_id, ep.adapter, ep.base_url) for ep in registry))
    if _manager is not None and not _manager.closed:
        if identity != _manager_identity:
            raise ValueError("Local resource identity cannot change in a running coordinator")
        return _manager
    _manager = LocalResourceManager(specs, registry, state_path)
    _manager_identity = identity
    return _manager


async def close_local_resource_manager() -> None:
    """Close the process coordinator at engine shutdown."""
    if _manager is not None:
        await _manager.close()
