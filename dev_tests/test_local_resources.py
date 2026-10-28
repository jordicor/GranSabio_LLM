"""Offline GPU coordination tests: temporary state and fake/native HTTP backends only."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI

import json_utils as json
from ai_service import _ollama_concurrency_depth, find_terminal_refusal
from local_endpoint_registry import LocalEndpointRegistry, LocalInferenceEndpoint
from local_resource_backend import BARRIER_PROTOCOL, LocalResourceBackend, LocalResourceError
from local_resources import LocalResourceManager, LocalResourceSpec, load_resource_specs


class OfflineBackend:
    def __init__(self):
        self.models = [{"name": "model:latest", "size_vram": 10}]
        self.idle = True
        self.released = True
        self.release_done = True
        self.instance_id = "comfy-process-1"
        self.unloads = 0
        self.unload_entered = None
        self.unload_continue = None

    async def loaded_models(self, endpoint):
        return list(self.models)

    async def unload_ollama(self, endpoint, model=None):
        self.unloads += 1
        if self.unload_entered:
            self.unload_entered.set()
            await self.unload_continue.wait()
        self.models = [item for item in self.models if model is not None and item["name"] != model]
        return True

    async def barrier_idle(self, url, gpu_uuid):
        return self.idle

    async def barrier_status(self, url, gpu_uuid):
        return {"protocol": BARRIER_PROTOCOL, "instance_id": self.instance_id,
                "gpu_uuid": gpu_uuid, "idle": self.idle, "memory_released": self.released}

    async def release_barrier(self, url, gpu_uuid, request_id, instance_id):
        if self.instance_id != instance_id:
            raise LocalResourceError("resource_service_instance_changed", "offline restart")
        if not self.idle:
            raise LocalResourceError("resource_busy", "offline active queue")
        return self.release_done

    async def close(self):
        pass


def registry():
    return LocalEndpointRegistry((
        LocalInferenceEndpoint("local_primary", "ollama", "http://ollama.test:11434"),
        LocalInferenceEndpoint("other", "ollama", "http://other.test:11434"),
    ), "offline")


def specs():
    return (LocalResourceSpec(id="gpu", gpu_uuid="GPU-offline", endpoint_ids=("local_primary",), comfyui_url="http://comfy.test:8188"),)


@pytest.fixture
async def manager(tmp_path):
    result = LocalResourceManager(specs(), registry(), tmp_path / "state.sqlite3", OfflineBackend())
    yield result
    await result.close()


async def reserve(manager, request_id=None):
    return await manager.operate("gpu", "reserve", request_id or str(uuid4()), purpose="comfyui")


@pytest.mark.asyncio
async def test_reservation_drains_admitted_work_and_rejects_new_work(manager):
    request_id = str(uuid4())
    async with manager.admission("local_primary"):
        pending = await reserve(manager, request_id)
        assert pending["state"] == "draining" and not pending["complete"]
        assert pending["reservation"]["reservation_id"] == request_id
        assert not pending["reservation"]["ready"]
        assert manager.backend.unloads == 0
        with pytest.raises(LocalResourceError, match="resource_busy"):
            async with manager.admission("local_primary"):
                pytest.fail("new local work entered")
        async with manager.admission("other"):
            pass
    ready = await reserve(manager, request_id)
    assert ready["state"] == "reserved" and ready["complete"]
    assert manager.backend.models == []
    assert await reserve(manager, request_id) == ready
    assert manager.backend.unloads == 1
    with pytest.raises(LocalResourceError, match="resource_busy"):
        await reserve(manager)


@pytest.mark.asyncio
async def test_parallel_acquire_has_one_owner_without_blocking_http_wait(manager):
    manager.backend.unload_entered = asyncio.Event()
    manager.backend.unload_continue = asyncio.Event()
    request_id = str(uuid4())
    first = asyncio.create_task(reserve(manager, request_id))
    await asyncio.wait_for(manager.backend.unload_entered.wait(), 1)
    assert not (await reserve(manager, request_id))["complete"]
    with pytest.raises(LocalResourceError, match="resource_busy"):
        await reserve(manager)
    with pytest.raises(LocalResourceError, match="resource_busy"):
        async with manager.admission("local_primary"):
            pass
    manager.backend.unload_continue.set()
    assert (await first)["complete"]


@pytest.mark.asyncio
async def test_manual_unload_refuses_busy_without_interrupting_work(manager):
    async with manager.admission("local_primary"):
        with pytest.raises(LocalResourceError, match="resource_busy"):
            await manager.operate("gpu", "unload", str(uuid4()), target="ollama", endpoint_id="local_primary")
        assert manager.backend.unloads == 0
    manager.backend.idle = False
    with pytest.raises(LocalResourceError, match="resource_busy"):
        await manager.operate("gpu", "unload", str(uuid4()), target="ollama", endpoint_id="local_primary")
    assert manager.backend.unloads == 0
    assert manager._state("gpu")["operation"] is None


@pytest.mark.asyncio
async def test_release_needs_correlated_memory_ack_and_does_not_resurrect_old_reservation(manager):
    request_id = str(uuid4())
    acquired = await reserve(manager, request_id)
    reservation_id = acquired["reservation"]["reservation_id"]
    release_id = str(uuid4())
    manager.backend.release_done = False
    pending = await manager.operate("gpu", "release", release_id, reservation_id=reservation_id)
    assert not pending["complete"]
    assert not (await reserve(manager, request_id))["complete"]
    with pytest.raises(LocalResourceError, match="resource_busy"):
        async with manager.admission("local_primary"):
            pass
    manager.backend.release_done = True
    released = await manager.operate("gpu", "release", release_id, reservation_id=reservation_id)
    assert released["state"] == "idle" and released["reservation"] is None
    assert await manager.operate("gpu", "release", release_id, reservation_id=reservation_id) == released
    old_retry = await reserve(manager, request_id)
    assert old_retry["state"] == "released" and old_retry["reservation"] is None
    async with manager.admission("local_primary"):
        pass


@pytest.mark.asyncio
async def test_restart_retains_ready_reservation_and_idempotency(tmp_path):
    path = tmp_path / "state.sqlite3"
    first = LocalResourceManager(specs(), registry(), path, OfflineBackend())
    request_id = str(uuid4())
    acquired = await reserve(first, request_id)
    await first.close()
    second = LocalResourceManager(specs(), registry(), path, OfflineBackend())
    try:
        assert await reserve(second, request_id) == acquired
        with pytest.raises(LocalResourceError, match="resource_busy"):
            async with second.admission("local_primary"):
                pass
        with pytest.raises(LocalResourceError, match="idempotency_conflict"):
            await second.operate("gpu", "unload", request_id, target="ollama", endpoint_id="local_primary")
    finally:
        await second.close()


@pytest.mark.asyncio
async def test_cancelled_dispatch_and_interrupted_process_remain_closed(tmp_path):
    path = tmp_path / "state.sqlite3"
    first = LocalResourceManager(specs(), registry(), path, OfflineBackend())
    with pytest.raises(asyncio.CancelledError):
        async with first.admission("local_primary"):
            raise asyncio.CancelledError()
    assert first._state("gpu")["state"] == "uncertain"
    await first.close()
    second = LocalResourceManager(specs(), registry(), path, OfflineBackend())
    try:
        with pytest.raises(LocalResourceError, match="resource_busy"):
            async with second.admission("local_primary"):
                pass
        # A persisted in-flight counter models abrupt process termination:
        # normal completion never clears it after that process has died.
        second._persist("gpu", {**second._state("gpu"), "active_dispatches": 1, "state": "idle"})
    finally:
        await second.close()
    third = LocalResourceManager(specs(), registry(), path, OfflineBackend())
    try:
        assert third._state("gpu")["state"] == "uncertain"
        assert third._state("gpu")["active_dispatches"] == 1
    finally:
        await third.close()


@pytest.mark.asyncio
async def test_cancelled_inference_cannot_be_unloaded_by_a_pending_acquire_retry(manager):
    request_id = str(uuid4())
    with pytest.raises(asyncio.CancelledError):
        async with manager.admission("local_primary"):
            assert not (await reserve(manager, request_id))["complete"]
            raise asyncio.CancelledError()
    with pytest.raises(LocalResourceError, match="reconciliation_required"):
        await reserve(manager, request_id)
    assert manager.backend.unloads == 0


@pytest.mark.asyncio
async def test_uncertain_lifecycle_response_is_reconciled_by_original_request(manager):
    request_id = str(uuid4())
    manager.backend.unload_ollama = AsyncMock(side_effect=[TimeoutError("lost unload reply"), True])
    with pytest.raises(TimeoutError):
        await reserve(manager, request_id)
    assert manager._state("gpu")["state"] == "uncertain"
    with pytest.raises(LocalResourceError, match="resource_busy"):
        await reserve(manager)
    assert (await reserve(manager, request_id))["state"] == "reserved"


@pytest.mark.asyncio
async def test_changed_comfy_instance_never_releases_reservation(manager):
    acquired = await reserve(manager)
    manager.backend.instance_id = "replacement-process"
    with pytest.raises(LocalResourceError, match="resource_service_instance_changed"):
        await manager.operate("gpu", "release", str(uuid4()), reservation_id=acquired["reservation"]["reservation_id"])
    assert manager._state("gpu")["state"] == "uncertain"
    assert manager._state("gpu")["reservation"]


@pytest.mark.asyncio
async def test_duplicate_coordinator_and_changed_mapping_fail_closed(tmp_path):
    path = tmp_path / "state.sqlite3"
    first = LocalResourceManager(specs(), registry(), path, OfflineBackend())
    try:
        with pytest.raises(ValueError, match="single coordinator"):
            LocalResourceManager(specs(), registry(), path, OfflineBackend())
    finally:
        await first.close()
    changed = (specs()[0].model_copy(update={"gpu_uuid": "different-GPU"}),)
    with pytest.raises(ValueError, match="changed identity"):
        LocalResourceManager(changed, registry(), path, OfflineBackend())


@pytest.mark.asyncio
async def test_unverified_comfy_memory_blocks_first_local_dispatch(manager):
    manager.backend.released = False
    with pytest.raises(LocalResourceError, match="memory_not_released"):
        async with manager.admission("local_primary"):
            pass
    assert manager._state("gpu")["active_dispatches"] == 0


@pytest.mark.asyncio
async def test_predispatch_refusal_does_not_invent_uncertain_gpu_work(manager):
    from local_execution_policy import LocalExecutionPolicyError

    with pytest.raises(LocalExecutionPolicyError):
        async with manager.admission("local_primary"):
            raise LocalExecutionPolicyError("offline_refusal", "No provider dispatch occurred")
    assert manager._state("gpu")["state"] == "idle"
    assert manager._state("gpu")["active_dispatches"] == 0


@pytest.mark.parametrize("change", ["duplicate_endpoint", "duplicate_gpu", "arbitrary_url", "unknown_endpoint"])
def test_invalid_resource_mapping_is_rejected(tmp_path, change):
    resource = specs()[0].model_dump(mode="json")
    resources = [resource]
    if change == "duplicate_endpoint":
        resource["endpoint_ids"] *= 2
    elif change == "duplicate_gpu":
        resources.append({**resource, "id": "another", "endpoint_ids": ["other"]})
    elif change == "arbitrary_url":
        resource["external_service_url"] = "http://user:secret@comfy.test/?target=elsewhere"
    else:
        resource["endpoint_ids"] = ["unknown"]
    path = tmp_path / "resources.json"
    path.write_text(json.dumps({"resources": resources}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_resource_specs(str(path), registry())


@pytest.mark.asyncio
async def test_native_ollama_ack_is_not_unload_confirmation():
    requests = []

    def respond(request):
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(200, json={"models": [{"name": "model:latest", "size_vram": 42}]})
        assert json.loads(request.content) == {"model": "model:latest", "keep_alive": 0, "stream": False}
        return httpx.Response(200, json={"done": True})

    backend = LocalResourceBackend(httpx.AsyncClient(transport=httpx.MockTransport(respond)))
    try:
        assert not await backend.unload_ollama(registry().get("local_primary"))
        assert [(req.method, req.url.path) for req in requests] == [("GET", "/api/ps"), ("POST", "/api/generate"), ("GET", "/api/ps")]
    finally:
        await backend.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["pending", "wrong_request", "wrong_gpu", "wrong_instance", "busy", "failed", "completed"])
async def test_comfy_release_uses_worker_ack_not_http_ack(outcome):
    request_id = str(uuid4())
    status = {"protocol": BARRIER_PROTOCOL, "instance_id": "instance", "gpu_uuid": "GPU-offline", "idle": True, "memory_released": True}

    def respond(request):
        if request.method == "POST":
            return httpx.Response(200, json={"accepted": True})
        if request.url.path.endswith("resource-barrier"):
            return httpx.Response(200, json=status)
        ack = {**status, "request_id": request_id, "state": "completed"}
        if outcome == "pending":
            ack.update(state="pending", memory_released=False)
        elif outcome == "wrong_request":
            ack["request_id"] = str(uuid4())
        elif outcome == "wrong_gpu":
            ack["gpu_uuid"] = "wrong-gpu"
        elif outcome == "wrong_instance":
            ack["instance_id"] = "new-instance"
        elif outcome == "busy":
            ack["idle"] = False
        elif outcome == "failed":
            ack["state"] = "failed"
        return httpx.Response(200, json=ack)

    backend = LocalResourceBackend(httpx.AsyncClient(transport=httpx.MockTransport(respond)))
    try:
        if outcome.startswith("wrong") or outcome == "failed":
            with pytest.raises(LocalResourceError):
                await backend.release_barrier("http://comfy.test", "GPU-offline", request_id, "instance")
        else:
            assert await backend.release_barrier("http://comfy.test", "GPU-offline", request_id, "instance") is (outcome == "completed")
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_stock_comfyui_cannot_grant_reservation(tmp_path):
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(404, text="no barrier installed")

    backend = LocalResourceBackend(httpx.AsyncClient(transport=httpx.MockTransport(respond)))
    manager = LocalResourceManager(specs(), registry(), tmp_path / "state.sqlite3", backend)
    try:
        with pytest.raises(LocalResourceError, match="backend_unavailable"):
            await reserve(manager)
        assert not manager._state("gpu")["reservation"]["ready"]
        assert all(request.method == "GET" for request in calls)
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_client_cancels_pending_acquire_without_canceling_active_inference(manager):
    acquire_id, release_id = str(uuid4()), str(uuid4())
    async with manager.admission("local_primary"):
        assert not (await reserve(manager, acquire_id))["complete"]
        canceled = await manager.operate("gpu", "release", release_id, reservation_id=acquire_id)
        assert canceled["complete"] and canceled["state"] == "idle"
        assert canceled["reservation"] is None and canceled["active_dispatches"] == 1
        assert manager.backend.unloads == 0
        assert (await reserve(manager, acquire_id))["state"] == "canceled"
        assert await manager.operate("gpu", "release", release_id, reservation_id=acquire_id) == canceled
    assert manager._state("gpu")["active_dispatches"] == 0


@pytest.mark.asyncio
async def test_cancel_pending_unload_keeps_gate_closed_until_native_completion(manager):
    acquire_id, release_id = str(uuid4()), str(uuid4())
    manager.backend.unload_ollama = AsyncMock(side_effect=[False, False, True])
    assert not (await reserve(manager, acquire_id))["complete"]
    assert not (await manager.operate("gpu", "release", release_id, reservation_id=acquire_id))["complete"]
    with pytest.raises(LocalResourceError, match="resource_busy"):
        async with manager.admission("local_primary"):
            pass
    done = await manager.operate("gpu", "release", release_id, reservation_id=acquire_id)
    assert done["complete"] and done["reservation"] is None
    assert (await reserve(manager, acquire_id))["state"] == "canceled"


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("vision", [False, True])
@pytest.mark.parametrize("qa_slot_held", [False, True])
async def test_normal_generation_cannot_bypass_reserved_gpu(manager, monkeypatch, streaming, vision, qa_slot_held):
    from dev_tests.test_local_execution_transport import _install_config, _runtime, _service
    from models import ImageData

    create = AsyncMock()
    service = _service(monkeypatch, create)
    service._local_resource_manager = manager
    _install_config(monkeypatch, capabilities=("text", "vision"))
    await reserve(manager)
    token = _ollama_concurrency_depth.set(1 if qa_slot_held else 0)
    try:
        kwargs = {"prompt": "offline", "model": "writer-local", "local_execution": _runtime(), "dispatch_call_id": "generation.main"}
        if vision:
            kwargs["images"] = [ImageData(base64_data="aW1hZ2U=", mime_type="image/png", original_filename="offline.png", size_bytes=5)]
        with pytest.raises(LocalResourceError, match="resource_busy") as caught:
            if streaming:
                _ = [chunk async for chunk in service.generate_content_stream(**kwargs)]
            else:
                await service.generate_content(**kwargs)
        assert find_terminal_refusal(caught.value) is caught.value
        create.assert_not_awaited()
    finally:
        _ollama_concurrency_depth.reset(token)


@pytest.mark.asyncio
async def test_tool_turn_cannot_bypass_reserved_gpu(manager, monkeypatch):
    from dev_tests.test_local_execution_transport import _runtime, _service

    service = _service(monkeypatch, AsyncMock())
    service._local_resource_manager = manager
    service._serialize_tool_loop_tool_defs = lambda *_args, **_kwargs: ""
    service._build_openai_compatible_tool_params = lambda *_args, **_kwargs: {}
    service._get_tool_loop_call_budget = lambda *_args, **_kwargs: 4
    service._build_trace_metrics = lambda *_args, **_kwargs: {}
    turn = AsyncMock()
    monkeypatch.setattr(service, "_stream_openai_compatible_tool_turn", turn)
    await reserve(manager)
    with pytest.raises(LocalResourceError, match="resource_busy"):
        await service._run_openai_compatible_validation_tool_loop(
            provider="ollama", model_id="writer-local", prompt="offline",
            validation_callback=lambda _text: None, temperature=0.2, max_tokens=32,
            system_prompt="system", request_timeout=None, reasoning_effort=None,
            json_output=False, json_schema=None, usage_callback=None, usage_extra=None,
            images=None, max_rounds=1, retries_enabled=False, local_execution=_runtime(),
            dispatch_call_id="generation.main",
        )
    turn.assert_not_awaited()


@pytest.mark.asyncio
async def test_logprobs_probe_cannot_bypass_reserved_gpu(manager, monkeypatch):
    from dev_tests.test_local_execution_transport import _runtime, _service

    create = AsyncMock()
    service = _service(monkeypatch, create)
    service._local_resource_manager = manager
    monkeypatch.setattr("ai_service.config", SimpleNamespace(
        model_specs={}, get_model_info=lambda _model: {"provider": "ollama", "model_id": "writer-local"},
    ))
    monkeypatch.setattr("model_capability_registry.model_supports", lambda **_kwargs: SimpleNamespace(supported=True))
    await reserve(manager)
    with pytest.raises(LocalResourceError, match="resource_busy"):
        await service.generate_with_logprobs(
            prompt="offline", model="writer-local", endpoint_id="local_primary",
            local_execution=_runtime(), dispatch_call_id="generation.main",
        )
    create.assert_not_awaited()


@pytest.mark.asyncio
async def test_comfy_post_ack_is_pending_until_worker_finishes_and_is_not_reposted():
    request_id = str(uuid4())
    requests = []
    submitted = False
    finished = False
    status = {"protocol": BARRIER_PROTOCOL, "instance_id": "instance", "gpu_uuid": "GPU-offline", "idle": True, "memory_released": False}

    def respond(request):
        nonlocal submitted
        requests.append(request)
        if request.method == "POST":
            submitted = True
            return httpx.Response(200, json={"accepted": True})
        if request.url.path.endswith("resource-barrier"):
            return httpx.Response(200, json=status)
        if not submitted:
            return httpx.Response(404)
        return httpx.Response(200, json={**status, "request_id": request_id,
            "state": "completed" if finished else "pending", "memory_released": finished})

    backend = LocalResourceBackend(httpx.AsyncClient(transport=httpx.MockTransport(respond)))
    try:
        assert not await backend.release_barrier("http://comfy.test", "GPU-offline", request_id, "instance")
        finished = True
        assert await backend.release_barrier("http://comfy.test", "GPU-offline", request_id, "instance")
        assert sum(request.method == "POST" for request in requests) == 1
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_resource_api_validates_identity_busy_and_idempotency(manager, monkeypatch):
    from core import local_resource_routes as routes

    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[routes.resource_manager] = lambda: manager
    app.dependency_overrides[routes.require_internal_ip] = lambda: "127.0.0.1"
    monkeypatch.setattr(routes, "config", SimpleNamespace(LOCAL_RESOURCES_CONFIG_PATH="offline"))
    monkeypatch.setattr(routes, "get_ai_service", lambda: SimpleNamespace(_local_resource_manager=manager))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://offline") as client:
        response = await client.get("/local-resources")
        assert response.status_code == 200 and response.json()["resources"][0]["endpoint_ids"] == ["local_primary"]
        invalid = await client.post("/local-resources/gpu/reserve", json={"request_id": str(uuid4()), "url": "http://arbitrary"})
        assert invalid.status_code == 422
        request_id = str(uuid4())
        async with manager.admission("local_primary"):
            pending = await client.post("/local-resources/gpu/reserve", json={"request_id": request_id})
            assert pending.status_code == 202 and pending.headers["Retry-After"] == "1"
        ready = await client.post("/local-resources/gpu/reserve", json={"request_id": request_id})
        assert ready.status_code == 200 and ready.json()["state"] == "reserved"
        busy = await client.post("/local-resources/gpu/unload", json={"request_id": str(uuid4()), "target": "ollama", "endpoint_id": "local_primary"})
        assert busy.status_code == 409
        bad_release = await client.post("/local-resources/gpu/release", json={"request_id": str(uuid4()), "reservation_id": str(uuid4())})
        assert bad_release.status_code == 409


@pytest.mark.asyncio
async def test_unconfigured_resource_feature_does_not_construct_services(monkeypatch):
    from core import local_resource_routes as routes

    monkeypatch.setattr(routes, "config", SimpleNamespace(LOCAL_RESOURCES_CONFIG_PATH=""))
    monkeypatch.setattr(routes, "get_ai_service", lambda: pytest.fail("disabled resource feature constructed AI clients"))
    assert await routes.list_local_resources() == {"enabled": False, "resources": []}


@pytest.mark.asyncio
async def test_other_application_uses_same_verified_protocol_and_legacy_client_still_works(tmp_path):
    resource = LocalResourceSpec(id="gpu", gpu_uuid="GPU-offline", endpoint_ids=("local_primary",),
                                 external_service_url="http://renderer.test")
    manager = LocalResourceManager((resource,), registry(), tmp_path / "state.sqlite3", OfflineBackend())
    try:
        acquired = await manager.operate("gpu", "reserve", str(uuid4()), purpose="video-render")
        assert acquired["complete"] and acquired["reservation"]["purpose"] == "video-render"
        assert acquired["reservation"]["service_instance_id"] == acquired["reservation"]["comfy_instance_id"]
        inventory = (await manager.status())[0]
        assert inventory["external_service_url"] == inventory["comfyui_url"] == "http://renderer.test"
        assert inventory["external_service"] == inventory["comfyui"]
        await manager.operate("gpu", "release", str(uuid4()), reservation_id=acquired["reservation"]["reservation_id"])
        unloaded = await manager.operate("gpu", "unload", str(uuid4()), target="external")
        assert unloaded["complete"] and unloaded["state"] == "idle"
    finally:
        await manager.close()


def test_resource_config_accepts_legacy_alias_but_rejects_ambiguous_urls():
    from core.local_resource_routes import ReserveRequest, UnloadRequest

    original = specs()[0]
    assert original.external_service_url == "http://comfy.test:8188"
    assert ReserveRequest(request_id=uuid4(), purpose="speech-synthesis").purpose == "speech-synthesis"
    assert UnloadRequest(request_id=uuid4(), target="comfyui").target == "comfyui"
    with pytest.raises(ValueError):
        LocalResourceSpec(**original.model_dump(), comfyui_url="http://another.test")
    with pytest.raises(ValueError):
        ReserveRequest(request_id=uuid4(), purpose="   ")
