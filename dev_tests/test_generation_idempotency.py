"""Regression tests for SaaS generation admission idempotency."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from core import generation_routes
from core.generation.idempotency import (
    GenerationIdempotencyRegistry,
    IdempotencyCapacityError,
    IdempotencyConflictError,
)
from local_execution_policy import LocalExecutionPolicyError
from models import ContentRequest, GenerationInitResponse


def _request(prompt: str = "Write a useful test article") -> ContentRequest:
    return ContentRequest(prompt=prompt)


def _response(session_id: str = "session-1") -> GenerationInitResponse:
    return GenerationInitResponse(
        status="initialized",
        session_id=session_id,
        project_id=session_id,
    )


@pytest.mark.asyncio
async def test_registry_waits_then_replays_completed_response() -> None:
    registry = GenerationIdempotencyRegistry(ttl_seconds=600, max_entries=100)
    owner = await registry.claim("request-1", "fingerprint")
    duplicate = await registry.claim("request-1", "fingerprint")

    assert owner.action == "execute"
    assert duplicate.action == "wait"

    await registry.complete(owner.entry, _response().model_dump(mode="json"))
    await registry.wait(duplicate.entry)
    replay = await registry.claim("request-1", "fingerprint")

    assert replay.action == "replay"
    assert replay.entry.response_payload["session_id"] == "session-1"


@pytest.mark.asyncio
async def test_registry_rejects_request_id_reuse_with_changed_payload() -> None:
    registry = GenerationIdempotencyRegistry(ttl_seconds=600, max_entries=100)
    await registry.claim("request-1", "first")

    with pytest.raises(IdempotencyConflictError):
        await registry.claim("request-1", "second")


@pytest.mark.asyncio
async def test_generate_replays_sequential_duplicate(monkeypatch: pytest.MonkeyPatch) -> None:
    await generation_routes.generation_idempotency_registry.clear()
    generate_once = AsyncMock(return_value=_response())
    monkeypatch.setattr(generation_routes, "_generate_content_once", generate_once)

    first = await generation_routes.generate_content(_request(), "saas-sequential")
    second = await generation_routes.generate_content(_request(), "saas-sequential")

    assert first == second
    assert second.session_id == "session-1"
    generate_once.assert_awaited_once()


@pytest.mark.asyncio
async def test_generate_deduplicates_concurrent_submissions(monkeypatch: pytest.MonkeyPatch) -> None:
    await generation_routes.generation_idempotency_registry.clear()
    entered = asyncio.Event()
    release = asyncio.Event()
    call_count = 0

    async def generate_once(_request_value: ContentRequest) -> GenerationInitResponse:
        nonlocal call_count
        call_count += 1
        entered.set()
        await release.wait()
        return _response("session-concurrent")

    monkeypatch.setattr(generation_routes, "_generate_content_once", generate_once)
    first_task = asyncio.create_task(
        generation_routes.generate_content(_request(), "saas-concurrent")
    )
    await entered.wait()
    second_task = asyncio.create_task(
        generation_routes.generate_content(_request(), "saas-concurrent")
    )
    await asyncio.sleep(0)
    release.set()

    first, second = await asyncio.gather(first_task, second_task)

    assert first.session_id == second.session_id == "session-concurrent"
    assert call_count == 1


@pytest.mark.asyncio
async def test_generate_rejects_changed_payload_for_same_request_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await generation_routes.generation_idempotency_registry.clear()
    generate_once = AsyncMock(return_value=_response())
    monkeypatch.setattr(generation_routes, "_generate_content_once", generate_once)
    await generation_routes.generate_content(_request(), "saas-conflict")

    with pytest.raises(HTTPException) as caught:
        await generation_routes.generate_content(
            _request("Write a materially different test article"),
            "saas-conflict",
        )

    assert caught.value.status_code == 409
    generate_once.assert_awaited_once()


@pytest.mark.asyncio
async def test_lookup_returns_reconciliation_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    await generation_routes.generation_idempotency_registry.clear()
    monkeypatch.setattr(
        generation_routes,
        "_generate_content_once",
        AsyncMock(return_value=_response("session-lookup")),
    )
    await generation_routes.generate_content(_request(), "saas-lookup")

    snapshot = await generation_routes.get_saas_generation_request("saas-lookup")
    missing = await generation_routes.get_saas_generation_request("saas-missing")

    assert snapshot == {
        "found": True,
        "request_id": "saas-lookup",
        "state": "completed",
        "status": "initialized",
        "session_id": "session-lookup",
        "project_id": "session-lookup",
        "init_response": _response("session-lookup").model_dump(mode="json"),
    }
    assert missing == {"found": False, "request_id": "saas-missing"}


@pytest.mark.asyncio
async def test_direct_call_without_header_bypasses_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await generation_routes.generation_idempotency_registry.clear()
    generate_once = AsyncMock(side_effect=[_response("session-a"), _response("session-b")])
    monkeypatch.setattr(generation_routes, "_generate_content_once", generate_once)

    first = await generation_routes.generate_content(_request())
    second = await generation_routes.generate_content(_request())

    assert first.session_id == "session-a"
    assert second.session_id == "session-b"
    assert generate_once.await_count == 2


def _capacity_refusal() -> generation_routes.TransientAdmissionRefusal:
    return generation_routes.TransientAdmissionRefusal(
        status_code=503,
        detail="Generation capacity reached: too many active sessions. Retry later.",
        headers={"Retry-After": "5"},
    )


@pytest.mark.asyncio
async def test_capacity_refusal_is_not_stored_under_the_request_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await generation_routes.generation_idempotency_registry.clear()
    generate_once = AsyncMock(side_effect=[_capacity_refusal(), _response("session-after-capacity")])
    monkeypatch.setattr(generation_routes, "_generate_content_once", generate_once)

    with pytest.raises(HTTPException) as caught:
        await generation_routes.generate_content(_request(), "saas-capacity")

    assert caught.value.status_code == 503
    assert caught.value.headers == {"Retry-After": "5"}
    assert await generation_routes.get_saas_generation_request("saas-capacity") == {
        "found": False,
        "request_id": "saas-capacity",
    }

    admitted = await generation_routes.generate_content(_request(), "saas-capacity")
    replayed = await generation_routes.generate_content(_request(), "saas-capacity")

    assert admitted.session_id == replayed.session_id == "session-after-capacity"
    assert generate_once.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        HTTPException(status_code=400, detail="llm_accent_guard is not supported"),
        HTTPException(status_code=409, detail="Project hard stop is still in progress"),
        HTTPException(status_code=422, detail=[{"loc": ["body", "prompt"], "msg": "invalid"}]),
        HTTPException(status_code=503, detail="A plain 503 the payload determined"),
    ],
)
async def test_payload_determined_errors_are_still_replayed(
    monkeypatch: pytest.MonkeyPatch,
    error: HTTPException,
) -> None:
    await generation_routes.generation_idempotency_registry.clear()
    generate_once = AsyncMock(side_effect=error)
    monkeypatch.setattr(generation_routes, "_generate_content_once", generate_once)

    for _attempt in range(2):
        with pytest.raises(HTTPException) as caught:
            await generation_routes.generate_content(_request(), "saas-rejected")
        assert caught.value.status_code == error.status_code
        assert caught.value.detail == error.detail

    generate_once.assert_awaited_once()
    snapshot = await generation_routes.get_saas_generation_request("saas-rejected")
    assert snapshot["state"] == "error"
    assert snapshot["error"] == {"status_code": error.status_code, "detail": error.detail}


@pytest.mark.asyncio
async def test_preflight_rejection_response_is_replayed(monkeypatch: pytest.MonkeyPatch) -> None:
    await generation_routes.generation_idempotency_registry.clear()
    rejected = GenerationInitResponse(status="preflight_rejected", session_id=None, project_id="project-1")
    generate_once = AsyncMock(return_value=rejected)
    monkeypatch.setattr(generation_routes, "_generate_content_once", generate_once)

    first = await generation_routes.generate_content(_request(), "saas-preflight")
    second = await generation_routes.generate_content(_request(), "saas-preflight")

    assert first == second == rejected
    generate_once.assert_awaited_once()


@pytest.mark.asyncio
async def test_duplicate_waiter_reclaims_after_capacity_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await generation_routes.generation_idempotency_registry.clear()
    entered = asyncio.Event()
    release = asyncio.Event()
    call_count = 0

    async def generate_once(_request_value: ContentRequest) -> GenerationInitResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            entered.set()
            await release.wait()
            raise _capacity_refusal()
        return _response("session-reclaimed")

    monkeypatch.setattr(generation_routes, "_generate_content_once", generate_once)
    first_task = asyncio.create_task(
        generation_routes.generate_content(_request(), "saas-reclaim")
    )
    await entered.wait()
    second_task = asyncio.create_task(
        generation_routes.generate_content(_request(), "saas-reclaim")
    )
    third_task = asyncio.create_task(
        generation_routes.generate_content(_request(), "saas-reclaim")
    )
    await asyncio.sleep(0)
    release.set()

    first, second, third = await asyncio.gather(
        first_task, second_task, third_task, return_exceptions=True
    )

    assert isinstance(first, HTTPException)
    assert first.status_code == 503
    assert second.session_id == third.session_id == "session-reclaimed"
    assert call_count == 2


@pytest.mark.asyncio
async def test_registry_full_is_a_transient_refusal_that_stores_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await generation_routes.generation_idempotency_registry.clear()
    generate_once = AsyncMock(return_value=_response("session-after-full"))
    monkeypatch.setattr(generation_routes, "_generate_content_once", generate_once)
    real_claim = generation_routes.generation_idempotency_registry.claim
    monkeypatch.setattr(
        generation_routes.generation_idempotency_registry,
        "claim",
        AsyncMock(side_effect=IdempotencyCapacityError()),
    )

    with pytest.raises(generation_routes.TransientAdmissionRefusal) as caught:
        await generation_routes.generate_content(_request(), "saas-full")

    assert caught.value.status_code == 503
    generate_once.assert_not_awaited()
    monkeypatch.setattr(generation_routes.generation_idempotency_registry, "claim", real_claim)
    assert await generation_routes.get_saas_generation_request("saas-full") == {
        "found": False,
        "request_id": "saas-full",
    }
    admitted = await generation_routes.generate_content(_request(), "saas-full")
    assert admitted.session_id == "session-after-full"
    generate_once.assert_awaited_once()


@pytest.mark.parametrize(
    "code",
    ["local_endpoint_unavailable", "local_model_not_announced", "local_discovery_invalid"],
)
def test_local_runtime_unavailability_is_a_transient_refusal(code: str) -> None:
    error = generation_routes._local_policy_http_error(LocalExecutionPolicyError(code, "detail"))

    assert isinstance(error, generation_routes.TransientAdmissionRefusal)
    assert error.status_code == 503
    assert error.detail == {"code": code, "detail": "detail"}


@pytest.mark.parametrize(
    "code",
    ["local_capability_missing", "local_feedback_memory_unsupported", "local_unknown_call"],
)
def test_local_policy_payload_errors_stay_deterministic(code: str) -> None:
    error = generation_routes._local_policy_http_error(LocalExecutionPolicyError(code, "detail"))

    assert not isinstance(error, generation_routes.TransientAdmissionRefusal)
    assert error.status_code == 400
