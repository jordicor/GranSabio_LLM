"""Regression tests for SaaS generation admission idempotency."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from core import generation_routes
from core.generation.idempotency import (
    GenerationIdempotencyRegistry,
    IdempotencyConflictError,
)
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
