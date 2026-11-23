"""Focused offline regressions for the November 2026 security corrections."""

import asyncio
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import openai
import pytest
from fastapi import FastAPI

import json_utils
from ai_runtime.usage import normalize_usage
from ai_service import AIService
from core.app_state import app
from core.cancellation import CancellationRegistry
from core.generation_processor import _claim_gran_sabio_call
from feedback_memory import FeedbackDatabase
from personal_subscription_transport import overload_retry_delay
from provider_errors import ProviderErrorKind, classify_provider_exception


@pytest.mark.parametrize("ascii_only", [True, False])
def test_json_unicode_and_integer_keys_have_independent_options(ascii_only):
    payload = {7: "Espa\u00f1a \U0001f680 \u6d77", "escaped": "\\n\n"}
    encoded = json_utils.dumps_bytes(payload, ensure_ascii=ascii_only, sort_keys=True)
    assert encoded.isascii() is ascii_only
    assert json_utils.loads(encoded) == {"7": payload[7], "escaped": payload["escaped"]}
    with pytest.raises(TypeError):
        json_utils.dumps_bytes({"unsupported": object()}, ensure_ascii=ascii_only)


@pytest.mark.parametrize("error,kind", [
    (openai.APITimeoutError(request=httpx.Request("POST", "https://provider.invalid")), ProviderErrorKind.TIMEOUT),
    (openai.APIConnectionError(request=httpx.Request("POST", "https://provider.invalid")), ProviderErrorKind.TRANSIENT_NETWORK),
])
def test_sdk_transport_failures_use_types_and_hide_private_messages(error, kind):
    failure = classify_provider_exception(error, provider="openai", model_id="fixture", operation="generate")
    assert failure.kind == kind
    assert failure.retryable
    assert "provider.invalid" not in failure.message


@pytest.mark.parametrize("vectors", [[[]], [[0.0, 0.0]], [[float("nan"), 1.0]], [[True, 1.0]], [[1.0], [1.0, 2.0]]])
async def test_invalid_embeddings_fail_after_one_call_without_a_second_provider(monkeypatch, vectors):
    service = AIService.__new__(AIService)
    create = AsyncMock(return_value=SimpleNamespace(data=[
        SimpleNamespace(index=i, embedding=vector) for i, vector in enumerate(vectors)]))
    service.openai_client = SimpleNamespace(embeddings=SimpleNamespace(create=create))
    service.google_new_client = AsyncMock()
    service._admit_embedding_batch = AsyncMock()
    service._record_provider_health_failure = AsyncMock()
    monkeypatch.setattr("ai_service.resolve_embedding_model", lambda *a: ("fixture", "openai"))
    with pytest.raises(ValueError, match="Embedding response"):
        await service.get_embeddings(["fixture"] * len(vectors), model="fixture")
    assert create.await_count == 1
    assert not service.google_new_client.mock_calls


def test_gemini_cache_counts_survive_repeated_usage_normalization():
    usage = {"prompt_token_count": 100, "cached_content_token_count": 60,
             "candidates_token_count": 10, "thoughts_token_count": 5, "total_token_count": 115}
    result = normalize_usage(usage)
    assert result["input_tokens"] == 100
    assert result["cached_input_tokens"] == 60
    assert result["output_tokens"] == 15
    assert normalize_usage(result) == result


async def test_denied_client_is_rejected_without_receiving_its_request_body():
    # Exercise the actual registered middleware order without starting providers.
    isolated = FastAPI(middleware=app.user_middleware)
    receive = AsyncMock(side_effect=AssertionError("Denied request body must not be read"))
    send = AsyncMock()
    await isolated({"type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"},
                    "http_version": "1.1", "method": "POST", "scheme": "http", "path": "/generate",
                    "raw_path": b"/generate", "query_string": b"", "root_path": "",
                    "headers": [(b"host", b"localhost"), (b"content-length", b"8000000")],
                    "client": ("203.0.113.10", 1234), "server": ("localhost", 8000)}, receive, send)
    assert send.call_args_list[0].args[0]["status"] == 403
    receive.assert_not_awaited()


async def test_project_admission_cleanup_preserves_concurrent_admissions():
    registry = CancellationRegistry()
    await registry.begin_project_admission("project")
    await registry.begin_project_admission("project")
    await registry.end_project_admission("project")
    assert registry._projects["project"].idle_since is None
    await registry.end_project_admission("project")
    assert registry._projects["project"].idle_since is not None
    registry._projects["project"].idle_since = datetime.utcnow() - timedelta(days=1)
    await registry.cleanup_inactive_projects()
    assert "project" not in registry._projects


def test_gran_sabio_limits_count_inline_and_final_reviews_together():
    request = SimpleNamespace(gran_sabio_call_limit_per_iteration=2, gran_sabio_call_limit_per_session=3)
    session = {"current_iteration": 1}
    assert _claim_gran_sabio_call(session, request)
    assert _claim_gran_sabio_call(session, request)
    assert not _claim_gran_sabio_call(session, request)
    session["current_iteration"] = 2
    assert _claim_gran_sabio_call(session, request)
    assert not _claim_gran_sabio_call(session, request)
    assert session["gran_sabio_escalation_count"] == 3


async def test_feedback_writes_do_not_lose_concurrent_category_updates(tmp_path):
    database = FeedbackDatabase(str(tmp_path / "feedback.sqlite"))
    await database.initialize()
    try:
        await database.create_session("test", "hash")
        await asyncio.gather(*(database.upsert_category(
            "test", "concept", "label", "style", "low", [str(i)], [], [0.1], total_iterations=1)
            for i in range(2)))
        categories = await database.get_categories("test")
        assert len(categories) == 1
        assert categories[0]["occurrences"] == 2
    finally:
        await database.close()


def test_subscription_retry_after_is_bounded_and_attempt_zero_is_invalid():
    error = SimpleNamespace(response=SimpleNamespace(headers={"retry-after": "86400"}))
    assert overload_retry_delay(error, 1) == 60
    with pytest.raises(ValueError):
        overload_retry_delay(error, 0)
