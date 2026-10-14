"""A refusal raised inside a provider SDK's HTTP transport stays terminal.

Provider SDKs report any exception raised inside their HTTP transport as a
connection error (``raise APIConnectionError(...) from err``). An admission or
budget refusal raised at that boundary must reach the generation's terminal
payload unchanged: never classified, retried, sent to a fallback or counted
against provider health. No network or provider calls; no environment variables.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from uuid import uuid4

import httpx
import openai
import pytest

import context_budget
from ai_service import AIRequestError, AIService, find_terminal_refusal
from arbiter import Arbiter, ArbiterContext, LayerEditHistory, ProposedEdit
from config import config
from context_budget import CapacityConfidence, TokenLimitsResolution
from core.app_state import pop_session, register_session
from core.cancellation import cancellation_registry
from core.generation_processor import _build_generation_error_result, _generate_full_content
from local_execution_policy import LocalExecutionPolicyError
from model_aliasing import ModelAliasRegistry
from models import ContentRequest, GenerationStatus, QALayer
from provider_budget import ProviderBudgetError
from qa_engine import QAEngine


class _SpendRefused(ProviderBudgetError):
    """A spending authority's refusal of one provider attempt."""

    error_code = "spend_refused"


def _refusal() -> _SpendRefused:
    # "unavailable" is a transient marker for message-based retry checks; a
    # refusal must stay terminal whatever its text says.
    return _SpendRefused("Spending authority unavailable for another provider request.")


def _sdk_connection_error() -> openai.APIConnectionError:
    return openai.APIConnectionError(request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"))


def _wrapped(refusal: BaseException, outer: Exception) -> Exception:
    """Raise ``outer`` from ``refusal``, as a provider SDK wraps a transport exception."""

    try:
        try:
            raise refusal
        except type(refusal) as transport_error:
            raise outer from transport_error
    except type(outer) as wrapped:
        return wrapped


def _as_sdk_connection_error(refusal: BaseException) -> openai.APIConnectionError:
    return _wrapped(refusal, _sdk_connection_error())


# Classified as a retryable transient network failure when it carries no refusal.
def _transient() -> ConnectionError:
    return ConnectionError("Connection reset by peer")


def _service(monkeypatch: pytest.MonkeyPatch) -> tuple[AIService, AsyncMock]:
    service = AIService.__new__(AIService)
    health_failure = AsyncMock()
    monkeypatch.setattr(service, "_record_provider_health_failure", health_failure)
    monkeypatch.setattr(service, "_record_provider_health_success", AsyncMock())
    monkeypatch.setattr(service, "_max_retry_attempts", lambda: 3)
    monkeypatch.setattr(service, "_calculate_retry_delay", lambda _attempt: 0.0)
    return service, health_failure


def _fake_model_specs() -> dict:
    return {
        "default_models": {},
        "aliases": {},
        "model_specifications": {
            "fake": {
                "fake-generator": {
                    "model_id": "fake-generator",
                    "input_tokens": 128_000,
                    "output_tokens": 16_000,
                    "context_window": 128_000,
                    "capabilities": ["text"],
                    "enabled": True,
                    "is_test_model": True,
                }
            }
        },
    }


def _refused_stream(wrapped: BaseException, calls: list[int]):
    async def stream(*_args, **_kwargs):
        calls.append(1)
        raise wrapped
        yield ""  # pragma: no cover - makes this an async generator

    return stream


# ---------------------------------------------------------------------------
# The helper
# ---------------------------------------------------------------------------


def test_find_terminal_refusal_walks_explicit_and_implicit_chains():
    refusal = _refusal()
    outer = RuntimeError("GPT-5 streaming initialization failed")
    outer.__cause__ = _as_sdk_connection_error(refusal)
    assert find_terminal_refusal(outer) is refusal
    assert find_terminal_refusal(refusal) is refusal

    policy = LocalExecutionPolicyError("local_endpoint_unknown", "Unknown local endpoint.")
    try:
        try:
            raise policy
        except LocalExecutionPolicyError:
            raise ValueError("cleanup failed")
    except ValueError as implicit:
        assert find_terminal_refusal(implicit) is policy

    assert find_terminal_refusal(_sdk_connection_error()) is None


def test_find_terminal_refusal_visits_a_cyclic_chain_once():
    first, second = RuntimeError("first"), RuntimeError("second")
    first.__cause__ = second
    second.__context__ = first
    assert find_terminal_refusal(first) is None


# ---------------------------------------------------------------------------
# Retry wrappers and the streaming loop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("wrapper", ["_execute_with_retries", "_execute_without_retries"])
@pytest.mark.parametrize("outer", [_sdk_connection_error, _transient], ids=["sdk-connection-error", "transient"])
async def test_retry_wrappers_reraise_a_wrapped_refusal_once_without_health_failure(monkeypatch, wrapper, outer):
    service, health_failure = _service(monkeypatch)
    refusal = _refusal()
    operation = AsyncMock(side_effect=_wrapped(refusal, outer()))

    with pytest.raises(_SpendRefused) as raised:
        await getattr(service, wrapper)(operation, provider="openai", model_id="gpt-test", action="generation")

    assert raised.value is refusal
    assert operation.await_count == 1
    health_failure.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_same_transient_failure_without_a_refusal_is_still_retried(monkeypatch):
    service, health_failure = _service(monkeypatch)
    operation = AsyncMock(side_effect=_transient())

    with pytest.raises(AIRequestError):
        await service._execute_with_retries(operation, provider="openai", model_id="gpt-test", action="generation")

    assert operation.await_count == 3
    health_failure.assert_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("outer", [_sdk_connection_error, _transient], ids=["sdk-connection-error", "transient"])
async def test_streaming_loop_reraises_a_wrapped_refusal_once(monkeypatch, outer):
    service, health_failure = _service(monkeypatch)
    monkeypatch.setattr(config, "model_specs", _fake_model_specs())
    monkeypatch.setattr(config, "MAX_RETRIES", 3)
    monkeypatch.setattr(config, "RETRY_DELAY", 0.0)
    refusal, calls = _refusal(), []
    monkeypatch.setattr(service, "_stream_fake", _refused_stream(_wrapped(refusal, outer()), calls))

    with pytest.raises(_SpendRefused) as raised:
        async for _chunk in service.generate_content_stream(
            prompt="Write one sentence.", model="fake-generator", max_tokens=128,
        ):
            pass

    assert raised.value is refusal
    assert calls == [1]
    health_failure.assert_not_awaited()


# ---------------------------------------------------------------------------
# Fallbacks that would otherwise send a new request or swallow the refusal
# ---------------------------------------------------------------------------


@pytest.fixture
def initialized_service():
    """AIService with provider SDK initialization patched out."""

    with patch("ai_service.aiohttp.TCPConnector"), patch("ai_service.aiohttp.ClientTimeout"), \
            patch.object(AIService, "_initialize_clients"):
        instance = AIService()
    for attribute in ("openai_client", "anthropic_client", "google_new_client", "xai_client",
                      "openrouter_client", "ollama_client"):
        setattr(instance, attribute, None)
    return instance


@pytest.mark.asyncio
async def test_gemini_stream_refusal_never_falls_back_to_a_new_request(initialized_service):
    # google-genai sends the request on first iteration and propagates transport
    # exceptions unwrapped.
    refusal = _refusal()

    class _RefusedStream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise refusal

    async def generate_content_stream(**_kwargs):
        return _RefusedStream()

    initialized_service.google_new_client = SimpleNamespace(
        aio=SimpleNamespace(models=SimpleNamespace(generate_content_stream=generate_content_stream)))
    initialized_service._generate_gemini_new_sdk = AsyncMock()

    with pytest.raises(_SpendRefused) as raised:
        async for _chunk in initialized_service._stream_gemini_new_sdk(
                "Write one sentence.", "gemini-2.5-flash", 0.5, 128, ""):
            pass

    assert raised.value is refusal
    initialized_service._generate_gemini_new_sdk.assert_not_awaited()


@pytest.mark.asyncio
async def test_embedding_refusals_are_not_swallowed_or_sent_to_the_fallback_model(initialized_service, monkeypatch):
    health_failure = AsyncMock()
    monkeypatch.setattr(initialized_service, "_record_provider_health_failure", health_failure)
    openai_refusal, google_refusal = _refusal(), _refusal()
    initialized_service.openai_client = MagicMock()
    initialized_service.openai_client.embeddings.create = AsyncMock(
        side_effect=_as_sdk_connection_error(openai_refusal))
    initialized_service.google_new_client = MagicMock()
    initialized_service.google_new_client.aio.models.embed_content = AsyncMock(side_effect=google_refusal)

    with patch("ai_service.resolve_call", return_value=SimpleNamespace(model="fallback-emb")):
        with pytest.raises(_SpendRefused) as raised:
            await initialized_service._get_openai_embeddings(["a chunk"], model="primary-emb")
    assert raised.value is openai_refusal
    assert initialized_service.openai_client.embeddings.create.await_count == 1

    with pytest.raises(_SpendRefused) as raised:
        await initialized_service._get_google_embeddings(["a chunk"], model="text-embedding-004")
    assert raised.value is google_refusal
    health_failure.assert_not_awaited()


# ---------------------------------------------------------------------------
# Review phase: quality review and arbiter
# ---------------------------------------------------------------------------


def _verified_limits(model: str) -> TokenLimitsResolution:
    return TokenLimitsResolution(
        model=model, provider="openai", max_input_tokens=None, max_output_tokens=None,
        max_total_context_tokens=1_000_000, output_accounting="within_total",
        reasoning_accounting="within_output", provenance="manual_curated",
        confidence=CapacityConfidence.VERIFIED,
    )


@pytest.mark.asyncio
async def test_quality_review_propagates_the_refusal_instead_of_model_unavailable(monkeypatch):
    service, health_failure = _service(monkeypatch)
    refusal = _refusal()
    operation = AsyncMock(side_effect=_as_sdk_connection_error(refusal))

    async def evaluate_content(*_args, **_kwargs):
        return await service._execute_with_retries(
            operation, provider="openai", model_id="qa-a", action="qa_evaluation")

    ai_service = MagicMock()
    ai_service._native_token_count = AsyncMock(return_value=None)
    ai_service.model_call_concurrency_slot = None
    bypass = MagicMock()
    bypass.should_bypass_qa_layer = Mock(return_value=False)
    engine = QAEngine(ai_service=ai_service, bypass_engine=bypass)
    engine.evaluate_content = evaluate_content
    models = ["qa-a"]
    request = ContentRequest(prompt="Evaluate the content.", content_type="biography",
                             generator_model="gen-model", qa_models=models)
    request._client_named_qa_models = True

    with patch.object(context_budget, "resolve_token_limits", side_effect=_verified_limits), \
            patch.object(config.__class__, "resolve_qa_max_tokens", return_value=6000):
        with pytest.raises(_SpendRefused) as raised:
            await engine._evaluate_single_semantic_layer(
                content="short content",
                layer=QALayer(name="Accuracy", description="d", criteria="Be accurate.", min_score=7.0, order=1),
                qa_models=models,
                qa_model_names=models,
                original_request=request,
                model_alias_registry=ModelAliasRegistry.from_request(request),
            )

    assert raised.value is refusal
    assert operation.await_count == 1
    health_failure.assert_not_awaited()


@pytest.mark.asyncio
async def test_arbiter_propagates_the_refusal_unwrapped():
    refusal = _refusal()
    ai_service = MagicMock()
    ai_service.generate_content = AsyncMock(side_effect=refusal)
    edit = SimpleNamespace(edit_type=SimpleNamespace(value="replace"), issue_severity=SimpleNamespace(value="minor"),
                           issue_description="Test issue", edit_instruction="Fix it", start_marker="marker_p1",
                           start_word_index=None, confidence=1.0)
    context = ArbiterContext(
        original_prompt="Write a biography", content_type="biography", system_prompt=None,
        layer_name="Accuracy", layer_criteria="Check factual accuracy", layer_min_score=7.0,
        current_content="This is the content to evaluate.", content_excerpt=None,
        proposed_edits=[ProposedEdit(edit=edit, source_evaluator="Evaluator A", source_score=7.0,
                                     paragraph_key="p1")],
        evaluator_scores={"gpt-4o": 6.5}, layer_history=LayerEditHistory(layer_name="Accuracy"),
        gran_sabio_model="claude-opus-4-5-20251101", qa_model_count=2, arbiter_tools_mode="never",
    )

    with pytest.raises(_SpendRefused) as raised:
        await Arbiter(ai_service=ai_service).arbitrate(context)

    assert raised.value is refusal


# ---------------------------------------------------------------------------
# Draft phase end to end: the terminal payload carries the refusal's code
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("outer", [_sdk_connection_error, _transient], ids=["sdk-connection-error", "transient"])
async def test_draft_refusal_ends_as_a_terminal_admission_payload_with_its_code(monkeypatch, outer):
    service, health_failure = _service(monkeypatch)
    monkeypatch.setattr(config, "model_specs", _fake_model_specs())
    monkeypatch.setattr(config, "MAX_RETRIES", 3)
    monkeypatch.setattr(config, "RETRY_STREAMING_AFTER_PARTIAL", True)
    monkeypatch.setattr(config, "RETRY_DELAY", 0.0)
    refusal, calls = _refusal(), []
    monkeypatch.setattr(service, "_stream_fake", _refused_stream(_wrapped(refusal, outer()), calls))
    session_id = f"refusal-session-{uuid4()}"
    session = {
        "session_id": session_id, "status": GenerationStatus.GENERATING, "current_phase": "generation",
        "current_iteration": 1, "verbose_log": [], "generation_content": "", "partial_content": "",
    }
    request = ContentRequest(
        prompt="Write a deterministic refusal test response.", generator_model="fake-generator",
        max_tokens=128, max_iterations=3, qa_models=[], qa_layers=[], generation_tools_mode="never",
        auto_qa={"enabled": False}, long_text_mode="off",
    )

    await cancellation_registry.register_session(session_id, None, None)
    await register_session(session_id, session)
    try:
        with pytest.raises(_SpendRefused) as raised:
            await _generate_full_content(
                final_prompt="Generate a deterministic refusal response.", request=request,
                ai_service=service, usage_tracker=None, session_id=session_id, session=session,
                iteration=0, json_output_requested=False,
            )
    finally:
        await pop_session(session_id)
        await cancellation_registry.unregister_session(session_id)

    assert raised.value is refusal
    assert calls == [1]
    health_failure.assert_not_awaited()
    terminal = _build_generation_error_result(session, request, raised.value)
    assert terminal["status"] == GenerationStatus.FAILED.value
    assert terminal["error_type"] == session["error_type"] == "context_admission"
    assert terminal["error_code"] == session["error_code"] == "spend_refused"
    assert terminal["context_admission"]["error_code"] == "spend_refused"
    assert terminal["failure_reason"].startswith("Provider budget admission failed")
    assert terminal["context_admission"]["detail"] == str(refusal)
