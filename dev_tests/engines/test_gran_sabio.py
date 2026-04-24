"""
Tests for gran_sabio.py - Gran Sabio Engine Module.

The Gran Sabio Engine is the final escalation system that resolves conflicts
and makes ultimate decisions when the standard QA process cannot reach
consensus. After Phase 4 of the shared tool-loop refactor, the three live
methods (``review_minority_deal_breakers``, ``regenerate_content``,
``review_iterations``) consume JSON_STRUCTURED payloads through the reusable
``call_ai_with_validation_tools`` entry point. Legacy tag parsing and the
dead ``handle_model_conflict`` method are gone.

Test areas covered here (smoke/unit level — deeper invariants live in
``test_gran_sabio_tool_loop.py`` and ``test_gran_sabio_contract_preservation.py``):
1. Helper functions (streaming retry helpers).
2. Model capacity validation.
3. ``_pick_best_iteration`` / formatting helpers.
4. ``review_minority_deal_breakers`` happy paths, cancellation, exception path.
5. ``regenerate_content`` happy paths, cancellation, exception path.
6. ``review_iterations`` happy paths, cancellation, fallback notes.
"""

import pytest
from unittest.mock import AsyncMock, Mock, MagicMock, patch

from gran_sabio import (
    GranSabioEngine,
    GranSabioProcessCancelled,
    _is_retryable_streaming_error,
    _extract_error_reason,
    _extract_provider,
)
from ai_service import AIRequestError
from models import ContentRequest


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_ai_service():
    """Mocked AI service for Gran Sabio tests. Tool loop is opted out via
    ``gransabio_tools_mode='never'`` on each request so the single-shot
    ``generate_content`` path drives the tests."""

    service = MagicMock()
    service.generate_content = AsyncMock(return_value="{}")
    service.generate_content_stream = AsyncMock()
    # Tool loop call surface — not exercised in single-shot tests but kept
    # callable to catch accidental routing.
    service.call_ai_with_validation_tools = AsyncMock(
        return_value=("{}", MagicMock(tools_skipped_reason="no_tool_support", payload=None))
    )
    return service


@pytest.fixture
def mock_config():
    """Mock config with model specs."""
    with patch('gran_sabio.config') as mock_cfg:
        mock_cfg.model_specs = {
            "model_specifications": {
                "anthropic": {
                    "claude-sonnet-4-20250514": {
                        "input_tokens": 200000,
                        "max_tokens": 8192,
                    },
                },
                "openai": {
                    "gpt-4o": {
                        "input_tokens": 128000,
                        "max_tokens": 16384,
                    },
                },
            },
            "aliases": {},
        }
        mock_cfg.MAX_RETRIES = 3
        mock_cfg.RETRY_DELAY = 1
        mock_cfg.RETRY_STREAMING_AFTER_PARTIAL = True
        mock_cfg.GRAN_SABIO_SYSTEM_PROMPT = "You are Gran Sabio, the final arbiter."
        mock_cfg.GRAN_SABIO_REGENERATE_MAX_TOOL_ROUNDS = 5
        mock_cfg.GRAN_SABIO_DECISION_MAX_TOOL_ROUNDS = 2
        mock_cfg.GRAN_SABIO_ESCALATION_MAX_TOOL_ROUNDS = 4
        mock_cfg.get_model_info = Mock(return_value={
            "input_tokens": 200000,
            "max_tokens": 8192,
            "provider": "anthropic",
        })
        mock_cfg._get_thinking_budget_config = Mock(return_value={
            "supported": True,
            "default_tokens": 10000,
        })
        yield mock_cfg


@pytest.fixture
def sample_content_request():
    """Sample content request with tool loop disabled for deterministic tests."""
    return ContentRequest(
        prompt="Write a test article about software testing",
        content_type="article",
        generator_model="gpt-4o",
        temperature=0.7,
        max_tokens=4000,
        min_words=500,
        max_words=1000,
        qa_layers=[],
        qa_models=[],
        gransabio_tools_mode="never",
    )


@pytest.fixture
def sample_minority_deal_breakers():
    """Sample minority deal-breakers data."""
    return {
        "details": [
            {
                "layer": "Accuracy",
                "model": "gpt-4o",
                "reason": "Found factual inconsistency",
                "score_given": 5.0,
                "layer_deal_breaker_criteria": "invents facts",
                "layer_min_score": 7.0,
            }
        ],
        "total_evaluations": 3,
        "qa_configuration": {
            "layer_name": "Accuracy",
            "description": "Factual accuracy verification",
            "criteria": "Check for factual errors",
            "deal_breaker_criteria": "invents facts",
            "min_score": 7.0,
        },
        "iteration": 1,
    }


@pytest.fixture
def sample_iterations():
    """Sample iterations data for review."""
    return [
        {
            "iteration": 1,
            "content": "First attempt content",
            "content_word_count": 500,
            "content_char_count": 3000,
            "consensus": {"average_score": 6.5, "deal_breakers_count": 1},
            "qa_results": {
                "Accuracy": {
                    "gpt-4o": Mock(score=6.5, deal_breaker=True, feedback="Issues found")
                }
            },
            "qa_layers_config": [{"name": "Accuracy", "min_score": 7.0}],
        },
        {
            "iteration": 2,
            "content": "Second attempt content with improvements",
            "content_word_count": 550,
            "content_char_count": 3300,
            "consensus": {"average_score": 7.8, "deal_breakers_count": 0},
            "qa_results": {
                "Accuracy": {
                    "gpt-4o": Mock(score=7.8, deal_breaker=False, feedback="Good quality")
                }
            },
            "qa_layers_config": [{"name": "Accuracy", "min_score": 7.0}],
        },
    ]


# ============================================================================
# Test Area 1: Helper Functions (Streaming Retry)
# ============================================================================


class TestIsRetryableStreamingError:
    def test_ai_request_error_is_retryable(self):
        exc = AIRequestError(
            provider="openai",
            model="gpt-4o",
            attempts=3,
            max_attempts=3,
            cause=Exception("API error"),
        )
        assert _is_retryable_streaming_error(exc) is True

    def test_status_429_is_retryable(self):
        exc = Exception("Rate limit")
        exc.status = 429
        assert _is_retryable_streaming_error(exc) is True

    def test_timeout_message_is_retryable(self):
        exc = Exception("Connection timeout occurred")
        assert _is_retryable_streaming_error(exc) is True

    def test_non_retryable_error(self):
        exc = Exception("Invalid input parameter")
        assert _is_retryable_streaming_error(exc) is False


class TestExtractErrorReason:
    def test_extracts_from_ai_request_error_with_message(self):
        cause = Mock()
        cause.message = "API rate limit exceeded"
        exc = AIRequestError(
            provider="openai",
            model="gpt-4o",
            attempts=3,
            max_attempts=3,
            cause=cause,
        )
        assert _extract_error_reason(exc) == "API rate limit exceeded"

    def test_truncates_long_messages(self):
        exc = Exception("x" * 500)
        assert len(_extract_error_reason(exc)) <= 200


class TestExtractProvider:
    def test_extracts_from_ai_request_error(self):
        exc = AIRequestError(
            provider="openai",
            model="gpt-4o",
            attempts=3,
            max_attempts=3,
            cause=Exception("test"),
        )
        assert _extract_provider(exc) == "openai"

    def test_returns_none_for_unknown(self):
        assert _extract_provider(Exception("Generic error")) is None


# ============================================================================
# Test Area 2: Model Capacity Validation
# ============================================================================


class TestModelCapacityValidation:
    def test_get_configured_default_model_success(self, mock_config):
        with patch('gran_sabio.get_default_models', return_value={"gran_sabio": "claude-sonnet-4-20250514"}):
            engine = GranSabioEngine(ai_service=MagicMock())
            assert engine._get_configured_default_model() == "claude-sonnet-4-20250514"

    def test_get_configured_default_model_not_configured(self):
        with patch('gran_sabio.get_default_models', return_value={}):
            engine = GranSabioEngine(ai_service=MagicMock())
            with pytest.raises(RuntimeError, match="not configured"):
                engine._get_configured_default_model()

    def test_get_default_thinking_tokens_supported(self, mock_config):
        engine = GranSabioEngine(ai_service=MagicMock())
        assert engine._get_default_thinking_tokens("claude-sonnet-4-20250514") == 10000

    def test_ensure_adequate_model_capacity_sufficient(self, mock_config):
        engine = GranSabioEngine(ai_service=MagicMock())
        assert engine._ensure_adequate_model_capacity(1000, "claude-sonnet-4-20250514", None) == "claude-sonnet-4-20250514"

    def test_resolve_model_alias(self, mock_config):
        mock_config.model_specs = {"aliases": {"claude": "claude-sonnet-4-20250514"}}
        engine = GranSabioEngine(ai_service=MagicMock())
        assert engine._resolve_model_alias("claude") == "claude-sonnet-4-20250514"


# ============================================================================
# Test Area 3: Pick-best / formatting helpers
# ============================================================================


class TestPickBestIteration:
    def test_empty_returns_none(self):
        assert GranSabioEngine._pick_best_iteration([]) is None

    def test_picks_highest_consensus(self, sample_iterations):
        best = GranSabioEngine._pick_best_iteration(sample_iterations)
        assert best is not None
        assert best["iteration"] == 2


class TestFormattingHelpers:
    def test_build_deal_breaker_context_empty(self):
        engine = GranSabioEngine(ai_service=MagicMock())
        result = engine._build_deal_breaker_context({"details": []})
        assert "No deal-breakers" in result

    def test_format_iteration_details(self, sample_iterations):
        engine = GranSabioEngine(ai_service=MagicMock())
        result = engine._format_iteration_details(sample_iterations)
        assert "Iteration 1" in result
        assert "Iteration 2" in result


# ============================================================================
# Test Area 4: review_minority_deal_breakers()
# ============================================================================


class TestReviewMinorityDealBreakers:
    APPROVE_JSON = (
        '{"decision": "APPROVED", "reason": "False positive", "score": 8.5, '
        '"modifications_made": false, "final_content": null}'
    )
    REJECT_JSON = (
        '{"decision": "REJECTED", "reason": "Invented facts", "score": 3.0, '
        '"modifications_made": false, "final_content": null}'
    )

    @pytest.mark.asyncio
    async def test_approves_false_positive_deal_breaker(
        self, mock_ai_service, mock_config, sample_content_request, sample_minority_deal_breakers
    ):
        mock_ai_service.generate_content = AsyncMock(return_value=self.APPROVE_JSON)

        with patch('gran_sabio.get_default_models', return_value={"gran_sabio": "claude-sonnet-4-20250514"}):
            engine = GranSabioEngine(ai_service=mock_ai_service)
            result = await engine.review_minority_deal_breakers(
                session_id="test-session",
                content="Test content",
                minority_deal_breakers=sample_minority_deal_breakers,
                original_request=sample_content_request,
            )

        assert result.approved is True
        assert result.final_score == 8.5
        # Matrix row 2: approved + no modifications -> original content.
        assert result.final_content == "Test content"

    @pytest.mark.asyncio
    async def test_rejects_real_deal_breaker(
        self, mock_ai_service, mock_config, sample_content_request, sample_minority_deal_breakers
    ):
        mock_ai_service.generate_content = AsyncMock(return_value=self.REJECT_JSON)

        with patch('gran_sabio.get_default_models', return_value={"gran_sabio": "claude-sonnet-4-20250514"}):
            engine = GranSabioEngine(ai_service=mock_ai_service)
            result = await engine.review_minority_deal_breakers(
                session_id="test-session",
                content="Content with invented facts",
                minority_deal_breakers=sample_minority_deal_breakers,
                original_request=sample_content_request,
            )

        assert result.approved is False
        assert result.final_score == 3.0
        # Matrix row 5 (v8 change): rejected -> original content, NOT "".
        assert result.final_content == "Content with invented facts"

    @pytest.mark.asyncio
    async def test_handles_cancellation(
        self, mock_ai_service, mock_config, sample_content_request, sample_minority_deal_breakers
    ):
        cancel_callback = AsyncMock(return_value=True)

        with patch('gran_sabio.get_default_models', return_value={"gran_sabio": "claude-sonnet-4-20250514"}):
            engine = GranSabioEngine(ai_service=mock_ai_service)

            with pytest.raises(GranSabioProcessCancelled):
                await engine.review_minority_deal_breakers(
                    session_id="test-session",
                    content="Test content",
                    minority_deal_breakers=sample_minority_deal_breakers,
                    original_request=sample_content_request,
                    cancel_callback=cancel_callback,
                )

    @pytest.mark.asyncio
    async def test_handles_api_error(
        self, mock_ai_service, mock_config, sample_content_request, sample_minority_deal_breakers
    ):
        mock_ai_service.generate_content = AsyncMock(side_effect=Exception("API unavailable"))

        with patch('gran_sabio.get_default_models', return_value={"gran_sabio": "claude-sonnet-4-20250514"}):
            engine = GranSabioEngine(ai_service=mock_ai_service)
            result = await engine.review_minority_deal_breakers(
                session_id="test-session",
                content="Test content",
                minority_deal_breakers=sample_minority_deal_breakers,
                original_request=sample_content_request,
            )

        # Matrix row 8: exception path returns `content` as defensive default.
        assert result.approved is False
        assert result.final_content == "Test content"
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_approves_with_modifications(
        self, mock_ai_service, mock_config, sample_content_request, sample_minority_deal_breakers
    ):
        payload = (
            '{"decision": "APPROVED", "reason": "Minor typo fixed", "score": 8.0, '
            '"modifications_made": true, "final_content": "Corrected content here."}'
        )
        mock_ai_service.generate_content = AsyncMock(return_value=payload)

        with patch('gran_sabio.get_default_models', return_value={"gran_sabio": "claude-sonnet-4-20250514"}):
            engine = GranSabioEngine(ai_service=mock_ai_service)
            result = await engine.review_minority_deal_breakers(
                session_id="test-session",
                content="Content with typo",
                minority_deal_breakers=sample_minority_deal_breakers,
                original_request=sample_content_request,
            )

        # Matrix row 1: approved + modifications_made -> LLM's final_content.
        assert result.approved is True
        assert result.modifications_made is True
        assert result.final_content == "Corrected content here."


# ============================================================================
# Test Area 5: regenerate_content()
# ============================================================================


class TestRegenerateContent:
    @pytest.mark.asyncio
    async def test_generates_new_content(
        self, mock_ai_service, mock_config, sample_content_request
    ):
        mock_ai_service.generate_content = AsyncMock(
            return_value="Newly generated content for testing."
        )

        with patch('gran_sabio.get_default_models', return_value={"gran_sabio": "claude-sonnet-4-20250514"}):
            engine = GranSabioEngine(ai_service=mock_ai_service)
            result = await engine.regenerate_content(
                session_id="test-session",
                original_request=sample_content_request,
            )

        assert result.approved is True
        assert result.final_content == "Newly generated content for testing."
        assert result.final_score is None

    @pytest.mark.asyncio
    async def test_includes_previous_attempts_context(
        self, mock_ai_service, mock_config, sample_content_request
    ):
        mock_ai_service.generate_content = AsyncMock(return_value="New content")
        previous_attempts = ["First failed attempt", "Second failed attempt"]

        with patch('gran_sabio.get_default_models', return_value={"gran_sabio": "claude-sonnet-4-20250514"}):
            engine = GranSabioEngine(ai_service=mock_ai_service)
            await engine.regenerate_content(
                session_id="test-session",
                original_request=sample_content_request,
                previous_attempts=previous_attempts,
            )

        call_args = mock_ai_service.generate_content.call_args
        prompt = call_args.kwargs.get("prompt", call_args.args[0] if call_args.args else "")
        assert "PREVIOUS ATTEMPTS" in prompt or "Attempt" in prompt

    @pytest.mark.asyncio
    async def test_handles_cancellation(
        self, mock_ai_service, mock_config, sample_content_request
    ):
        cancel_callback = AsyncMock(return_value=True)

        with patch('gran_sabio.get_default_models', return_value={"gran_sabio": "claude-sonnet-4-20250514"}):
            engine = GranSabioEngine(ai_service=mock_ai_service)

            with pytest.raises(GranSabioProcessCancelled):
                await engine.regenerate_content(
                    session_id="test-session",
                    original_request=sample_content_request,
                    cancel_callback=cancel_callback,
                )

    @pytest.mark.asyncio
    async def test_handles_generation_error_with_previous_attempts(
        self, mock_ai_service, mock_config, sample_content_request
    ):
        mock_ai_service.generate_content = AsyncMock(side_effect=Exception("Model error"))
        previous_attempts = ["first attempt", "second attempt"]

        with patch('gran_sabio.get_default_models', return_value={"gran_sabio": "claude-sonnet-4-20250514"}):
            engine = GranSabioEngine(ai_service=mock_ai_service)
            result = await engine.regenerate_content(
                session_id="test-session",
                original_request=sample_content_request,
                previous_attempts=previous_attempts,
            )

        # Matrix row 9 (v9 H1 change): exception path returns previous_attempts[-1].
        assert result.approved is False
        assert result.final_content == "second attempt"
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_handles_generation_error_without_previous_attempts(
        self, mock_ai_service, mock_config, sample_content_request
    ):
        mock_ai_service.generate_content = AsyncMock(side_effect=Exception("Model error"))

        with patch('gran_sabio.get_default_models', return_value={"gran_sabio": "claude-sonnet-4-20250514"}):
            engine = GranSabioEngine(ai_service=mock_ai_service)
            result = await engine.regenerate_content(
                session_id="test-session",
                original_request=sample_content_request,
            )

        # Matrix row 9 edge: no previous attempts -> "".
        assert result.approved is False
        assert result.final_content == ""


# ============================================================================
# Test Area 6: review_iterations()
# ============================================================================


class TestReviewIterations:
    APPROVE_JSON = (
        '{"decision": "APPROVE", "reason": "Second iteration meets all criteria", '
        '"score": 8.5, "modifications_made": false, "final_content": null}'
    )
    REJECT_JSON = (
        '{"decision": "REJECT", "reason": "Critical issues", "score": 4.0, '
        '"modifications_made": false, "final_content": null}'
    )

    @pytest.mark.asyncio
    async def test_approves_best_iteration(
        self, mock_ai_service, mock_config, sample_content_request, sample_iterations
    ):
        mock_ai_service.generate_content = AsyncMock(return_value=self.APPROVE_JSON)

        with patch('gran_sabio.get_default_models', return_value={"gran_sabio": "claude-sonnet-4-20250514"}):
            engine = GranSabioEngine(ai_service=mock_ai_service)
            result = await engine.review_iterations(
                session_id="test-session",
                iterations=sample_iterations,
                original_request=sample_content_request,
            )

        assert result.approved is True
        assert result.final_score == 8.5
        # Matrix row 3: APPROVE -> best iteration content fallback.
        assert result.final_content == "Second attempt content with improvements"

    @pytest.mark.asyncio
    async def test_rejects_all_iterations(
        self, mock_ai_service, mock_config, sample_content_request, sample_iterations
    ):
        mock_ai_service.generate_content = AsyncMock(return_value=self.REJECT_JSON)

        with patch('gran_sabio.get_default_models', return_value={"gran_sabio": "claude-sonnet-4-20250514"}):
            engine = GranSabioEngine(ai_service=mock_ai_service)
            result = await engine.review_iterations(
                session_id="test-session",
                iterations=sample_iterations,
                original_request=sample_content_request,
            )

        # Matrix row 6 (v8 change): REJECT -> best iteration content, NOT "".
        assert result.approved is False
        assert result.final_content == "Second attempt content with improvements"

    @pytest.mark.asyncio
    async def test_handles_cancellation(
        self, mock_ai_service, mock_config, sample_content_request, sample_iterations
    ):
        cancel_callback = AsyncMock(return_value=True)

        with patch('gran_sabio.get_default_models', return_value={"gran_sabio": "claude-sonnet-4-20250514"}):
            engine = GranSabioEngine(ai_service=mock_ai_service)

            with pytest.raises(GranSabioProcessCancelled):
                await engine.review_iterations(
                    session_id="test-session",
                    iterations=sample_iterations,
                    original_request=sample_content_request,
                    cancel_callback=cancel_callback,
                )

    @pytest.mark.asyncio
    async def test_includes_fallback_notes(
        self, mock_ai_service, mock_config, sample_content_request, sample_iterations
    ):
        mock_ai_service.generate_content = AsyncMock(return_value=self.APPROVE_JSON)

        fallback_notes = ["Note 1: Previous issue", "Note 2: Consideration"]

        with patch('gran_sabio.get_default_models', return_value={"gran_sabio": "claude-sonnet-4-20250514"}):
            engine = GranSabioEngine(ai_service=mock_ai_service)
            await engine.review_iterations(
                session_id="test-session",
                iterations=sample_iterations,
                original_request=sample_content_request,
                fallback_notes=fallback_notes,
            )

        call_args = mock_ai_service.generate_content.call_args
        prompt = call_args.kwargs.get("prompt", "")
        assert "FALLBACK NOTES" in prompt

    @pytest.mark.asyncio
    async def test_handles_api_error_with_iterations(
        self, mock_ai_service, mock_config, sample_content_request, sample_iterations
    ):
        mock_ai_service.generate_content = AsyncMock(side_effect=Exception("API error"))

        with patch('gran_sabio.get_default_models', return_value={"gran_sabio": "claude-sonnet-4-20250514"}):
            engine = GranSabioEngine(ai_service=mock_ai_service)
            result = await engine.review_iterations(
                session_id="test-session",
                iterations=sample_iterations,
                original_request=sample_content_request,
            )

        # Matrix row 10 (v9 H1 change): exception path -> best iteration content.
        assert result.approved is False
        assert result.final_content == "Second attempt content with improvements"
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_handles_api_error_without_iterations(
        self, mock_ai_service, mock_config, sample_content_request
    ):
        mock_ai_service.generate_content = AsyncMock(side_effect=Exception("API error"))

        with patch('gran_sabio.get_default_models', return_value={"gran_sabio": "claude-sonnet-4-20250514"}):
            engine = GranSabioEngine(ai_service=mock_ai_service)
            result = await engine.review_iterations(
                session_id="test-session",
                iterations=[],
                original_request=sample_content_request,
            )

        # Matrix row 10 edge: empty iterations -> "".
        assert result.approved is False
        assert result.final_content == ""
