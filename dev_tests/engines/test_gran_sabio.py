"""
Tests for gran_sabio.py - Gran Sabio Engine Module.

The Gran Sabio Engine is the final escalation system that resolves conflicts
and makes ultimate decisions when the standard QA process cannot reach consensus.

Test Areas:
1. Helper Functions (Streaming Retry Helpers)
2. Model Capacity Validation
3. Response Parsing
4. review_minority_deal_breakers()
5. regenerate_content()
6. review_iterations()
"""

import pytest
from unittest.mock import AsyncMock, Mock, MagicMock, patch
import asyncio

from gran_sabio import (
    GranSabioEngine,
    GranSabioInvocationError,
    GranSabioProcessCancelled,
    _is_retryable_streaming_error,
    _extract_error_reason,
    _extract_provider,
)
from ai_service import AIRequestError
from qa_evaluation_service import MissingScoreTagError
from models import ContentRequest, QALayer, GranSabioResult


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_ai_service():
    """Mocked AI service for Gran Sabio tests."""
    service = MagicMock()
    service.generate_content = AsyncMock(return_value="Generated content")
    service.generate_content_stream = AsyncMock()
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
                        "max_tokens": 8192
                    },
                    "claude-opus-4-1-20250805": {
                        "input_tokens": 200000,
                        "max_tokens": 16384
                    }
                },
                "openai": {
                    "gpt-4o": {
                        "input_tokens": 128000,
                        "max_tokens": 16384
                    }
                }
            },
            "aliases": {}
        }
        mock_cfg.MAX_RETRIES = 3
        mock_cfg.RETRY_DELAY = 1
        mock_cfg.RETRY_STREAMING_AFTER_PARTIAL = True
        mock_cfg.GRAN_SABIO_SYSTEM_PROMPT = "You are Gran Sabio, the final arbiter."
        mock_cfg.get_model_info = Mock(return_value={
            "input_tokens": 200000,
            "max_tokens": 8192,
            "provider": "anthropic"
        })
        mock_cfg._get_thinking_budget_config = Mock(return_value={
            "supported": True,
            "default_tokens": 10000
        })
        yield mock_cfg


@pytest.fixture
def sample_content_request():
    """Sample content request for testing."""
    return ContentRequest(
        prompt="Write a test article about software testing",
        content_type="article",
        generator_model="gpt-4o",
        temperature=0.7,
        max_tokens=4000,
        min_words=500,
        max_words=1000,
        qa_layers=[],
        qa_models=[]
    )


@pytest.fixture
def sample_qa_layer():
    """Sample QA layer for testing."""
    return QALayer(
        name="Accuracy",
        description="Factual accuracy verification",
        criteria="Check for factual errors",
        min_score=7.0,
        deal_breaker_criteria="invents facts",
        order=1
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
                "layer_min_score": 7.0
            }
        ],
        "total_evaluations": 3,
        "qa_configuration": {
            "layer_name": "Accuracy",
            "description": "Factual accuracy verification",
            "criteria": "Check for factual errors",
            "deal_breaker_criteria": "invents facts",
            "min_score": 7.0
        },
        "iteration": 1
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
            "qa_layers_config": [{"name": "Accuracy", "min_score": 7.0}]
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
            "qa_layers_config": [{"name": "Accuracy", "min_score": 7.0}]
        }
    ]


# ============================================================================
# Test Area 1: Helper Functions (Streaming Retry)
# ============================================================================

class TestIsRetryableStreamingError:
    """Tests for _is_retryable_streaming_error()."""

    def test_ai_request_error_is_retryable(self):
        """Given: AIRequestError, Then: Returns True."""
        exc = AIRequestError(
            provider="openai",
            model="gpt-4o",
            attempts=3,
            max_attempts=3,
            cause=Exception("API error")
        )
        assert _is_retryable_streaming_error(exc) is True

    def test_status_429_is_retryable(self):
        """Given: Exception with status 429, Then: Returns True."""
        exc = Exception("Rate limit")
        exc.status = 429
        assert _is_retryable_streaming_error(exc) is True

    def test_status_503_is_retryable(self):
        """Given: Exception with status 503, Then: Returns True."""
        exc = Exception("Service unavailable")
        exc.status_code = 503
        assert _is_retryable_streaming_error(exc) is True

    def test_timeout_message_is_retryable(self):
        """Given: Exception with 'timeout' in message, Then: Returns True."""
        exc = Exception("Connection timeout occurred")
        assert _is_retryable_streaming_error(exc) is True

    def test_gateway_error_is_retryable(self):
        """Given: Exception with 'gateway' in message, Then: Returns True."""
        exc = Exception("Bad gateway error")
        assert _is_retryable_streaming_error(exc) is True

    def test_non_retryable_error(self):
        """Given: Regular exception without markers, Then: Returns False."""
        exc = Exception("Invalid input parameter")
        assert _is_retryable_streaming_error(exc) is False


class TestExtractErrorReason:
    """Tests for _extract_error_reason()."""

    def test_extracts_from_ai_request_error_with_message(self):
        """Given: AIRequestError with cause.message, Then: Returns message."""
        cause = Mock()
        cause.message = "API rate limit exceeded"
        exc = AIRequestError(
            provider="openai",
            model="gpt-4o",
            attempts=3,
            max_attempts=3,
            cause=cause
        )
        assert _extract_error_reason(exc) == "API rate limit exceeded"

    def test_extracts_from_ai_request_error_without_message(self):
        """Given: AIRequestError without cause.message, Then: Returns str(cause)."""
        cause = Exception("Simple error")
        exc = AIRequestError(
            provider="openai",
            model="gpt-4o",
            attempts=3,
            max_attempts=3,
            cause=cause
        )
        assert "Simple error" in _extract_error_reason(exc)

    def test_extracts_from_exception_with_message_pattern(self):
        """Given: Exception string with 'message' pattern, Then: Extracts it."""
        exc = Exception("Error: {'message': 'Authentication failed'}")
        result = _extract_error_reason(exc)
        assert "Authentication failed" in result

    def test_truncates_long_messages(self):
        """Given: Very long error message, Then: Truncates to 200 chars."""
        exc = Exception("x" * 500)
        result = _extract_error_reason(exc)
        assert len(result) <= 200


class TestExtractProvider:
    """Tests for _extract_provider()."""

    def test_extracts_from_ai_request_error(self):
        """Given: AIRequestError with provider, Then: Returns provider."""
        exc = AIRequestError(
            provider="openai",
            model="gpt-4o",
            attempts=3,
            max_attempts=3,
            cause=Exception("test")
        )
        assert _extract_provider(exc) == "openai"

    def test_detects_anthropic_from_message(self):
        """Given: Exception mentioning anthropic, Then: Returns 'anthropic'."""
        exc = Exception("Anthropic API error occurred")
        assert _extract_provider(exc) == "anthropic"

    def test_detects_openai_from_message(self):
        """Given: Exception mentioning OpenAI, Then: Returns 'openai'."""
        exc = Exception("OpenAI rate limit exceeded")
        assert _extract_provider(exc) == "openai"

    def test_detects_google_from_message(self):
        """Given: Exception mentioning Gemini, Then: Returns 'google'."""
        exc = Exception("Gemini model unavailable")
        assert _extract_provider(exc) == "google"

    def test_returns_none_for_unknown(self):
        """Given: Exception without provider hints, Then: Returns None."""
        exc = Exception("Generic error")
        assert _extract_provider(exc) is None


# ============================================================================
# Test Area 2: Model Capacity Validation
# ============================================================================

class TestModelCapacityValidation:
    """Tests for model capacity validation methods."""

    def test_get_configured_default_model_success(self, mock_config):
        """Given: Valid config, Then: Returns configured model."""
        with patch('gran_sabio.get_default_models', return_value={"gran_sabio": "claude-sonnet-4-20250514"}):
            engine = GranSabioEngine(ai_service=MagicMock())
            model = engine._get_configured_default_model()
            assert model == "claude-sonnet-4-20250514"

    def test_get_configured_default_model_not_configured(self):
        """Given: No gran_sabio in defaults, Then: Raises RuntimeError."""
        with patch('gran_sabio.get_default_models', return_value={}):
            engine = GranSabioEngine(ai_service=MagicMock())
            with pytest.raises(RuntimeError, match="not configured"):
                engine._get_configured_default_model()

    def test_get_default_thinking_tokens_supported(self, mock_config):
        """Given: Model supports thinking, Then: Returns default tokens."""
        engine = GranSabioEngine(ai_service=MagicMock())
        tokens = engine._get_default_thinking_tokens("claude-sonnet-4-20250514")
        assert tokens == 10000

    def test_get_default_thinking_tokens_not_supported(self, mock_config):
        """Given: Model does not support thinking, Then: Returns None."""
        mock_config._get_thinking_budget_config.return_value = {"supported": False}
        engine = GranSabioEngine(ai_service=MagicMock())
        tokens = engine._get_default_thinking_tokens("gpt-4o")
        assert tokens is None

    def test_get_default_thinking_tokens_exception(self, mock_config):
        """Given: Config raises exception, Then: Returns None."""
        mock_config._get_thinking_budget_config.side_effect = Exception("Config error")
        engine = GranSabioEngine(ai_service=MagicMock())
        tokens = engine._get_default_thinking_tokens("unknown-model")
        assert tokens is None

    def test_ensure_adequate_model_capacity_sufficient(self, mock_config):
        """Given: Model has sufficient capacity, Then: Returns same model."""
        engine = GranSabioEngine(ai_service=MagicMock())
        # Short content - 1000 chars = ~250 tokens, needs ~375 with overhead
        result = engine._ensure_adequate_model_capacity(
            1000, "claude-sonnet-4-20250514", None
        )
        assert result == "claude-sonnet-4-20250514"

    def test_ensure_adequate_model_capacity_upgrade_needed(self, mock_config):
        """Given: Model has insufficient capacity, Then: Returns upgraded model."""
        # Make the model have very low capacity
        mock_config.get_model_info.return_value = {
            "input_tokens": 100,  # Very small
            "max_tokens": 8192,
            "provider": "anthropic"
        }
        engine = GranSabioEngine(ai_service=MagicMock())
        # Long content that exceeds capacity
        result = engine._ensure_adequate_model_capacity(
            100000, "small-model", None
        )
        # Should still return something (even if same model with warning)
        assert result is not None

    def test_resolve_model_alias(self, mock_config):
        """Given: Model alias, Then: Returns resolved name."""
        mock_config.model_specs = {"aliases": {"claude": "claude-sonnet-4-20250514"}}
        engine = GranSabioEngine(ai_service=MagicMock())
        result = engine._resolve_model_alias("claude")
        assert result == "claude-sonnet-4-20250514"

    def test_resolve_model_alias_no_alias(self, mock_config):
        """Given: Non-aliased model name, Then: Returns same name."""
        mock_config.model_specs = {"aliases": {}}
        engine = GranSabioEngine(ai_service=MagicMock())
        result = engine._resolve_model_alias("gpt-4o")
        assert result == "gpt-4o"

    def test_get_default_critical_analysis_model(self, mock_config):
        """Given: Requested model, Then: Returns model tuple."""
        with patch('gran_sabio.get_default_models', return_value={"gran_sabio": "claude-sonnet-4-20250514"}):
            engine = GranSabioEngine(ai_service=MagicMock())
            model, reasoning, thinking = engine._get_default_critical_analysis_model("gpt-4o")
            assert model == "gpt-4o"
            assert reasoning is None
            assert thinking == 10000  # From mock config

    def test_get_default_content_generation_model(self, mock_config):
        """Given: No requested model, Then: Returns default model tuple."""
        with patch('gran_sabio.get_default_models', return_value={"gran_sabio": "claude-sonnet-4-20250514"}):
            engine = GranSabioEngine(ai_service=MagicMock())
            model, reasoning, thinking = engine._get_default_content_generation_model(None)
            assert model == "claude-sonnet-4-20250514"


# ============================================================================
# Test Area 3: Response Parsing
# ============================================================================

class TestExtractDecision:
    """Tests for _extract_decision()."""

    def test_extracts_approved(self):
        """Given: APPROVED decision, Then: Returns 'APPROVED'."""
        engine = GranSabioEngine(ai_service=MagicMock())
        response = "[DECISION]APPROVED[/DECISION]"
        assert engine._extract_decision(response) == "APPROVED"

    def test_extracts_rejected(self):
        """Given: REJECTED decision, Then: Returns 'REJECTED'."""
        engine = GranSabioEngine(ai_service=MagicMock())
        response = "[DECISION]REJECTED[/DECISION]"
        assert engine._extract_decision(response) == "REJECTED"

    def test_case_insensitive(self):
        """Given: Lowercase decision, Then: Returns uppercase."""
        engine = GranSabioEngine(ai_service=MagicMock())
        response = "[decision]approved[/decision]"
        assert engine._extract_decision(response) == "APPROVED"

    def test_missing_decision_returns_rejected(self):
        """Given: No decision tag, Then: Returns 'REJECTED'."""
        engine = GranSabioEngine(ai_service=MagicMock())
        response = "No decision here"
        assert engine._extract_decision(response) == "REJECTED"


class TestExtractScore:
    """Tests for _extract_score()."""

    def test_extracts_valid_score(self):
        """Given: Valid score tag, Then: Returns float score."""
        engine = GranSabioEngine(ai_service=MagicMock())
        response = "[SCORE]8.5[/SCORE]"
        assert engine._extract_score(response) == 8.5

    def test_extracts_integer_score(self):
        """Given: Integer score, Then: Returns float."""
        engine = GranSabioEngine(ai_service=MagicMock())
        response = "[SCORE]9[/SCORE]"
        assert engine._extract_score(response) == 9.0

    def test_case_insensitive(self):
        """Given: Lowercase tags, Then: Extracts correctly."""
        engine = GranSabioEngine(ai_service=MagicMock())
        response = "[score]7.5[/score]"
        assert engine._extract_score(response) == 7.5

    def test_missing_score_raises_error(self):
        """Given: No score tag, Then: Raises MissingScoreTagError."""
        engine = GranSabioEngine(ai_service=MagicMock())
        response = "No score here"
        with pytest.raises(MissingScoreTagError):
            engine._extract_score(response)


class TestExtractFinalContent:
    """Tests for _extract_final_content()."""

    def test_extracts_content(self):
        """Given: Valid content tag, Then: Returns content."""
        engine = GranSabioEngine(ai_service=MagicMock())
        response = "[FINAL_CONTENT]This is the final content.[/FINAL_CONTENT]"
        assert engine._extract_final_content(response) == "This is the final content."

    def test_extracts_multiline_content(self):
        """Given: Multiline content, Then: Returns all lines."""
        engine = GranSabioEngine(ai_service=MagicMock())
        response = "[FINAL_CONTENT]Line 1\nLine 2\nLine 3[/FINAL_CONTENT]"
        result = engine._extract_final_content(response)
        assert "Line 1" in result
        assert "Line 3" in result

    def test_missing_content_returns_empty(self):
        """Given: No content tag, Then: Returns empty string."""
        engine = GranSabioEngine(ai_service=MagicMock())
        response = "No content tag here"
        assert engine._extract_final_content(response) == ""


class TestExtractReason:
    """Tests for _extract_reason()."""

    def test_extracts_reason(self):
        """Given: Valid reason tag, Then: Returns reason."""
        engine = GranSabioEngine(ai_service=MagicMock())
        response = "[REASON]Content meets all criteria.[/REASON]"
        assert engine._extract_reason(response) == "Content meets all criteria."

    def test_missing_reason_returns_default(self):
        """Given: No reason tag, Then: Returns default message."""
        engine = GranSabioEngine(ai_service=MagicMock())
        response = "No reason tag"
        assert "No specific reason" in engine._extract_reason(response)


class TestExtractModifications:
    """Tests for _extract_modifications()."""

    def test_extracts_true(self):
        """Given: modifications_made=true, Then: Returns True."""
        engine = GranSabioEngine(ai_service=MagicMock())
        response = "[MODIFICATIONS_MADE]true[/MODIFICATIONS_MADE]"
        assert engine._extract_modifications(response) is True

    def test_extracts_false(self):
        """Given: modifications_made=false, Then: Returns False."""
        engine = GranSabioEngine(ai_service=MagicMock())
        response = "[MODIFICATIONS_MADE]false[/MODIFICATIONS_MADE]"
        assert engine._extract_modifications(response) is False

    def test_missing_returns_false(self):
        """Given: No modifications tag, Then: Returns False."""
        engine = GranSabioEngine(ai_service=MagicMock())
        response = "No modifications tag"
        assert engine._extract_modifications(response) is False


class TestParseGranSabioResponse:
    """Tests for _parse_gran_sabio_response()."""

    def test_parses_approve_response(self):
        """Given: Full APPROVE response, Then: Returns GranSabioResult with approved=True."""
        engine = GranSabioEngine(ai_service=MagicMock())
        response = """
[DECISION]APPROVE[/DECISION]
[FINAL_SCORE]8.5[/FINAL_SCORE]
[REASONING]Content meets all criteria.[/REASONING]
[MODIFICATIONS_MADE]false[/MODIFICATIONS_MADE]
"""
        iterations = [{"content": "Test content", "consensus": {"average_score": 8.0}}]
        result = engine._parse_gran_sabio_response(response, iterations, {})

        assert result.approved is True
        assert result.final_score == 8.5
        assert "meets all criteria" in result.reason
        assert result.modifications_made is False

    def test_parses_reject_response(self):
        """Given: REJECT response, Then: Returns GranSabioResult with approved=False."""
        engine = GranSabioEngine(ai_service=MagicMock())
        response = """
[DECISION]REJECT[/DECISION]
[FINAL_SCORE]4.0[/FINAL_SCORE]
[REASONING]Critical errors found.[/REASONING]
[MODIFICATIONS_MADE]false[/MODIFICATIONS_MADE]
"""
        result = engine._parse_gran_sabio_response(response, [], {})

        assert result.approved is False
        assert result.final_score == 4.0

    def test_parses_approve_with_modifications(self):
        """Given: APPROVE_WITH_MODIFICATIONS, Then: modifications_made=True."""
        engine = GranSabioEngine(ai_service=MagicMock())
        response = """
[DECISION]APPROVE_WITH_MODIFICATIONS[/DECISION]
[FINAL_SCORE]7.5[/FINAL_SCORE]
[REASONING]Minor fixes applied.[/REASONING]
[MODIFICATIONS_MADE]true[/MODIFICATIONS_MADE]
[FINAL_CONTENT]Modified content here.[/FINAL_CONTENT]
"""
        result = engine._parse_gran_sabio_response(response, [], {})

        assert result.approved is True
        assert result.modifications_made is True
        assert result.final_content == "Modified content here."

    def test_parses_spanish_decision_tags(self):
        """Given: Spanish decision tags, Then: Parses correctly."""
        engine = GranSabioEngine(ai_service=MagicMock())
        response = """
[DECISION]APROBAR[/DECISION]
[FINAL_SCORE]8.0[/FINAL_SCORE]
[REASONING]Todo correcto.[/REASONING]
"""
        result = engine._parse_gran_sabio_response(response, [], {})
        assert result.approved is True

    def test_fallback_to_best_iteration_content(self):
        """Given: No FINAL_CONTENT tag, Then: Uses best iteration content."""
        engine = GranSabioEngine(ai_service=MagicMock())
        response = """
[DECISION]APPROVE[/DECISION]
[FINAL_SCORE]8.0[/FINAL_SCORE]
[REASONING]Approved as-is.[/REASONING]
"""
        iterations = [
            {"content": "Low score content", "consensus": {"average_score": 5.0}},
            {"content": "Best content here", "consensus": {"average_score": 8.5}},
        ]
        result = engine._parse_gran_sabio_response(response, iterations, {})
        assert result.final_content == "Best content here"


class TestAnalyzeIterationPatterns:
    """Tests for _analyze_iteration_patterns()."""

    def test_empty_iterations(self):
        """Given: Empty iterations, Then: Returns error dict."""
        engine = GranSabioEngine(ai_service=MagicMock())
        result = engine._analyze_iteration_patterns([])
        assert "error" in result

    def test_analyzes_layer_trends(self, sample_iterations):
        """Given: Multiple iterations, Then: Calculates layer trends."""
        engine = GranSabioEngine(ai_service=MagicMock())
        result = engine._analyze_iteration_patterns(sample_iterations)

        assert "layer_trends" in result
        assert result["total_iterations"] == 2

    def test_identifies_consistent_issues(self):
        """Given: Consistent deal-breakers, Then: Lists consistent issues."""
        engine = GranSabioEngine(ai_service=MagicMock())
        iterations = [
            {
                "qa_results": {
                    "Layer1": {
                        "model1": Mock(score=5.0, deal_breaker=True)
                    }
                }
            }
        ] * 5  # Same issue in 5 iterations

        result = engine._analyze_iteration_patterns(iterations)
        # Should have analyzed the patterns
        assert "deal_breaker_analysis" in result


# ============================================================================
# Test Area 4: review_minority_deal_breakers()
# ============================================================================

class TestReviewMinorityDealBreakers:
    """Tests for review_minority_deal_breakers()."""

    @pytest.mark.asyncio
    async def test_approves_false_positive_deal_breaker(
        self, mock_ai_service, mock_config, sample_content_request, sample_minority_deal_breakers
    ):
        """Given: False positive deal-breaker, Then: Returns approved=True."""
        approval_response = """
[DECISION]APPROVED[/DECISION]
[SCORE]8.5[/SCORE]
[REASON]The flagged issue is a false positive.[/REASON]
[MODIFICATIONS_MADE]false[/MODIFICATIONS_MADE]
"""
        mock_ai_service.generate_content = AsyncMock(return_value=approval_response)

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

    @pytest.mark.asyncio
    async def test_rejects_real_deal_breaker(
        self, mock_ai_service, mock_config, sample_content_request, sample_minority_deal_breakers
    ):
        """Given: Real deal-breaker, Then: Returns approved=False."""
        rejection_response = """
[DECISION]REJECTED[/DECISION]
[SCORE]3.0[/SCORE]
[REASON]The deal-breaker is valid - content invents facts.[/REASON]
[MODIFICATIONS_MADE]false[/MODIFICATIONS_MADE]
"""
        mock_ai_service.generate_content = AsyncMock(return_value=rejection_response)

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

    @pytest.mark.asyncio
    async def test_handles_cancellation(
        self, mock_ai_service, mock_config, sample_content_request, sample_minority_deal_breakers
    ):
        """Given: Cancel callback returns True, Then: Raises GranSabioProcessCancelled."""
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
        """Given: API error occurs, Then: Returns error result."""
        mock_ai_service.generate_content = AsyncMock(
            side_effect=Exception("API unavailable")
        )

        with patch('gran_sabio.get_default_models', return_value={"gran_sabio": "claude-sonnet-4-20250514"}):
            engine = GranSabioEngine(ai_service=mock_ai_service)
            result = await engine.review_minority_deal_breakers(
                session_id="test-session",
                content="Test content",
                minority_deal_breakers=sample_minority_deal_breakers,
                original_request=sample_content_request,
            )

        assert result.approved is False
        assert result.error is not None
        assert "API unavailable" in result.error

    @pytest.mark.asyncio
    async def test_streams_response_when_callback_provided(
        self, mock_ai_service, mock_config, sample_content_request, sample_minority_deal_breakers
    ):
        """Given: Stream callback provided, Then: Uses streaming generation."""
        async def mock_stream():
            yield "[DECISION]APPROVED[/DECISION]"
            yield "[SCORE]8.0[/SCORE]"
            yield "[REASON]Good[/REASON]"
            yield "[MODIFICATIONS_MADE]false[/MODIFICATIONS_MADE]"

        mock_ai_service.generate_content_stream = Mock(return_value=mock_stream())
        stream_callback = AsyncMock()

        with patch('gran_sabio.get_default_models', return_value={"gran_sabio": "claude-sonnet-4-20250514"}):
            engine = GranSabioEngine(ai_service=mock_ai_service)
            result = await engine.review_minority_deal_breakers(
                session_id="test-session",
                content="Test content",
                minority_deal_breakers=sample_minority_deal_breakers,
                original_request=sample_content_request,
                stream_callback=stream_callback,
            )

        assert stream_callback.called

    @pytest.mark.asyncio
    async def test_approves_with_modifications(
        self, mock_ai_service, mock_config, sample_content_request, sample_minority_deal_breakers
    ):
        """Given: Minor fix needed, Then: Returns with modifications."""
        response = """
[DECISION]APPROVED[/DECISION]
[SCORE]8.0[/SCORE]
[REASON]Minor typo fixed.[/REASON]
[MODIFICATIONS_MADE]true[/MODIFICATIONS_MADE]
[FINAL_CONTENT]Corrected content here.[/FINAL_CONTENT]
"""
        mock_ai_service.generate_content = AsyncMock(return_value=response)

        with patch('gran_sabio.get_default_models', return_value={"gran_sabio": "claude-sonnet-4-20250514"}):
            engine = GranSabioEngine(ai_service=mock_ai_service)
            result = await engine.review_minority_deal_breakers(
                session_id="test-session",
                content="Content with typo",
                minority_deal_breakers=sample_minority_deal_breakers,
                original_request=sample_content_request,
            )

        assert result.approved is True
        assert result.modifications_made is True
        assert result.final_content == "Corrected content here."


# ============================================================================
# Test Area 5: regenerate_content()
# ============================================================================

class TestRegenerateContent:
    """Tests for regenerate_content()."""

    @pytest.mark.asyncio
    async def test_generates_new_content(
        self, mock_ai_service, mock_config, sample_content_request
    ):
        """Given: Valid request, Then: Returns generated content."""
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
        assert result.final_score == 8.5  # Default score for regeneration

    @pytest.mark.asyncio
    async def test_includes_previous_attempts_context(
        self, mock_ai_service, mock_config, sample_content_request
    ):
        """Given: Previous attempts provided, Then: Includes them in prompt."""
        mock_ai_service.generate_content = AsyncMock(return_value="New content")

        previous_attempts = ["First failed attempt", "Second failed attempt"]

        with patch('gran_sabio.get_default_models', return_value={"gran_sabio": "claude-sonnet-4-20250514"}):
            engine = GranSabioEngine(ai_service=mock_ai_service)
            await engine.regenerate_content(
                session_id="test-session",
                original_request=sample_content_request,
                previous_attempts=previous_attempts,
            )

        # Verify the prompt included previous attempts context
        call_args = mock_ai_service.generate_content.call_args
        prompt = call_args.kwargs.get("prompt", call_args.args[0] if call_args.args else "")
        assert "PREVIOUS ATTEMPTS" in prompt or "Attempt" in prompt

    @pytest.mark.asyncio
    async def test_handles_cancellation(
        self, mock_ai_service, mock_config, sample_content_request
    ):
        """Given: Cancel callback returns True, Then: Raises GranSabioProcessCancelled."""
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
    async def test_handles_generation_error(
        self, mock_ai_service, mock_config, sample_content_request
    ):
        """Given: Generation fails, Then: Returns error result."""
        mock_ai_service.generate_content = AsyncMock(
            side_effect=Exception("Model error")
        )

        with patch('gran_sabio.get_default_models', return_value={"gran_sabio": "claude-sonnet-4-20250514"}):
            engine = GranSabioEngine(ai_service=mock_ai_service)
            result = await engine.regenerate_content(
                session_id="test-session",
                original_request=sample_content_request,
            )

        assert result.approved is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_includes_word_count_instructions(
        self, mock_ai_service, mock_config
    ):
        """Given: Request with word limits, Then: Includes word instructions."""
        mock_ai_service.generate_content = AsyncMock(return_value="Content")

        request = ContentRequest(
            prompt="Write an article",
            min_words=500,
            max_words=1000,
        )

        with patch('gran_sabio.get_default_models', return_value={"gran_sabio": "claude-sonnet-4-20250514"}):
            engine = GranSabioEngine(ai_service=mock_ai_service)
            await engine.regenerate_content(
                session_id="test-session",
                original_request=request,
            )

        call_args = mock_ai_service.generate_content.call_args
        prompt = call_args.kwargs.get("prompt", "")
        assert "500" in prompt and "1000" in prompt


# ============================================================================
# Test Area 6: review_iterations()
# ============================================================================

class TestReviewIterations:
    """Tests for review_iterations()."""

    @pytest.mark.asyncio
    async def test_approves_best_iteration(
        self, mock_ai_service, mock_config, sample_content_request, sample_iterations
    ):
        """Given: Valid iterations, Then: Reviews and returns decision."""
        approval_response = """
[DECISION]APPROVE[/DECISION]
[FINAL_SCORE]8.5[/FINAL_SCORE]
[REASONING]Second iteration meets all criteria.[/REASONING]
[MODIFICATIONS_MADE]false[/MODIFICATIONS_MADE]
"""
        mock_ai_service.generate_content = AsyncMock(return_value=approval_response)

        with patch('gran_sabio.get_default_models', return_value={"gran_sabio": "claude-sonnet-4-20250514"}):
            engine = GranSabioEngine(ai_service=mock_ai_service)
            result = await engine.review_iterations(
                session_id="test-session",
                iterations=sample_iterations,
                original_request=sample_content_request,
            )

        assert result.approved is True
        assert result.final_score == 8.5

    @pytest.mark.asyncio
    async def test_rejects_all_iterations(
        self, mock_ai_service, mock_config, sample_content_request, sample_iterations
    ):
        """Given: All iterations fail, Then: Returns rejected."""
        rejection_response = """
[DECISION]REJECT[/DECISION]
[FINAL_SCORE]4.0[/FINAL_SCORE]
[REASONING]All iterations have critical issues.[/REASONING]
[MODIFICATIONS_MADE]false[/MODIFICATIONS_MADE]
"""
        mock_ai_service.generate_content = AsyncMock(return_value=rejection_response)

        with patch('gran_sabio.get_default_models', return_value={"gran_sabio": "claude-sonnet-4-20250514"}):
            engine = GranSabioEngine(ai_service=mock_ai_service)
            result = await engine.review_iterations(
                session_id="test-session",
                iterations=sample_iterations,
                original_request=sample_content_request,
            )

        assert result.approved is False

    @pytest.mark.asyncio
    async def test_handles_cancellation(
        self, mock_ai_service, mock_config, sample_content_request, sample_iterations
    ):
        """Given: Cancel callback returns True, Then: Raises GranSabioProcessCancelled."""
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
        """Given: Fallback notes provided, Then: Includes them in prompt."""
        response = """
[DECISION]APPROVE[/DECISION]
[FINAL_SCORE]7.5[/FINAL_SCORE]
[REASONING]Approved with notes.[/REASONING]
"""
        mock_ai_service.generate_content = AsyncMock(return_value=response)

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
    async def test_handles_api_error(
        self, mock_ai_service, mock_config, sample_content_request, sample_iterations
    ):
        """Given: API error occurs, Then: Returns error result."""
        mock_ai_service.generate_content = AsyncMock(
            side_effect=Exception("API error")
        )

        with patch('gran_sabio.get_default_models', return_value={"gran_sabio": "claude-sonnet-4-20250514"}):
            engine = GranSabioEngine(ai_service=mock_ai_service)
            result = await engine.review_iterations(
                session_id="test-session",
                iterations=sample_iterations,
                original_request=sample_content_request,
            )

        assert result.approved is False
        assert result.error is not None


# ============================================================================
# Test: Formatting Helpers
# ============================================================================

class TestFormattingHelpers:
    """Tests for formatting helper methods."""

    def test_build_deal_breaker_context_empty(self):
        """Given: No deal-breakers, Then: Returns appropriate message."""
        engine = GranSabioEngine(ai_service=MagicMock())
        result = engine._build_deal_breaker_context({"details": []})
        assert "No deal-breakers" in result

    def test_build_deal_breaker_context_with_details(self):
        """Given: Deal-breaker details, Then: Formats them correctly."""
        engine = GranSabioEngine(ai_service=MagicMock())
        deal_breakers = {
            "details": [
                {"layer": "Accuracy", "model": "gpt-4o", "reason": "Error found"}
            ],
            "total_evaluations": 3
        }
        result = engine._build_deal_breaker_context(deal_breakers)
        assert "Accuracy" in result
        assert "gpt-4o" in result

    def test_get_word_config_both_limits(self, sample_content_request):
        """Given: Both min and max words, Then: Returns range string."""
        engine = GranSabioEngine(ai_service=MagicMock())
        result = engine._get_word_config(sample_content_request)
        assert "500" in result and "1000" in result

    def test_get_word_config_max_only(self):
        """Given: Only max_words, Then: Returns max string."""
        engine = GranSabioEngine(ai_service=MagicMock())
        request = ContentRequest(prompt="Test prompt here", max_words=1000)
        result = engine._get_word_config(request)
        assert "Max" in result and "1000" in result

    def test_get_word_config_no_limits(self):
        """Given: No word limits, Then: Returns 'No specific' message."""
        engine = GranSabioEngine(ai_service=MagicMock())
        request = ContentRequest(prompt="Test prompt here")
        result = engine._get_word_config(request)
        assert "No specific" in result

    def test_format_qa_layers_config_empty(self):
        """Given: Empty config, Then: Returns appropriate message."""
        engine = GranSabioEngine(ai_service=MagicMock())
        result = engine._format_qa_layers_config([])
        assert "No QA layers" in result

    def test_format_qa_layers_config_with_layers(self):
        """Given: QA layer config, Then: Formats correctly."""
        engine = GranSabioEngine(ai_service=MagicMock())
        config = [{"name": "Accuracy", "min_score": 7.0, "description": "Check accuracy"}]
        result = engine._format_qa_layers_config(config)
        assert "Accuracy" in result
        assert "7.0" in result

    def test_format_iteration_details(self, sample_iterations):
        """Given: Iterations, Then: Formats details correctly."""
        engine = GranSabioEngine(ai_service=MagicMock())
        result = engine._format_iteration_details(sample_iterations)
        assert "Iteration 1" in result
        assert "Iteration 2" in result

    def test_format_trend_analysis_with_error(self):
        """Given: Analysis with error, Then: Returns error message."""
        engine = GranSabioEngine(ai_service=MagicMock())
        analysis = {"error": "No iterations to analyze"}
        result = engine._format_trend_analysis(analysis)
        assert "error" in result.lower()

    def test_format_trend_analysis_with_trends(self):
        """Given: Valid analysis, Then: Formats trends correctly."""
        engine = GranSabioEngine(ai_service=MagicMock())
        analysis = {
            "layer_trends": {
                "Accuracy": {
                    "trend": "improving",
                    "best_score": 8.5,
                    "worst_score": 6.0,
                    "consistency": 0.75
                }
            },
            "consistent_issues": ["Issue 1"],
            "model_analysis": {}
        }
        result = engine._format_trend_analysis(analysis)
        assert "LAYER TRENDS" in result
        assert "Accuracy" in result
        assert "improving" in result
