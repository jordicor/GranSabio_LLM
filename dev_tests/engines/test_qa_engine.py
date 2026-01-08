"""
Tests for qa_engine.py - Multi-layer Quality Assurance Engine.

This module tests the QAEngine which evaluates content through multiple
QA layers using different AI models with deal-breaker detection and
Gran Sabio escalation.

Functions tested:
- calculate_qa_timeout_for_model(): Single model timeout calculation
- calculate_comprehensive_qa_timeout(): Multi-layer timeout calculation
- QAEngine.__init__(): Initialization with AI service and bypass engine
- QAEngine._should_request_edit_info(): Edit info request logic
- QAEngine._increment_model_failure(): Failure tracking
- QAEngine._reset_model_failure(): Failure reset
- QAEngine._qa_failure_threshold(): Failure threshold getter
- QAEngine.evaluate_content(): Single evaluation
- QAEngine.evaluate_all_layers(): Non-progress evaluation
- QAEngine.evaluate_all_layers_with_progress(): Progress-tracked evaluation
- QAEngine._check_deal_breaker_consensus(): Consensus checking
- QAEngine._create_iteration_stop_result(): Iteration stop result
- QAEngine._create_gran_sabio_modified_result(): Modified content result
- QAEngine._calculate_summary(): Summary statistics
- QAEngine._identify_critical_issues(): Issue identification
- QAEngine.validate_layers(): Layer validation
"""

import pytest
import asyncio
from unittest.mock import Mock, MagicMock, AsyncMock, patch
from datetime import datetime

from qa_engine import (
    QAEngine,
    QAProcessCancelled,
    QAModelUnavailableError,
    calculate_qa_timeout_for_model,
    calculate_comprehensive_qa_timeout,
)
from models import QALayer, QAEvaluation, QAModelConfig


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_ai_service():
    """Create a mocked AI service."""
    service = MagicMock()
    service.generate_content = AsyncMock(return_value="AI evaluation response")
    service.generate_content_stream = AsyncMock()
    return service


@pytest.fixture
def mock_bypass_engine():
    """Create a mocked bypass engine."""
    bypass = MagicMock()
    bypass.should_bypass_qa_layer = Mock(return_value=False)
    bypass.bypass_layer_evaluation = Mock(return_value={})
    return bypass


@pytest.fixture
def mock_qa_evaluator():
    """Create a mocked QA evaluation service."""
    evaluator = MagicMock()
    evaluator.evaluate_content = AsyncMock()
    return evaluator


@pytest.fixture
def qa_engine(mock_ai_service, mock_bypass_engine):
    """Create a QAEngine with mocked dependencies."""
    engine = QAEngine(ai_service=mock_ai_service, bypass_engine=mock_bypass_engine)
    return engine


@pytest.fixture
def sample_qa_layer():
    """Create a sample QA layer."""
    return QALayer(
        name="Test Quality",
        description="Test quality check",
        criteria="Check for test quality issues",
        min_score=7.0,
        order=1
    )


@pytest.fixture
def sample_qa_layer_deal_breaker():
    """Create a sample deal-breaker QA layer."""
    return QALayer(
        name="Accuracy",
        description="Factual accuracy check",
        criteria="Check for factual errors",
        min_score=8.0,
        order=1,
        is_deal_breaker=True,
        deal_breaker_criteria="Contains fabricated or false information"
    )


@pytest.fixture
def sample_layers():
    """Create sample QA layers for testing."""
    return [
        QALayer(
            name="Accuracy",
            description="Factual accuracy check",
            criteria="Check for factual errors",
            min_score=7.0,
            order=1
        ),
        QALayer(
            name="Style",
            description="Writing style check",
            criteria="Check writing quality",
            min_score=6.0,
            order=2
        ),
        QALayer(
            name="Tone",
            description="Tone consistency check",
            criteria="Check tone appropriateness",
            min_score=5.0,
            order=3
        )
    ]


@pytest.fixture
def sample_qa_evaluation():
    """Create a sample passing QA evaluation."""
    return QAEvaluation(
        model="gpt-4o",
        layer="Test Quality",
        score=8.5,
        feedback="Content meets quality standards",
        deal_breaker=False,
        passes_score=True
    )


@pytest.fixture
def sample_qa_evaluation_deal_breaker():
    """Create a sample deal-breaker QA evaluation."""
    return QAEvaluation(
        model="gpt-4o",
        layer="Accuracy",
        score=3.0,
        feedback="Critical factual errors detected",
        deal_breaker=True,
        deal_breaker_reason="Contains fabricated statistics",
        passes_score=False
    )


@pytest.fixture
def sample_qa_model_config():
    """Create a sample QA model config."""
    return QAModelConfig(
        model="gpt-4o",
        max_tokens=8000,
        reasoning_effort=None,
        thinking_budget_tokens=None,
        temperature=0.3
    )


@pytest.fixture
def sample_qa_model_config_reasoning():
    """Create a QA model config with reasoning effort."""
    return QAModelConfig(
        model="o3-mini",
        max_tokens=16000,
        reasoning_effort="medium",
        thinking_budget_tokens=None,
        temperature=None
    )


@pytest.fixture
def sample_content_request():
    """Create a sample content request."""
    request = Mock()
    request.content_type = "article"
    request.smart_editing_mode = "auto"
    request.extra_verbose = False
    request.gran_sabio_call_limit_per_iteration = -1
    return request


# ============================================================================
# Test: Timeout Calculation
# ============================================================================

class TestCalculateQATimeoutForModel:
    """Tests for calculate_qa_timeout_for_model() function."""

    def test_non_reasoning_model_returns_base_timeout(self):
        """
        Given: A non-reasoning model
        When: calculate_qa_timeout_for_model() is called
        Then: Returns base QA timeout
        """
        with patch('qa_engine.config') as mock_config:
            mock_config.get_reasoning_timeout_seconds = Mock(return_value=None)
            mock_config.QA_BASE_TIMEOUT = 120

            timeout = calculate_qa_timeout_for_model("gpt-4o")

            assert timeout == 120.0

    def test_reasoning_model_applies_multiplier(self):
        """
        Given: A reasoning model with specific timeout
        When: calculate_qa_timeout_for_model() is called
        Then: Returns base timeout multiplied by QA multiplier
        """
        with patch('qa_engine.config') as mock_config:
            mock_config.get_reasoning_timeout_seconds = Mock(return_value=300)
            mock_config.QA_TIMEOUT_MULTIPLIER = 1.5

            timeout = calculate_qa_timeout_for_model("o3-mini", reasoning_effort="medium")

            assert timeout == 450.0  # 300 * 1.5

    def test_with_thinking_budget_tokens(self):
        """
        Given: A model with thinking budget tokens
        When: calculate_qa_timeout_for_model() is called
        Then: Passes thinking_budget_tokens to config
        """
        with patch('qa_engine.config') as mock_config:
            mock_config.get_reasoning_timeout_seconds = Mock(return_value=180)
            mock_config.QA_TIMEOUT_MULTIPLIER = 2.0

            timeout = calculate_qa_timeout_for_model(
                "claude-sonnet-4",
                thinking_budget_tokens=10000
            )

            mock_config.get_reasoning_timeout_seconds.assert_called_once_with(
                "claude-sonnet-4", None, 10000
            )
            assert timeout == 360.0  # 180 * 2.0

    def test_zero_reasoning_timeout_uses_base(self):
        """
        Given: A model with zero reasoning timeout
        When: calculate_qa_timeout_for_model() is called
        Then: Falls back to base timeout
        """
        with patch('qa_engine.config') as mock_config:
            mock_config.get_reasoning_timeout_seconds = Mock(return_value=0)
            mock_config.QA_BASE_TIMEOUT = 90

            timeout = calculate_qa_timeout_for_model("gpt-4o-mini")

            assert timeout == 90.0

    def test_negative_reasoning_timeout_uses_base(self):
        """
        Given: A model with negative reasoning timeout
        When: calculate_qa_timeout_for_model() is called
        Then: Falls back to base timeout
        """
        with patch('qa_engine.config') as mock_config:
            mock_config.get_reasoning_timeout_seconds = Mock(return_value=-10)
            mock_config.QA_BASE_TIMEOUT = 120

            timeout = calculate_qa_timeout_for_model("gpt-4o")

            assert timeout == 120.0


class TestCalculateComprehensiveQATimeout:
    """Tests for calculate_comprehensive_qa_timeout() function."""

    def test_single_layer_single_model(self):
        """
        Given: One layer and one model
        When: calculate_comprehensive_qa_timeout() is called
        Then: Returns model timeout plus margin
        """
        layers = [QALayer(name="Test", description="Test", criteria="Test", min_score=7.0, order=1)]
        models = ["gpt-4o"]

        with patch('qa_engine.config') as mock_config:
            mock_config.get_reasoning_timeout_seconds = Mock(return_value=None)
            mock_config.QA_BASE_TIMEOUT = 120
            mock_config.QA_COMPREHENSIVE_TIMEOUT_MARGIN = 60

            timeout = calculate_comprehensive_qa_timeout(layers, models)

            # 120 * 1 layer + 60 margin = 180
            assert timeout == 180.0

    def test_multiple_layers_takes_max_timeout(self):
        """
        Given: Multiple layers and models
        When: calculate_comprehensive_qa_timeout() is called
        Then: Takes max model timeout multiplied by layer count
        """
        layers = [
            QALayer(name="L1", description="L1", criteria="L1", min_score=7.0, order=1),
            QALayer(name="L2", description="L2", criteria="L2", min_score=7.0, order=2),
        ]
        models = ["gpt-4o", "claude-sonnet-4"]

        with patch('qa_engine.config') as mock_config:
            mock_config.get_reasoning_timeout_seconds = Mock(return_value=None)
            mock_config.QA_BASE_TIMEOUT = 100
            mock_config.QA_COMPREHENSIVE_TIMEOUT_MARGIN = 30

            timeout = calculate_comprehensive_qa_timeout(layers, models)

            # max(100, 100) * 2 layers + 30 margin = 230
            assert timeout == 230.0

    def test_with_qa_model_config_objects(self):
        """
        Given: QAModelConfig objects instead of strings
        When: calculate_comprehensive_qa_timeout() is called
        Then: Extracts model name and config from objects
        """
        layers = [QALayer(name="Test", description="Test", criteria="Test", min_score=7.0, order=1)]
        models = [
            QAModelConfig(model="o3-mini", reasoning_effort="high"),
        ]

        with patch('qa_engine.config') as mock_config:
            mock_config.get_reasoning_timeout_seconds = Mock(return_value=600)
            mock_config.QA_TIMEOUT_MULTIPLIER = 1.5
            mock_config.QA_COMPREHENSIVE_TIMEOUT_MARGIN = 60

            timeout = calculate_comprehensive_qa_timeout(layers, models)

            # 600 * 1.5 * 1 layer + 60 margin = 960
            assert timeout == 960.0


# ============================================================================
# Test: QAEngine Initialization
# ============================================================================

class TestQAEngineInit:
    """Tests for QAEngine.__init__()."""

    def test_init_with_provided_ai_service(self, mock_ai_service, mock_bypass_engine):
        """
        Given: An AI service is provided
        When: QAEngine is initialized
        Then: Uses provided AI service
        """
        engine = QAEngine(ai_service=mock_ai_service, bypass_engine=mock_bypass_engine)

        assert engine.ai_service is mock_ai_service
        assert engine.bypass_engine is mock_bypass_engine

    def test_init_without_ai_service_uses_default(self, mock_bypass_engine):
        """
        Given: No AI service is provided
        When: QAEngine is initialized
        Then: Gets default AI service from get_ai_service()
        """
        with patch('qa_engine.get_ai_service') as mock_get_service:
            mock_service = MagicMock()
            mock_get_service.return_value = mock_service

            engine = QAEngine(bypass_engine=mock_bypass_engine)

            mock_get_service.assert_called_once()
            assert engine.ai_service is mock_service

    def test_init_creates_qa_evaluator(self, mock_ai_service, mock_bypass_engine):
        """
        Given: Dependencies provided
        When: QAEngine is initialized
        Then: Creates QAEvaluationService with AI service
        """
        engine = QAEngine(ai_service=mock_ai_service, bypass_engine=mock_bypass_engine)

        assert engine.qa_evaluator is not None

    def test_init_creates_failure_tracker(self, mock_ai_service, mock_bypass_engine):
        """
        Given: Dependencies provided
        When: QAEngine is initialized
        Then: Creates empty failure tracker dict
        """
        engine = QAEngine(ai_service=mock_ai_service, bypass_engine=mock_bypass_engine)

        assert engine._qa_failure_tracker == {}


# ============================================================================
# Test: Should Request Edit Info
# ============================================================================

class TestShouldRequestEditInfo:
    """Tests for QAEngine._should_request_edit_info()."""

    def test_mode_never_returns_false(self, qa_engine):
        """
        Given: mode='never'
        When: _should_request_edit_info() is called
        Then: Returns False regardless of content_type
        """
        result = qa_engine._should_request_edit_info(mode="never", content_type="biography")

        assert result is False

    def test_mode_always_returns_true(self, qa_engine):
        """
        Given: mode='always'
        When: _should_request_edit_info() is called
        Then: Returns True regardless of content_type
        """
        result = qa_engine._should_request_edit_info(mode="always", content_type="other")

        assert result is True

    def test_mode_auto_biography_returns_true(self, qa_engine):
        """
        Given: mode='auto' and content_type='biography'
        When: _should_request_edit_info() is called
        Then: Returns True (biography is editable)
        """
        result = qa_engine._should_request_edit_info(mode="auto", content_type="biography")

        assert result is True

    def test_mode_auto_article_returns_true(self, qa_engine):
        """
        Given: mode='auto' and content_type='article'
        When: _should_request_edit_info() is called
        Then: Returns True (article is editable)
        """
        result = qa_engine._should_request_edit_info(mode="auto", content_type="article")

        assert result is True

    def test_mode_auto_other_returns_false(self, qa_engine):
        """
        Given: mode='auto' and content_type='other'
        When: _should_request_edit_info() is called
        Then: Returns False (other is not editable)
        """
        result = qa_engine._should_request_edit_info(mode="auto", content_type="other")

        assert result is False

    def test_mode_auto_all_editable_types(self, qa_engine):
        """
        Given: mode='auto' and various editable content types
        When: _should_request_edit_info() is called
        Then: Returns True for all editable types
        """
        editable_types = ["biography", "article", "script", "story", "essay", "blog", "novel"]

        for content_type in editable_types:
            result = qa_engine._should_request_edit_info(mode="auto", content_type=content_type)
            assert result is True, f"Expected True for {content_type}"


# ============================================================================
# Test: Failure Tracking
# ============================================================================

class TestFailureTracking:
    """Tests for failure tracking methods."""

    def test_increment_model_failure_without_session(self, qa_engine):
        """
        Given: No session_id
        When: _increment_model_failure() is called
        Then: Returns 1 (no tracking)
        """
        result = qa_engine._increment_model_failure(None, "gpt-4o")

        assert result == 1

    def test_increment_model_failure_first_failure(self, qa_engine):
        """
        Given: First failure for model in session
        When: _increment_model_failure() is called
        Then: Returns 1
        """
        result = qa_engine._increment_model_failure("session-123", "gpt-4o")

        assert result == 1
        assert qa_engine._qa_failure_tracker["session-123"]["gpt-4o"] == 1

    def test_increment_model_failure_multiple_failures(self, qa_engine):
        """
        Given: Multiple failures for same model
        When: _increment_model_failure() is called repeatedly
        Then: Returns incrementing counts
        """
        qa_engine._increment_model_failure("session-123", "gpt-4o")
        qa_engine._increment_model_failure("session-123", "gpt-4o")
        result = qa_engine._increment_model_failure("session-123", "gpt-4o")

        assert result == 3

    def test_reset_model_failure_clears_count(self, qa_engine):
        """
        Given: Model has recorded failures
        When: _reset_model_failure() is called
        Then: Removes model from tracker
        """
        qa_engine._increment_model_failure("session-123", "gpt-4o")
        qa_engine._increment_model_failure("session-123", "gpt-4o")

        qa_engine._reset_model_failure("session-123", "gpt-4o")

        assert "gpt-4o" not in qa_engine._qa_failure_tracker.get("session-123", {})

    def test_reset_model_failure_cleans_empty_session(self, qa_engine):
        """
        Given: Model is only one tracked for session
        When: _reset_model_failure() is called
        Then: Removes session from tracker
        """
        qa_engine._increment_model_failure("session-123", "gpt-4o")

        qa_engine._reset_model_failure("session-123", "gpt-4o")

        assert "session-123" not in qa_engine._qa_failure_tracker

    def test_qa_failure_threshold_returns_config_value(self, qa_engine):
        """
        Given: Config has QA_MODEL_FAILURE_THRESHOLD
        When: _qa_failure_threshold() is called
        Then: Returns config value
        """
        with patch('qa_engine.config') as mock_config:
            mock_config.QA_MODEL_FAILURE_THRESHOLD = 3

            result = qa_engine._qa_failure_threshold()

            assert result == 3

    def test_qa_failure_threshold_defaults_to_5(self, qa_engine):
        """
        Given: Config attribute is missing
        When: _qa_failure_threshold() is called
        Then: Returns 5 as default
        """
        with patch('qa_engine.config') as mock_config:
            del mock_config.QA_MODEL_FAILURE_THRESHOLD

            result = qa_engine._qa_failure_threshold()

            assert result == 5


# ============================================================================
# Test: Evaluate Content
# ============================================================================

class TestEvaluateContent:
    """Tests for QAEngine.evaluate_content()."""

    @pytest.mark.asyncio
    async def test_evaluate_content_with_string_model(
        self, qa_engine, sample_qa_layer, sample_qa_evaluation
    ):
        """
        Given: Model provided as string
        When: evaluate_content() is called
        Then: Converts to QAModelConfig and evaluates
        """
        qa_engine.qa_evaluator.evaluate_content = AsyncMock(return_value=sample_qa_evaluation)

        result = await qa_engine.evaluate_content(
            content="Test content",
            layer=sample_qa_layer,
            model="gpt-4o"
        )

        assert result == sample_qa_evaluation
        qa_engine.qa_evaluator.evaluate_content.assert_called_once()

    @pytest.mark.asyncio
    async def test_evaluate_content_with_model_config(
        self, qa_engine, sample_qa_layer, sample_qa_evaluation, sample_qa_model_config
    ):
        """
        Given: Model provided as QAModelConfig
        When: evaluate_content() is called
        Then: Uses config directly
        """
        qa_engine.qa_evaluator.evaluate_content = AsyncMock(return_value=sample_qa_evaluation)

        result = await qa_engine.evaluate_content(
            content="Test content",
            layer=sample_qa_layer,
            model=sample_qa_model_config
        )

        assert result == sample_qa_evaluation

    @pytest.mark.asyncio
    async def test_evaluate_content_passes_all_parameters(
        self, qa_engine, sample_qa_layer, sample_qa_evaluation
    ):
        """
        Given: Various parameters provided
        When: evaluate_content() is called
        Then: Passes all parameters to evaluator
        """
        qa_engine.qa_evaluator.evaluate_content = AsyncMock(return_value=sample_qa_evaluation)
        original_request = Mock()
        stream_callback = Mock()
        usage_callback = Mock()

        await qa_engine.evaluate_content(
            content="Test content",
            layer=sample_qa_layer,
            model="gpt-4o",
            original_request=original_request,
            extra_verbose=True,
            stream_callback=stream_callback,
            usage_callback=usage_callback,
            request_edit_info=True,
            marker_mode="phrase",
            marker_length=6
        )

        call_kwargs = qa_engine.qa_evaluator.evaluate_content.call_args.kwargs
        assert call_kwargs["content"] == "Test content"
        assert call_kwargs["criteria"] == sample_qa_layer.criteria
        assert call_kwargs["model"] == "gpt-4o"
        assert call_kwargs["extra_verbose"] is True

    @pytest.mark.asyncio
    async def test_evaluate_content_handles_exception(self, qa_engine, sample_qa_layer):
        """
        Given: Evaluator raises exception
        When: evaluate_content() is called
        Then: Logs error and re-raises
        """
        qa_engine.qa_evaluator.evaluate_content = AsyncMock(
            side_effect=Exception("API Error")
        )

        with pytest.raises(Exception, match="API Error"):
            await qa_engine.evaluate_content(
                content="Test content",
                layer=sample_qa_layer,
                model="gpt-4o"
            )

    @pytest.mark.asyncio
    async def test_evaluate_content_with_images_vision_model(
        self, qa_engine, sample_qa_evaluation
    ):
        """
        Given: Layer has include_input_images=True and model supports vision
        When: evaluate_content() is called with images
        Then: Passes images to evaluator
        """
        layer = QALayer(
            name="Visual Check",
            description="Check visual accuracy",
            criteria="Verify image descriptions",
            min_score=7.0,
            order=1,
            include_input_images=True
        )
        qa_engine.qa_evaluator.evaluate_content = AsyncMock(return_value=sample_qa_evaluation)

        mock_images = [Mock()]

        with patch('qa_engine.config') as mock_config:
            mock_config.get_model_info = Mock(return_value={
                "capabilities": ["vision", "streaming"]
            })

            await qa_engine.evaluate_content(
                content="Test content",
                layer=layer,
                model="gpt-4o",
                input_images=mock_images
            )

            call_kwargs = qa_engine.qa_evaluator.evaluate_content.call_args.kwargs
            assert call_kwargs["input_images"] == mock_images

    @pytest.mark.asyncio
    async def test_evaluate_content_without_images_non_vision_model(
        self, qa_engine, sample_qa_evaluation
    ):
        """
        Given: Layer has include_input_images=True but model doesn't support vision
        When: evaluate_content() is called with images
        Then: Passes None for images
        """
        layer = QALayer(
            name="Visual Check",
            description="Check visual accuracy",
            criteria="Verify image descriptions",
            min_score=7.0,
            order=1,
            include_input_images=True
        )
        qa_engine.qa_evaluator.evaluate_content = AsyncMock(return_value=sample_qa_evaluation)

        mock_images = [Mock()]

        with patch('qa_engine.config') as mock_config:
            mock_config.get_model_info = Mock(return_value={
                "capabilities": ["streaming"]  # No vision
            })

            await qa_engine.evaluate_content(
                content="Test content",
                layer=layer,
                model="gpt-4o-mini",
                input_images=mock_images
            )

            call_kwargs = qa_engine.qa_evaluator.evaluate_content.call_args.kwargs
            assert call_kwargs["input_images"] is None


# ============================================================================
# Test: Deal-Breaker Consensus
# ============================================================================

class TestDealBreakerConsensus:
    """Tests for _check_deal_breaker_consensus()."""

    def test_no_deal_breakers_no_stop(self, qa_engine):
        """
        Given: No deal-breakers in results
        When: _check_deal_breaker_consensus() is called
        Then: Returns immediate_stop=False
        """
        layer_results = {
            "gpt-4o": QAEvaluation(
                model="gpt-4o", layer="Test", score=8.0,
                feedback="Good", deal_breaker=False, passes_score=True
            ),
            "claude-sonnet-4": QAEvaluation(
                model="claude-sonnet-4", layer="Test", score=7.5,
                feedback="OK", deal_breaker=False, passes_score=True
            )
        }

        result = qa_engine._check_deal_breaker_consensus(
            layer_results, ["gpt-4o", "claude-sonnet-4"]
        )

        assert result["immediate_stop"] is False
        assert result["deal_breaker_count"] == 0

    def test_minority_deal_breakers_no_stop(self, qa_engine):
        """
        Given: Minority of models have deal-breakers (1/3)
        When: _check_deal_breaker_consensus() is called
        Then: Returns immediate_stop=False
        """
        layer_results = {
            "gpt-4o": QAEvaluation(
                model="gpt-4o", layer="Test", score=3.0,
                feedback="Bad", deal_breaker=True,
                deal_breaker_reason="Error", passes_score=False
            ),
            "claude-sonnet-4": QAEvaluation(
                model="claude-sonnet-4", layer="Test", score=7.5,
                feedback="OK", deal_breaker=False, passes_score=True
            ),
            "gemini-pro": QAEvaluation(
                model="gemini-pro", layer="Test", score=8.0,
                feedback="Good", deal_breaker=False, passes_score=True
            )
        }

        result = qa_engine._check_deal_breaker_consensus(
            layer_results, ["gpt-4o", "claude-sonnet-4", "gemini-pro"]
        )

        assert result["immediate_stop"] is False
        assert result["deal_breaker_count"] == 1

    def test_majority_deal_breakers_stops(self, qa_engine):
        """
        Given: Majority of models have deal-breakers (2/3)
        When: _check_deal_breaker_consensus() is called
        Then: Returns immediate_stop=True
        """
        layer_results = {
            "gpt-4o": QAEvaluation(
                model="gpt-4o", layer="Test", score=3.0,
                feedback="Bad", deal_breaker=True,
                deal_breaker_reason="Error 1", passes_score=False
            ),
            "claude-sonnet-4": QAEvaluation(
                model="claude-sonnet-4", layer="Test", score=2.0,
                feedback="Bad", deal_breaker=True,
                deal_breaker_reason="Error 2", passes_score=False
            ),
            "gemini-pro": QAEvaluation(
                model="gemini-pro", layer="Test", score=8.0,
                feedback="Good", deal_breaker=False, passes_score=True
            )
        }

        result = qa_engine._check_deal_breaker_consensus(
            layer_results, ["gpt-4o", "claude-sonnet-4", "gemini-pro"]
        )

        assert result["immediate_stop"] is True
        assert result["deal_breaker_count"] == 2
        assert len(result["deal_breaker_details"]) == 2

    def test_all_deal_breakers_stops(self, qa_engine):
        """
        Given: All models have deal-breakers
        When: _check_deal_breaker_consensus() is called
        Then: Returns immediate_stop=True
        """
        layer_results = {
            "gpt-4o": QAEvaluation(
                model="gpt-4o", layer="Test", score=2.0,
                feedback="Bad", deal_breaker=True,
                deal_breaker_reason="Error 1", passes_score=False
            ),
            "claude-sonnet-4": QAEvaluation(
                model="claude-sonnet-4", layer="Test", score=1.0,
                feedback="Bad", deal_breaker=True,
                deal_breaker_reason="Error 2", passes_score=False
            )
        }

        result = qa_engine._check_deal_breaker_consensus(
            layer_results, ["gpt-4o", "claude-sonnet-4"]
        )

        assert result["immediate_stop"] is True
        assert result["deal_breaker_count"] == 2

    def test_tie_with_even_models_no_stop(self, qa_engine):
        """
        Given: 50/50 tie (1/2 deal-breakers)
        When: _check_deal_breaker_consensus() is called
        Then: Returns immediate_stop=False (tie doesn't stop)
        """
        layer_results = {
            "gpt-4o": QAEvaluation(
                model="gpt-4o", layer="Test", score=3.0,
                feedback="Bad", deal_breaker=True,
                deal_breaker_reason="Error", passes_score=False
            ),
            "claude-sonnet-4": QAEvaluation(
                model="claude-sonnet-4", layer="Test", score=7.5,
                feedback="OK", deal_breaker=False, passes_score=True
            )
        }

        result = qa_engine._check_deal_breaker_consensus(
            layer_results, ["gpt-4o", "claude-sonnet-4"]
        )

        assert result["immediate_stop"] is False  # Tie doesn't stop
        assert result["deal_breaker_count"] == 1
        assert result["majority_threshold"] == 1.0

    def test_returns_deal_breaker_details(self, qa_engine):
        """
        Given: Deal-breakers in results
        When: _check_deal_breaker_consensus() is called
        Then: Returns details with model and reason
        """
        layer_results = {
            "gpt-4o": QAEvaluation(
                model="gpt-4o", layer="Test", score=3.0,
                feedback="Bad", deal_breaker=True,
                deal_breaker_reason="Contains fabricated info", passes_score=False
            )
        }

        result = qa_engine._check_deal_breaker_consensus(layer_results, ["gpt-4o"])

        assert len(result["deal_breaker_details"]) == 1
        assert result["deal_breaker_details"][0]["model"] == "gpt-4o"
        assert result["deal_breaker_details"][0]["reason"] == "Contains fabricated info"


# ============================================================================
# Test: Create Result Structures
# ============================================================================

class TestCreateIterationStopResult:
    """Tests for _create_iteration_stop_result()."""

    def test_creates_result_with_force_iteration(self, qa_engine):
        """
        Given: Partial results and consensus info
        When: _create_iteration_stop_result() is called
        Then: Creates result with force_iteration=True
        """
        partial_results = {
            "Layer1": {
                "gpt-4o": QAEvaluation(
                    model="gpt-4o", layer="Layer1", score=3.0,
                    feedback="Bad", deal_breaker=True, passes_score=False
                )
            }
        }
        consensus_info = {
            "deal_breaker_count": 1,
            "total_evaluated": 1,
            "deal_breaker_details": [{"model": "gpt-4o", "reason": "Error"}]
        }

        result = qa_engine._create_iteration_stop_result(partial_results, consensus_info)

        assert result["summary"]["force_iteration"] is True
        assert result["summary"]["has_deal_breakers"] is True

    def test_calculates_average_score(self, qa_engine):
        """
        Given: Multiple evaluations with scores
        When: _create_iteration_stop_result() is called
        Then: Calculates correct average score
        """
        partial_results = {
            "Layer1": {
                "gpt-4o": QAEvaluation(
                    model="gpt-4o", layer="Layer1", score=4.0,
                    feedback="OK", deal_breaker=True, passes_score=False
                ),
                "claude-sonnet-4": QAEvaluation(
                    model="claude-sonnet-4", layer="Layer1", score=6.0,
                    feedback="OK", deal_breaker=False, passes_score=False
                )
            }
        }
        consensus_info = {
            "deal_breaker_count": 1,
            "total_evaluated": 2,
            "deal_breaker_details": []
        }

        result = qa_engine._create_iteration_stop_result(partial_results, consensus_info)

        assert result["summary"]["average_score"] == 5.0  # (4+6)/2
        assert result["summary"]["min_score"] == 4.0
        assert result["summary"]["max_score"] == 6.0

    def test_handles_none_scores(self, qa_engine):
        """
        Given: Evaluations with None scores
        When: _create_iteration_stop_result() is called
        Then: Excludes None scores from calculations
        """
        partial_results = {
            "Layer1": {
                "gpt-4o": QAEvaluation(
                    model="gpt-4o", layer="Layer1", score=None,
                    feedback="Invalid", deal_breaker=False, passes_score=False
                ),
                "claude-sonnet-4": QAEvaluation(
                    model="claude-sonnet-4", layer="Layer1", score=8.0,
                    feedback="Good", deal_breaker=False, passes_score=True
                )
            }
        }
        consensus_info = {
            "deal_breaker_count": 0,
            "total_evaluated": 2,
            "deal_breaker_details": []
        }

        result = qa_engine._create_iteration_stop_result(partial_results, consensus_info)

        assert result["summary"]["average_score"] == 8.0
        assert result["summary"]["total_evaluations"] == 1


class TestCreateGranSabioModifiedResult:
    """Tests for _create_gran_sabio_modified_result()."""

    def test_creates_result_with_modified_content(self, qa_engine):
        """
        Given: Partial results and modified content
        When: _create_gran_sabio_modified_result() is called
        Then: Includes modified content and metadata
        """
        partial_results = {
            "Layer1": {
                "gpt-4o": QAEvaluation(
                    model="gpt-4o", layer="Layer1", score=7.0,
                    feedback="OK", deal_breaker=False, passes_score=True
                )
            }
        }

        result = qa_engine._create_gran_sabio_modified_result(
            partial_results=partial_results,
            modified_content="Modified test content",
            reason="Fixed factual error",
            score=8.5
        )

        assert result["gran_sabio_modified_content"] == "Modified test content"
        assert result["gran_sabio_modification_reason"] == "Fixed factual error"
        assert result["gran_sabio_score"] == 8.5
        assert result["summary"]["gran_sabio_modified"] is True
        assert result["summary"]["force_iteration"] is True

    def test_sets_no_deal_breakers(self, qa_engine):
        """
        Given: Gran Sabio approved with modifications
        When: _create_gran_sabio_modified_result() is called
        Then: Sets has_deal_breakers=False
        """
        partial_results = {}

        result = qa_engine._create_gran_sabio_modified_result(
            partial_results=partial_results,
            modified_content="Fixed content",
            reason="Corrected",
            score=9.0
        )

        assert result["summary"]["has_deal_breakers"] is False
        assert result["summary"]["deal_breakers_count"] == 0


# ============================================================================
# Test: Evaluate All Layers with Progress
# ============================================================================

class TestEvaluateAllLayersWithProgress:
    """Tests for evaluate_all_layers_with_progress()."""

    @pytest.mark.asyncio
    async def test_evaluates_layers_in_order(
        self, qa_engine, sample_layers, sample_qa_evaluation
    ):
        """
        Given: Multiple layers
        When: evaluate_all_layers_with_progress() is called
        Then: Evaluates layers sorted by order
        """
        qa_engine.qa_evaluator.evaluate_content = AsyncMock(return_value=sample_qa_evaluation)

        with patch('qa_engine.calculate_qa_timeout_for_model', return_value=60):
            results = await qa_engine.evaluate_all_layers_with_progress(
                content="Test content",
                layers=sample_layers,
                qa_models=["gpt-4o"]
            )

        assert "Accuracy" in results
        assert "Style" in results
        assert "Tone" in results

    @pytest.mark.asyncio
    async def test_calls_progress_callback(self, qa_engine, sample_qa_layer, sample_qa_evaluation):
        """
        Given: Progress callback provided
        When: evaluate_all_layers_with_progress() is called
        Then: Calls progress callback with updates
        """
        qa_engine.qa_evaluator.evaluate_content = AsyncMock(return_value=sample_qa_evaluation)
        progress_callback = AsyncMock()

        with patch('qa_engine.calculate_qa_timeout_for_model', return_value=60):
            await qa_engine.evaluate_all_layers_with_progress(
                content="Test content",
                layers=[sample_qa_layer],
                qa_models=["gpt-4o"],
                progress_callback=progress_callback
            )

        assert progress_callback.call_count >= 2  # At least layer start and result

    @pytest.mark.asyncio
    async def test_uses_bypass_engine_when_applicable(
        self, qa_engine, sample_qa_layer, sample_qa_evaluation
    ):
        """
        Given: Bypass engine says layer can be bypassed
        When: evaluate_all_layers_with_progress() is called
        Then: Uses bypass engine instead of AI evaluation
        """
        qa_engine.bypass_engine.should_bypass_qa_layer = Mock(return_value=True)
        qa_engine.bypass_engine.bypass_layer_evaluation = Mock(return_value={
            "gpt-4o": sample_qa_evaluation
        })

        with patch('qa_engine.calculate_qa_timeout_for_model', return_value=60):
            results = await qa_engine.evaluate_all_layers_with_progress(
                content="Test content",
                layers=[sample_qa_layer],
                qa_models=["gpt-4o"]
            )

        qa_engine.bypass_engine.should_bypass_qa_layer.assert_called_once()
        qa_engine.bypass_engine.bypass_layer_evaluation.assert_called_once()
        assert "Test Quality" in results

    @pytest.mark.asyncio
    async def test_handles_cancellation(self, qa_engine, sample_qa_layer, sample_qa_evaluation):
        """
        Given: Cancel callback returns True
        When: evaluate_all_layers_with_progress() is called
        Then: Raises QAProcessCancelled
        """
        qa_engine.qa_evaluator.evaluate_content = AsyncMock(return_value=sample_qa_evaluation)
        cancel_callback = AsyncMock(return_value=True)

        with patch('qa_engine.calculate_qa_timeout_for_model', return_value=60):
            with pytest.raises(QAProcessCancelled):
                await qa_engine.evaluate_all_layers_with_progress(
                    content="Test content",
                    layers=[sample_qa_layer],
                    qa_models=["gpt-4o"],
                    cancel_callback=cancel_callback
                )

    @pytest.mark.asyncio
    async def test_handles_timeout(self, qa_engine, sample_qa_layer):
        """
        Given: Evaluation times out
        When: evaluate_all_layers_with_progress() is called
        Then: Creates deal-breaker evaluation with timeout reason
        """
        async def slow_evaluation(*args, **kwargs):
            await asyncio.sleep(10)

        qa_engine.qa_evaluator.evaluate_content = AsyncMock(side_effect=slow_evaluation)

        with patch('qa_engine.calculate_qa_timeout_for_model', return_value=0.01):
            results = await qa_engine.evaluate_all_layers_with_progress(
                content="Test content",
                layers=[sample_qa_layer],
                qa_models=["gpt-4o"]
            )

        # With single model timeout causes majority deal-breaker (1/1)
        # Result is wrapped in iteration stop structure
        qa_results = results.get("qa_results", results)
        assert "Test Quality" in qa_results
        evaluation = qa_results["Test Quality"]["gpt-4o"]
        assert evaluation.deal_breaker is True
        assert "Timeout" in evaluation.feedback

    @pytest.mark.asyncio
    async def test_handles_value_error_single_model(self, qa_engine, sample_qa_layer):
        """
        Given: Single model returns invalid JSON
        When: evaluate_all_layers_with_progress() is called
        Then: Raises ValueError
        """
        qa_engine.qa_evaluator.evaluate_content = AsyncMock(
            side_effect=ValueError("Invalid JSON response")
        )

        with patch('qa_engine.calculate_qa_timeout_for_model', return_value=60):
            with pytest.raises(ValueError, match="invalid JSON"):
                await qa_engine.evaluate_all_layers_with_progress(
                    content="Test content",
                    layers=[sample_qa_layer],
                    qa_models=["gpt-4o"]
                )

    @pytest.mark.asyncio
    async def test_handles_value_error_multiple_models(
        self, qa_engine, sample_qa_layer, sample_qa_evaluation
    ):
        """
        Given: One model returns invalid JSON with multiple models
        When: evaluate_all_layers_with_progress() is called
        Then: Skips failed model, continues with others
        """
        call_count = [0]

        async def mock_evaluate(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise ValueError("Invalid JSON")
            return sample_qa_evaluation

        qa_engine.qa_evaluator.evaluate_content = AsyncMock(side_effect=mock_evaluate)

        with patch('qa_engine.calculate_qa_timeout_for_model', return_value=60):
            results = await qa_engine.evaluate_all_layers_with_progress(
                content="Test content",
                layers=[sample_qa_layer],
                qa_models=["gpt-4o", "claude-sonnet-4"]
            )

        # First model should have placeholder, second should succeed
        assert "Test Quality" in results
        assert results["Test Quality"]["gpt-4o"].score is None
        assert results["Test Quality"]["claude-sonnet-4"].score == 8.5

    @pytest.mark.asyncio
    async def test_majority_deal_breaker_stops_early(
        self, qa_engine, sample_qa_layer
    ):
        """
        Given: Majority of models report deal-breakers
        When: evaluate_all_layers_with_progress() is called
        Then: Returns early with force_iteration result
        """
        deal_breaker_eval = QAEvaluation(
            model="gpt-4o", layer="Test Quality", score=2.0,
            feedback="Critical error", deal_breaker=True,
            deal_breaker_reason="Fabricated info", passes_score=False
        )

        qa_engine.qa_evaluator.evaluate_content = AsyncMock(return_value=deal_breaker_eval)

        with patch('qa_engine.calculate_qa_timeout_for_model', return_value=60):
            results = await qa_engine.evaluate_all_layers_with_progress(
                content="Test content",
                layers=[sample_qa_layer],
                qa_models=["gpt-4o", "claude-sonnet-4", "gemini-pro"]
            )

        assert results.get("summary", {}).get("force_iteration") is True


# ============================================================================
# Test: Gran Sabio Escalation
# ============================================================================

class TestGranSabioEscalation:
    """Tests for Gran Sabio escalation within evaluate_all_layers_with_progress()."""

    @pytest.mark.asyncio
    async def test_escalates_minority_deal_breaker(self, qa_engine, sample_qa_layer):
        """
        Given: Minority deal-breaker (1/3) and gran_sabio_engine provided
        When: evaluate_all_layers_with_progress() is called
        Then: Escalates to Gran Sabio
        """
        # First call returns deal-breaker, rest return passing
        call_count = [0]

        async def mock_evaluate(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return QAEvaluation(
                    model="gpt-4o", layer="Test Quality", score=2.0,
                    feedback="Error", deal_breaker=True,
                    deal_breaker_reason="Fabricated", passes_score=False
                )
            return QAEvaluation(
                model="other", layer="Test Quality", score=8.0,
                feedback="Good", deal_breaker=False, passes_score=True
            )

        qa_engine.qa_evaluator.evaluate_content = AsyncMock(side_effect=mock_evaluate)

        gran_sabio = MagicMock()
        gran_sabio.review_minority_deal_breakers = AsyncMock(return_value=MagicMock(
            approved=True,
            error=None,
            reason="False positive",
            final_score=8.0,
            final_content=None,
            modifications_made=False
        ))

        with patch('qa_engine.calculate_qa_timeout_for_model', return_value=60):
            with patch('deal_breaker_tracker.get_tracker') as mock_tracker:
                mock_tracker.return_value.record_escalation = Mock(return_value="esc-123")
                mock_tracker.return_value.complete_escalation = Mock()

                results = await qa_engine.evaluate_all_layers_with_progress(
                    content="Test content",
                    layers=[sample_qa_layer],
                    qa_models=["gpt-4o", "claude-sonnet-4", "gemini-pro"],
                    gran_sabio_engine=gran_sabio
                )

        gran_sabio.review_minority_deal_breakers.assert_called_once()

    @pytest.mark.asyncio
    async def test_gran_sabio_confirms_deal_breaker(self, qa_engine, sample_qa_layer):
        """
        Given: Gran Sabio confirms deal-breaker as real (minority 1/3)
        When: evaluate_all_layers_with_progress() is called
        Then: Returns force_iteration result with gran_sabio_confirmed
        """
        # Only first model returns deal-breaker (1/3 = minority)
        call_count = [0]

        async def mock_evaluate(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return QAEvaluation(
                    model="gpt-4o", layer="Test Quality", score=2.0,
                    feedback="Error", deal_breaker=True,
                    deal_breaker_reason="Fabricated", passes_score=False
                )
            return QAEvaluation(
                model="other", layer="Test Quality", score=8.0,
                feedback="Good", deal_breaker=False, passes_score=True
            )

        qa_engine.qa_evaluator.evaluate_content = AsyncMock(side_effect=mock_evaluate)

        gran_sabio = MagicMock()
        gran_sabio.review_minority_deal_breakers = AsyncMock(return_value=MagicMock(
            approved=False,  # Not approved = deal-breaker is real
            error=None,
            reason="Confirmed fabrication"
        ))

        with patch('qa_engine.calculate_qa_timeout_for_model', return_value=60):
            with patch('deal_breaker_tracker.get_tracker') as mock_tracker:
                mock_tracker.return_value.record_escalation = Mock(return_value="esc-123")
                mock_tracker.return_value.complete_escalation = Mock()

                results = await qa_engine.evaluate_all_layers_with_progress(
                    content="Test content",
                    layers=[sample_qa_layer],
                    qa_models=["gpt-4o", "claude-sonnet-4", "gemini-pro"],
                    gran_sabio_engine=gran_sabio
                )

        assert results.get("summary", {}).get("force_iteration") is True
        assert results.get("consensus_info", {}).get("gran_sabio_confirmed") is True
        gran_sabio.review_minority_deal_breakers.assert_called_once()

    @pytest.mark.asyncio
    async def test_gran_sabio_returns_modified_content(self, qa_engine, sample_qa_layer):
        """
        Given: Gran Sabio approves with modifications (minority 1/3 deal-breaker)
        When: evaluate_all_layers_with_progress() is called
        Then: Returns result with modified content
        """
        # Only first model returns deal-breaker (1/3 = minority)
        call_count = [0]

        async def mock_evaluate(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return QAEvaluation(
                    model="gpt-4o", layer="Test Quality", score=4.0,
                    feedback="Minor error", deal_breaker=True,
                    deal_breaker_reason="Small issue", passes_score=False
                )
            return QAEvaluation(
                model="other", layer="Test Quality", score=8.0,
                feedback="Good", deal_breaker=False, passes_score=True
            )

        qa_engine.qa_evaluator.evaluate_content = AsyncMock(side_effect=mock_evaluate)

        gran_sabio = MagicMock()
        gran_sabio.review_minority_deal_breakers = AsyncMock(return_value=MagicMock(
            approved=True,
            error=None,
            reason="Fixed the issue",
            final_score=9.0,
            final_content="Corrected content here",
            modifications_made=True
        ))

        with patch('qa_engine.calculate_qa_timeout_for_model', return_value=60):
            with patch('deal_breaker_tracker.get_tracker') as mock_tracker:
                mock_tracker.return_value.record_escalation = Mock(return_value="esc-123")
                mock_tracker.return_value.complete_escalation = Mock()

                results = await qa_engine.evaluate_all_layers_with_progress(
                    content="Test content",
                    layers=[sample_qa_layer],
                    qa_models=["gpt-4o", "claude-sonnet-4", "gemini-pro"],
                    gran_sabio_engine=gran_sabio
                )

        assert results.get("gran_sabio_modified_content") == "Corrected content here"
        assert results.get("summary", {}).get("gran_sabio_modified") is True
        gran_sabio.review_minority_deal_breakers.assert_called_once()

    @pytest.mark.asyncio
    async def test_respects_escalation_limit(
        self, qa_engine, sample_qa_layer, sample_content_request
    ):
        """
        Given: Escalation limit is set and reached
        When: evaluate_all_layers_with_progress() continues after limit
        Then: Does not escalate to Gran Sabio
        """
        sample_content_request.gran_sabio_call_limit_per_iteration = 0

        qa_engine.qa_evaluator.evaluate_content = AsyncMock(return_value=QAEvaluation(
            model="gpt-4o", layer="Test Quality", score=4.0,
            feedback="Error", deal_breaker=True,
            deal_breaker_reason="Issue", passes_score=False
        ))

        gran_sabio = MagicMock()
        gran_sabio.review_minority_deal_breakers = AsyncMock()

        with patch('qa_engine.calculate_qa_timeout_for_model', return_value=60):
            await qa_engine.evaluate_all_layers_with_progress(
                content="Test content",
                layers=[sample_qa_layer],
                qa_models=["gpt-4o", "claude-sonnet-4", "gemini-pro"],
                gran_sabio_engine=gran_sabio,
                original_request=sample_content_request
            )

        gran_sabio.review_minority_deal_breakers.assert_not_called()


# ============================================================================
# Test: Summary and Statistics
# ============================================================================

class TestCalculateSummary:
    """Tests for _calculate_summary()."""

    def test_calculates_average_score(self, qa_engine, sample_layers):
        """
        Given: QA results with various scores
        When: _calculate_summary() is called
        Then: Calculates correct average
        """
        qa_results = {
            "Accuracy": {
                "gpt-4o": QAEvaluation(
                    model="gpt-4o", layer="Accuracy", score=8.0,
                    feedback="Good", deal_breaker=False, passes_score=True
                ),
                "claude-sonnet-4": QAEvaluation(
                    model="claude-sonnet-4", layer="Accuracy", score=9.0,
                    feedback="Great", deal_breaker=False, passes_score=True
                )
            },
            "Style": {
                "gpt-4o": QAEvaluation(
                    model="gpt-4o", layer="Style", score=7.0,
                    feedback="OK", deal_breaker=False, passes_score=True
                )
            }
        }

        summary = qa_engine._calculate_summary(qa_results, sample_layers)

        assert summary["average_score"] == 8.0  # (8+9+7)/3
        assert summary["min_score"] == 7.0
        assert summary["max_score"] == 9.0
        assert summary["total_evaluations"] == 3

    def test_counts_deal_breakers(self, qa_engine, sample_layers):
        """
        Given: QA results with deal-breakers
        When: _calculate_summary() is called
        Then: Counts and lists deal-breakers
        """
        qa_results = {
            "Accuracy": {
                "gpt-4o": QAEvaluation(
                    model="gpt-4o", layer="Accuracy", score=3.0,
                    feedback="Bad", deal_breaker=True,
                    reason="Error", passes_score=False
                )
            }
        }

        summary = qa_engine._calculate_summary(qa_results, sample_layers)

        assert summary["deal_breakers_count"] == 1
        assert summary["has_deal_breakers"] is True
        assert len(summary["deal_breakers"]) == 1

    def test_handles_empty_results(self, qa_engine, sample_layers):
        """
        Given: Empty QA results
        When: _calculate_summary() is called
        Then: Returns zeroes
        """
        qa_results = {}

        summary = qa_engine._calculate_summary(qa_results, sample_layers)

        assert summary["average_score"] == 0.0
        assert summary["total_evaluations"] == 0


class TestIdentifyCriticalIssues:
    """Tests for _identify_critical_issues()."""

    def test_identifies_low_layer_score(self, qa_engine):
        """
        Given: Layer average below 4.0
        When: _identify_critical_issues() is called
        Then: Reports low_layer_score issue
        """
        qa_results = {
            "Accuracy": {
                "gpt-4o": QAEvaluation(
                    model="gpt-4o", layer="Accuracy", score=2.0,
                    feedback="Bad", deal_breaker=False, passes_score=False
                ),
                "claude-sonnet-4": QAEvaluation(
                    model="claude-sonnet-4", layer="Accuracy", score=3.0,
                    feedback="Bad", deal_breaker=False, passes_score=False
                )
            }
        }

        issues = qa_engine._identify_critical_issues(qa_results)

        low_score_issues = [i for i in issues if i["type"] == "low_layer_score"]
        assert len(low_score_issues) == 1
        assert low_score_issues[0]["layer"] == "Accuracy"

    def test_identifies_model_disagreement(self, qa_engine):
        """
        Given: Model scores differ by more than 5 points
        When: _identify_critical_issues() is called
        Then: Reports model_disagreement issue
        """
        qa_results = {
            "Style": {
                "gpt-4o": QAEvaluation(
                    model="gpt-4o", layer="Style", score=3.0,
                    feedback="Bad", deal_breaker=False, passes_score=False
                ),
                "claude-sonnet-4": QAEvaluation(
                    model="claude-sonnet-4", layer="Style", score=9.0,
                    feedback="Great", deal_breaker=False, passes_score=True
                )
            }
        }

        issues = qa_engine._identify_critical_issues(qa_results)

        disagreement_issues = [i for i in issues if i["type"] == "model_disagreement"]
        assert len(disagreement_issues) == 1
        assert disagreement_issues[0]["score_range"] == 6.0

    def test_identifies_deal_breaker(self, qa_engine):
        """
        Given: Deal-breaker evaluation
        When: _identify_critical_issues() is called
        Then: Reports deal_breaker issue
        """
        qa_results = {
            "Accuracy": {
                "gpt-4o": QAEvaluation(
                    model="gpt-4o", layer="Accuracy", score=2.0,
                    feedback="Critical", deal_breaker=True, passes_score=False
                )
            }
        }

        issues = qa_engine._identify_critical_issues(qa_results)

        db_issues = [i for i in issues if i["type"] == "deal_breaker"]
        assert len(db_issues) == 1
        assert "gpt-4o" in db_issues[0]["models"]

    def test_identifies_missing_model_scores(self, qa_engine):
        """
        Given: Evaluation with None score
        When: _identify_critical_issues() is called
        Then: Reports missing_model_scores issue
        """
        qa_results = {
            "Accuracy": {
                "gpt-4o": QAEvaluation(
                    model="gpt-4o", layer="Accuracy", score=None,
                    feedback="Invalid JSON", deal_breaker=False, passes_score=False
                )
            }
        }

        issues = qa_engine._identify_critical_issues(qa_results)

        missing_issues = [i for i in issues if i["type"] == "missing_model_scores"]
        assert len(missing_issues) == 1
        assert "gpt-4o" in missing_issues[0]["models"]


# ============================================================================
# Test: Validate Layers
# ============================================================================

class TestValidateLayers:
    """Tests for validate_layers()."""

    @pytest.mark.asyncio
    async def test_empty_layers_invalid(self, qa_engine):
        """
        Given: Empty layers list
        When: validate_layers() is called
        Then: Returns invalid with error
        """
        result = await qa_engine.validate_layers([])

        assert result["is_valid"] is False
        assert "No QA layers provided" in result["errors"]

    @pytest.mark.asyncio
    async def test_duplicate_names_invalid(self, qa_engine):
        """
        Given: Layers with duplicate names
        When: validate_layers() is called
        Then: Returns invalid with error
        """
        layers = [
            QALayer(name="Same", description="D1", criteria="C1", min_score=7.0, order=1),
            QALayer(name="Same", description="D2", criteria="C2", min_score=7.0, order=2)
        ]

        result = await qa_engine.validate_layers(layers)

        assert result["is_valid"] is False
        assert "Duplicate layer names" in result["errors"][0]

    @pytest.mark.asyncio
    async def test_duplicate_orders_warning(self, qa_engine):
        """
        Given: Layers with duplicate orders
        When: validate_layers() is called
        Then: Returns warning
        """
        layers = [
            QALayer(name="L1", description="D1", criteria="C1", min_score=7.0, order=1),
            QALayer(name="L2", description="D2", criteria="C2", min_score=7.0, order=1)
        ]

        result = await qa_engine.validate_layers(layers)

        assert result["is_valid"] is True
        assert any("Duplicate layer orders" in w for w in result["warnings"])

    @pytest.mark.asyncio
    async def test_invalid_min_score_invalid(self, qa_engine):
        """
        Given: Layer with min_score > 10
        When: validate_layers() is called
        Then: Returns invalid with error
        """
        # Note: This might be caught by Pydantic validation first
        # Testing the engine's validation logic
        layer = QALayer(name="Test", description="D", criteria="C", min_score=7.0, order=1)
        # Manually set invalid score to bypass Pydantic
        layer.min_score = 15.0

        result = await qa_engine.validate_layers([layer])

        assert result["is_valid"] is False
        assert any("invalid min_score" in e for e in result["errors"])

    @pytest.mark.asyncio
    async def test_no_deal_breaker_layers_warning(self, qa_engine):
        """
        Given: No layers with is_deal_breaker=True
        When: validate_layers() is called
        Then: Returns warning
        """
        layers = [
            QALayer(name="L1", description="D1", criteria="C1", min_score=7.0, order=1),
            QALayer(name="L2", description="D2", criteria="C2", min_score=7.0, order=2)
        ]

        result = await qa_engine.validate_layers(layers)

        assert result["is_valid"] is True
        assert any("No deal-breaker layers" in w for w in result["warnings"])

    @pytest.mark.asyncio
    async def test_valid_layers_pass(self, qa_engine):
        """
        Given: Valid layers configuration
        When: validate_layers() is called
        Then: Returns valid
        """
        layers = [
            QALayer(
                name="Accuracy", description="D1", criteria="C1",
                min_score=7.0, order=1, is_deal_breaker=True
            ),
            QALayer(name="Style", description="D2", criteria="C2", min_score=6.0, order=2)
        ]

        result = await qa_engine.validate_layers(layers)

        assert result["is_valid"] is True
        assert len(result["errors"]) == 0
