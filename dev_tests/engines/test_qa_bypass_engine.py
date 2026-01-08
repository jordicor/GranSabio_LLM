"""
Tests for qa_bypass_engine.py - Algorithmic QA Bypass Engine.

This module tests the QABypassEngine which handles predictable deal-breaker
scenarios algorithmically without requiring expensive AI model calls.

Functions tested:
- __init__(): Initializes evaluator name lists
- can_bypass_layer(): Determines if layer can be bypassed
- bypass_layer_evaluation(): Dispatches to specific bypass methods
- _bypass_word_count_evaluation(): Word count algorithmic evaluation
- _bypass_phrase_frequency_evaluation(): Phrase frequency evaluation
- _bypass_lexical_diversity_evaluation(): Lexical diversity evaluation
- _bypass_cumulative_repetition_evaluation(): Cumulative repetition evaluation
- should_bypass_qa_layer(): Wrapper for bypass determination
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

from qa_bypass_engine import QABypassEngine
from models import QALayer, QAEvaluation


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def bypass_engine():
    """Create a fresh QABypassEngine instance."""
    return QABypassEngine()


@pytest.fixture
def word_count_layer():
    """Create a Word Count Enforcement QA layer."""
    return QALayer(
        name="Word Count Enforcement",
        description="Enforces word count limits",
        criteria="Content must be within word count range",
        min_score=7.0,
        order=0
    )


@pytest.fixture
def phrase_frequency_layer():
    """Create a Phrase Frequency Guard QA layer."""
    return QALayer(
        name="Phrase Frequency Guard",
        description="Checks for phrase repetition",
        criteria="Detect overused phrases",
        min_score=7.0,
        order=1
    )


@pytest.fixture
def lexical_diversity_layer():
    """Create a Lexical Diversity Guard QA layer."""
    return QALayer(
        name="Lexical Diversity Guard",
        description="Analyzes vocabulary diversity",
        criteria="Check lexical diversity metrics",
        min_score=7.0,
        order=2
    )


@pytest.fixture
def cumulative_repetition_layer():
    """Create a Cumulative Repetition Guard QA layer."""
    return QALayer(
        name="Cumulative Repetition Guard",
        description="Checks repetition across accumulated content",
        criteria="Detect cumulative phrase repetition",
        min_score=7.0,
        order=3
    )


@pytest.fixture
def generic_layer():
    """Create a generic QA layer that cannot be bypassed."""
    return QALayer(
        name="Content Quality",
        description="General content quality check",
        criteria="Check overall content quality",
        min_score=7.0,
        order=5
    )


@pytest.fixture
def mock_word_count_config():
    """Create a mock word count enforcement config."""
    config = Mock()
    config.enabled = True
    config.flexibility_percent = 10
    config.direction = "both"
    config.severity = "important"
    config.target_field = None
    config.model_dump = Mock(return_value={
        "enabled": True,
        "flexibility_percent": 10,
        "direction": "both",
        "severity": "important",
        "target_field": None
    })
    return config


@pytest.fixture
def mock_phrase_frequency_config():
    """Create a mock phrase frequency config."""
    config = Mock()
    config.enabled = True
    config.to_settings = Mock(return_value=Mock(rules=[]))
    return config


@pytest.fixture
def mock_lexical_diversity_config():
    """Create a mock lexical diversity config."""
    config = Mock()
    config.enabled = True
    config.to_settings = Mock(return_value=Mock(
        top_words_k=50,
        language=None
    ))
    return config


@pytest.fixture
def mock_request_with_word_count(mock_word_count_config):
    """Create a mock request with word count enforcement."""
    request = Mock()
    request.word_count_enforcement = mock_word_count_config
    request.min_words = 400
    request.max_words = 600
    return request


@pytest.fixture
def mock_request_with_phrase_frequency(mock_phrase_frequency_config):
    """Create a mock request with phrase frequency config."""
    request = Mock()
    request.phrase_frequency = mock_phrase_frequency_config
    return request


@pytest.fixture
def mock_request_with_lexical_diversity(mock_lexical_diversity_config):
    """Create a mock request with lexical diversity config."""
    request = Mock()
    request.lexical_diversity = mock_lexical_diversity_config
    return request


@pytest.fixture
def mock_request_with_cumulative(mock_phrase_frequency_config):
    """Create a mock request with cumulative repetition data."""
    request = Mock()
    request.cumulative_text = "Previous chapter content. " * 100
    request.cumulative_word_count = 400
    request.phrase_frequency = mock_phrase_frequency_config
    return request


# ============================================================================
# TestInit
# ============================================================================

class TestInit:
    """Tests for QABypassEngine.__init__()."""

    def test_initializes_word_count_evaluator_names(self, bypass_engine):
        """
        Given: A new QABypassEngine instance
        When: Initialized
        Then: Has evaluator_names list with Arithmos evaluators
        """
        assert hasattr(bypass_engine, 'evaluator_names')
        assert len(bypass_engine.evaluator_names) == 5
        assert "Arithmos-Prime" in bypass_engine.evaluator_names

    def test_initializes_phrase_evaluator_names(self, bypass_engine):
        """
        Given: A new QABypassEngine instance
        When: Initialized
        Then: Has phrase_evaluator_names list with FraseGuard evaluators
        """
        assert hasattr(bypass_engine, 'phrase_evaluator_names')
        assert len(bypass_engine.phrase_evaluator_names) == 5
        assert "FraseGuard-Alpha" in bypass_engine.phrase_evaluator_names

    def test_initializes_lexical_evaluator_names(self, bypass_engine):
        """
        Given: A new QABypassEngine instance
        When: Initialized
        Then: Has lexical_evaluator_names list with LexiGuard evaluators
        """
        assert hasattr(bypass_engine, 'lexical_evaluator_names')
        assert len(bypass_engine.lexical_evaluator_names) == 5
        assert "LexiGuard-Alpha" in bypass_engine.lexical_evaluator_names

    def test_initializes_cumulative_evaluator_names(self, bypass_engine):
        """
        Given: A new QABypassEngine instance
        When: Initialized
        Then: Has cumulative_evaluator_names list with CumulGuard evaluators
        """
        assert hasattr(bypass_engine, 'cumulative_evaluator_names')
        assert len(bypass_engine.cumulative_evaluator_names) == 5
        assert "CumulGuard-Alpha" in bypass_engine.cumulative_evaluator_names


# ============================================================================
# TestCanBypassLayer
# ============================================================================

class TestCanBypassLayer:
    """Tests for can_bypass_layer()."""

    def test_word_count_layer_with_enabled_config_returns_true(
        self, bypass_engine, word_count_layer, mock_request_with_word_count
    ):
        """
        Given: Word Count Enforcement layer with enabled config
        When: can_bypass_layer() is called
        Then: Returns True
        """
        result = bypass_engine.can_bypass_layer(word_count_layer, mock_request_with_word_count)
        assert result is True

    def test_word_count_layer_with_disabled_config_returns_false(
        self, bypass_engine, word_count_layer
    ):
        """
        Given: Word Count Enforcement layer with disabled config
        When: can_bypass_layer() is called
        Then: Returns False
        """
        request = Mock()
        config = Mock()
        config.enabled = False
        request.word_count_enforcement = config

        result = bypass_engine.can_bypass_layer(word_count_layer, request)
        assert result is False

    def test_word_count_layer_without_config_returns_false(
        self, bypass_engine, word_count_layer
    ):
        """
        Given: Word Count Enforcement layer without config
        When: can_bypass_layer() is called
        Then: Returns False
        """
        request = Mock()
        request.word_count_enforcement = None

        result = bypass_engine.can_bypass_layer(word_count_layer, request)
        assert result is False

    def test_phrase_frequency_layer_with_enabled_config_returns_true(
        self, bypass_engine, phrase_frequency_layer, mock_request_with_phrase_frequency
    ):
        """
        Given: Phrase Frequency Guard layer with enabled config
        When: can_bypass_layer() is called
        Then: Returns True
        """
        result = bypass_engine.can_bypass_layer(phrase_frequency_layer, mock_request_with_phrase_frequency)
        assert result is True

    def test_phrase_frequency_layer_with_disabled_config_returns_false(
        self, bypass_engine, phrase_frequency_layer
    ):
        """
        Given: Phrase Frequency Guard layer with disabled config
        When: can_bypass_layer() is called
        Then: Returns False
        """
        request = Mock()
        config = Mock()
        config.enabled = False
        request.phrase_frequency = config

        result = bypass_engine.can_bypass_layer(phrase_frequency_layer, request)
        assert result is False

    def test_lexical_diversity_layer_with_enabled_config_returns_true(
        self, bypass_engine, lexical_diversity_layer, mock_request_with_lexical_diversity
    ):
        """
        Given: Lexical Diversity Guard layer with enabled config
        When: can_bypass_layer() is called
        Then: Returns True
        """
        result = bypass_engine.can_bypass_layer(lexical_diversity_layer, mock_request_with_lexical_diversity)
        assert result is True

    def test_lexical_diversity_layer_with_disabled_config_returns_false(
        self, bypass_engine, lexical_diversity_layer
    ):
        """
        Given: Lexical Diversity Guard layer with disabled config
        When: can_bypass_layer() is called
        Then: Returns False
        """
        request = Mock()
        config = Mock()
        config.enabled = False
        request.lexical_diversity = config

        result = bypass_engine.can_bypass_layer(lexical_diversity_layer, request)
        assert result is False

    def test_cumulative_repetition_layer_with_all_requirements_returns_true(
        self, bypass_engine, cumulative_repetition_layer, mock_request_with_cumulative
    ):
        """
        Given: Cumulative Repetition Guard layer with all required data
        When: can_bypass_layer() is called
        Then: Returns True
        """
        result = bypass_engine.can_bypass_layer(cumulative_repetition_layer, mock_request_with_cumulative)
        assert result is True

    def test_cumulative_repetition_layer_without_cumulative_text_returns_false(
        self, bypass_engine, cumulative_repetition_layer, mock_phrase_frequency_config
    ):
        """
        Given: Cumulative Repetition Guard layer without cumulative_text
        When: can_bypass_layer() is called
        Then: Returns False
        """
        request = Mock()
        request.cumulative_text = None
        request.phrase_frequency = mock_phrase_frequency_config

        result = bypass_engine.can_bypass_layer(cumulative_repetition_layer, request)
        assert result is False

    def test_generic_layer_returns_false(
        self, bypass_engine, generic_layer, mock_request_with_word_count
    ):
        """
        Given: Generic QA layer (not bypassable)
        When: can_bypass_layer() is called
        Then: Returns False
        """
        result = bypass_engine.can_bypass_layer(generic_layer, mock_request_with_word_count)
        assert result is False


# ============================================================================
# TestBypassLayerEvaluation
# ============================================================================

class TestBypassLayerEvaluation:
    """Tests for bypass_layer_evaluation()."""

    @patch('qa_bypass_engine.check_word_count_compliance')
    def test_word_count_layer_calls_word_count_bypass(
        self, mock_compliance, bypass_engine, word_count_layer, mock_request_with_word_count
    ):
        """
        Given: Word Count Enforcement layer
        When: bypass_layer_evaluation() is called
        Then: Calls word count bypass method
        """
        mock_compliance.return_value = {
            'actual_count': 500,
            'required_min': 360,
            'required_max': 660,
            'target_min': 400,
            'target_max': 600,
            'score': 10.0,
            'severity': 'important',
            'target_field': None,
            'is_json_extraction': False
        }

        result = bypass_engine.bypass_layer_evaluation(
            "word " * 500,
            word_count_layer,
            ["gpt-4o", "claude-sonnet-4"],
            mock_request_with_word_count
        )

        assert len(result) == 2
        assert "gpt-4o" in result
        assert "claude-sonnet-4" in result
        mock_compliance.assert_called_once()

    @patch('qa_bypass_engine.analyze_phrase_frequency')
    def test_phrase_frequency_layer_calls_phrase_bypass(
        self, mock_analyze, bypass_engine, phrase_frequency_layer, mock_request_with_phrase_frequency
    ):
        """
        Given: Phrase Frequency Guard layer
        When: bypass_layer_evaluation() is called
        Then: Calls phrase frequency bypass method
        """
        mock_result = Mock()
        mock_result.issues = []
        mock_result.analyzer_output = {"meta": {}, "summary": {"top_by_count": {}}}
        mock_analyze.return_value = mock_result

        result = bypass_engine.bypass_layer_evaluation(
            "test content",
            phrase_frequency_layer,
            ["gpt-4o"],
            mock_request_with_phrase_frequency
        )

        assert len(result) == 1
        mock_analyze.assert_called_once()

    @patch('qa_bypass_engine.analyze_text_lexical_diversity')
    def test_lexical_diversity_layer_calls_lexical_bypass(
        self, mock_analyze, bypass_engine, lexical_diversity_layer, mock_request_with_lexical_diversity
    ):
        """
        Given: Lexical Diversity Guard layer
        When: bypass_layer_evaluation() is called
        Then: Calls lexical diversity bypass method
        """
        mock_result = Mock()
        mock_result.decision_label = "GREEN"
        mock_result.score = 9.0
        mock_result.deal_breaker = False
        mock_result.adjusted_grades = {"mtld": "GREEN"}
        mock_result.analysis = {"metrics": {}, "meta": {}, "top_words": [], "windows": []}
        mock_analyze.return_value = mock_result

        result = bypass_engine.bypass_layer_evaluation(
            "test content with diverse vocabulary",
            lexical_diversity_layer,
            ["gpt-4o"],
            mock_request_with_lexical_diversity
        )

        assert len(result) == 1
        mock_analyze.assert_called_once()

    def test_unsupported_layer_returns_empty_dict(
        self, bypass_engine, generic_layer, mock_request_with_word_count
    ):
        """
        Given: Unsupported layer type
        When: bypass_layer_evaluation() is called
        Then: Returns empty dict and logs warning
        """
        result = bypass_engine.bypass_layer_evaluation(
            "test content",
            generic_layer,
            ["gpt-4o"],
            mock_request_with_word_count
        )

        assert result == {}


# ============================================================================
# TestBypassWordCountEvaluation
# ============================================================================

class TestBypassWordCountEvaluation:
    """Tests for _bypass_word_count_evaluation()."""

    @patch('qa_bypass_engine.check_word_count_compliance')
    def test_perfect_compliance_returns_score_10(
        self, mock_compliance, bypass_engine, word_count_layer, mock_request_with_word_count
    ):
        """
        Given: Content with perfect word count compliance
        When: _bypass_word_count_evaluation() is called
        Then: Returns score 10.0 with no deal_breaker
        """
        mock_compliance.return_value = {
            'actual_count': 500,
            'required_min': 360,
            'required_max': 660,
            'target_min': 400,
            'target_max': 600,
            'score': 10.0,
            'severity': 'important',
            'target_field': None,
            'is_json_extraction': False
        }

        result = bypass_engine._bypass_word_count_evaluation(
            "word " * 500,
            word_count_layer,
            ["gpt-4o"],
            mock_request_with_word_count
        )

        assert result["gpt-4o"].score == 10.0
        assert result["gpt-4o"].deal_breaker is False
        assert "PERFECT COMPLIANCE" in result["gpt-4o"].feedback

    @patch('qa_bypass_engine.check_word_count_compliance')
    def test_within_flexibility_buffer_returns_partial_score(
        self, mock_compliance, bypass_engine, word_count_layer, mock_request_with_word_count
    ):
        """
        Given: Content within flexibility buffer but outside target
        When: _bypass_word_count_evaluation() is called
        Then: Returns partial score (between 5-10)
        """
        mock_compliance.return_value = {
            'actual_count': 375,
            'required_min': 360,
            'required_max': 660,
            'target_min': 400,
            'target_max': 600,
            'score': 6.5,
            'severity': 'important',
            'target_field': None,
            'is_json_extraction': False
        }

        result = bypass_engine._bypass_word_count_evaluation(
            "word " * 375,
            word_count_layer,
            ["gpt-4o"],
            mock_request_with_word_count
        )

        assert result["gpt-4o"].score == 6.5
        assert result["gpt-4o"].deal_breaker is False
        assert "ACCEPTABLE" in result["gpt-4o"].feedback

    @patch('qa_bypass_engine.check_word_count_compliance')
    def test_outside_absolute_limits_returns_deal_breaker(
        self, mock_compliance, bypass_engine, word_count_layer, mock_request_with_word_count
    ):
        """
        Given: Content outside absolute limits (score = 0)
        When: _bypass_word_count_evaluation() is called
        Then: Returns score 0.0 with deal_breaker=True
        """
        mock_compliance.return_value = {
            'actual_count': 200,
            'required_min': 360,
            'required_max': 660,
            'target_min': 400,
            'target_max': 600,
            'score': 0.0,
            'severity': 'deal_breaker',
            'target_field': None,
            'is_json_extraction': False
        }

        result = bypass_engine._bypass_word_count_evaluation(
            "word " * 200,
            word_count_layer,
            ["gpt-4o"],
            mock_request_with_word_count
        )

        assert result["gpt-4o"].score == 0.0
        assert result["gpt-4o"].deal_breaker is True
        assert "VIOLATION" in result["gpt-4o"].feedback

    @patch('qa_bypass_engine.check_word_count_compliance')
    def test_replicates_evaluation_for_all_models(
        self, mock_compliance, bypass_engine, word_count_layer, mock_request_with_word_count
    ):
        """
        Given: Multiple QA models
        When: _bypass_word_count_evaluation() is called
        Then: Returns same evaluation for all models
        """
        mock_compliance.return_value = {
            'actual_count': 500,
            'required_min': 360,
            'required_max': 660,
            'target_min': 400,
            'target_max': 600,
            'score': 10.0,
            'severity': 'important',
            'target_field': None,
            'is_json_extraction': False
        }

        models = ["gpt-4o", "claude-sonnet-4", "gemini-pro"]
        result = bypass_engine._bypass_word_count_evaluation(
            "word " * 500,
            word_count_layer,
            models,
            mock_request_with_word_count
        )

        assert len(result) == 3
        # All should reference the same evaluation object
        assert result["gpt-4o"] is result["claude-sonnet-4"]
        assert result["gpt-4o"] is result["gemini-pro"]

    @patch('qa_bypass_engine.check_word_count_compliance')
    def test_includes_target_field_info_when_json_extraction(
        self, mock_compliance, bypass_engine, word_count_layer, mock_request_with_word_count
    ):
        """
        Given: Content with JSON extraction target_field
        When: _bypass_word_count_evaluation() is called
        Then: Feedback includes target_field info
        """
        mock_compliance.return_value = {
            'actual_count': 500,
            'required_min': 360,
            'required_max': 660,
            'target_min': 400,
            'target_max': 600,
            'score': 10.0,
            'severity': 'important',
            'target_field': 'content',
            'is_json_extraction': True
        }

        result = bypass_engine._bypass_word_count_evaluation(
            '{"content": "test content..."}',
            word_count_layer,
            ["gpt-4o"],
            mock_request_with_word_count
        )

        assert "content" in result["gpt-4o"].feedback
        assert "JSON extraction" in result["gpt-4o"].feedback

    @patch('qa_bypass_engine.check_word_count_compliance')
    def test_evaluation_has_correct_layer_name(
        self, mock_compliance, bypass_engine, word_count_layer, mock_request_with_word_count
    ):
        """
        Given: Word count layer
        When: _bypass_word_count_evaluation() is called
        Then: Evaluation has correct layer name
        """
        mock_compliance.return_value = {
            'actual_count': 500,
            'required_min': 360,
            'required_max': 660,
            'target_min': 400,
            'target_max': 600,
            'score': 10.0,
            'severity': 'important',
            'target_field': None,
            'is_json_extraction': False
        }

        result = bypass_engine._bypass_word_count_evaluation(
            "word " * 500,
            word_count_layer,
            ["gpt-4o"],
            mock_request_with_word_count
        )

        assert result["gpt-4o"].layer == "Word Count Enforcement"

    @patch('qa_bypass_engine.check_word_count_compliance')
    def test_evaluation_includes_algorithmic_model_name(
        self, mock_compliance, bypass_engine, word_count_layer, mock_request_with_word_count
    ):
        """
        Given: Word count layer
        When: _bypass_word_count_evaluation() is called
        Then: Model name indicates algorithmic evaluation
        """
        mock_compliance.return_value = {
            'actual_count': 500,
            'required_min': 360,
            'required_max': 660,
            'target_min': 400,
            'target_max': 600,
            'score': 10.0,
            'severity': 'important',
            'target_field': None,
            'is_json_extraction': False
        }

        result = bypass_engine._bypass_word_count_evaluation(
            "word " * 500,
            word_count_layer,
            ["gpt-4o"],
            mock_request_with_word_count
        )

        assert "Arithmos" in result["gpt-4o"].model
        assert "Algorithmic" in result["gpt-4o"].model


# ============================================================================
# TestBypassPhraseFrequencyEvaluation
# ============================================================================

class TestBypassPhraseFrequencyEvaluation:
    """Tests for _bypass_phrase_frequency_evaluation()."""

    @patch('qa_bypass_engine.analyze_phrase_frequency')
    def test_no_issues_returns_score_10(
        self, mock_analyze, bypass_engine, phrase_frequency_layer, mock_request_with_phrase_frequency
    ):
        """
        Given: Content with no phrase frequency issues
        When: _bypass_phrase_frequency_evaluation() is called
        Then: Returns score 10.0
        """
        mock_result = Mock()
        mock_result.issues = []
        mock_result.analyzer_output = {"meta": {"total_tokens": 100}, "summary": {"top_by_count": {}}}
        mock_analyze.return_value = mock_result

        result = bypass_engine._bypass_phrase_frequency_evaluation(
            "diverse content with varied vocabulary",
            phrase_frequency_layer,
            ["gpt-4o"],
            mock_request_with_phrase_frequency
        )

        assert result["gpt-4o"].score == 10.0
        assert result["gpt-4o"].deal_breaker is False

    @patch('qa_bypass_engine.analyze_phrase_frequency')
    def test_deal_breaker_issues_returns_low_score(
        self, mock_analyze, bypass_engine, phrase_frequency_layer, mock_request_with_phrase_frequency
    ):
        """
        Given: Content with deal-breaker phrase frequency issues
        When: _bypass_phrase_frequency_evaluation() is called
        Then: Returns low score with deal_breaker=True
        """
        issue = Mock()
        issue.phrase = "repeated phrase"
        issue.n = 2
        issue.count = 20
        issue.limit = 5
        issue.severity = "deal_breaker"
        issue.rule_label = "test_rule"
        issue.repeat_ratio_tokens = 0.1
        issue.guidance = "Vary your language"

        mock_result = Mock()
        mock_result.issues = [issue]
        mock_result.analyzer_output = {"meta": {"total_tokens": 100}, "summary": {"top_by_count": {}}}
        mock_analyze.return_value = mock_result

        result = bypass_engine._bypass_phrase_frequency_evaluation(
            "repeated phrase " * 20,
            phrase_frequency_layer,
            ["gpt-4o"],
            mock_request_with_phrase_frequency
        )

        assert result["gpt-4o"].score == 2.0
        assert result["gpt-4o"].deal_breaker is True

    @patch('qa_bypass_engine.analyze_phrase_frequency')
    def test_warning_issues_returns_medium_score(
        self, mock_analyze, bypass_engine, phrase_frequency_layer, mock_request_with_phrase_frequency
    ):
        """
        Given: Content with warning-level phrase frequency issues
        When: _bypass_phrase_frequency_evaluation() is called
        Then: Returns medium score without deal_breaker
        """
        issue = Mock()
        issue.phrase = "somewhat repeated"
        issue.n = 2
        issue.count = 8
        issue.limit = 5
        issue.severity = "warning"
        issue.rule_label = "test_rule"
        issue.repeat_ratio_tokens = None
        issue.guidance = None

        mock_result = Mock()
        mock_result.issues = [issue]
        mock_result.analyzer_output = {"meta": {"total_tokens": 100}, "summary": {"top_by_count": {}}}
        mock_analyze.return_value = mock_result

        result = bypass_engine._bypass_phrase_frequency_evaluation(
            "content with somewhat repeated phrases",
            phrase_frequency_layer,
            ["gpt-4o"],
            mock_request_with_phrase_frequency
        )

        assert result["gpt-4o"].score >= 5.0
        assert result["gpt-4o"].deal_breaker is False

    def test_disabled_config_returns_empty(
        self, bypass_engine, phrase_frequency_layer
    ):
        """
        Given: Disabled phrase frequency config
        When: _bypass_phrase_frequency_evaluation() is called
        Then: Returns empty dict
        """
        request = Mock()
        request.phrase_frequency = None

        result = bypass_engine._bypass_phrase_frequency_evaluation(
            "test content",
            phrase_frequency_layer,
            ["gpt-4o"],
            request
        )

        assert result == {}


# ============================================================================
# TestBypassLexicalDiversityEvaluation
# ============================================================================

class TestBypassLexicalDiversityEvaluation:
    """Tests for _bypass_lexical_diversity_evaluation()."""

    @patch('qa_bypass_engine.analyze_text_lexical_diversity')
    def test_green_decision_returns_high_score(
        self, mock_analyze, bypass_engine, lexical_diversity_layer, mock_request_with_lexical_diversity
    ):
        """
        Given: Content with GREEN lexical diversity decision
        When: _bypass_lexical_diversity_evaluation() is called
        Then: Returns high score without deal_breaker
        """
        mock_result = Mock()
        mock_result.decision_label = "GREEN"
        mock_result.score = 9.0
        mock_result.deal_breaker = False
        mock_result.adjusted_grades = {"mtld": "GREEN", "hdd": "GREEN"}
        mock_result.analysis = {
            "metrics": {"mtld": 80.5, "hdd": 0.85},
            "meta": {"total_tokens": 500},
            "top_words": [{"word": "the", "count": 20, "freq": 0.04}],
            "windows": []
        }
        mock_analyze.return_value = mock_result

        result = bypass_engine._bypass_lexical_diversity_evaluation(
            "diverse content with rich vocabulary",
            lexical_diversity_layer,
            ["gpt-4o"],
            mock_request_with_lexical_diversity
        )

        assert result["gpt-4o"].score == 9.0
        assert result["gpt-4o"].deal_breaker is False
        assert "GREEN" in result["gpt-4o"].feedback

    @patch('qa_bypass_engine.analyze_text_lexical_diversity')
    def test_red_decision_triggers_deal_breaker(
        self, mock_analyze, bypass_engine, lexical_diversity_layer, mock_request_with_lexical_diversity
    ):
        """
        Given: Content with RED lexical diversity decision
        When: _bypass_lexical_diversity_evaluation() is called
        Then: Returns deal_breaker=True
        """
        mock_result = Mock()
        mock_result.decision_label = "RED"
        mock_result.score = 3.0
        mock_result.deal_breaker = True
        mock_result.adjusted_grades = {"mtld": "RED", "hdd": "RED"}
        mock_result.analysis = {
            "metrics": {"mtld": 20.5, "hdd": 0.45},
            "meta": {},
            "top_words": [],
            "windows": []
        }
        mock_analyze.return_value = mock_result

        result = bypass_engine._bypass_lexical_diversity_evaluation(
            "repetitive repetitive repetitive content",
            lexical_diversity_layer,
            ["gpt-4o"],
            mock_request_with_lexical_diversity
        )

        assert result["gpt-4o"].deal_breaker is True
        assert result["gpt-4o"].deal_breaker_reason is not None

    @patch('qa_bypass_engine.analyze_text_lexical_diversity')
    def test_includes_metadata_payload(
        self, mock_analyze, bypass_engine, lexical_diversity_layer, mock_request_with_lexical_diversity
    ):
        """
        Given: Lexical diversity analysis
        When: _bypass_lexical_diversity_evaluation() is called
        Then: Evaluation includes metadata payload
        """
        mock_result = Mock()
        mock_result.decision_label = "AMBER"
        mock_result.score = 6.0
        mock_result.deal_breaker = False
        mock_result.adjusted_grades = {"mtld": "AMBER"}
        mock_result.analysis = {
            "metrics": {"mtld": 50.0},
            "meta": {"language_hint": "en"},
            "top_words": [{"word": "test", "count": 10, "freq": 0.05}],
            "windows": []
        }
        mock_analyze.return_value = mock_result

        result = bypass_engine._bypass_lexical_diversity_evaluation(
            "test content",
            lexical_diversity_layer,
            ["gpt-4o"],
            mock_request_with_lexical_diversity
        )

        assert result["gpt-4o"].metadata is not None
        assert "decision" in result["gpt-4o"].metadata
        assert "adjusted_grades" in result["gpt-4o"].metadata

    def test_disabled_config_returns_empty(
        self, bypass_engine, lexical_diversity_layer
    ):
        """
        Given: Disabled lexical diversity config
        When: _bypass_lexical_diversity_evaluation() is called
        Then: Returns empty dict
        """
        request = Mock()
        request.lexical_diversity = None

        result = bypass_engine._bypass_lexical_diversity_evaluation(
            "test content",
            lexical_diversity_layer,
            ["gpt-4o"],
            request
        )

        assert result == {}


# ============================================================================
# TestBypassCumulativeRepetitionEvaluation
# ============================================================================

class TestBypassCumulativeRepetitionEvaluation:
    """Tests for _bypass_cumulative_repetition_evaluation()."""

    @patch('qa_bypass_engine.analyze_phrase_frequency')
    @patch('qa_bypass_engine.count_words')
    def test_no_issues_returns_score_10(
        self, mock_count, mock_analyze, bypass_engine, cumulative_repetition_layer, mock_request_with_cumulative
    ):
        """
        Given: Cumulative content with no repetition issues
        When: _bypass_cumulative_repetition_evaluation() is called
        Then: Returns score 10.0
        """
        mock_count.return_value = 100
        mock_result = Mock()
        mock_result.issues = []
        mock_result.analyzer_output = {"meta": {}, "summary": {}}
        mock_analyze.return_value = mock_result

        result = bypass_engine._bypass_cumulative_repetition_evaluation(
            "new chapter content",
            cumulative_repetition_layer,
            ["gpt-4o"],
            mock_request_with_cumulative
        )

        assert result["gpt-4o"].score == 10.0
        assert result["gpt-4o"].deal_breaker is False
        assert "CUMULATIVE ANALYSIS" in result["gpt-4o"].feedback

    @patch('qa_bypass_engine.analyze_phrase_frequency')
    @patch('qa_bypass_engine.count_words')
    def test_deal_breaker_issues_returns_low_score(
        self, mock_count, mock_analyze, bypass_engine, cumulative_repetition_layer, mock_request_with_cumulative
    ):
        """
        Given: Cumulative content with deal-breaker repetition
        When: _bypass_cumulative_repetition_evaluation() is called
        Then: Returns low score with deal_breaker=True
        """
        mock_count.return_value = 100

        issue = Mock()
        issue.phrase = "repeated across chapters"
        issue.n = 3
        issue.count = 50
        issue.limit = 10
        issue.severity = "deal_breaker"
        issue.rule_label = "cumulative_rule"
        issue.repeat_ratio_tokens = 0.15
        issue.guidance = "Vary your chapter openings"

        mock_result = Mock()
        mock_result.issues = [issue]
        mock_result.analyzer_output = {"meta": {}, "summary": {}}
        mock_analyze.return_value = mock_result

        result = bypass_engine._bypass_cumulative_repetition_evaluation(
            "new chapter with repeated across chapters",
            cumulative_repetition_layer,
            ["gpt-4o"],
            mock_request_with_cumulative
        )

        assert result["gpt-4o"].score == 2.0
        assert result["gpt-4o"].deal_breaker is True

    def test_missing_cumulative_text_returns_empty(
        self, bypass_engine, cumulative_repetition_layer, mock_phrase_frequency_config
    ):
        """
        Given: Request without cumulative_text
        When: _bypass_cumulative_repetition_evaluation() is called
        Then: Returns empty dict
        """
        request = Mock()
        request.cumulative_text = None
        request.phrase_frequency = mock_phrase_frequency_config

        result = bypass_engine._bypass_cumulative_repetition_evaluation(
            "test content",
            cumulative_repetition_layer,
            ["gpt-4o"],
            request
        )

        assert result == {}


# ============================================================================
# TestShouldBypassQALayer
# ============================================================================

class TestShouldBypassQALayer:
    """Tests for should_bypass_qa_layer()."""

    def test_returns_true_for_bypassable_layer(
        self, bypass_engine, word_count_layer, mock_request_with_word_count
    ):
        """
        Given: Bypassable layer
        When: should_bypass_qa_layer() is called
        Then: Returns True
        """
        result = bypass_engine.should_bypass_qa_layer(
            word_count_layer,
            mock_request_with_word_count
        )
        assert result is True

    def test_returns_false_for_non_bypassable_layer(
        self, bypass_engine, generic_layer, mock_request_with_word_count
    ):
        """
        Given: Non-bypassable layer
        When: should_bypass_qa_layer() is called
        Then: Returns False
        """
        result = bypass_engine.should_bypass_qa_layer(
            generic_layer,
            mock_request_with_word_count
        )
        assert result is False

    def test_logs_when_extra_verbose_true(
        self, bypass_engine, word_count_layer, mock_request_with_word_count
    ):
        """
        Given: extra_verbose=True
        When: should_bypass_qa_layer() is called for bypassable layer
        Then: Logs info message (no error raised)
        """
        # Should not raise any exceptions
        result = bypass_engine.should_bypass_qa_layer(
            word_count_layer,
            mock_request_with_word_count,
            extra_verbose=True
        )
        assert result is True

    def test_no_log_when_extra_verbose_false(
        self, bypass_engine, word_count_layer, mock_request_with_word_count
    ):
        """
        Given: extra_verbose=False (default)
        When: should_bypass_qa_layer() is called
        Then: No verbose logging (function completes normally)
        """
        result = bypass_engine.should_bypass_qa_layer(
            word_count_layer,
            mock_request_with_word_count,
            extra_verbose=False
        )
        assert result is True
