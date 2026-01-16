"""
Tests for per-layer smart-edit flow in generation_processor.py.

This module tests the new per-layer smart-edit implementation where each QA layer
is processed sequentially with iterative smart-edit before moving to the next layer.

Functions tested:
- _calculate_layer_avg_score(): Calculate average score from layer results
- _extract_edits_from_layer_results(): Extract TextEditRange from evaluations
- _check_layer_passed(): Check if layer score >= min_score
- _process_single_layer_with_edits(): Process one layer with iterative editing
- _process_all_layers_with_edits(): Orchestrate all layers sequentially

See SMART_EDIT_PER_LAYER_DESIGN.md for full design documentation.
"""

import pytest
import asyncio
from unittest.mock import Mock, MagicMock, AsyncMock, patch
from datetime import datetime
from typing import Dict, Any, List, Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models import QALayer, QAEvaluation, ContentRequest
from smart_edit import TextEditRange, SeverityLevel, OperationType as EditType


# ============================================================================
# Fixtures
# ============================================================================

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
def sample_qa_layer_high_min():
    """Create a QA layer with high min_score."""
    return QALayer(
        name="Strict Quality",
        description="Strict quality check",
        criteria="Check for quality issues with high bar",
        min_score=9.0,
        order=2
    )


@pytest.fixture
def sample_qa_layer_deal_breaker():
    """Create a deal-breaker QA layer."""
    return QALayer(
        name="Factual Accuracy",
        description="Check factual accuracy",
        criteria="All facts must be verifiable",
        min_score=8.0,
        order=1,
        is_deal_breaker=True
    )


@pytest.fixture
def passing_evaluation():
    """Create a passing QA evaluation."""
    return QAEvaluation(
        model="gpt-4o",
        layer="Test Quality",
        score=8.5,
        feedback="Content passes quality check",
        deal_breaker=False,
        passes_score=True,
        identified_issues=None
    )


@pytest.fixture
def failing_evaluation():
    """Create a failing QA evaluation with issues."""
    return QAEvaluation(
        model="gpt-4o",
        layer="Test Quality",
        score=5.0,
        feedback="Content has quality issues",
        deal_breaker=False,
        passes_score=False,
        identified_issues=[
            TextEditRange(
                marker_mode="phrase",
                paragraph_start="The quick brown fox",
                paragraph_end="over the lazy dog",
                issue_type="quality",
                issue_description="Needs improvement",
                issue_severity=SeverityLevel.MAJOR,
                edit_type=EditType.REPLACE,
            )
        ]
    )


@pytest.fixture
def failing_evaluation_no_edits():
    """Create a failing evaluation without edit suggestions."""
    return QAEvaluation(
        model="gpt-4o",
        layer="Test Quality",
        score=6.0,
        feedback="Content has issues but no specific edits suggested",
        deal_breaker=False,
        passes_score=False,
        identified_issues=[]
    )


@pytest.fixture
def deal_breaker_evaluation():
    """Create a deal-breaker evaluation."""
    return QAEvaluation(
        model="gpt-4o",
        layer="Factual Accuracy",
        score=3.0,
        feedback="Critical factual error detected",
        deal_breaker=True,
        deal_breaker_reason="Incorrect birth year stated",
        passes_score=False,
        identified_issues=None
    )


@pytest.fixture
def sample_content():
    """Sample content for testing."""
    return """The quick brown fox jumps over the lazy dog. This is a test paragraph
with multiple sentences for evaluation purposes. It contains enough text to be
meaningful for smart-edit operations.

Another paragraph follows here. This paragraph also contains multiple sentences
and provides additional content for testing the per-layer editing flow."""


@pytest.fixture
def sample_content_request():
    """Create a sample ContentRequest."""
    return ContentRequest(
        prompt="Write a test article",
        generator_model="gpt-4o",
        content_type="article",
        max_iterations=3,
        max_edit_rounds_per_layer=5,
        qa_layers=[],
        qa_models=["gpt-4o"],
    )


@pytest.fixture
def mock_ai_service():
    """Create a mocked AI service."""
    service = MagicMock()
    service.generate_content = AsyncMock(return_value="Edited content result")
    return service


@pytest.fixture
def mock_qa_engine():
    """Create a mocked QA engine."""
    engine = MagicMock()
    engine._evaluate_single_semantic_layer = AsyncMock()
    return engine


@pytest.fixture
def mock_session():
    """Create a mock session dict."""
    return {
        "session_id": "test-session-123",
        "status": "processing",
        "marker_config": {},
    }


# ============================================================================
# Test: _calculate_layer_avg_score
# ============================================================================

class TestCalculateLayerAvgScore:
    """Tests for _calculate_layer_avg_score() function."""

    def test_single_evaluation_returns_its_score(self, passing_evaluation):
        """
        Given: Layer results with one evaluation
        When: _calculate_layer_avg_score() is called
        Then: Returns that evaluation's score
        """
        from core.generation_processor import _calculate_layer_avg_score

        layer_results = {"gpt-4o": passing_evaluation}

        result = _calculate_layer_avg_score(layer_results)

        assert result == 8.5

    def test_multiple_evaluations_returns_average(self):
        """
        Given: Layer results with multiple evaluations
        When: _calculate_layer_avg_score() is called
        Then: Returns average of all scores
        """
        from core.generation_processor import _calculate_layer_avg_score

        layer_results = {
            "gpt-4o": QAEvaluation(
                model="gpt-4o", layer="Test", score=8.0,
                feedback="Good", deal_breaker=False, passes_score=True
            ),
            "claude-sonnet-4": QAEvaluation(
                model="claude-sonnet-4", layer="Test", score=7.0,
                feedback="OK", deal_breaker=False, passes_score=True
            ),
        }

        result = _calculate_layer_avg_score(layer_results)

        assert result == 7.5  # (8.0 + 7.0) / 2

    def test_empty_results_returns_zero(self):
        """
        Given: Empty layer results
        When: _calculate_layer_avg_score() is called
        Then: Returns 0.0
        """
        from core.generation_processor import _calculate_layer_avg_score

        result = _calculate_layer_avg_score({})

        assert result == 0.0

    def test_evaluation_with_none_score_excluded(self):
        """
        Given: Layer results with some None scores
        When: _calculate_layer_avg_score() is called
        Then: Only includes valid scores in average
        """
        from core.generation_processor import _calculate_layer_avg_score

        layer_results = {
            "gpt-4o": QAEvaluation(
                model="gpt-4o", layer="Test", score=8.0,
                feedback="Good", deal_breaker=False, passes_score=True
            ),
            "claude-sonnet-4": QAEvaluation(
                model="claude-sonnet-4", layer="Test", score=None,
                feedback="Error", deal_breaker=False, passes_score=False
            ),
        }

        result = _calculate_layer_avg_score(layer_results)

        assert result == 8.0  # Only gpt-4o's score counts


# ============================================================================
# Test: _extract_edits_from_layer_results
# ============================================================================

class TestExtractEditsFromLayerResults:
    """Tests for _extract_edits_from_layer_results() function."""

    def test_extracts_issues_from_evaluation(self, failing_evaluation):
        """
        Given: Layer results with identified issues
        When: _extract_edits_from_layer_results() is called
        Then: Returns list of TextEditRange objects
        """
        from core.generation_processor import _extract_edits_from_layer_results

        layer_results = {"gpt-4o": failing_evaluation}

        result = _extract_edits_from_layer_results(layer_results)

        assert len(result) == 1
        assert isinstance(result[0], TextEditRange)
        assert result[0].paragraph_start == "The quick brown fox"

    def test_returns_empty_for_no_issues(self, passing_evaluation):
        """
        Given: Layer results with no identified issues
        When: _extract_edits_from_layer_results() is called
        Then: Returns empty list
        """
        from core.generation_processor import _extract_edits_from_layer_results

        layer_results = {"gpt-4o": passing_evaluation}

        result = _extract_edits_from_layer_results(layer_results)

        assert result == []

    def test_respects_max_edits_limit(self):
        """
        Given: Layer results with many issues
        When: _extract_edits_from_layer_results() is called with max_edits
        Then: Returns at most max_edits items
        """
        from core.generation_processor import _extract_edits_from_layer_results

        # Create evaluation with 5 issues
        issues = [
            TextEditRange(
                marker_mode="phrase",
                paragraph_start=f"Start {i}",
                paragraph_end=f"End {i}",
                issue_type="quality",
                issue_description=f"Issue {i}",
                issue_severity=SeverityLevel.MINOR,
            )
            for i in range(5)
        ]

        evaluation = QAEvaluation(
            model="gpt-4o", layer="Test", score=5.0,
            feedback="Multiple issues", deal_breaker=False,
            passes_score=False, identified_issues=issues
        )

        layer_results = {"gpt-4o": evaluation}

        result = _extract_edits_from_layer_results(layer_results, max_edits=3)

        assert len(result) == 3

    def test_combines_issues_from_multiple_models(self):
        """
        Given: Layer results from multiple models with issues
        When: _extract_edits_from_layer_results() is called
        Then: Combines and deduplicates issues from all models
        """
        from core.generation_processor import _extract_edits_from_layer_results

        issue1 = TextEditRange(
            marker_mode="phrase",
            paragraph_start="Start A",
            paragraph_end="End A",
            issue_type="quality",
            issue_description="Issue A",
            issue_severity=SeverityLevel.MAJOR,
        )
        issue2 = TextEditRange(
            marker_mode="phrase",
            paragraph_start="Start B",
            paragraph_end="End B",
            issue_type="tone",
            issue_description="Issue B",
            issue_severity=SeverityLevel.MINOR,
        )

        eval1 = QAEvaluation(
            model="gpt-4o", layer="Test", score=5.0,
            feedback="Issue A", deal_breaker=False,
            passes_score=False, identified_issues=[issue1]
        )
        eval2 = QAEvaluation(
            model="claude-sonnet-4", layer="Test", score=6.0,
            feedback="Issue B", deal_breaker=False,
            passes_score=False, identified_issues=[issue2]
        )

        layer_results = {"gpt-4o": eval1, "claude-sonnet-4": eval2}

        result = _extract_edits_from_layer_results(layer_results)

        assert len(result) == 2


# ============================================================================
# Test: _check_layer_passed
# ============================================================================

class TestCheckLayerPassed:
    """Tests for _check_layer_passed() function."""

    def test_returns_true_when_score_above_min(self, passing_evaluation):
        """
        Given: Layer results with score above min_score
        When: _check_layer_passed() is called
        Then: Returns True
        """
        from core.generation_processor import _check_layer_passed

        layer_results = {"gpt-4o": passing_evaluation}

        result = _check_layer_passed(layer_results, min_score=7.0)

        assert result is True

    def test_returns_false_when_score_below_min(self, failing_evaluation):
        """
        Given: Layer results with score below min_score
        When: _check_layer_passed() is called
        Then: Returns False
        """
        from core.generation_processor import _check_layer_passed

        layer_results = {"gpt-4o": failing_evaluation}

        result = _check_layer_passed(layer_results, min_score=7.0)

        assert result is False

    def test_returns_true_when_score_equals_min(self):
        """
        Given: Layer results with score exactly at min_score
        When: _check_layer_passed() is called
        Then: Returns True
        """
        from core.generation_processor import _check_layer_passed

        evaluation = QAEvaluation(
            model="gpt-4o", layer="Test", score=7.0,
            feedback="Borderline", deal_breaker=False, passes_score=True
        )
        layer_results = {"gpt-4o": evaluation}

        result = _check_layer_passed(layer_results, min_score=7.0)

        assert result is True

    def test_returns_false_for_empty_results(self):
        """
        Given: Empty layer results
        When: _check_layer_passed() is called
        Then: Returns False (0.0 < any min_score)
        """
        from core.generation_processor import _check_layer_passed

        result = _check_layer_passed({}, min_score=7.0)

        assert result is False


# ============================================================================
# Test: _process_single_layer_with_edits
# ============================================================================

class TestProcessSingleLayerWithEdits:
    """Tests for _process_single_layer_with_edits() async function."""

    @pytest.mark.asyncio
    async def test_layer_passes_first_round_no_edits_needed(
        self, sample_qa_layer, mock_qa_engine, mock_ai_service,
        sample_content, sample_content_request, mock_session, passing_evaluation
    ):
        """
        Given: Layer that passes on first evaluation
        When: _process_single_layer_with_edits() is called
        Then: Returns immediately with passed=True and original content
        """
        from core.generation_processor import _process_single_layer_with_edits

        # Mock QA engine to return passing evaluation
        mock_qa_engine._evaluate_single_semantic_layer = AsyncMock(
            return_value=({"gpt-4o": passing_evaluation}, None)
        )

        edited_content, layer_results, passed, deal_breaker_info = await _process_single_layer_with_edits(
            content=sample_content,
            layer=sample_qa_layer,
            qa_engine=mock_qa_engine,
            qa_models=["gpt-4o"],
            qa_model_names=["gpt-4o"],
            request=sample_content_request,
            session=mock_session,
            session_id="test-session",
            usage_tracker=None,
            phase_logger=None,
            max_rounds=5,
            ai_service=mock_ai_service,
        )

        assert passed is True
        assert deal_breaker_info is None
        assert edited_content == sample_content  # No changes
        assert "gpt-4o" in layer_results
        # Only one evaluation call (passed first try)
        assert mock_qa_engine._evaluate_single_semantic_layer.call_count == 1

    @pytest.mark.asyncio
    async def test_layer_fails_with_deal_breaker_stops_immediately(
        self, sample_qa_layer_deal_breaker, mock_qa_engine, mock_ai_service,
        sample_content, sample_content_request, mock_session, deal_breaker_evaluation
    ):
        """
        Given: Layer with deal-breaker detected
        When: _process_single_layer_with_edits() is called
        Then: Returns immediately with deal_breaker_info populated
        """
        from core.generation_processor import _process_single_layer_with_edits

        deal_breaker_info_from_qa = {
            "layer": "Factual Accuracy",
            "reason": "Critical factual error",
            "models": ["gpt-4o"]
        }

        mock_qa_engine._evaluate_single_semantic_layer = AsyncMock(
            return_value=({"gpt-4o": deal_breaker_evaluation}, deal_breaker_info_from_qa)
        )

        edited_content, layer_results, passed, deal_breaker_info = await _process_single_layer_with_edits(
            content=sample_content,
            layer=sample_qa_layer_deal_breaker,
            qa_engine=mock_qa_engine,
            qa_models=["gpt-4o"],
            qa_model_names=["gpt-4o"],
            request=sample_content_request,
            session=mock_session,
            session_id="test-session",
            usage_tracker=None,
            phase_logger=None,
            max_rounds=5,
            ai_service=mock_ai_service,
        )

        assert passed is False
        assert deal_breaker_info is not None
        assert deal_breaker_info["layer"] == "Factual Accuracy"
        # Only one evaluation (stopped at deal-breaker)
        assert mock_qa_engine._evaluate_single_semantic_layer.call_count == 1

    @pytest.mark.asyncio
    async def test_layer_fails_no_edits_continues_to_max_rounds(
        self, sample_qa_layer, mock_qa_engine, mock_ai_service,
        sample_content, sample_content_request, mock_session, failing_evaluation_no_edits
    ):
        """
        Given: Layer that fails but has no edits suggested
        When: _process_single_layer_with_edits() is called
        Then: Continues re-evaluating until max_rounds reached
        """
        from core.generation_processor import _process_single_layer_with_edits

        mock_qa_engine._evaluate_single_semantic_layer = AsyncMock(
            return_value=({"gpt-4o": failing_evaluation_no_edits}, None)
        )

        max_rounds = 3

        edited_content, layer_results, passed, deal_breaker_info = await _process_single_layer_with_edits(
            content=sample_content,
            layer=sample_qa_layer,
            qa_engine=mock_qa_engine,
            qa_models=["gpt-4o"],
            qa_model_names=["gpt-4o"],
            request=sample_content_request,
            session=mock_session,
            session_id="test-session",
            usage_tracker=None,
            phase_logger=None,
            max_rounds=max_rounds,
            ai_service=mock_ai_service,
        )

        assert passed is False
        assert deal_breaker_info is None
        # Should have tried max_rounds times
        assert mock_qa_engine._evaluate_single_semantic_layer.call_count == max_rounds

    @pytest.mark.asyncio
    async def test_layer_passes_after_applying_edits(
        self, sample_qa_layer, mock_qa_engine, mock_ai_service,
        sample_content, sample_content_request, mock_session,
        failing_evaluation, passing_evaluation
    ):
        """
        Given: Layer that fails first, then passes after edits
        When: _process_single_layer_with_edits() is called
        Then: Applies edits and returns with passed=True
        """
        from core.generation_processor import _process_single_layer_with_edits

        # First call fails with edits, second call passes
        call_count = 0

        async def mock_evaluate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ({"gpt-4o": failing_evaluation}, None)
            else:
                return ({"gpt-4o": passing_evaluation}, None)

        mock_qa_engine._evaluate_single_semantic_layer = AsyncMock(side_effect=mock_evaluate)

        # Mock _generate_smart_edits to return edited content
        with patch('core.generation_processor._generate_smart_edits', new_callable=AsyncMock) as mock_smart_edit, \
             patch('core.generation_processor._locate_edit_segment', return_value=(0, 50)):

            mock_smart_edit.return_value = "Edited content after smart edit"

            edited_content, layer_results, passed, deal_breaker_info = await _process_single_layer_with_edits(
                content=sample_content,
                layer=sample_qa_layer,
                qa_engine=mock_qa_engine,
                qa_models=["gpt-4o"],
                qa_model_names=["gpt-4o"],
                request=sample_content_request,
                session=mock_session,
                session_id="test-session",
                usage_tracker=None,
                phase_logger=None,
                max_rounds=5,
                ai_service=mock_ai_service,
            )

        assert passed is True
        assert deal_breaker_info is None
        assert edited_content == "Edited content after smart edit"
        # Two evaluations: fail then pass
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_cancellation_during_layer_processing(
        self, sample_qa_layer, mock_qa_engine, mock_ai_service,
        sample_content, sample_content_request, mock_session
    ):
        """
        Given: Cancellation requested during processing
        When: _process_single_layer_with_edits() is called
        Then: Returns early with empty results
        """
        from core.generation_processor import _process_single_layer_with_edits

        async def cancel_callback():
            return True  # Always cancelled

        edited_content, layer_results, passed, deal_breaker_info = await _process_single_layer_with_edits(
            content=sample_content,
            layer=sample_qa_layer,
            qa_engine=mock_qa_engine,
            qa_models=["gpt-4o"],
            qa_model_names=["gpt-4o"],
            request=sample_content_request,
            session=mock_session,
            session_id="test-session",
            usage_tracker=None,
            phase_logger=None,
            max_rounds=5,
            ai_service=mock_ai_service,
            cancel_callback=cancel_callback,
        )

        assert passed is False
        assert layer_results == {}
        # No QA evaluation should happen
        mock_qa_engine._evaluate_single_semantic_layer.assert_not_called()


# ============================================================================
# Test: _process_all_layers_with_edits
# ============================================================================

class TestProcessAllLayersWithEdits:
    """Tests for _process_all_layers_with_edits() async function."""

    @pytest.mark.asyncio
    async def test_all_layers_pass(
        self, mock_qa_engine, mock_ai_service,
        sample_content, sample_content_request, mock_session, passing_evaluation
    ):
        """
        Given: Multiple layers that all pass
        When: _process_all_layers_with_edits() is called
        Then: Returns all_passed=True and processes all layers
        """
        from core.generation_processor import _process_all_layers_with_edits

        layers = [
            QALayer(name="Layer1", description="L1", criteria="C1", min_score=7.0, order=1),
            QALayer(name="Layer2", description="L2", criteria="C2", min_score=7.0, order=2),
        ]

        mock_qa_engine._evaluate_single_semantic_layer = AsyncMock(
            return_value=({"gpt-4o": passing_evaluation}, None)
        )

        (
            final_content, all_qa_results, all_passed,
            deal_breaker_info, layers_summary
        ) = await _process_all_layers_with_edits(
            content=sample_content,
            qa_layers=layers,
            qa_engine=mock_qa_engine,
            qa_models=["gpt-4o"],
            qa_model_names=["gpt-4o"],
            request=sample_content_request,
            session=mock_session,
            session_id="test-session",
            usage_tracker=None,
            phase_logger=None,
            ai_service=mock_ai_service,
        )

        assert all_passed is True
        assert deal_breaker_info is None
        assert len(layers_summary) == 2
        assert layers_summary["Layer1"]["passed"] is True
        assert layers_summary["Layer2"]["passed"] is True

    @pytest.mark.asyncio
    async def test_one_layer_fails_continues_to_next(
        self, mock_qa_engine, mock_ai_service,
        sample_content, sample_content_request, mock_session,
        failing_evaluation_no_edits, passing_evaluation
    ):
        """
        Given: First layer fails, second passes
        When: _process_all_layers_with_edits() is called
        Then: Continues processing all layers, returns all_passed=False
        """
        from core.generation_processor import _process_all_layers_with_edits

        layers = [
            QALayer(name="Layer1", description="L1", criteria="C1", min_score=7.0, order=1),
            QALayer(name="Layer2", description="L2", criteria="C2", min_score=7.0, order=2),
        ]

        # First layer fails (with no edits, so max rounds quickly), second passes
        call_count = 0
        max_rounds = sample_content_request.max_edit_rounds_per_layer

        async def mock_evaluate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # First max_rounds calls are for Layer1 (all fail)
            if call_count <= max_rounds:
                return ({"gpt-4o": failing_evaluation_no_edits}, None)
            # Then Layer2 passes
            return ({"gpt-4o": passing_evaluation}, None)

        mock_qa_engine._evaluate_single_semantic_layer = AsyncMock(side_effect=mock_evaluate)

        (
            final_content, all_qa_results, all_passed,
            deal_breaker_info, layers_summary
        ) = await _process_all_layers_with_edits(
            content=sample_content,
            qa_layers=layers,
            qa_engine=mock_qa_engine,
            qa_models=["gpt-4o"],
            qa_model_names=["gpt-4o"],
            request=sample_content_request,
            session=mock_session,
            session_id="test-session",
            usage_tracker=None,
            phase_logger=None,
            ai_service=mock_ai_service,
        )

        assert all_passed is False  # Layer1 failed
        assert deal_breaker_info is None
        assert layers_summary["Layer1"]["passed"] is False
        assert layers_summary["Layer2"]["passed"] is True

    @pytest.mark.asyncio
    async def test_deal_breaker_in_layer_continues_to_next(
        self, mock_qa_engine, mock_ai_service,
        sample_content, sample_content_request, mock_session,
        deal_breaker_evaluation, passing_evaluation
    ):
        """
        Given: Deal-breaker in first layer
        When: _process_all_layers_with_edits() is called
        Then: Records deal-breaker but continues to next layers
        """
        from core.generation_processor import _process_all_layers_with_edits

        layers = [
            QALayer(name="Accuracy", description="L1", criteria="C1", min_score=7.0, order=1, is_deal_breaker=True),
            QALayer(name="Quality", description="L2", criteria="C2", min_score=7.0, order=2),
        ]

        call_count = 0

        async def mock_evaluate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First layer has deal-breaker
                return (
                    {"gpt-4o": deal_breaker_evaluation},
                    {"layer": "Accuracy", "reason": "Critical error", "models": ["gpt-4o"]}
                )
            # Subsequent layers pass
            return ({"gpt-4o": passing_evaluation}, None)

        mock_qa_engine._evaluate_single_semantic_layer = AsyncMock(side_effect=mock_evaluate)

        (
            final_content, all_qa_results, all_passed,
            deal_breaker_info, layers_summary
        ) = await _process_all_layers_with_edits(
            content=sample_content,
            qa_layers=layers,
            qa_engine=mock_qa_engine,
            qa_models=["gpt-4o"],
            qa_model_names=["gpt-4o"],
            request=sample_content_request,
            session=mock_session,
            session_id="test-session",
            usage_tracker=None,
            phase_logger=None,
            ai_service=mock_ai_service,
        )

        assert all_passed is False
        assert deal_breaker_info is not None
        assert deal_breaker_info["layer"] == "Accuracy"
        assert layers_summary["Accuracy"]["deal_breaker"] is True
        # Second layer should still be processed
        assert "Quality" in layers_summary

    @pytest.mark.asyncio
    async def test_layers_processed_in_order(
        self, mock_qa_engine, mock_ai_service,
        sample_content, sample_content_request, mock_session, passing_evaluation
    ):
        """
        Given: Layers with different order values
        When: _process_all_layers_with_edits() is called
        Then: Processes layers in ascending order
        """
        from core.generation_processor import _process_all_layers_with_edits

        # Layers provided out of order
        layers = [
            QALayer(name="Third", description="L3", criteria="C3", min_score=7.0, order=3),
            QALayer(name="First", description="L1", criteria="C1", min_score=7.0, order=1),
            QALayer(name="Second", description="L2", criteria="C2", min_score=7.0, order=2),
        ]

        processed_order = []

        async def mock_evaluate(content, layer, *args, **kwargs):
            processed_order.append(layer.name)
            return ({"gpt-4o": passing_evaluation}, None)

        mock_qa_engine._evaluate_single_semantic_layer = AsyncMock(side_effect=mock_evaluate)

        await _process_all_layers_with_edits(
            content=sample_content,
            qa_layers=layers,
            qa_engine=mock_qa_engine,
            qa_models=["gpt-4o"],
            qa_model_names=["gpt-4o"],
            request=sample_content_request,
            session=mock_session,
            session_id="test-session",
            usage_tracker=None,
            phase_logger=None,
            ai_service=mock_ai_service,
        )

        assert processed_order == ["First", "Second", "Third"]

    @pytest.mark.asyncio
    async def test_layers_summary_contains_correct_data(
        self, mock_qa_engine, mock_ai_service,
        sample_content, sample_content_request, mock_session, passing_evaluation
    ):
        """
        Given: Multiple layers processed
        When: _process_all_layers_with_edits() is called
        Then: layers_summary contains score, min_score, passed, order for each layer
        """
        from core.generation_processor import _process_all_layers_with_edits

        layers = [
            QALayer(name="Layer1", description="L1", criteria="C1", min_score=7.0, order=1),
            QALayer(name="Layer2", description="L2", criteria="C2", min_score=8.0, order=2),
        ]

        mock_qa_engine._evaluate_single_semantic_layer = AsyncMock(
            return_value=({"gpt-4o": passing_evaluation}, None)
        )

        (
            final_content, all_qa_results, all_passed,
            deal_breaker_info, layers_summary
        ) = await _process_all_layers_with_edits(
            content=sample_content,
            qa_layers=layers,
            qa_engine=mock_qa_engine,
            qa_models=["gpt-4o"],
            qa_model_names=["gpt-4o"],
            request=sample_content_request,
            session=mock_session,
            session_id="test-session",
            usage_tracker=None,
            phase_logger=None,
            ai_service=mock_ai_service,
        )

        # Check Layer1 summary
        assert "Layer1" in layers_summary
        assert "score" in layers_summary["Layer1"]
        assert "min_score" in layers_summary["Layer1"]
        assert "passed" in layers_summary["Layer1"]
        assert "order" in layers_summary["Layer1"]
        assert layers_summary["Layer1"]["min_score"] == 7.0
        assert layers_summary["Layer1"]["order"] == 1

        # Check Layer2 summary
        assert "Layer2" in layers_summary
        assert layers_summary["Layer2"]["min_score"] == 8.0
        assert layers_summary["Layer2"]["order"] == 2


# ============================================================================
# Test: Integration with process_content_generation
# ============================================================================

class TestPerLayerIntegration:
    """Integration tests verifying the per-layer flow works with main generation."""

    @pytest.mark.asyncio
    async def test_qa_comprehensive_result_structure(
        self, mock_qa_engine, mock_ai_service,
        sample_content, sample_content_request, mock_session, passing_evaluation
    ):
        """
        Given: Per-layer processing completes
        When: Building qa_comprehensive_result
        Then: Structure is compatible with downstream code (consensus, etc.)
        """
        from core.generation_processor import _process_all_layers_with_edits

        layers = [
            QALayer(name="Quality", description="L1", criteria="C1", min_score=7.0, order=1),
        ]

        mock_qa_engine._evaluate_single_semantic_layer = AsyncMock(
            return_value=({"gpt-4o": passing_evaluation}, None)
        )

        (
            final_content, all_qa_results, all_passed,
            deal_breaker_info, layers_summary
        ) = await _process_all_layers_with_edits(
            content=sample_content,
            qa_layers=layers,
            qa_engine=mock_qa_engine,
            qa_models=["gpt-4o"],
            qa_model_names=["gpt-4o"],
            request=sample_content_request,
            session=mock_session,
            session_id="test-session",
            usage_tracker=None,
            phase_logger=None,
            ai_service=mock_ai_service,
        )

        # Build qa_comprehensive_result as done in process_content_generation
        total_score = 0.0
        total_evals = 0
        critical_issues = []

        for layer_name, layer_data in layers_summary.items():
            total_score += layer_data.get("score", 0.0)
            total_evals += 1
            if layer_data.get("deal_breaker"):
                critical_issues.append({
                    "layer": layer_name,
                    "description": f"Deal-breaker in layer '{layer_name}'",
                    "score": layer_data.get("score", 0.0),
                })

        avg_score = total_score / total_evals if total_evals > 0 else 0.0

        qa_comprehensive_result = {
            "summary": {
                "has_deal_breakers": deal_breaker_info is not None,
                "force_iteration": not all_passed,
                "average_score": avg_score,
                "total_evaluations": total_evals,
                "layers_summary": layers_summary,
            },
            "qa_results": all_qa_results,
            "critical_issues": critical_issues,
            "evidence_grounding": None,
        }

        # Verify structure
        assert "summary" in qa_comprehensive_result
        assert "qa_results" in qa_comprehensive_result
        assert "critical_issues" in qa_comprehensive_result
        assert qa_comprehensive_result["summary"]["has_deal_breakers"] is False
        assert qa_comprehensive_result["summary"]["force_iteration"] is False
        assert qa_comprehensive_result["summary"]["average_score"] == 8.5
        assert qa_comprehensive_result["summary"]["total_evaluations"] == 1


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
