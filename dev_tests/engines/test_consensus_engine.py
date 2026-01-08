"""
Tests for consensus_engine.py - Consensus Calculation Engine.

This module tests the ConsensusEngine which calculates consensus from
multiple AI evaluations and determines whether content meets quality standards.

Functions tested:
- __init__(): Initializes with AI service
- calculate_consensus(): Main consensus calculation
- _calculate_model_averages(): Per-model average calculation
- _determine_approval(): Approval logic
- _collect_feedback_details(): Feedback aggregation
- _safe_get_evaluation_attr(): Safe attribute getter
- advanced_consensus_analysis(): AI-based analysis
- _build_consensus_analysis_prompt(): Prompt building
- _parse_consensus_analysis(): Response parsing
- _identify_model_disagreements(): Disagreement detection
- _calculate_confidence_metrics(): Confidence metrics
"""

import pytest
from unittest.mock import Mock, MagicMock, AsyncMock, patch
from datetime import datetime

from consensus_engine import ConsensusEngine
from models import QALayer, QAEvaluation, ConsensusResult


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_ai_service():
    """Create a mocked AI service."""
    service = MagicMock()
    service.generate_content = AsyncMock(return_value="AI analysis response")
    return service


@pytest.fixture
def consensus_engine(mock_ai_service):
    """Create a ConsensusEngine with mocked AI service."""
    return ConsensusEngine(ai_service=mock_ai_service)


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
def sample_qa_results():
    """Create sample QA results with passing scores."""
    return {
        "Accuracy": {
            "gpt-4o": QAEvaluation(
                model="gpt-4o",
                layer="Accuracy",
                score=8.5,
                feedback="Content is factually accurate",
                deal_breaker=False,
                passes_score=True
            ),
            "claude-sonnet-4": QAEvaluation(
                model="claude-sonnet-4",
                layer="Accuracy",
                score=8.0,
                feedback="Good accuracy overall",
                deal_breaker=False,
                passes_score=True
            )
        },
        "Style": {
            "gpt-4o": QAEvaluation(
                model="gpt-4o",
                layer="Style",
                score=7.5,
                feedback="Writing style is clear",
                deal_breaker=False,
                passes_score=True
            ),
            "claude-sonnet-4": QAEvaluation(
                model="claude-sonnet-4",
                layer="Style",
                score=7.0,
                feedback="Style is acceptable",
                deal_breaker=False,
                passes_score=True
            )
        }
    }


@pytest.fixture
def sample_qa_results_with_deal_breaker():
    """Create sample QA results with a deal-breaker."""
    return {
        "Accuracy": {
            "gpt-4o": QAEvaluation(
                model="gpt-4o",
                layer="Accuracy",
                score=3.0,
                feedback="Critical factual errors detected",
                deal_breaker=True,
                deal_breaker_reason="Contains fabricated statistics",
                passes_score=False
            ),
            "claude-sonnet-4": QAEvaluation(
                model="claude-sonnet-4",
                layer="Accuracy",
                score=4.0,
                feedback="Multiple accuracy issues",
                deal_breaker=False,
                passes_score=False
            )
        }
    }


@pytest.fixture
def sample_qa_results_low_scores():
    """Create sample QA results with scores below minimum."""
    return {
        "Accuracy": {
            "gpt-4o": QAEvaluation(
                model="gpt-4o",
                layer="Accuracy",
                score=5.5,
                feedback="Some accuracy issues",
                deal_breaker=False,
                passes_score=False
            ),
            "claude-sonnet-4": QAEvaluation(
                model="claude-sonnet-4",
                layer="Accuracy",
                score=6.0,
                feedback="Needs improvement",
                deal_breaker=False,
                passes_score=False
            )
        }
    }


@pytest.fixture
def sample_qa_results_high_variance():
    """Create sample QA results with high variance between models."""
    return {
        "Style": {
            "gpt-4o": QAEvaluation(
                model="gpt-4o",
                layer="Style",
                score=9.5,
                feedback="Excellent style",
                deal_breaker=False,
                passes_score=True
            ),
            "claude-sonnet-4": QAEvaluation(
                model="claude-sonnet-4",
                layer="Style",
                score=4.0,
                feedback="Poor style choices",
                deal_breaker=False,
                passes_score=False
            )
        }
    }


# ============================================================================
# TestInit
# ============================================================================

class TestInit:
    """Tests for ConsensusEngine.__init__()."""

    def test_initializes_with_provided_ai_service(self, mock_ai_service):
        """
        Given: An AI service instance
        When: ConsensusEngine is initialized
        Then: Uses the provided AI service
        """
        engine = ConsensusEngine(ai_service=mock_ai_service)
        assert engine.ai_service is mock_ai_service

    @patch('consensus_engine.get_ai_service')
    def test_initializes_with_default_ai_service(self, mock_get_service):
        """
        Given: No AI service provided
        When: ConsensusEngine is initialized
        Then: Uses get_ai_service() to get default
        """
        mock_service = MagicMock()
        mock_get_service.return_value = mock_service

        engine = ConsensusEngine()

        mock_get_service.assert_called_once()
        assert engine.ai_service is mock_service


# ============================================================================
# TestCalculateConsensus
# ============================================================================

class TestCalculateConsensus:
    """Tests for calculate_consensus()."""

    @pytest.mark.asyncio
    async def test_calculates_layer_averages(
        self, consensus_engine, sample_layers, sample_qa_results
    ):
        """
        Given: QA results for multiple layers
        When: calculate_consensus() is called
        Then: Returns correct layer averages
        """
        result = await consensus_engine.calculate_consensus(
            content="Test content",
            qa_results=sample_qa_results,
            layers=sample_layers[:2]  # Only Accuracy and Style
        )

        assert "Accuracy" in result.layer_averages
        assert "Style" in result.layer_averages
        assert result.layer_averages["Accuracy"] == 8.25  # (8.5 + 8.0) / 2
        assert result.layer_averages["Style"] == 7.25  # (7.5 + 7.0) / 2

    @pytest.mark.asyncio
    async def test_calculates_overall_average(
        self, consensus_engine, sample_layers, sample_qa_results
    ):
        """
        Given: QA results
        When: calculate_consensus() is called
        Then: Returns correct overall average score
        """
        result = await consensus_engine.calculate_consensus(
            content="Test content",
            qa_results=sample_qa_results,
            layers=sample_layers[:2]
        )

        # (8.5 + 8.0 + 7.5 + 7.0) / 4 = 7.75
        assert result.average_score == 7.75

    @pytest.mark.asyncio
    async def test_calculates_per_model_averages(
        self, consensus_engine, sample_layers, sample_qa_results
    ):
        """
        Given: QA results from multiple models
        When: calculate_consensus() is called
        Then: Returns correct per-model averages
        """
        result = await consensus_engine.calculate_consensus(
            content="Test content",
            qa_results=sample_qa_results,
            layers=sample_layers[:2]
        )

        assert "gpt-4o" in result.per_model_averages
        assert "claude-sonnet-4" in result.per_model_averages
        assert result.per_model_averages["gpt-4o"] == 8.0  # (8.5 + 7.5) / 2
        assert result.per_model_averages["claude-sonnet-4"] == 7.5  # (8.0 + 7.0) / 2

    @pytest.mark.asyncio
    async def test_detects_deal_breakers(
        self, consensus_engine, sample_layers, sample_qa_results_with_deal_breaker
    ):
        """
        Given: QA results with deal-breaker
        When: calculate_consensus() is called
        Then: Populates deal_breakers list
        """
        result = await consensus_engine.calculate_consensus(
            content="Test content",
            qa_results=sample_qa_results_with_deal_breaker,
            layers=sample_layers[:1]
        )

        assert len(result.deal_breakers) == 1
        assert "Accuracy" in result.deal_breakers[0]
        assert "gpt-4o" in result.deal_breakers[0]

    @pytest.mark.asyncio
    async def test_approved_when_all_criteria_met(
        self, consensus_engine, sample_layers, sample_qa_results
    ):
        """
        Given: QA results meeting all criteria
        When: calculate_consensus() is called
        Then: approved=True
        """
        result = await consensus_engine.calculate_consensus(
            content="Test content",
            qa_results=sample_qa_results,
            layers=sample_layers[:2]
        )

        assert result.approved is True

    @pytest.mark.asyncio
    async def test_rejected_when_deal_breaker_present(
        self, consensus_engine, sample_layers, sample_qa_results_with_deal_breaker
    ):
        """
        Given: QA results with deal-breaker
        When: calculate_consensus() is called
        Then: approved=False
        """
        result = await consensus_engine.calculate_consensus(
            content="Test content",
            qa_results=sample_qa_results_with_deal_breaker,
            layers=sample_layers[:1]
        )

        assert result.approved is False

    @pytest.mark.asyncio
    async def test_rejected_when_layer_below_minimum(
        self, consensus_engine, sample_layers, sample_qa_results_low_scores
    ):
        """
        Given: QA results with scores below layer minimum
        When: calculate_consensus() is called
        Then: approved=False
        """
        result = await consensus_engine.calculate_consensus(
            content="Test content",
            qa_results=sample_qa_results_low_scores,
            layers=sample_layers[:1]  # Accuracy requires 7.0
        )

        # Average is 5.75, below min_score of 7.0
        assert result.approved is False

    @pytest.mark.asyncio
    async def test_returns_total_evaluations_count(
        self, consensus_engine, sample_layers, sample_qa_results
    ):
        """
        Given: QA results
        When: calculate_consensus() is called
        Then: Returns correct total evaluation count
        """
        result = await consensus_engine.calculate_consensus(
            content="Test content",
            qa_results=sample_qa_results,
            layers=sample_layers[:2]
        )

        assert result.total_evaluations == 4  # 2 models x 2 layers

    @pytest.mark.asyncio
    async def test_handles_empty_qa_results(
        self, consensus_engine, sample_layers
    ):
        """
        Given: Empty QA results
        When: calculate_consensus() is called
        Then: Returns result with zero scores
        """
        result = await consensus_engine.calculate_consensus(
            content="Test content",
            qa_results={},
            layers=sample_layers[:1]
        )

        assert result.average_score == 0.0
        assert result.total_evaluations == 0


# ============================================================================
# TestCalculateModelAverages
# ============================================================================

class TestCalculateModelAverages:
    """Tests for _calculate_model_averages()."""

    def test_calculates_averages_per_model(self, consensus_engine, sample_qa_results):
        """
        Given: QA results from multiple models
        When: _calculate_model_averages() is called
        Then: Returns average score per model
        """
        result = consensus_engine._calculate_model_averages(sample_qa_results)

        assert result["gpt-4o"] == 8.0  # (8.5 + 7.5) / 2
        assert result["claude-sonnet-4"] == 7.5  # (8.0 + 7.0) / 2

    def test_handles_single_model(self, consensus_engine):
        """
        Given: QA results from single model
        When: _calculate_model_averages() is called
        Then: Returns that model's average
        """
        qa_results = {
            "Layer1": {
                "gpt-4o": QAEvaluation(
                    model="gpt-4o", layer="Layer1", score=9.0,
                    feedback="Good", deal_breaker=False, passes_score=True
                )
            },
            "Layer2": {
                "gpt-4o": QAEvaluation(
                    model="gpt-4o", layer="Layer2", score=7.0,
                    feedback="OK", deal_breaker=False, passes_score=True
                )
            }
        }

        result = consensus_engine._calculate_model_averages(qa_results)

        assert len(result) == 1
        assert result["gpt-4o"] == 8.0

    def test_handles_empty_results(self, consensus_engine):
        """
        Given: Empty QA results
        When: _calculate_model_averages() is called
        Then: Returns empty dict
        """
        result = consensus_engine._calculate_model_averages({})
        assert result == {}


# ============================================================================
# TestDetermineApproval
# ============================================================================

class TestDetermineApproval:
    """Tests for _determine_approval()."""

    @pytest.mark.asyncio
    async def test_rejects_with_deal_breakers(self, consensus_engine, sample_layers):
        """
        Given: Deal-breakers present
        When: _determine_approval() is called
        Then: Returns False
        """
        result = await consensus_engine._determine_approval(
            content="Test",
            qa_results={},
            layers=sample_layers,
            layer_averages={"Accuracy": 9.0},
            average_score=9.0,
            deal_breakers=["Critical issue found"]
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_rejects_when_layer_below_minimum(self, consensus_engine, sample_layers):
        """
        Given: Layer score below minimum
        When: _determine_approval() is called
        Then: Returns False
        """
        result = await consensus_engine._determine_approval(
            content="Test",
            qa_results={},
            layers=sample_layers[:1],  # Accuracy min_score=7.0
            layer_averages={"Accuracy": 5.0},
            average_score=5.0,
            deal_breakers=[]
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_approves_when_all_criteria_met(self, consensus_engine, sample_layers):
        """
        Given: All criteria met
        When: _determine_approval() is called
        Then: Returns True
        """
        result = await consensus_engine._determine_approval(
            content="Test",
            qa_results={},
            layers=sample_layers[:1],
            layer_averages={"Accuracy": 8.0},
            average_score=8.0,
            deal_breakers=[]
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_approves_at_exact_minimum_score(self, consensus_engine, sample_layers):
        """
        Given: Score exactly at minimum
        When: _determine_approval() is called
        Then: Returns True (not rejected)
        """
        result = await consensus_engine._determine_approval(
            content="Test",
            qa_results={},
            layers=sample_layers[:1],  # Accuracy min_score=7.0
            layer_averages={"Accuracy": 7.0},
            average_score=7.0,
            deal_breakers=[]
        )

        assert result is True


# ============================================================================
# TestCollectFeedbackDetails
# ============================================================================

class TestCollectFeedbackDetails:
    """Tests for _collect_feedback_details()."""

    def test_collects_feedback_by_layer(self, consensus_engine, sample_qa_results):
        """
        Given: QA results
        When: _collect_feedback_details() is called
        Then: Returns feedback organized by layer
        """
        layer_averages = {"Accuracy": 8.25, "Style": 7.25}
        feedback_by_layer, actionable = consensus_engine._collect_feedback_details(
            sample_qa_results, layer_averages
        )

        assert len(feedback_by_layer) == 2
        layer_names = [f["layer"] for f in feedback_by_layer]
        assert "Accuracy" in layer_names
        assert "Style" in layer_names

    def test_includes_model_feedback(self, consensus_engine, sample_qa_results):
        """
        Given: QA results with feedback
        When: _collect_feedback_details() is called
        Then: Each layer includes model feedback
        """
        layer_averages = {"Accuracy": 8.25, "Style": 7.25}
        feedback_by_layer, _ = consensus_engine._collect_feedback_details(
            sample_qa_results, layer_averages
        )

        accuracy_layer = next(f for f in feedback_by_layer if f["layer"] == "Accuracy")
        assert len(accuracy_layer["model_feedback"]) == 2

    def test_collects_actionable_feedback(self, consensus_engine, sample_qa_results):
        """
        Given: QA results with feedback
        When: _collect_feedback_details() is called
        Then: Returns actionable feedback list
        """
        layer_averages = {"Accuracy": 8.25}
        _, actionable = consensus_engine._collect_feedback_details(
            sample_qa_results, layer_averages
        )

        assert len(actionable) > 0
        assert any("Accuracy" in item for item in actionable)

    def test_includes_deal_breaker_reasons(
        self, consensus_engine, sample_qa_results_with_deal_breaker
    ):
        """
        Given: QA results with deal-breaker
        When: _collect_feedback_details() is called
        Then: Includes deal-breaker reasons
        """
        layer_averages = {"Accuracy": 3.5}
        feedback_by_layer, _ = consensus_engine._collect_feedback_details(
            sample_qa_results_with_deal_breaker, layer_averages
        )

        accuracy_layer = next(f for f in feedback_by_layer if f["layer"] == "Accuracy")
        assert len(accuracy_layer["deal_breakers"]) > 0


# ============================================================================
# TestSafeGetEvaluationAttr
# ============================================================================

class TestSafeGetEvaluationAttr:
    """Tests for _safe_get_evaluation_attr()."""

    def test_gets_attribute_from_object(self):
        """
        Given: Object with attribute
        When: _safe_get_evaluation_attr() is called
        Then: Returns attribute value
        """
        evaluation = QAEvaluation(
            model="test", layer="test", score=8.0,
            feedback="test", deal_breaker=False, passes_score=True
        )

        result = ConsensusEngine._safe_get_evaluation_attr(evaluation, "score")
        assert result == 8.0

    def test_gets_value_from_dict(self):
        """
        Given: Dict with key
        When: _safe_get_evaluation_attr() is called
        Then: Returns dict value
        """
        evaluation = {"score": 7.5, "feedback": "test"}

        result = ConsensusEngine._safe_get_evaluation_attr(evaluation, "score")
        assert result == 7.5

    def test_returns_default_for_missing_attribute(self):
        """
        Given: Object without attribute
        When: _safe_get_evaluation_attr() is called with default
        Then: Returns default value
        """
        evaluation = QAEvaluation(
            model="test", layer="test", score=8.0,
            feedback="test", deal_breaker=False, passes_score=True
        )

        result = ConsensusEngine._safe_get_evaluation_attr(
            evaluation, "nonexistent", default="default_value"
        )
        assert result == "default_value"

    def test_returns_default_for_missing_dict_key(self):
        """
        Given: Dict without key
        When: _safe_get_evaluation_attr() is called with default
        Then: Returns default value
        """
        evaluation = {"other_key": "value"}

        result = ConsensusEngine._safe_get_evaluation_attr(
            evaluation, "score", default=0.0
        )
        assert result == 0.0


# ============================================================================
# TestIdentifyModelDisagreements
# ============================================================================

class TestIdentifyModelDisagreements:
    """Tests for _identify_model_disagreements()."""

    def test_identifies_high_variance_disagreements(
        self, consensus_engine, sample_qa_results_high_variance
    ):
        """
        Given: QA results with high score variance
        When: _identify_model_disagreements() is called
        Then: Returns disagreement info
        """
        disagreements = consensus_engine._identify_model_disagreements(
            sample_qa_results_high_variance
        )

        assert len(disagreements) == 1
        assert disagreements[0]["layer"] == "Style"
        assert disagreements[0]["score_range"] == 5.5  # 9.5 - 4.0

    def test_no_disagreements_for_similar_scores(
        self, consensus_engine, sample_qa_results
    ):
        """
        Given: QA results with similar scores
        When: _identify_model_disagreements() is called
        Then: Returns empty list
        """
        disagreements = consensus_engine._identify_model_disagreements(sample_qa_results)

        # Accuracy: 8.5 vs 8.0 (0.5 diff), Style: 7.5 vs 7.0 (0.5 diff)
        # Both below 3.0 threshold
        assert len(disagreements) == 0

    def test_categorizes_severity_correctly(self, consensus_engine):
        """
        Given: QA results with different variance levels
        When: _identify_model_disagreements() is called
        Then: Categorizes severity correctly
        """
        qa_results = {
            "Layer1": {
                "model_a": QAEvaluation(
                    model="model_a", layer="Layer1", score=9.0,
                    feedback="Great", deal_breaker=False, passes_score=True
                ),
                "model_b": QAEvaluation(
                    model="model_b", layer="Layer1", score=2.5,
                    feedback="Poor", deal_breaker=False, passes_score=False
                )
            }
        }

        disagreements = consensus_engine._identify_model_disagreements(qa_results)

        assert len(disagreements) == 1
        assert disagreements[0]["severity"] == "high"  # 6.5 > 5.0

    def test_handles_single_model_per_layer(self, consensus_engine):
        """
        Given: Only one model per layer
        When: _identify_model_disagreements() is called
        Then: Returns empty list (no disagreement possible)
        """
        qa_results = {
            "Layer1": {
                "gpt-4o": QAEvaluation(
                    model="gpt-4o", layer="Layer1", score=8.0,
                    feedback="Good", deal_breaker=False, passes_score=True
                )
            }
        }

        disagreements = consensus_engine._identify_model_disagreements(qa_results)
        assert len(disagreements) == 0


# ============================================================================
# TestCalculateConfidenceMetrics
# ============================================================================

class TestCalculateConfidenceMetrics:
    """Tests for _calculate_confidence_metrics()."""

    def test_calculates_overall_variance(self, consensus_engine, sample_qa_results):
        """
        Given: QA results
        When: _calculate_confidence_metrics() is called
        Then: Returns overall_variance metric
        """
        metrics = consensus_engine._calculate_confidence_metrics(sample_qa_results)

        assert "overall_variance" in metrics
        assert metrics["overall_variance"] >= 0

    def test_calculates_consistency_score(self, consensus_engine, sample_qa_results):
        """
        Given: QA results
        When: _calculate_confidence_metrics() is called
        Then: Returns consistency_score between 0 and 1
        """
        metrics = consensus_engine._calculate_confidence_metrics(sample_qa_results)

        assert "consistency_score" in metrics
        assert 0 <= metrics["consistency_score"] <= 1

    def test_calculates_layer_consistency(self, consensus_engine, sample_qa_results):
        """
        Given: QA results
        When: _calculate_confidence_metrics() is called
        Then: Returns layer_consistency metric
        """
        metrics = consensus_engine._calculate_confidence_metrics(sample_qa_results)

        assert "layer_consistency" in metrics
        assert 0 <= metrics["layer_consistency"] <= 1

    def test_calculates_overall_confidence(self, consensus_engine, sample_qa_results):
        """
        Given: QA results
        When: _calculate_confidence_metrics() is called
        Then: Returns overall_confidence as weighted average
        """
        metrics = consensus_engine._calculate_confidence_metrics(sample_qa_results)

        assert "overall_confidence" in metrics
        # Weighted average: 0.6 * consistency + 0.4 * layer_consistency
        expected = metrics["consistency_score"] * 0.6 + metrics["layer_consistency"] * 0.4
        assert abs(metrics["overall_confidence"] - expected) < 0.001

    def test_high_variance_gives_low_confidence(self, consensus_engine):
        """
        Given: QA results with high variance
        When: _calculate_confidence_metrics() is called
        Then: Returns low confidence score
        """
        qa_results = {
            "Layer1": {
                "model_a": QAEvaluation(
                    model="model_a", layer="Layer1", score=10.0,
                    feedback="Perfect", deal_breaker=False, passes_score=True
                ),
                "model_b": QAEvaluation(
                    model="model_b", layer="Layer1", score=1.0,
                    feedback="Terrible", deal_breaker=False, passes_score=False
                )
            }
        }

        metrics = consensus_engine._calculate_confidence_metrics(qa_results)

        # High variance should result in lower consistency
        assert metrics["consistency_score"] < 0.5


# ============================================================================
# TestParseConsensusAnalysis
# ============================================================================

class TestParseConsensusAnalysis:
    """Tests for _parse_consensus_analysis()."""

    def test_extracts_consensus_score(self, consensus_engine):
        """
        Given: Analysis with score tag
        When: _parse_consensus_analysis() is called
        Then: Extracts consensus score
        """
        analysis = "[CONSENSUS_SCORE]8.5[/CONSENSUS_SCORE]"
        result = consensus_engine._parse_consensus_analysis(analysis)

        assert result["consensus_score"] == 8.5

    def test_extracts_recommendation(self, consensus_engine):
        """
        Given: Analysis with recommendation tag
        When: _parse_consensus_analysis() is called
        Then: Extracts recommendation
        """
        analysis = "[RECOMMENDATION]APPROVE[/RECOMMENDATION]"
        result = consensus_engine._parse_consensus_analysis(analysis)

        assert result["recommendation"] == "APPROVE"

    def test_extracts_analysis_text(self, consensus_engine):
        """
        Given: Analysis with analysis tag
        When: _parse_consensus_analysis() is called
        Then: Extracts analysis text
        """
        analysis = "[ANALYSIS]The content is well-written.[/ANALYSIS]"
        result = consensus_engine._parse_consensus_analysis(analysis)

        assert result["analysis"] == "The content is well-written."

    def test_extracts_improvements(self, consensus_engine):
        """
        Given: Analysis with improvements tag
        When: _parse_consensus_analysis() is called
        Then: Extracts improvements text
        """
        analysis = "[IMPROVEMENTS]Consider adding more examples.[/IMPROVEMENTS]"
        result = consensus_engine._parse_consensus_analysis(analysis)

        assert result["improvements"] == "Consider adding more examples."

    def test_handles_missing_tags(self, consensus_engine):
        """
        Given: Analysis without expected tags
        When: _parse_consensus_analysis() is called
        Then: Returns defaults for missing values
        """
        analysis = "Just plain text without tags"
        result = consensus_engine._parse_consensus_analysis(analysis)

        assert result["consensus_score"] is None
        assert result["recommendation"] == "UNKNOWN"
        assert result["analysis"] == ""
        assert result["improvements"] == ""

    def test_handles_multiline_content(self, consensus_engine):
        """
        Given: Analysis with multiline content
        When: _parse_consensus_analysis() is called
        Then: Extracts full multiline content
        """
        analysis = """[ANALYSIS]
Line 1
Line 2
Line 3
[/ANALYSIS]"""
        result = consensus_engine._parse_consensus_analysis(analysis)

        assert "Line 1" in result["analysis"]
        assert "Line 2" in result["analysis"]
        assert "Line 3" in result["analysis"]


# ============================================================================
# TestAdvancedConsensusAnalysis
# ============================================================================

class TestAdvancedConsensusAnalysis:
    """Tests for advanced_consensus_analysis()."""

    @pytest.mark.asyncio
    async def test_calls_ai_service(
        self, consensus_engine, mock_ai_service, sample_layers, sample_qa_results
    ):
        """
        Given: QA results
        When: advanced_consensus_analysis() is called
        Then: Calls AI service for analysis
        """
        mock_ai_service.generate_content.return_value = """
[CONSENSUS_SCORE]8.0[/CONSENSUS_SCORE]
[RECOMMENDATION]APPROVE[/RECOMMENDATION]
[ANALYSIS]Content meets all criteria.[/ANALYSIS]
[IMPROVEMENTS]None needed.[/IMPROVEMENTS]
"""

        result = await consensus_engine.advanced_consensus_analysis(
            content="Test content",
            qa_results=sample_qa_results,
            layers=sample_layers[:2]
        )

        mock_ai_service.generate_content.assert_called_once()
        assert "consensus_analysis" in result

    @pytest.mark.asyncio
    async def test_returns_structured_analysis(
        self, consensus_engine, mock_ai_service, sample_layers, sample_qa_results
    ):
        """
        Given: AI returns structured response
        When: advanced_consensus_analysis() is called
        Then: Parses and returns structured analysis
        """
        mock_ai_service.generate_content.return_value = """
[CONSENSUS_SCORE]7.5[/CONSENSUS_SCORE]
[RECOMMENDATION]CONDITIONAL[/RECOMMENDATION]
[ANALYSIS]Good but needs minor fixes.[/ANALYSIS]
[IMPROVEMENTS]Fix typos in paragraph 2.[/IMPROVEMENTS]
"""

        result = await consensus_engine.advanced_consensus_analysis(
            content="Test content",
            qa_results=sample_qa_results,
            layers=sample_layers[:2]
        )

        assert result["structured_analysis"]["consensus_score"] == 7.5
        assert result["structured_analysis"]["recommendation"] == "CONDITIONAL"

    @pytest.mark.asyncio
    async def test_includes_disagreements_and_confidence(
        self, consensus_engine, mock_ai_service, sample_layers, sample_qa_results_high_variance
    ):
        """
        Given: QA results with variance
        When: advanced_consensus_analysis() is called
        Then: Includes disagreements and confidence metrics
        """
        mock_ai_service.generate_content.return_value = "[RECOMMENDATION]REJECT[/RECOMMENDATION]"

        result = await consensus_engine.advanced_consensus_analysis(
            content="Test content",
            qa_results=sample_qa_results_high_variance,
            layers=sample_layers[1:2]  # Style layer
        )

        assert "model_disagreements" in result
        assert "confidence_metrics" in result

    @pytest.mark.asyncio
    async def test_handles_ai_service_error(
        self, consensus_engine, mock_ai_service, sample_layers, sample_qa_results
    ):
        """
        Given: AI service raises exception
        When: advanced_consensus_analysis() is called
        Then: Returns error info
        """
        mock_ai_service.generate_content.side_effect = Exception("API error")

        result = await consensus_engine.advanced_consensus_analysis(
            content="Test content",
            qa_results=sample_qa_results,
            layers=sample_layers[:2]
        )

        assert "error" in result
        assert "API error" in result["error"]


# ============================================================================
# TestBuildConsensusAnalysisPrompt
# ============================================================================

class TestBuildConsensusAnalysisPrompt:
    """Tests for _build_consensus_analysis_prompt()."""

    def test_includes_content(self, consensus_engine, sample_layers, sample_qa_results):
        """
        Given: Content and QA results
        When: _build_consensus_analysis_prompt() is called
        Then: Prompt includes the content
        """
        prompt = consensus_engine._build_consensus_analysis_prompt(
            content="Test content here",
            qa_results=sample_qa_results,
            layers=sample_layers[:2]
        )

        assert "Test content here" in prompt

    def test_includes_layer_evaluations(
        self, consensus_engine, sample_layers, sample_qa_results
    ):
        """
        Given: QA results with layer evaluations
        When: _build_consensus_analysis_prompt() is called
        Then: Prompt includes layer information
        """
        prompt = consensus_engine._build_consensus_analysis_prompt(
            content="Test content",
            qa_results=sample_qa_results,
            layers=sample_layers[:2]
        )

        assert "Accuracy" in prompt
        assert "Style" in prompt

    def test_includes_response_format_instructions(
        self, consensus_engine, sample_layers, sample_qa_results
    ):
        """
        Given: Any input
        When: _build_consensus_analysis_prompt() is called
        Then: Prompt includes expected response format
        """
        prompt = consensus_engine._build_consensus_analysis_prompt(
            content="Test",
            qa_results=sample_qa_results,
            layers=sample_layers[:1]
        )

        assert "[CONSENSUS_SCORE]" in prompt
        assert "[RECOMMENDATION]" in prompt
        assert "[ANALYSIS]" in prompt
        assert "[IMPROVEMENTS]" in prompt
