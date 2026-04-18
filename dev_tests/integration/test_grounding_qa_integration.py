"""
Integration tests for Evidence Grounding with QA Engine.

Phase 7 of Strawberry Integration: Tests the complete integration of evidence
grounding as a special QA layer within the QA evaluation pipeline.

Tests verify:
- Evidence grounding executes as a special layer with configurable ordering
- Deal-breaker triggering stops evaluation and forces iteration
- Regeneration flow works correctly
- Results are properly integrated into qa_comprehensive_result
- Ordering respects on_flag auto-calculation and explicit overrides

All tests use mocked AI responses to avoid API calls.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import math

from qa_engine import QAEngine
from models import (
    EvidenceGroundingConfig,
    EvidenceGroundingResult,
    ExtractedClaim,
    ClaimBudgetResult,
    EvidenceSpan,
    SpanType,
    QALayer,
    QAEvaluation,
    ContentRequest,
)
from evidence_grounding import get_effective_order


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def mock_ai_service():
    """Create a mock AI service for testing."""
    service = MagicMock()
    service.generate_content = AsyncMock(return_value="Generated content")
    service.generate_with_logprobs = AsyncMock()
    return service


@pytest.fixture
def qa_engine(mock_ai_service):
    """Create a QAEngine with mocked AI service."""
    with patch('ai_service.get_ai_service', return_value=mock_ai_service):
        engine = QAEngine(ai_service=mock_ai_service)
        return engine


@pytest.fixture
def sample_semantic_layer():
    """Create a sample semantic QA layer."""
    return QALayer(
        name="Accuracy",
        description="Check factual accuracy",
        criteria="Content must be factually accurate",
        min_score=7.0,
        order=1,
    )


@pytest.fixture
def grounding_config_warn():
    """Grounding config with on_flag='warn' (runs last, order=999)."""
    return EvidenceGroundingConfig(
        enabled=True,
        model="gpt-4o-mini",
        max_claims=10,
        filter_trivial=True,
        min_claim_importance=0.5,
        budget_gap_threshold=0.5,
        max_flagged_claims=2,
        on_flag="warn",
    )


@pytest.fixture
def grounding_config_deal_breaker():
    """Grounding config with on_flag='deal_breaker' (runs first, order=0)."""
    return EvidenceGroundingConfig(
        enabled=True,
        model="gpt-4o-mini",
        max_claims=10,
        filter_trivial=True,
        min_claim_importance=0.5,
        budget_gap_threshold=0.5,
        max_flagged_claims=2,
        on_flag="deal_breaker",
    )


@pytest.fixture
def grounding_config_regenerate():
    """Grounding config with on_flag='regenerate' (runs first, order=0)."""
    return EvidenceGroundingConfig(
        enabled=True,
        model="gpt-4o-mini",
        max_claims=10,
        max_flagged_claims=1,
        on_flag="regenerate",
    )


@pytest.fixture
def sample_claims_passed():
    """Sample claims that pass grounding (1 flagged < threshold 2)."""
    return [
        ExtractedClaim(
            idx=0,
            claim="Marie Curie was born in Warsaw in 1867",
            kind="factual",
            importance=0.9,
            cited_spans=["S0"],
            source_text="Marie Curie was born in Warsaw in 1867."
        ),
        ExtractedClaim(
            idx=1,
            claim="She studied physics in Paris",
            kind="factual",
            importance=0.8,
            cited_spans=["S1"],
            source_text="She studied physics in Paris."
        ),
    ]


@pytest.fixture
def budget_results_passed():
    """Budget results where only 1 claim is flagged (passes threshold of 2)."""
    return [
        ClaimBudgetResult(
            idx=0,
            claim="Marie Curie was born in Warsaw in 1867",
            cited_spans=["S0"],
            posterior_yes=0.95,
            prior_yes=0.15,
            required_bits=2.5,
            observed_bits=2.8,
            budget_gap=-0.3,
            flagged=False,
            confidence_delta=0.8,
        ),
        ClaimBudgetResult(
            idx=1,
            claim="She studied physics in Paris",
            cited_spans=["S1"],
            posterior_yes=0.88,
            prior_yes=0.82,
            required_bits=0.9,
            observed_bits=0.08,
            budget_gap=0.82,
            flagged=True,  # One flagged
            confidence_delta=0.06,
        ),
    ]


@pytest.fixture
def budget_results_failed():
    """Budget results where 2+ claims are flagged (fails threshold of 2)."""
    return [
        ClaimBudgetResult(
            idx=0,
            claim="Claim 1",
            cited_spans=["S0"],
            posterior_yes=0.85,
            prior_yes=0.80,
            required_bits=0.9,
            observed_bits=0.1,
            budget_gap=0.8,
            flagged=True,
            confidence_delta=0.05,
        ),
        ClaimBudgetResult(
            idx=1,
            claim="Claim 2",
            cited_spans=["S1"],
            posterior_yes=0.82,
            prior_yes=0.78,
            required_bits=0.85,
            observed_bits=0.08,
            budget_gap=0.77,
            flagged=True,
            confidence_delta=0.04,
        ),
    ]


@pytest.fixture
def mock_grounding_result_passed(budget_results_passed):
    """A grounding result that passes (1 flagged < threshold 2)."""
    return EvidenceGroundingResult(
        enabled=True,
        model_used="gpt-4o-mini",
        total_claims_extracted=2,
        claims_after_filter=2,
        claims_verified=2,
        claims=budget_results_passed,
        flagged_claims=1,
        max_budget_gap=0.82,
        passed=True,
        triggered_action=None,
        verification_time_ms=1500.0,
        tokens_used=500,
    )


@pytest.fixture
def mock_grounding_result_failed(budget_results_failed):
    """A grounding result that fails (2 flagged >= threshold 2)."""
    return EvidenceGroundingResult(
        enabled=True,
        model_used="gpt-4o-mini",
        total_claims_extracted=2,
        claims_after_filter=2,
        claims_verified=2,
        claims=budget_results_failed,
        flagged_claims=2,
        max_budget_gap=0.8,
        passed=False,
        triggered_action="deal_breaker",
        verification_time_ms=1200.0,
        tokens_used=450,
    )


# =============================================================================
# Tests for get_effective_order
# =============================================================================

class TestEffectiveOrderIntegration:
    """Tests for order calculation in integration context."""

    def test_warn_auto_order_is_999(self, grounding_config_warn):
        """
        Given: Config with on_flag='warn'
        When: get_effective_order is called
        Then: Returns 999 (runs after semantic QA)
        """
        assert get_effective_order(grounding_config_warn) == 999

    def test_deal_breaker_auto_order_is_0(self, grounding_config_deal_breaker):
        """
        Given: Config with on_flag='deal_breaker'
        When: get_effective_order is called
        Then: Returns 0 (fail-fast, runs first)
        """
        assert get_effective_order(grounding_config_deal_breaker) == 0

    def test_regenerate_auto_order_is_0(self, grounding_config_regenerate):
        """
        Given: Config with on_flag='regenerate'
        When: get_effective_order is called
        Then: Returns 0 (fail-fast, runs first)
        """
        assert get_effective_order(grounding_config_regenerate) == 0

    def test_explicit_order_overrides_auto(self):
        """
        Given: Config with explicit order=5
        When: get_effective_order is called
        Then: Returns 5, ignoring auto-calculation
        """
        config = EvidenceGroundingConfig(
            enabled=True,
            on_flag="deal_breaker",  # Would auto to 0
            order=5,  # Explicit override
        )
        assert get_effective_order(config) == 5


# =============================================================================
# Tests for Evidence Grounding as Special QA Layer
# =============================================================================

class TestGroundingAsQALayer:
    """Tests for evidence grounding execution within QA pipeline."""

    @pytest.mark.asyncio
    async def test_grounding_executes_in_pipeline(
        self, qa_engine, grounding_config_warn, sample_claims_passed, mock_grounding_result_passed
    ):
        """
        Given: Grounding enabled with on_flag='warn'
        When: evaluate_all_layers_with_progress is called
        Then: Grounding executes and results are in 'Evidence Grounding' key
        """
        # Mock the grounding engine
        qa_engine.grounding_engine.run_grounding_check = AsyncMock(
            return_value=mock_grounding_result_passed
        )

        results = await qa_engine.evaluate_all_layers_with_progress(
            content="Test content about Marie Curie.",
            layers=[],  # No semantic layers
            qa_models=[],
            evidence_grounding_config=grounding_config_warn,
            context_for_grounding="Marie Curie was born in Warsaw in 1867.",
        )

        # Verify grounding was executed
        assert "Evidence Grounding" in results
        assert "evidence_grounding_logprobs" in results["Evidence Grounding"]

        qa_eval = results["Evidence Grounding"]["evidence_grounding_logprobs"]
        assert isinstance(qa_eval, QAEvaluation)
        assert qa_eval.layer == "Evidence Grounding"
        assert qa_eval.model == "evidence_grounding_logprobs"

    @pytest.mark.asyncio
    async def test_grounding_result_contains_metadata(
        self, qa_engine, grounding_config_warn, mock_grounding_result_passed
    ):
        """
        Given: Grounding passes
        When: QAEvaluation is created
        Then: Metadata contains full grounding_result
        """
        qa_engine.grounding_engine.run_grounding_check = AsyncMock(
            return_value=mock_grounding_result_passed
        )

        results = await qa_engine.evaluate_all_layers_with_progress(
            content="Test content",
            layers=[],
            qa_models=[],
            evidence_grounding_config=grounding_config_warn,
            context_for_grounding="Test context",
        )

        qa_eval = results["Evidence Grounding"]["evidence_grounding_logprobs"]
        assert qa_eval.metadata is not None
        assert "grounding_result" in qa_eval.metadata

        grounding_data = qa_eval.metadata["grounding_result"]
        assert grounding_data["passed"] is True
        assert grounding_data["flagged_claims"] == 1

    @pytest.mark.asyncio
    async def test_grounding_passed_score_is_10(
        self, qa_engine, grounding_config_warn, mock_grounding_result_passed
    ):
        """
        Given: Grounding passes
        When: QAEvaluation is created
        Then: Score is 10.0
        """
        qa_engine.grounding_engine.run_grounding_check = AsyncMock(
            return_value=mock_grounding_result_passed
        )

        results = await qa_engine.evaluate_all_layers_with_progress(
            content="Test content",
            layers=[],
            qa_models=[],
            evidence_grounding_config=grounding_config_warn,
            context_for_grounding="Test context",
        )

        qa_eval = results["Evidence Grounding"]["evidence_grounding_logprobs"]
        assert qa_eval.score == 10.0
        assert qa_eval.passes_score is True
        assert qa_eval.deal_breaker is False

    @pytest.mark.asyncio
    async def test_grounding_warn_failed_score_is_5(self, qa_engine, mock_grounding_result_failed):
        """
        Given: Grounding fails with on_flag='warn'
        When: QAEvaluation is created
        Then: Score is 5.0 (warning, not deal-breaker)
        """
        # Modify the result to be warn action
        mock_grounding_result_failed.triggered_action = "warn"

        config = EvidenceGroundingConfig(
            enabled=True,
            max_flagged_claims=2,
            on_flag="warn",  # Warning only
        )

        qa_engine.grounding_engine.run_grounding_check = AsyncMock(
            return_value=mock_grounding_result_failed
        )

        results = await qa_engine.evaluate_all_layers_with_progress(
            content="Test content",
            layers=[],
            qa_models=[],
            evidence_grounding_config=config,
            context_for_grounding="Test context",
        )

        qa_eval = results["Evidence Grounding"]["evidence_grounding_logprobs"]
        assert qa_eval.score == 5.0
        assert qa_eval.deal_breaker is False  # warn doesn't trigger deal_breaker


# =============================================================================
# Tests for Deal-Breaker Triggering
# =============================================================================

class TestGroundingDealBreaker:
    """Tests for deal-breaker behavior in evidence grounding."""

    @pytest.mark.asyncio
    async def test_deal_breaker_stops_evaluation(
        self, qa_engine, grounding_config_deal_breaker, sample_semantic_layer,
        mock_grounding_result_failed
    ):
        """
        Given: Grounding fails with on_flag='deal_breaker'
        When: evaluate_all_layers_with_progress is called
        Then: Evaluation stops and returns iteration_stop result
        """
        # Configure failed result with deal_breaker action
        mock_grounding_result_failed.triggered_action = "deal_breaker"

        qa_engine.grounding_engine.run_grounding_check = AsyncMock(
            return_value=mock_grounding_result_failed
        )

        # Mock semantic evaluation (should NOT be called)
        qa_engine._evaluate_single_layer = AsyncMock()

        results = await qa_engine.evaluate_all_layers_with_progress(
            content="Test content",
            layers=[sample_semantic_layer],  # Has semantic layer
            qa_models=["gpt-4o"],
            evidence_grounding_config=grounding_config_deal_breaker,
            context_for_grounding="Test context",
        )

        # Verify it's an iteration_stop result
        assert "qa_results" in results
        assert "summary" in results
        assert results["summary"]["has_deal_breakers"] is True

        # Verify grounding result is in qa_results
        assert "Evidence Grounding" in results["qa_results"]

        # Verify semantic layer was NOT evaluated (grounding ran first with order=0)
        # The semantic layer has order=1, so it would run after grounding
        # Since grounding triggered deal_breaker, semantic should not execute
        qa_engine._evaluate_single_layer.assert_not_called()

    @pytest.mark.asyncio
    async def test_deal_breaker_qa_eval_has_flag(
        self, qa_engine, grounding_config_deal_breaker, mock_grounding_result_failed
    ):
        """
        Given: Grounding fails with on_flag='deal_breaker'
        When: QAEvaluation is created
        Then: deal_breaker=True and score=0.0
        """
        mock_grounding_result_failed.triggered_action = "deal_breaker"

        qa_engine.grounding_engine.run_grounding_check = AsyncMock(
            return_value=mock_grounding_result_failed
        )

        results = await qa_engine.evaluate_all_layers_with_progress(
            content="Test content",
            layers=[],
            qa_models=[],
            evidence_grounding_config=grounding_config_deal_breaker,
            context_for_grounding="Test context",
        )

        qa_eval = results["qa_results"]["Evidence Grounding"]["evidence_grounding_logprobs"]
        assert qa_eval.deal_breaker is True
        assert qa_eval.score == 0.0
        # Verify deal_breaker_reason mentions claims/grounding
        reason_lower = qa_eval.deal_breaker_reason.lower()
        assert "claims" in reason_lower or "grounding" in reason_lower


# =============================================================================
# Tests for Regeneration Flow
# =============================================================================

class TestGroundingRegeneration:
    """Tests for regeneration behavior in evidence grounding."""

    @pytest.mark.asyncio
    async def test_regenerate_stops_evaluation(
        self, qa_engine, grounding_config_regenerate, sample_semantic_layer,
        mock_grounding_result_failed
    ):
        """
        Given: Grounding fails with on_flag='regenerate'
        When: evaluate_all_layers_with_progress is called
        Then: Evaluation stops and returns iteration_stop result
        """
        mock_grounding_result_failed.triggered_action = "regenerate"
        mock_grounding_result_failed.flagged_claims = 1  # threshold is 1

        qa_engine.grounding_engine.run_grounding_check = AsyncMock(
            return_value=mock_grounding_result_failed
        )

        results = await qa_engine.evaluate_all_layers_with_progress(
            content="Test content",
            layers=[sample_semantic_layer],
            qa_models=["gpt-4o"],
            evidence_grounding_config=grounding_config_regenerate,
            context_for_grounding="Test context",
        )

        # Verify it triggers iteration stop
        assert "qa_results" in results
        assert results["summary"]["has_deal_breakers"] is True

    @pytest.mark.asyncio
    async def test_regenerate_order_is_zero(self, grounding_config_regenerate):
        """
        Given: Config with on_flag='regenerate'
        When: Order is calculated
        Then: Order is 0 (fail-fast)
        """
        assert get_effective_order(grounding_config_regenerate) == 0


# =============================================================================
# Tests for Execution Order
# =============================================================================

class TestGroundingExecutionOrder:
    """Tests for correct execution order of grounding vs semantic layers."""

    @pytest.mark.asyncio
    async def test_deal_breaker_runs_before_semantic_blocks_semantic(
        self, qa_engine, mock_grounding_result_failed
    ):
        """
        Given: Grounding with on_flag='deal_breaker' (order=0) fails, and semantic layer (order=1)
        When: evaluate_all_layers_with_progress is called
        Then: Grounding runs first, fails, and semantic layer is NOT evaluated
        """
        mock_grounding_result_failed.triggered_action = "deal_breaker"

        grounding_called = False
        async def track_grounding(*args, **kwargs):
            nonlocal grounding_called
            grounding_called = True
            return mock_grounding_result_failed

        qa_engine.grounding_engine.run_grounding_check = track_grounding

        config = EvidenceGroundingConfig(
            enabled=True,
            max_flagged_claims=2,
            on_flag="deal_breaker",  # order=0
        )

        semantic_layer = QALayer(
            name="Semantic",
            description="Test",
            criteria="Test",
            min_score=7.0,
            order=1,  # After grounding (order=0)
        )

        results = await qa_engine.evaluate_all_layers_with_progress(
            content="Test content",
            layers=[semantic_layer],
            qa_models=["gpt-4o"],
            evidence_grounding_config=config,
            context_for_grounding="Test context",
        )

        # Verify grounding ran and stopped evaluation
        assert grounding_called is True
        # Semantic layer should NOT be in results (grounding stopped evaluation)
        assert "Semantic" not in results.get("qa_results", results)
        # Grounding result should be present
        assert "Evidence Grounding" in results.get("qa_results", results)

    @pytest.mark.asyncio
    async def test_warn_grounding_executes_with_no_semantic_layers(
        self, qa_engine, grounding_config_warn, mock_grounding_result_passed
    ):
        """
        Given: Grounding with on_flag='warn' (order=999) and no semantic layers
        When: evaluate_all_layers_with_progress is called
        Then: Grounding executes successfully
        """
        qa_engine.grounding_engine.run_grounding_check = AsyncMock(
            return_value=mock_grounding_result_passed
        )

        results = await qa_engine.evaluate_all_layers_with_progress(
            content="Test content",
            layers=[],  # No semantic layers
            qa_models=[],
            evidence_grounding_config=grounding_config_warn,
            context_for_grounding="Test context",
        )

        # Grounding should have executed
        assert "Evidence Grounding" in results
        qa_engine.grounding_engine.run_grounding_check.assert_called_once()

    @pytest.mark.asyncio
    async def test_execution_plan_order_is_correct(self, qa_engine, mock_grounding_result_passed):
        """
        Given: Grounding with explicit order=5 and semantic layers at order=1 and order=10
        When: Execution plan is built
        Then: Items are sorted correctly by order

        This test verifies the ordering logic by checking that grounding
        would execute between the two semantic layers based on order values.
        """
        # Create layers
        layer_1 = QALayer(name="Layer1", description="Test", criteria="Test", min_score=7.0, order=1)
        layer_10 = QALayer(name="Layer10", description="Test", criteria="Test", min_score=7.0, order=10)

        config = EvidenceGroundingConfig(
            enabled=True,
            on_flag="warn",  # Would auto to 999
            order=5,  # Explicit: between 1 and 10
        )

        # Test the order calculation
        assert get_effective_order(config) == 5
        assert layer_1.order == 1
        assert layer_10.order == 10

        # Verify the expected sorting order
        items = [
            {"type": "layer", "order": layer_1.order, "name": "layer_1"},
            {"type": "grounding", "order": get_effective_order(config), "name": "grounding"},
            {"type": "layer", "order": layer_10.order, "name": "layer_10"},
        ]
        sorted_items = sorted(items, key=lambda x: x["order"])

        assert sorted_items[0]["name"] == "layer_1"
        assert sorted_items[1]["name"] == "grounding"
        assert sorted_items[2]["name"] == "layer_10"

    @pytest.mark.asyncio
    async def test_grounding_order_with_multiple_configs(self):
        """
        Given: Different grounding configurations
        When: Effective order is calculated
        Then: Order respects on_flag and explicit overrides
        """
        # Test various configurations
        test_cases = [
            (EvidenceGroundingConfig(enabled=True, on_flag="deal_breaker"), 0),
            (EvidenceGroundingConfig(enabled=True, on_flag="regenerate"), 0),
            (EvidenceGroundingConfig(enabled=True, on_flag="warn"), 999),
            (EvidenceGroundingConfig(enabled=True, on_flag="deal_breaker", order=50), 50),
            (EvidenceGroundingConfig(enabled=True, on_flag="warn", order=1), 1),
            (EvidenceGroundingConfig(enabled=True, on_flag="regenerate", order=100), 100),
        ]

        for config, expected_order in test_cases:
            assert get_effective_order(config) == expected_order, \
                f"Config {config.on_flag} with order={config.order} should have effective order {expected_order}"


# =============================================================================
# Tests for Comprehensive Evaluation Integration
# =============================================================================

class TestComprehensiveEvaluationIntegration:
    """Tests for evaluate_content_comprehensive with evidence grounding."""

    @pytest.mark.asyncio
    async def test_comprehensive_includes_grounding_result(
        self, qa_engine, grounding_config_warn, mock_grounding_result_passed
    ):
        """
        Given: Grounding enabled
        When: evaluate_content_comprehensive is called
        Then: Result includes evidence_grounding field
        """
        qa_engine.grounding_engine.run_grounding_check = AsyncMock(
            return_value=mock_grounding_result_passed
        )

        result = await qa_engine.evaluate_content_comprehensive(
            content="Test content about Marie Curie.",
            layers=[],
            qa_models=[],
            evidence_grounding_config=grounding_config_warn,
            context_for_grounding="Marie Curie was born in Warsaw.",
        )

        assert "evidence_grounding" in result
        grounding_data = result["evidence_grounding"]
        assert grounding_data is not None
        assert grounding_data["passed"] is True
        assert grounding_data["flagged_claims"] == 1

    @pytest.mark.asyncio
    async def test_comprehensive_force_iteration_still_includes_grounding_result(
        self, qa_engine, grounding_config_deal_breaker, mock_grounding_result_failed
    ):
        """
        Given: Grounding triggers an early stop
        When: evaluate_content_comprehensive short-circuits
        Then: The returned payload still exposes evidence_grounding details
        """
        mock_grounding_result_failed.triggered_action = "deal_breaker"
        qa_engine.grounding_engine.run_grounding_check = AsyncMock(
            return_value=mock_grounding_result_failed
        )

        result = await qa_engine.evaluate_content_comprehensive(
            content="Test content about Marie Curie.",
            layers=[],
            qa_models=[],
            evidence_grounding_config=grounding_config_deal_breaker,
            context_for_grounding="Marie Curie was born in Warsaw.",
        )

        assert result["summary"]["force_iteration"] is True
        assert result["evidence_grounding"] is not None
        assert result["evidence_grounding"]["passed"] is False
        assert result["evidence_grounding"]["triggered_action"] == "deal_breaker"

    @pytest.mark.asyncio
    async def test_comprehensive_no_grounding_when_disabled(self, qa_engine):
        """
        Given: Grounding not enabled
        When: evaluate_content_comprehensive is called
        Then: evidence_grounding is None
        """
        result = await qa_engine.evaluate_content_comprehensive(
            content="Test content",
            layers=[],
            qa_models=[],
            evidence_grounding_config=None,  # Not enabled
        )

        assert result.get("evidence_grounding") is None


# =============================================================================
# Tests for Progress and Stream Callbacks
# =============================================================================

class TestGroundingCallbacks:
    """Tests for callback invocation during grounding."""

    @pytest.mark.asyncio
    async def test_progress_callback_invoked(
        self, qa_engine, grounding_config_warn, mock_grounding_result_passed
    ):
        """
        Given: Progress callback provided
        When: Grounding executes
        Then: Progress callback is invoked with grounding messages
        """
        progress_messages = []

        async def progress_callback(msg):
            progress_messages.append(msg)

        qa_engine.grounding_engine.run_grounding_check = AsyncMock(
            return_value=mock_grounding_result_passed
        )

        await qa_engine.evaluate_all_layers_with_progress(
            content="Test content",
            layers=[],
            qa_models=[],
            evidence_grounding_config=grounding_config_warn,
            context_for_grounding="Test context",
            progress_callback=progress_callback,
        )

        # Should have progress messages about grounding
        assert len(progress_messages) > 0
        assert any("grounding" in msg.lower() for msg in progress_messages)

    @pytest.mark.asyncio
    async def test_stream_callback_passed_to_engine(
        self, qa_engine, grounding_config_warn, mock_grounding_result_passed
    ):
        """
        Given: Stream callback provided
        When: Grounding executes
        Then: Stream callback is passed to grounding engine
        """
        stream_events = []

        async def stream_callback(event):
            stream_events.append(event)

        qa_engine.grounding_engine.run_grounding_check = AsyncMock(
            return_value=mock_grounding_result_passed
        )

        await qa_engine.evaluate_all_layers_with_progress(
            content="Test content",
            layers=[],
            qa_models=[],
            evidence_grounding_config=grounding_config_warn,
            context_for_grounding="Test context",
            stream_callback=stream_callback,
        )

        # Verify run_grounding_check was called with stream_callback
        call_kwargs = qa_engine.grounding_engine.run_grounding_check.call_args[1]
        assert "stream_callback" in call_kwargs


# =============================================================================
# Tests for Edge Cases
# =============================================================================

class TestGroundingEdgeCases:
    """Tests for edge cases in grounding integration."""

    @pytest.mark.asyncio
    async def test_no_claims_extracted_passes(self, qa_engine, grounding_config_deal_breaker):
        """
        Given: No claims extracted from content
        When: Grounding executes
        Then: Passes (nothing to fail)
        """
        empty_result = EvidenceGroundingResult(
            enabled=True,
            model_used="gpt-4o-mini",
            total_claims_extracted=0,
            claims_after_filter=0,
            claims_verified=0,
            claims=[],
            flagged_claims=0,
            max_budget_gap=0.0,
            passed=True,
            triggered_action=None,
            verification_time_ms=100.0,
            tokens_used=50,
        )

        qa_engine.grounding_engine.run_grounding_check = AsyncMock(return_value=empty_result)

        results = await qa_engine.evaluate_all_layers_with_progress(
            content="Simple decorative text.",
            layers=[],
            qa_models=[],
            evidence_grounding_config=grounding_config_deal_breaker,
            context_for_grounding="Some context",
        )

        qa_eval = results["Evidence Grounding"]["evidence_grounding_logprobs"]
        assert qa_eval.score == 10.0
        assert qa_eval.deal_breaker is False

    @pytest.mark.asyncio
    async def test_grounding_error_conservative_pass(
        self, qa_engine, grounding_config_deal_breaker
    ):
        """
        Given: Grounding engine raises an error
        When: Error is caught and conservative result returned
        Then: Grounding passes (don't block on errors)
        """
        # Create a result that simulates error recovery
        error_result = EvidenceGroundingResult(
            enabled=True,
            model_used="gpt-4o-mini",
            total_claims_extracted=0,
            claims_after_filter=0,
            claims_verified=0,
            claims=[],
            flagged_claims=0,
            max_budget_gap=0.0,
            passed=True,  # Conservative pass
            triggered_action=None,
            verification_time_ms=50.0,
            tokens_used=0,
        )

        qa_engine.grounding_engine.run_grounding_check = AsyncMock(return_value=error_result)

        results = await qa_engine.evaluate_all_layers_with_progress(
            content="Test content",
            layers=[],
            qa_models=[],
            evidence_grounding_config=grounding_config_deal_breaker,
            context_for_grounding="Test context",
        )

        qa_eval = results["Evidence Grounding"]["evidence_grounding_logprobs"]
        assert qa_eval.deal_breaker is False

    @pytest.mark.asyncio
    async def test_grounding_disabled_not_executed(self, qa_engine):
        """
        Given: Grounding config with enabled=False
        When: evaluate_all_layers_with_progress is called
        Then: Grounding is not executed
        """
        config = EvidenceGroundingConfig(enabled=False)

        qa_engine.grounding_engine.run_grounding_check = AsyncMock()

        results = await qa_engine.evaluate_all_layers_with_progress(
            content="Test content",
            layers=[],
            qa_models=[],
            evidence_grounding_config=config,
            context_for_grounding="Test context",
        )

        # Grounding should not be in results
        assert "Evidence Grounding" not in results
        qa_engine.grounding_engine.run_grounding_check.assert_not_called()


# =============================================================================
# Tests for Context Building
# =============================================================================

class TestGroundingContext:
    """Tests for context handling in grounding."""

    @pytest.mark.asyncio
    async def test_uses_provided_context(
        self, qa_engine, grounding_config_warn, mock_grounding_result_passed
    ):
        """
        Given: Explicit context_for_grounding provided
        When: Grounding executes
        Then: Uses the provided context
        """
        qa_engine.grounding_engine.run_grounding_check = AsyncMock(
            return_value=mock_grounding_result_passed
        )

        await qa_engine.evaluate_all_layers_with_progress(
            content="Test content",
            layers=[],
            qa_models=[],
            evidence_grounding_config=grounding_config_warn,
            context_for_grounding="Explicit test context",
        )

        call_kwargs = qa_engine.grounding_engine.run_grounding_check.call_args[1]
        assert call_kwargs["context"] == "Explicit test context"

    @pytest.mark.asyncio
    async def test_falls_back_to_prompt_when_no_context(
        self, qa_engine, grounding_config_warn, mock_grounding_result_passed
    ):
        """
        Given: No context_for_grounding but original_request has prompt
        When: Grounding executes
        Then: Falls back to prompt from original_request
        """
        qa_engine.grounding_engine.run_grounding_check = AsyncMock(
            return_value=mock_grounding_result_passed
        )

        original_request = MagicMock()
        original_request.prompt = "Prompt from request"

        await qa_engine.evaluate_all_layers_with_progress(
            content="Test content",
            layers=[],
            qa_models=[],
            evidence_grounding_config=grounding_config_warn,
            context_for_grounding=None,  # Not provided
            original_request=original_request,
        )

        call_kwargs = qa_engine.grounding_engine.run_grounding_check.call_args[1]
        assert call_kwargs["context"] == "Prompt from request"
