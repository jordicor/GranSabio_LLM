"""
Tests for evidence_grounding/grounding_engine.py - Grounding Engine (Phase 5).

Phase 5 of Strawberry Integration: Orchestrates the complete evidence grounding
verification pipeline and integrates with the QA engine.

Tests use mocked AI responses to avoid API calls.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from evidence_grounding.grounding_engine import (
    GroundingEngine,
    get_effective_order,
    run_grounding_check,
)
from models import (
    ClaimBudgetResult,
    EvidenceGroundingConfig,
    EvidenceGroundingResult,
    ExtractedClaim,
)

# =============================================================================
# Tests for get_effective_order
# =============================================================================

class TestGetEffectiveOrder:
    """Tests for automatic order calculation based on on_flag."""

    def test_explicit_order_returns_value(self):
        """
        Given: Config with explicit order set
        When: get_effective_order is called
        Then: Returns the explicit order value
        """
        config = EvidenceGroundingConfig(enabled=True, order=5)
        assert get_effective_order(config) == 5

    def test_deal_breaker_returns_zero(self):
        """
        Given: Config with on_flag="deal_breaker" and no explicit order
        When: get_effective_order is called
        Then: Returns 0 (fail-fast)
        """
        config = EvidenceGroundingConfig(enabled=True, on_flag="deal_breaker")
        assert get_effective_order(config) == 0

    def test_regenerate_returns_zero(self):
        """
        Given: Config with on_flag="regenerate" and no explicit order
        When: get_effective_order is called
        Then: Returns 0 (fail-fast)
        """
        config = EvidenceGroundingConfig(enabled=True, on_flag="regenerate")
        assert get_effective_order(config) == 0

    def test_warn_returns_999(self):
        """
        Given: Config with on_flag="warn" and no explicit order
        When: get_effective_order is called
        Then: Returns 999 (verification-only at end)
        """
        config = EvidenceGroundingConfig(enabled=True, on_flag="warn")
        assert get_effective_order(config) == 999

    def test_explicit_order_overrides_auto(self):
        """
        Given: Config with on_flag="deal_breaker" but explicit order=10
        When: get_effective_order is called
        Then: Returns explicit order (10), not auto (0)
        """
        config = EvidenceGroundingConfig(enabled=True, on_flag="deal_breaker", order=10)
        assert get_effective_order(config) == 10

    def test_order_zero_explicit(self):
        """
        Given: Config with explicit order=0
        When: get_effective_order is called
        Then: Returns 0 (explicit value, not auto-calculated)
        """
        config = EvidenceGroundingConfig(enabled=True, on_flag="warn", order=0)
        assert get_effective_order(config) == 0


# =============================================================================
# Tests for GroundingEngine initialization
# =============================================================================

class TestGroundingEngineInit:
    """Tests for GroundingEngine initialization."""

    def test_init_with_ai_service(self):
        """
        Given: A mock AI service
        When: GroundingEngine is initialized
        Then: All components are created with the service
        """
        mock_ai_service = MagicMock()
        engine = GroundingEngine(mock_ai_service)

        assert engine.ai_service == mock_ai_service
        assert engine.claim_extractor is not None
        assert engine.evidence_matcher is not None
        assert engine.budget_scorer is not None


# =============================================================================
# Tests for GroundingEngine.run_grounding_check
# =============================================================================

class TestGroundingEngineRunCheck:
    """Tests for the main grounding check orchestration."""

    @pytest.fixture
    def mock_ai_service(self):
        """Create a mock AI service."""
        return MagicMock()

    @pytest.fixture
    def basic_config(self):
        """Basic grounding configuration."""
        return EvidenceGroundingConfig(
            enabled=True,
            max_claims=10,
            filter_trivial=True,
            min_claim_importance=0.5,
            budget_gap_threshold=0.5,
            max_flagged_claims=2,
            on_flag="warn",
        )

    @pytest.fixture
    def sample_claims(self):
        """Sample extracted claims."""
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
                claim="She won two Nobel Prizes",
                kind="factual",
                importance=0.8,
                cited_spans=["S1"],
                source_text="She won two Nobel Prizes."
            ),
        ]

    @pytest.fixture
    def sample_budget_results(self):
        """Sample budget scoring results."""
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
                claim="She won two Nobel Prizes",
                cited_spans=["S1"],
                posterior_yes=0.90,
                prior_yes=0.85,
                required_bits=0.8,
                observed_bits=0.05,
                budget_gap=0.75,
                flagged=True,
                confidence_delta=0.05,
            ),
        ]

    @pytest.mark.asyncio
    async def test_successful_grounding_check_passed(
        self, mock_ai_service, basic_config, sample_claims, sample_budget_results
    ):
        """
        Given: Content with well-grounded claims (1 flagged < threshold 2)
        When: run_grounding_check is called
        Then: Returns passed=True result
        """
        engine = GroundingEngine(mock_ai_service)

        # Mock claim extractor
        engine.claim_extractor.extract_claims = AsyncMock(return_value=sample_claims)

        # Mock budget scorer
        engine.budget_scorer.score_claims = AsyncMock(return_value=sample_budget_results)

        result = await engine.run_grounding_check(
            content="Marie Curie was born in Warsaw in 1867. She won two Nobel Prizes.",
            context="Historical facts about Marie Curie...",
            grounding_config=basic_config,
        )

        assert isinstance(result, EvidenceGroundingResult)
        assert result.enabled is True
        assert result.passed is True  # 1 flagged < threshold 2
        assert result.flagged_claims == 1
        assert result.triggered_action is None
        assert result.claims_verified == 2

    @pytest.mark.asyncio
    async def test_grounding_check_failed_threshold_exceeded(
        self, mock_ai_service, sample_claims
    ):
        """
        Given: Content with too many flagged claims (>= threshold)
        When: run_grounding_check is called
        Then: Returns passed=False with triggered_action
        """
        config = EvidenceGroundingConfig(
            enabled=True,
            max_flagged_claims=1,  # Threshold = 1, so 1 flagged >= 1 triggers
            on_flag="deal_breaker",
        )

        # Both claims flagged
        budget_results = [
            ClaimBudgetResult(
                idx=0, claim="Claim 1", cited_spans=["S0"],
                posterior_yes=0.9, prior_yes=0.85,
                required_bits=0.8, observed_bits=0.1, budget_gap=0.7,
                flagged=True, confidence_delta=0.05,
            ),
            ClaimBudgetResult(
                idx=1, claim="Claim 2", cited_spans=["S1"],
                posterior_yes=0.88, prior_yes=0.82,
                required_bits=0.9, observed_bits=0.08, budget_gap=0.82,
                flagged=True, confidence_delta=0.06,
            ),
        ]

        engine = GroundingEngine(mock_ai_service)
        engine.claim_extractor.extract_claims = AsyncMock(return_value=sample_claims)
        engine.budget_scorer.score_claims = AsyncMock(return_value=budget_results)

        result = await engine.run_grounding_check(
            content="Test content",
            context="Test context",
            grounding_config=config,
        )

        assert result.passed is False
        assert result.flagged_claims == 2
        assert result.triggered_action == "deal_breaker"

    @pytest.mark.asyncio
    async def test_grounding_check_no_claims_passes(self, mock_ai_service, basic_config):
        """
        Given: Content with no verifiable claims extracted
        When: run_grounding_check is called
        Then: Returns passed=True (nothing to fail)
        """
        engine = GroundingEngine(mock_ai_service)
        engine.claim_extractor.extract_claims = AsyncMock(return_value=[])

        result = await engine.run_grounding_check(
            content="Simple decorative text without facts.",
            context="Some context",
            grounding_config=basic_config,
        )

        assert result.passed is True
        assert result.claims_verified == 0
        assert result.flagged_claims == 0

    @pytest.mark.asyncio
    async def test_grounding_check_extraction_error_conservative_pass(
        self, mock_ai_service, basic_config
    ):
        """
        Given: Claim extraction fails with an exception
        When: run_grounding_check is called
        Then: Returns conservative passed=True (don't block on errors)
        """
        engine = GroundingEngine(mock_ai_service)
        engine.claim_extractor.extract_claims = AsyncMock(
            side_effect=Exception("API Error")
        )

        result = await engine.run_grounding_check(
            content="Test content",
            context="Test context",
            grounding_config=basic_config,
        )

        assert result.passed is True  # Conservative pass
        assert result.claims_verified == 0

    @pytest.mark.asyncio
    async def test_grounding_check_scoring_error_conservative_pass(
        self, mock_ai_service, basic_config, sample_claims
    ):
        """
        Given: Budget scoring fails with an exception
        When: run_grounding_check is called
        Then: Returns conservative passed=True
        """
        engine = GroundingEngine(mock_ai_service)
        engine.claim_extractor.extract_claims = AsyncMock(return_value=sample_claims)
        engine.budget_scorer.score_claims = AsyncMock(
            side_effect=Exception("Logprob API Error")
        )

        result = await engine.run_grounding_check(
            content="Test content",
            context="Test context",
            grounding_config=basic_config,
        )

        assert result.passed is True  # Conservative pass

    @pytest.mark.asyncio
    async def test_grounding_check_with_progress_callback(
        self, mock_ai_service, basic_config, sample_claims, sample_budget_results
    ):
        """
        Given: Progress callback provided
        When: run_grounding_check is called
        Then: Callback is invoked during execution
        """
        progress_messages = []

        async def progress_callback(msg):
            progress_messages.append(msg)

        engine = GroundingEngine(mock_ai_service)
        engine.claim_extractor.extract_claims = AsyncMock(return_value=sample_claims)
        engine.budget_scorer.score_claims = AsyncMock(return_value=sample_budget_results)

        await engine.run_grounding_check(
            content="Test content",
            context="Test context",
            grounding_config=basic_config,
            progress_callback=progress_callback,
        )

        assert len(progress_messages) > 0
        assert any("claim" in msg.lower() for msg in progress_messages)

    @pytest.mark.asyncio
    async def test_grounding_check_with_stream_callback(
        self, mock_ai_service, basic_config, sample_claims, sample_budget_results
    ):
        """
        Given: Stream callback provided
        When: run_grounding_check is called
        Then: Streaming events are emitted
        """
        stream_events = []

        async def stream_callback(event):
            stream_events.append(event)

        engine = GroundingEngine(mock_ai_service)
        engine.claim_extractor.extract_claims = AsyncMock(return_value=sample_claims)
        engine.budget_scorer.score_claims = AsyncMock(return_value=sample_budget_results)

        await engine.run_grounding_check(
            content="Test content",
            context="Test context",
            grounding_config=basic_config,
            stream_callback=stream_callback,
        )

        assert len(stream_events) > 0
        # Should have grounding_phase events
        phase_types = [e.get("type") for e in stream_events]
        assert "grounding_phase" in phase_types

    @pytest.mark.asyncio
    async def test_max_budget_gap_calculated_correctly(
        self, mock_ai_service, basic_config, sample_claims
    ):
        """
        Given: Multiple claims with different budget gaps
        When: run_grounding_check is called
        Then: max_budget_gap is the maximum across all claims
        """
        budget_results = [
            ClaimBudgetResult(
                idx=0, claim="Claim 1", cited_spans=["S0"],
                posterior_yes=0.9, prior_yes=0.1,
                required_bits=2.0, observed_bits=2.5, budget_gap=-0.5,
                flagged=False, confidence_delta=0.8,
            ),
            ClaimBudgetResult(
                idx=1, claim="Claim 2", cited_spans=["S1"],
                posterior_yes=0.85, prior_yes=0.80,
                required_bits=1.0, observed_bits=0.1, budget_gap=0.9,
                flagged=True, confidence_delta=0.05,
            ),
            ClaimBudgetResult(
                idx=2, claim="Claim 3", cited_spans=["S2"],
                posterior_yes=0.75, prior_yes=0.70,
                required_bits=0.8, observed_bits=0.5, budget_gap=0.3,
                flagged=False, confidence_delta=0.05,
            ),
        ]

        claims = sample_claims + [
            ExtractedClaim(
                idx=2, claim="Third claim", kind="factual",
                importance=0.7, cited_spans=["S2"], source_text="Third claim."
            )
        ]

        engine = GroundingEngine(mock_ai_service)
        engine.claim_extractor.extract_claims = AsyncMock(return_value=claims)
        engine.budget_scorer.score_claims = AsyncMock(return_value=budget_results)

        result = await engine.run_grounding_check(
            content="Test content",
            context="Test context",
            grounding_config=basic_config,
        )

        assert result.max_budget_gap == 0.9  # Maximum across all claims


# =============================================================================
# Tests for convenience function run_grounding_check
# =============================================================================

class TestRunGroundingCheckConvenience:
    """Tests for the standalone convenience function."""

    @pytest.mark.asyncio
    async def test_convenience_function_creates_engine(self):
        """
        Given: No explicit engine
        When: run_grounding_check convenience function is called
        Then: Creates engine internally and returns result
        """
        # Patch at the point where ai_service is imported in run_grounding_check
        with patch('ai_service.get_ai_service') as mock_get_service:
            mock_service = MagicMock()
            mock_get_service.return_value = mock_service

            with patch.object(GroundingEngine, 'run_grounding_check') as mock_run:
                mock_run.return_value = EvidenceGroundingResult(
                    enabled=True,
                    model_used="gpt-5-nano",
                    total_claims_extracted=0,
                    claims_after_filter=0,
                    claims_verified=0,
                    claims=[],
                    flagged_claims=0,
                    max_budget_gap=0.0,
                    passed=True,
                    triggered_action=None,
                    verification_time_ms=100.0,
                    tokens_used=0,
                )

                result = await run_grounding_check(
                    content="Test",
                    context="Context",
                )

                assert result.passed is True
                mock_get_service.assert_called_once()


# =============================================================================
# Tests for EvidenceGroundingConfig order field
# =============================================================================

class TestEvidenceGroundingConfigOrder:
    """Tests for the order field in EvidenceGroundingConfig."""

    def test_order_default_is_none(self):
        """
        Given: Default EvidenceGroundingConfig
        When: Accessing order field
        Then: Returns None (auto-calculated)
        """
        config = EvidenceGroundingConfig(enabled=True)
        assert config.order is None

    def test_order_can_be_set_explicitly(self):
        """
        Given: Config with explicit order
        When: Creating config
        Then: Order is set correctly
        """
        config = EvidenceGroundingConfig(enabled=True, order=5)
        assert config.order == 5

    def test_order_can_be_zero(self):
        """
        Given: Config with order=0
        When: Creating config
        Then: Order is 0 (not None)
        """
        config = EvidenceGroundingConfig(enabled=True, order=0)
        assert config.order == 0

    def test_order_negative_allowed(self):
        """
        Given: Config with negative order
        When: Creating config
        Then: Negative order is allowed (run before everything)
        """
        config = EvidenceGroundingConfig(enabled=True, order=-1)
        assert config.order == -1


# =============================================================================
# Tests for result structure
# =============================================================================

class TestEvidenceGroundingResultStructure:
    """Tests for the EvidenceGroundingResult model."""

    def test_result_contains_all_required_fields(self):
        """
        Given: A complete EvidenceGroundingResult
        When: Accessing fields
        Then: All fields are present and correctly typed
        """
        result = EvidenceGroundingResult(
            enabled=True,
            model_used="gpt-5-nano",
            total_claims_extracted=10,
            claims_after_filter=8,
            claims_verified=8,
            claims=[],
            flagged_claims=2,
            max_budget_gap=0.75,
            passed=True,
            triggered_action=None,
            verification_time_ms=1500.5,
            tokens_used=500,
        )

        assert result.enabled is True
        assert result.model_used == "gpt-5-nano"
        assert result.total_claims_extracted == 10
        assert result.claims_after_filter == 8
        assert result.claims_verified == 8
        assert result.flagged_claims == 2
        assert result.max_budget_gap == 0.75
        assert result.passed is True
        assert result.triggered_action is None
        assert result.verification_time_ms == 1500.5
        assert result.tokens_used == 500

    def test_result_model_dump_serializable(self):
        """
        Given: An EvidenceGroundingResult
        When: Calling model_dump()
        Then: Returns serializable dictionary
        """
        result = EvidenceGroundingResult(
            enabled=True,
            model_used="test-model",
            total_claims_extracted=5,
            claims_after_filter=3,
            claims_verified=3,
            claims=[
                ClaimBudgetResult(
                    idx=0, claim="Test claim", cited_spans=["S0"],
                    posterior_yes=0.9, prior_yes=0.5,
                    required_bits=1.0, observed_bits=0.8, budget_gap=0.2,
                    flagged=False, confidence_delta=0.4,
                )
            ],
            flagged_claims=0,
            max_budget_gap=0.2,
            passed=True,
            triggered_action=None,
            verification_time_ms=500.0,
            tokens_used=100,
        )

        data = result.model_dump()
        assert isinstance(data, dict)
        assert data["enabled"] is True
        assert len(data["claims"]) == 1
        assert data["claims"][0]["claim"] == "Test claim"
