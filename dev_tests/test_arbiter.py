"""
Test suite for Arbiter: Intelligent Conflict Resolver
=====================================================

This file contains tests for the Arbiter system that resolves conflicts
between edits proposed by different QA evaluators.

Tests are organized in phases matching the implementation plan:
- Phase 1: Data models and structure (this file)
- Phase 2: Conflict detection and resolution
- Phase 3: Integration with generation processor
- Phase 4: History injection and QA integration

Run tests with:
    python -m pytest dev_tests/test_arbiter.py -v
"""

import pytest
from typing import List

# Phase 1 imports - data models
from arbiter import (
    ConflictType,
    ArbiterDecision,
    EditDistribution,
    ProposedEdit,
    ConflictInfo,
    ArbiterEditDecision,
    EditRoundRecord,
    LayerEditHistory,
    ArbiterResult,
    ArbiterContext,
    Arbiter,
    _get_paragraph_key_for_history,
)


# =============================================================================
# PHASE 1 TESTS: Data Models and Structure
# =============================================================================


class TestConflictTypeEnum:
    """Tests for ConflictType enum."""

    def test_enum_values_exist(self):
        """Verify all expected conflict types are defined."""
        assert ConflictType.OPPOSITE_OPERATIONS == "opposite_operations"
        assert ConflictType.OPPOSITE_DIRECTIONS == "opposite_directions"
        assert ConflictType.SEVERITY_MISMATCH == "severity_mismatch"
        assert ConflictType.SEMANTIC_REDUNDANCY == "semantic_redundancy"
        assert ConflictType.CYCLE_DETECTED == "cycle_detected"

    def test_enum_is_string(self):
        """Verify enum values are strings (for JSON serialization)."""
        for conflict_type in ConflictType:
            assert isinstance(conflict_type.value, str)


class TestArbiterDecisionEnum:
    """Tests for ArbiterDecision enum."""

    def test_enum_values_exist(self):
        """Verify all expected decisions are defined."""
        assert ArbiterDecision.APPLY == "apply"
        assert ArbiterDecision.DISCARD == "discard"
        assert ArbiterDecision.MERGE == "merge"


class TestLayerEditHistory:
    """Tests for LayerEditHistory dataclass."""

    def test_empty_history_format(self):
        """Empty history should return empty string."""
        history = LayerEditHistory(layer_name="test_layer")
        assert history.format_for_prompt() == ""

    def test_add_round(self):
        """Should be able to add round records."""
        history = LayerEditHistory(layer_name="test_layer")
        record = EditRoundRecord(
            round_number=1,
            proposed_edits=[],
            conflicts_detected=[],
            decisions=[]
        )
        history.add_round(record)
        assert len(history.rounds) == 1
        assert history.rounds[0].round_number == 1

    def test_discarded_edit_keys_empty(self):
        """Empty history should return empty set of keys."""
        history = LayerEditHistory(layer_name="test_layer")
        assert history.get_discarded_edit_keys() == set()

    def test_applied_edit_keys_empty(self):
        """Empty history should return empty set of applied keys."""
        history = LayerEditHistory(layer_name="test_layer")
        assert history.get_applied_edit_keys() == set()


class TestEditRoundRecord:
    """Tests for EditRoundRecord dataclass."""

    def test_edits_applied_property_empty(self):
        """Empty decisions should return empty applied list."""
        record = EditRoundRecord(
            round_number=1,
            proposed_edits=[],
            conflicts_detected=[],
            decisions=[]
        )
        assert record.edits_applied == []

    def test_edits_discarded_property_empty(self):
        """Empty decisions should return empty discarded list."""
        record = EditRoundRecord(
            round_number=1,
            proposed_edits=[],
            conflicts_detected=[],
            decisions=[]
        )
        assert record.edits_discarded == []


class TestArbiterResult:
    """Tests for ArbiterResult Pydantic model."""

    def test_default_values(self):
        """ArbiterResult should have sensible defaults."""
        result = ArbiterResult()
        assert result.edits_to_apply == []
        assert result.edits_discarded == []
        assert result.conflicts_found == 0
        assert result.conflicts_resolved == 0
        assert result.arbiter_reasoning == ""

    def test_serialization(self):
        """ArbiterResult should serialize to dict/JSON."""
        result = ArbiterResult(
            conflicts_found=2,
            conflicts_resolved=1,
            arbiter_reasoning="Test reasoning"
        )
        data = result.model_dump()
        assert data["conflicts_found"] == 2
        assert data["conflicts_resolved"] == 1
        assert data["arbiter_reasoning"] == "Test reasoning"


class TestArbiterStub:
    """Tests for Arbiter class stub (Phase 1)."""

    def test_arbiter_init(self):
        """Arbiter should initialize with ai_service."""
        # Mock ai_service for testing
        mock_ai_service = object()
        arbiter = Arbiter(
            ai_service=mock_ai_service,
            model="gpt-5-mini"
        )
        assert arbiter.ai_service is mock_ai_service
        assert arbiter.model == "gpt-5-mini"


# =============================================================================
# PHASE 2 TESTS: Mock TextEditRange for testing
# =============================================================================


class MockOperationType:
    """Mock OperationType enum for testing."""
    def __init__(self, value: str):
        self.value = value


class MockSeverityLevel:
    """Mock SeverityLevel enum for testing."""
    def __init__(self, value: str):
        self.value = value


class MockTextEditRange:
    """Mock TextEditRange for testing conflict detection."""

    def __init__(
        self,
        edit_type: str = "replace",
        issue_severity: str = "minor",
        issue_description: str = "Test issue",
        edit_instruction: str = "Test instruction",
        start_marker: str = None,
        start_word_index: int = None,
        confidence: float = 1.0
    ):
        self.edit_type = MockOperationType(edit_type)
        self.issue_severity = MockSeverityLevel(issue_severity)
        self.issue_description = issue_description
        self.edit_instruction = edit_instruction
        self.start_marker = start_marker
        self.start_word_index = start_word_index
        self.confidence = confidence


def create_proposed_edit(
    edit_type: str = "replace",
    severity: str = "minor",
    source_model: str = "gpt-4o",
    paragraph_key: str = "para_1",
    description: str = "Test issue",
    start_marker: str = None,
    confidence: float = 1.0
) -> ProposedEdit:
    """Helper to create ProposedEdit for tests."""
    edit = MockTextEditRange(
        edit_type=edit_type,
        issue_severity=severity,
        issue_description=description,
        start_marker=start_marker or f"marker_{paragraph_key}",
        confidence=confidence
    )
    return ProposedEdit(
        edit=edit,
        source_model=source_model,
        source_score=7.0,
        paragraph_key=paragraph_key
    )


# =============================================================================
# PHASE 2 TESTS: Conflict Detection
# =============================================================================


class TestConflictDetection:
    """Tests for conflict detection."""

    @pytest.fixture
    def arbiter(self):
        """Create Arbiter instance for testing."""
        mock_ai_service = object()
        return Arbiter(ai_service=mock_ai_service)

    def test_detect_opposite_operations_delete_vs_replace(self, arbiter):
        """Should detect DELETE vs REPLACE on same fragment."""
        edits = [
            create_proposed_edit(edit_type="delete", source_model="gpt-4o", paragraph_key="p1"),
            create_proposed_edit(edit_type="replace", source_model="claude", paragraph_key="p1"),
        ]
        conflict = arbiter._detect_opposite_operations(edits, "p1")
        assert conflict is not None
        assert conflict.conflict_type == ConflictType.OPPOSITE_OPERATIONS
        assert len(conflict.involved_edits) == 2

    def test_detect_opposite_operations_delete_vs_rephrase(self, arbiter):
        """Should detect DELETE vs REPHRASE on same fragment."""
        edits = [
            create_proposed_edit(edit_type="delete", source_model="gpt-4o", paragraph_key="p1"),
            create_proposed_edit(edit_type="rephrase", source_model="gemini", paragraph_key="p1"),
        ]
        conflict = arbiter._detect_opposite_operations(edits, "p1")
        assert conflict is not None
        assert conflict.conflict_type == ConflictType.OPPOSITE_OPERATIONS

    def test_detect_opposite_directions_expand_vs_condense(self, arbiter):
        """Should detect EXPAND vs CONDENSE on same paragraph."""
        edits = [
            create_proposed_edit(edit_type="expand", source_model="gpt-4o", paragraph_key="p1"),
            create_proposed_edit(edit_type="condense", source_model="claude", paragraph_key="p1"),
        ]
        conflict = arbiter._detect_opposite_directions(edits, "p1")
        assert conflict is not None
        assert conflict.conflict_type == ConflictType.OPPOSITE_DIRECTIONS
        assert "EXPAND" in conflict.description or "CONTRACT" in conflict.description

    def test_detect_severity_mismatch(self, arbiter):
        """Should detect same issue with different severities (critical vs minor)."""
        edits = [
            create_proposed_edit(edit_type="replace", severity="critical", source_model="gpt-4o", paragraph_key="p1"),
            create_proposed_edit(edit_type="replace", severity="minor", source_model="claude", paragraph_key="p1"),
        ]
        conflict = arbiter._detect_severity_mismatch(edits, "p1")
        assert conflict is not None
        assert conflict.conflict_type == ConflictType.SEVERITY_MISMATCH

    def test_no_severity_mismatch_for_similar_severities(self, arbiter):
        """Should NOT flag severity mismatch for critical vs major."""
        edits = [
            create_proposed_edit(edit_type="replace", severity="critical", source_model="gpt-4o", paragraph_key="p1"),
            create_proposed_edit(edit_type="replace", severity="major", source_model="claude", paragraph_key="p1"),
        ]
        conflict = arbiter._detect_severity_mismatch(edits, "p1")
        assert conflict is None  # Only flags critical vs minor

    def test_detect_semantic_redundancy(self, arbiter):
        """Should detect multiple edits of same type on same paragraph."""
        edits = [
            create_proposed_edit(edit_type="rephrase", source_model="gpt-4o", paragraph_key="p1"),
            create_proposed_edit(edit_type="rephrase", source_model="claude", paragraph_key="p1"),
        ]
        conflict = arbiter._detect_semantic_redundancy(edits, "p1")
        assert conflict is not None
        assert conflict.conflict_type == ConflictType.SEMANTIC_REDUNDANCY
        assert "rephrase" in conflict.description

    def test_detect_cycle(self, arbiter):
        """Should detect previously discarded edit being re-proposed."""
        # Create history with a discarded edit
        history = LayerEditHistory(layer_name="test_layer")
        discarded_edit = MockTextEditRange(start_marker="the quick brown fox")
        record = EditRoundRecord(
            round_number=1,
            proposed_edits=[],
            conflicts_detected=[],
            decisions=[ArbiterEditDecision(
                edit=discarded_edit,
                decision=ArbiterDecision.DISCARD,
                reason="Test discard",
                source_model="gpt-4o"
            )]
        )
        history.add_round(record)

        # Now propose same edit again
        new_edit = create_proposed_edit(
            source_model="claude",
            paragraph_key="p1",
            start_marker="the quick brown fox"
        )
        conflicts = arbiter._detect_cycles([new_edit], "p1", history)
        assert len(conflicts) == 1
        assert conflicts[0].conflict_type == ConflictType.CYCLE_DETECTED

    def test_no_conflict_different_paragraphs(self, arbiter):
        """Edits on different paragraphs should not conflict."""
        edits = [
            create_proposed_edit(edit_type="delete", source_model="gpt-4o", paragraph_key="p1"),
            create_proposed_edit(edit_type="delete", source_model="claude", paragraph_key="p2"),
        ]
        history = LayerEditHistory(layer_name="test")
        conflicts = arbiter._detect_conflicts(edits, history)
        # No conflicts because different paragraphs
        assert len(conflicts) == 0

    def test_detect_conflicts_finds_multiple_issues(self, arbiter):
        """Should find multiple conflict types in same edit set."""
        edits = [
            create_proposed_edit(edit_type="delete", severity="critical", source_model="gpt-4o", paragraph_key="p1"),
            create_proposed_edit(edit_type="replace", severity="minor", source_model="claude", paragraph_key="p1"),
        ]
        history = LayerEditHistory(layer_name="test")
        conflicts = arbiter._detect_conflicts(edits, history)
        # Should detect: opposite_operations AND severity_mismatch
        conflict_types = {c.conflict_type for c in conflicts}
        assert ConflictType.OPPOSITE_OPERATIONS in conflict_types
        assert ConflictType.SEVERITY_MISMATCH in conflict_types

    def test_group_edits_by_paragraph(self, arbiter):
        """Should correctly group edits by paragraph key."""
        edits = [
            create_proposed_edit(paragraph_key="p1", source_model="gpt-4o"),
            create_proposed_edit(paragraph_key="p2", source_model="claude"),
            create_proposed_edit(paragraph_key="p1", source_model="gemini"),
        ]
        groups = arbiter._group_edits_by_paragraph(edits)
        assert len(groups) == 2
        assert len(groups["p1"]) == 2
        assert len(groups["p2"]) == 1


# =============================================================================
# PHASE 2 TESTS: Arbiter Resolution (Algorithmic)
# =============================================================================


class TestArbiterResolution:
    """Tests for Arbiter algorithmic resolution."""

    @pytest.fixture
    def arbiter(self):
        """Create Arbiter instance for testing."""
        mock_ai_service = object()
        return Arbiter(ai_service=mock_ai_service)

    def test_resolve_algorithmically_higher_severity_wins(self, arbiter):
        """Should prioritize higher severity edits."""
        edits = [
            create_proposed_edit(severity="minor", source_model="claude", paragraph_key="p1", confidence=1.0),
            create_proposed_edit(severity="critical", source_model="gpt-4o", paragraph_key="p1", confidence=1.0),
        ]
        conflicts = [ConflictInfo(
            conflict_type=ConflictType.SEVERITY_MISMATCH,
            paragraph_key="p1",
            involved_edits=edits,
            description="Test"
        )]
        decisions = arbiter._resolve_algorithmically(edits, conflicts)

        # Critical severity should win
        applied = [d for d in decisions if d.decision == ArbiterDecision.APPLY]
        discarded = [d for d in decisions if d.decision == ArbiterDecision.DISCARD]

        assert len(applied) == 1
        assert len(discarded) == 1
        assert applied[0].source_model == "gpt-4o"  # Critical severity
        assert discarded[0].source_model == "claude"  # Minor severity

    def test_resolve_algorithmically_first_wins_same_severity(self, arbiter):
        """With same severity, first processed wins."""
        edits = [
            create_proposed_edit(severity="major", source_model="gpt-4o", paragraph_key="p1", confidence=0.9),
            create_proposed_edit(severity="major", source_model="claude", paragraph_key="p1", confidence=0.8),
        ]
        conflicts = []
        decisions = arbiter._resolve_algorithmically(edits, conflicts)

        applied = [d for d in decisions if d.decision == ArbiterDecision.APPLY]
        discarded = [d for d in decisions if d.decision == ArbiterDecision.DISCARD]

        assert len(applied) == 1
        assert len(discarded) == 1
        # Higher confidence should win
        assert applied[0].source_model == "gpt-4o"

    def test_resolve_no_conflict_all_apply(self, arbiter):
        """Without conflicts, all unique edits should apply."""
        edits = [
            create_proposed_edit(paragraph_key="p1", source_model="gpt-4o"),
            create_proposed_edit(paragraph_key="p2", source_model="claude"),
            create_proposed_edit(paragraph_key="p3", source_model="gemini"),
        ]
        decisions = arbiter._resolve_algorithmically(edits, [])

        applied = [d for d in decisions if d.decision == ArbiterDecision.APPLY]
        assert len(applied) == 3


# =============================================================================
# PHASE 2 TESTS: Arbitrate Method (async tests)
# =============================================================================


class TestArbitrateMethod:
    """Tests for the main arbitrate() method."""

    @pytest.fixture
    def mock_ai_service(self):
        """Create a mock AI service that returns valid responses."""
        from unittest.mock import AsyncMock, MagicMock
        service = MagicMock()
        # Return a valid JSON response approving all edits
        service.generate_content = AsyncMock(return_value='{"reasoning": "All edits verified", "decisions": [{"edit_index": 0, "decision": "apply", "reason": "Aligned with request"}, {"edit_index": 1, "decision": "apply", "reason": "Aligned with request"}], "conflicts_resolved": []}')
        return service

    @pytest.fixture
    def arbiter(self, mock_ai_service):
        """Create Arbiter instance for testing."""
        return Arbiter(ai_service=mock_ai_service)

    @pytest.fixture
    def basic_context(self):
        """Create a basic ArbiterContext for testing."""
        return ArbiterContext(
            original_prompt="Write a biography",
            content_type="biography",
            system_prompt=None,
            layer_name="Accuracy",
            layer_criteria="Check factual accuracy",
            layer_min_score=7.0,
            current_content="This is the content to evaluate.",
            content_excerpt=None,
            proposed_edits=[],
            evaluator_scores={"gpt-4o": 6.5},
            layer_history=LayerEditHistory(layer_name="Accuracy"),
            gran_sabio_model="claude-opus-4-5-20251101",
            qa_model_count=2
        )

    async def test_arbitrate_no_edits_returns_empty(self, arbiter, basic_context):
        """Empty edit list should return empty result."""
        basic_context.proposed_edits = []
        result = await arbiter.arbitrate(basic_context)

        assert result.edits_to_apply == []
        assert result.conflicts_found == 0
        assert result.distribution == "consensus"

    async def test_arbitrate_always_calls_ai(self, arbiter, mock_ai_service, basic_context):
        """Arbiter should ALWAYS call AI to verify edits, even without conflicts."""
        basic_context.proposed_edits = [
            create_proposed_edit(paragraph_key="p1", source_model="gpt-4o"),
            create_proposed_edit(paragraph_key="p2", source_model="claude"),
        ]
        result = await arbiter.arbitrate(basic_context)

        # AI should have been called
        mock_ai_service.generate_content.assert_called_once()
        assert len(result.edits_to_apply) == 2
        assert "verified" in result.arbiter_reasoning.lower() or "aligned" in result.arbiter_reasoning.lower()

    async def test_arbitrate_with_conflicts_uses_algorithmic_fallback(self, basic_context):
        """With conflicts and failing AI, should use algorithmic fallback."""
        # Use a mock that fails
        arbiter = Arbiter(ai_service=object())
        # Create conflicting edits
        basic_context.proposed_edits = [
            create_proposed_edit(
                edit_type="delete",
                severity="critical",
                source_model="gpt-4o",
                paragraph_key="p1"
            ),
            create_proposed_edit(
                edit_type="replace",
                severity="minor",
                source_model="claude",
                paragraph_key="p1"
            ),
        ]
        basic_context.qa_model_count = 2
        result = await arbiter.arbitrate(basic_context)

        # Should detect conflict
        assert result.conflicts_found > 0
        # Should have applied one, discarded one
        assert len(result.edits_to_apply) == 1
        assert len(result.edits_discarded) == 1
        # Critical severity should win
        assert result.edits_to_apply[0].issue_severity.value == "critical"

    async def test_arbitrate_result_includes_distribution(self, arbiter, basic_context):
        """Result should include distribution classification."""
        basic_context.proposed_edits = [
            create_proposed_edit(paragraph_key="p1", source_model="gpt-4o"),
        ]
        basic_context.qa_model_count = 2  # Only 1 of 2 proposed = TIE

        result = await arbiter.arbitrate(basic_context)

        assert result.distribution in ["tie", "consensus", "minority", "majority", "conflict", "single_qa"]
        assert isinstance(result.escalated_to_gran_sabio, bool)
        assert result.model_used != ""


# =============================================================================
# PHASE 2 TESTS: Distribution Classification
# =============================================================================


class TestDistributionClassification:
    """Tests for edit distribution classification."""

    @pytest.fixture
    def arbiter(self):
        """Create Arbiter instance for testing."""
        return Arbiter(ai_service=object())

    def test_single_qa_model(self, arbiter):
        """With 1 QA model, should classify as SINGLE_QA."""
        edits = [create_proposed_edit(source_model="gpt-4o")]
        distribution = arbiter._classify_distribution(edits, qa_model_count=1, conflicts=[])
        assert distribution == EditDistribution.SINGLE_QA

    def test_two_models_both_propose_different_paragraphs(self, arbiter):
        """With 2 models proposing edits for different paragraphs, should be CONSENSUS."""
        edits = [
            create_proposed_edit(source_model="gpt-4o", paragraph_key="p1"),
            create_proposed_edit(source_model="claude", paragraph_key="p2"),
        ]
        distribution = arbiter._classify_distribution(edits, qa_model_count=2, conflicts=[])
        assert distribution == EditDistribution.CONSENSUS

    def test_two_models_only_one_proposes(self, arbiter):
        """With 2 models and only 1 proposes edit, should be TIE (disagreement)."""
        edits = [create_proposed_edit(source_model="gpt-4o", paragraph_key="p1")]
        distribution = arbiter._classify_distribution(edits, qa_model_count=2, conflicts=[])
        assert distribution == EditDistribution.TIE

    def test_two_models_with_conflict(self, arbiter):
        """With 2 models and conflict, should be TIE."""
        edits = [
            create_proposed_edit(source_model="gpt-4o", paragraph_key="p1"),
            create_proposed_edit(source_model="claude", paragraph_key="p1"),
        ]
        conflicts = [ConflictInfo(
            conflict_type=ConflictType.OPPOSITE_OPERATIONS,
            paragraph_key="p1",
            involved_edits=edits,
            description="Test"
        )]
        distribution = arbiter._classify_distribution(edits, qa_model_count=2, conflicts=conflicts)
        assert distribution == EditDistribution.TIE

    def test_three_models_minority(self, arbiter):
        """With 3 models and only 1 proposes, should be MINORITY."""
        edits = [create_proposed_edit(source_model="gpt-4o", paragraph_key="p1")]
        distribution = arbiter._classify_distribution(edits, qa_model_count=3, conflicts=[])
        assert distribution == EditDistribution.MINORITY

    def test_three_models_majority(self, arbiter):
        """With 3 models and 2 propose, should be MAJORITY."""
        edits = [
            create_proposed_edit(source_model="gpt-4o", paragraph_key="p1"),
            create_proposed_edit(source_model="claude", paragraph_key="p2"),
        ]
        distribution = arbiter._classify_distribution(edits, qa_model_count=3, conflicts=[])
        assert distribution == EditDistribution.MAJORITY

    def test_three_models_consensus(self, arbiter):
        """With 3 models and all propose, should be CONSENSUS."""
        edits = [
            create_proposed_edit(source_model="gpt-4o", paragraph_key="p1"),
            create_proposed_edit(source_model="claude", paragraph_key="p2"),
            create_proposed_edit(source_model="gemini", paragraph_key="p3"),
        ]
        distribution = arbiter._classify_distribution(edits, qa_model_count=3, conflicts=[])
        assert distribution == EditDistribution.CONSENSUS

    def test_three_models_with_conflict(self, arbiter):
        """With 3 models and conflict, should be CONFLICT."""
        edits = [
            create_proposed_edit(source_model="gpt-4o", paragraph_key="p1"),
            create_proposed_edit(source_model="claude", paragraph_key="p1"),
        ]
        conflicts = [ConflictInfo(
            conflict_type=ConflictType.OPPOSITE_OPERATIONS,
            paragraph_key="p1",
            involved_edits=edits,
            description="Test"
        )]
        distribution = arbiter._classify_distribution(edits, qa_model_count=3, conflicts=conflicts)
        assert distribution == EditDistribution.CONFLICT


class TestModelEscalation:
    """Tests for model escalation logic."""

    @pytest.fixture
    def arbiter(self):
        """Create Arbiter instance for testing."""
        return Arbiter(ai_service=object())

    def test_escalate_for_minority(self, arbiter):
        """MINORITY distribution should escalate to GranSabio."""
        model, escalated = arbiter._select_model_for_distribution(
            EditDistribution.MINORITY,
            gran_sabio_model="claude-opus"
        )
        assert escalated is True
        assert model == "claude-opus"

    def test_escalate_for_conflict(self, arbiter):
        """CONFLICT distribution should escalate to GranSabio."""
        model, escalated = arbiter._select_model_for_distribution(
            EditDistribution.CONFLICT,
            gran_sabio_model="claude-opus"
        )
        assert escalated is True
        assert model == "claude-opus"

    def test_escalate_for_tie(self, arbiter):
        """TIE distribution should escalate to GranSabio."""
        model, escalated = arbiter._select_model_for_distribution(
            EditDistribution.TIE,
            gran_sabio_model="claude-opus"
        )
        assert escalated is True
        assert model == "claude-opus"

    def test_no_escalate_for_consensus(self, arbiter):
        """CONSENSUS distribution should NOT escalate."""
        model, escalated = arbiter._select_model_for_distribution(
            EditDistribution.CONSENSUS,
            gran_sabio_model="claude-opus"
        )
        assert escalated is False

    def test_no_escalate_for_majority(self, arbiter):
        """MAJORITY distribution should NOT escalate."""
        model, escalated = arbiter._select_model_for_distribution(
            EditDistribution.MAJORITY,
            gran_sabio_model="claude-opus"
        )
        assert escalated is False

    def test_no_escalate_for_single_qa(self, arbiter):
        """SINGLE_QA distribution should NOT escalate."""
        model, escalated = arbiter._select_model_for_distribution(
            EditDistribution.SINGLE_QA,
            gran_sabio_model="claude-opus"
        )
        assert escalated is False

    def test_fallback_when_no_gran_sabio_model(self, arbiter):
        """Should fallback to economic model when gran_sabio_model not provided."""
        model, escalated = arbiter._select_model_for_distribution(
            EditDistribution.MINORITY,
            gran_sabio_model=None
        )
        # Falls back to arbiter's default model
        assert escalated is False


# =============================================================================
# PHASE 2 TESTS: Prompt Building
# =============================================================================


class TestPromptBuilding:
    """Tests for prompt building methods."""

    @pytest.fixture
    def arbiter(self):
        """Create Arbiter instance for testing."""
        return Arbiter(ai_service=object())

    def test_format_edits_for_prompt(self, arbiter):
        """Should format edits clearly for AI prompt."""
        edits = [
            create_proposed_edit(
                edit_type="delete",
                severity="critical",
                source_model="gpt-4o",
                paragraph_key="p1",
                description="Remove duplicate word"
            ),
        ]
        formatted = arbiter._format_edits_for_prompt(edits)

        assert "[0]" in formatted
        assert "DELETE" in formatted
        assert "critical" in formatted
        assert "gpt-4o" in formatted
        assert "Remove duplicate word" in formatted

    def test_format_conflicts_for_prompt(self, arbiter):
        """Should format conflicts clearly for AI prompt."""
        edits = [
            create_proposed_edit(source_model="gpt-4o", paragraph_key="p1"),
            create_proposed_edit(source_model="claude", paragraph_key="p1"),
        ]
        conflicts = [ConflictInfo(
            conflict_type=ConflictType.OPPOSITE_OPERATIONS,
            paragraph_key="p1",
            involved_edits=edits,
            description="DELETE vs REPLACE conflict"
        )]
        formatted = arbiter._format_conflicts_for_prompt(conflicts)

        assert "[Conflict 1]" in formatted
        assert "opposite_operations" in formatted
        assert "gpt-4o" in formatted
        assert "claude" in formatted

    def test_format_empty_edits(self, arbiter):
        """Empty edit list should produce clear message."""
        formatted = arbiter._format_edits_for_prompt([])
        assert "No edits proposed" in formatted

    def test_format_empty_conflicts(self, arbiter):
        """Empty conflict list should produce clear message."""
        formatted = arbiter._format_conflicts_for_prompt([])
        assert "No conflicts detected" in formatted


# =============================================================================
# PHASE 3 TESTS: Integration
# =============================================================================


class TestExtractProposedEditsFromLayerResults:
    """Tests for _extract_proposed_edits_from_layer_results function."""

    def test_extracts_edits_with_source_model(self):
        """Should extract edits preserving source model information."""
        from core.generation_processor import _extract_proposed_edits_from_layer_results
        from smart_edit import TextEditRange, SeverityLevel, OperationType
        from unittest.mock import MagicMock

        # Create mock evaluations with identified issues
        issue1 = TextEditRange(
            marker_mode="phrase",
            paragraph_start="The quick brown",
            paragraph_end="jumps over",
            edit_type=OperationType.REPLACE,
            issue_severity=SeverityLevel.MAJOR,
            issue_description="Test issue 1",
            edit_instruction="Fix it",
            confidence=0.9
        )
        issue2 = TextEditRange(
            marker_mode="phrase",
            paragraph_start="The lazy dog",
            paragraph_end="sleeps",
            edit_type=OperationType.DELETE,
            issue_severity=SeverityLevel.MINOR,
            issue_description="Test issue 2",
            edit_instruction="Remove it",
            confidence=0.8
        )

        eval1 = MagicMock()
        eval1.identified_issues = [issue1]
        eval1.score = 7.5

        eval2 = MagicMock()
        eval2.identified_issues = [issue2]
        eval2.score = 6.0

        layer_results = {
            "gpt-4o": eval1,
            "claude-sonnet": eval2
        }

        # Extract proposed edits
        proposed_edits = _extract_proposed_edits_from_layer_results(layer_results)

        # Verify source model is preserved
        assert len(proposed_edits) == 2
        models = {pe.source_model for pe in proposed_edits}
        assert "gpt-4o" in models
        assert "claude-sonnet" in models

        # Verify source scores are preserved
        gpt_edit = next(pe for pe in proposed_edits if pe.source_model == "gpt-4o")
        assert gpt_edit.source_score == 7.5

    def test_sorts_by_severity_and_confidence(self):
        """Should sort edits by severity (higher first) then confidence."""
        from core.generation_processor import _extract_proposed_edits_from_layer_results
        from smart_edit import TextEditRange, SeverityLevel, OperationType
        from unittest.mock import MagicMock

        critical_issue = TextEditRange(
            marker_mode="phrase",
            paragraph_start="Critical text",
            paragraph_end="end",
            edit_type=OperationType.DELETE,
            issue_severity=SeverityLevel.CRITICAL,
            issue_description="Critical issue",
            edit_instruction="Fix",
            confidence=0.7
        )
        minor_issue = TextEditRange(
            marker_mode="phrase",
            paragraph_start="Minor text",
            paragraph_end="end",
            edit_type=OperationType.REPLACE,
            issue_severity=SeverityLevel.MINOR,
            issue_description="Minor issue",
            edit_instruction="Fix",
            confidence=0.9
        )

        eval1 = MagicMock()
        eval1.identified_issues = [minor_issue]
        eval1.score = 8.0

        eval2 = MagicMock()
        eval2.identified_issues = [critical_issue]
        eval2.score = 5.0

        layer_results = {"model1": eval1, "model2": eval2}

        proposed_edits = _extract_proposed_edits_from_layer_results(layer_results)

        # Critical should come first despite lower confidence
        assert proposed_edits[0].edit.issue_severity == SeverityLevel.CRITICAL

    def test_respects_max_edits_limit(self):
        """Should respect max_edits parameter."""
        from core.generation_processor import _extract_proposed_edits_from_layer_results
        from smart_edit import TextEditRange, SeverityLevel, OperationType
        from unittest.mock import MagicMock

        issues = [
            TextEditRange(
                marker_mode="phrase",
                paragraph_start=f"Text {i}",
                paragraph_end="end",
                edit_type=OperationType.REPLACE,
                issue_severity=SeverityLevel.MINOR,
                issue_description=f"Issue {i}",
                edit_instruction="Fix",
                confidence=0.8
            )
            for i in range(10)
        ]

        eval1 = MagicMock()
        eval1.identified_issues = issues
        eval1.score = 6.0

        layer_results = {"model1": eval1}

        proposed_edits = _extract_proposed_edits_from_layer_results(layer_results, max_edits=3)

        assert len(proposed_edits) == 3


class TestArbiterIntegration:
    """Tests for Arbiter integration with generation processor."""

    def test_arbiter_instantiation_in_processor(self):
        """Arbiter should be created with correct parameters in processor."""
        # This test verifies the code structure - actual integration
        # requires mocking the full async flow
        from arbiter import Arbiter

        mock_ai_service = object()
        arbiter = Arbiter(ai_service=mock_ai_service, model="gpt-5-mini")

        assert arbiter.ai_service is mock_ai_service
        assert arbiter.model == "gpt-5-mini"

    def test_proposed_edit_paragraph_key_generation(self):
        """ProposedEdit should have correct paragraph_key."""
        from arbiter import ProposedEdit
        from smart_edit import TextEditRange, SeverityLevel, OperationType

        edit = TextEditRange(
            marker_mode="phrase",
            paragraph_start="The quick brown fox",
            paragraph_end="lazy dog",
            edit_type=OperationType.REPLACE,
            issue_severity=SeverityLevel.MAJOR,
            issue_description="Test",
            edit_instruction="Fix",
            confidence=0.9
        )

        proposed = ProposedEdit(
            edit=edit,
            source_model="gpt-4o",
            source_score=7.5,
            paragraph_key="The quick brown fox"
        )

        assert proposed.source_model == "gpt-4o"
        assert proposed.source_score == 7.5
        assert proposed.paragraph_key == "The quick brown fox"

    @pytest.mark.asyncio
    async def test_arbiter_result_contains_expected_fields(self):
        """ArbiterResult should contain all expected fields after arbitration."""
        from arbiter import Arbiter, ArbiterContext, LayerEditHistory, ArbiterResult
        from unittest.mock import AsyncMock, MagicMock

        # Create mock AI service
        mock_ai_service = MagicMock()
        mock_ai_service.generate_content = AsyncMock(return_value='{"reasoning": "Test", "decisions": [{"edit_index": 0, "decision": "apply", "reason": "OK"}], "conflicts_resolved": []}')

        arbiter = Arbiter(ai_service=mock_ai_service, model="gpt-4o-mini")

        context = ArbiterContext(
            original_prompt="Test prompt",
            content_type="general",
            system_prompt=None,
            layer_name="Test Layer",
            layer_criteria="Test criteria",
            layer_min_score=7.0,
            current_content="Test content",
            content_excerpt=None,
            proposed_edits=[create_proposed_edit(paragraph_key="p1")],
            evaluator_scores={"gpt-4o": 6.5},
            layer_history=LayerEditHistory(layer_name="Test Layer"),
            gran_sabio_model="claude-opus",
            qa_model_count=1
        )

        result = await arbiter.arbitrate(context)

        # Verify result structure
        assert isinstance(result, ArbiterResult)
        assert hasattr(result, 'edits_to_apply')
        assert hasattr(result, 'edits_discarded')
        assert hasattr(result, 'conflicts_found')
        assert hasattr(result, 'distribution')
        assert hasattr(result, 'escalated_to_gran_sabio')
        assert hasattr(result, 'model_used')


# =============================================================================
# PHASE 4 TESTS: History Injection (stubs for future implementation)
# =============================================================================


class TestHistoryInjection:
    """Tests for history injection - to be implemented in Phase 4."""

    @pytest.mark.skip(reason="Phase 4 - not yet implemented")
    async def test_layer_uses_arbiter_for_conflicts(self):
        """Layer processing should use Arbiter for conflicts."""
        pass

    @pytest.mark.skip(reason="Phase 4 - not yet implemented")
    async def test_history_passed_to_subsequent_rounds(self):
        """Edit history should be passed to subsequent rounds."""
        pass

    @pytest.mark.skip(reason="Phase 4 - not yet implemented")
    async def test_history_cleared_between_layers(self):
        """Edit history should be cleared when moving to new layer."""
        pass

    @pytest.mark.skip(reason="Phase 4 - not yet implemented")
    async def test_qa_receives_history_in_prompt(self):
        """QA evaluators should receive history in their prompts."""
        pass


# =============================================================================
# CONFIG TESTS
# =============================================================================


class TestArbiterConfig:
    """Tests for Arbiter configuration."""

    def test_arbiter_model_in_defaults(self):
        """Arbiter model should be in default_models."""
        from config import get_default_models
        defaults = get_default_models()
        assert "arbiter" in defaults
        assert defaults["arbiter"] == "gpt-5-mini"

    def test_arbiter_model_function(self):
        """_default_arbiter_model should return configured model."""
        from models import _default_arbiter_model
        model = _default_arbiter_model()
        assert model == "gpt-5-mini"

    def test_arbiter_config_values(self):
        """Arbiter config values should be accessible."""
        from config import config
        assert hasattr(config, 'ARBITER_MAX_TOKENS')
        assert hasattr(config, 'ARBITER_TEMPERATURE')
        assert hasattr(config, 'EDIT_HISTORY_MAX_ROUNDS')
        assert hasattr(config, 'EDIT_HISTORY_MAX_CHARS')


class TestContentRequestArbiterField:
    """Tests for arbiter_model field in ContentRequest."""

    def test_content_request_has_arbiter_model(self):
        """ContentRequest should have arbiter_model field."""
        from models import ContentRequest
        # Check field exists in model fields
        assert "arbiter_model" in ContentRequest.model_fields

    def test_content_request_default_arbiter_model(self):
        """ContentRequest should use default arbiter model."""
        from models import ContentRequest, _default_arbiter_model
        # Create minimal request
        request = ContentRequest(
            prompt="Test prompt",
            model="gpt-4o",
            qa_models=["gpt-4o"],
            qa_layers=[]
        )
        assert request.arbiter_model == _default_arbiter_model()

    def test_content_request_custom_arbiter_model(self):
        """ContentRequest should accept custom arbiter model."""
        from models import ContentRequest
        request = ContentRequest(
            prompt="Test prompt",
            model="gpt-4o",
            qa_models=["gpt-4o"],
            qa_layers=[],
            arbiter_model="claude-sonnet-4-5"
        )
        assert request.arbiter_model == "claude-sonnet-4-5"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
