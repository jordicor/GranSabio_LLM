"""Parser hardening regression for the Arbiter (§3.4.2 — 4 criteria).

The Phase 3 refactor eliminates the ``default apply`` fallback in
``_parse_arbiter_response`` and enforces fail-closed semantics. This file
covers all four criteria plus the arbitrate()-level behaviour: the whole
batch must be DISCARDED when any criterion fails and a
``arbiter_parse_error`` debug event must be emitted.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from arbiter import (
    Arbiter,
    ArbiterContext,
    ArbiterDecision,
    ArbiterParseError,
    ConflictInfo,
    LayerEditHistory,
    ProposedEdit,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _edit(description: str) -> SimpleNamespace:
    return SimpleNamespace(
        edit_type=SimpleNamespace(value="replace"),
        issue_severity=SimpleNamespace(value="minor"),
        issue_description=description,
        edit_instruction=description,
        start_marker="marker",
        start_word_index=None,
        confidence=1.0,
    )


def _proposed(description: str, evaluator: str = "Evaluator A", key: str = "p1") -> ProposedEdit:
    return ProposedEdit(
        edit=_edit(description),
        source_evaluator=evaluator,
        source_score=7.0,
        paragraph_key=key,
    )


def _two_edit_context() -> ArbiterContext:
    return ArbiterContext(
        original_prompt="Write a biography.",
        content_type="biography",
        system_prompt=None,
        layer_name="Accuracy",
        layer_criteria="Factual accuracy.",
        layer_min_score=7.0,
        current_content="Draft content.",
        content_excerpt=None,
        proposed_edits=[
            _proposed("First", "Evaluator A", "p1"),
            _proposed("Second", "Evaluator B", "p2"),
        ],
        evaluator_scores={"qa-a": 7.0, "qa-b": 7.0},
        layer_history=LayerEditHistory(layer_name="Accuracy"),
        gran_sabio_model=None,
        qa_model_count=2,
        arbiter_tools_mode="never",
    )


# ---------------------------------------------------------------------------
# Criterion 1: Coverage (missing index → fail-closed)
# ---------------------------------------------------------------------------


class TestCoverageCriterion:
    def test_missing_index_raises_arbiter_parse_error(self):
        """Index 1 missing from decisions must raise ``ArbiterParseError``."""
        arbiter = Arbiter(ai_service=object())
        proposed = [_proposed("a"), _proposed("b")]
        ai_response = {
            "reasoning": "Only one decision emitted.",
            "decisions": [
                {"edit_index": 0, "decision": "APPLY", "reason": "keep"},
            ],
        }
        with pytest.raises(ArbiterParseError) as exc_info:
            arbiter._parse_arbiter_response(ai_response, proposed, [])

        assert 1 in exc_info.value.missing_indices
        assert exc_info.value.duplicate_indices == []
        assert exc_info.value.out_of_range == []
        assert exc_info.value.invalid_decisions == []


# ---------------------------------------------------------------------------
# Criterion 2: Uniqueness (duplicate index → fail-closed)
# ---------------------------------------------------------------------------


class TestUniquenessCriterion:
    def test_duplicate_index_raises_arbiter_parse_error(self):
        """Two decisions for the same edit_index must fail-close."""
        arbiter = Arbiter(ai_service=object())
        proposed = [_proposed("a"), _proposed("b")]
        ai_response = {
            "reasoning": "Duplicate index.",
            "decisions": [
                {"edit_index": 0, "decision": "APPLY", "reason": "first"},
                {"edit_index": 0, "decision": "DISCARD", "reason": "second"},
            ],
        }
        with pytest.raises(ArbiterParseError) as exc_info:
            arbiter._parse_arbiter_response(ai_response, proposed, [])
        assert 0 in exc_info.value.duplicate_indices
        assert 1 in exc_info.value.missing_indices


# ---------------------------------------------------------------------------
# Criterion 3: Range (out-of-range → fail-closed)
# ---------------------------------------------------------------------------


class TestRangeCriterion:
    def test_out_of_range_index_raises_arbiter_parse_error(self):
        """edit_index >= N (or negative) must trip the range check."""
        arbiter = Arbiter(ai_service=object())
        proposed = [_proposed("a"), _proposed("b")]
        ai_response = {
            "reasoning": "Bad index.",
            "decisions": [
                {"edit_index": 0, "decision": "APPLY", "reason": "ok"},
                {"edit_index": 5, "decision": "DISCARD", "reason": "oob"},
            ],
        }
        with pytest.raises(ArbiterParseError) as exc_info:
            arbiter._parse_arbiter_response(ai_response, proposed, [])
        assert 5 in exc_info.value.out_of_range
        assert 1 in exc_info.value.missing_indices

    def test_negative_index_also_flags_out_of_range(self):
        arbiter = Arbiter(ai_service=object())
        proposed = [_proposed("a")]
        ai_response = {
            "reasoning": "Negative.",
            "decisions": [
                {"edit_index": -1, "decision": "APPLY", "reason": "x"},
            ],
        }
        with pytest.raises(ArbiterParseError) as exc_info:
            arbiter._parse_arbiter_response(ai_response, proposed, [])
        assert -1 in exc_info.value.out_of_range


# ---------------------------------------------------------------------------
# Criterion 4: Decision validity (invalid value → fail-closed)
# ---------------------------------------------------------------------------


class TestDecisionValidityCriterion:
    def test_merge_decision_value_is_rejected(self):
        """The legacy ``"merge"`` value must be classified as invalid."""
        arbiter = Arbiter(ai_service=object())
        proposed = [_proposed("a")]
        ai_response = {
            "reasoning": "Legacy merge.",
            "decisions": [
                {"edit_index": 0, "decision": "merge", "reason": "legacy"},
            ],
        }
        with pytest.raises(ArbiterParseError) as exc_info:
            arbiter._parse_arbiter_response(ai_response, proposed, [])
        invalid_values = [entry["decision"] for entry in exc_info.value.invalid_decisions]
        assert "merge" in invalid_values

    def test_lowercase_apply_is_rejected(self):
        """Schema mandates uppercase — no silent case normalisation."""
        arbiter = Arbiter(ai_service=object())
        proposed = [_proposed("a")]
        ai_response = {
            "reasoning": "Lowercase.",
            "decisions": [
                {"edit_index": 0, "decision": "apply", "reason": "case"},
            ],
        }
        with pytest.raises(ArbiterParseError) as exc_info:
            arbiter._parse_arbiter_response(ai_response, proposed, [])
        invalid_values = [entry["decision"] for entry in exc_info.value.invalid_decisions]
        assert "apply" in invalid_values

    def test_unknown_decision_value_is_rejected(self):
        arbiter = Arbiter(ai_service=object())
        proposed = [_proposed("a")]
        ai_response = {
            "reasoning": "Bogus.",
            "decisions": [
                {"edit_index": 0, "decision": "BOGUS", "reason": "x"},
            ],
        }
        with pytest.raises(ArbiterParseError) as exc_info:
            arbiter._parse_arbiter_response(ai_response, proposed, [])
        assert len(exc_info.value.invalid_decisions) == 1


# ---------------------------------------------------------------------------
# Happy path (no violations, no default-apply): every index must be explicit
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_full_coverage_all_apply(self):
        arbiter = Arbiter(ai_service=object())
        proposed = [_proposed("a"), _proposed("b")]
        ai_response = {
            "reasoning": "All clean.",
            "decisions": [
                {"edit_index": 0, "decision": "APPLY", "reason": "ok-a"},
                {"edit_index": 1, "decision": "APPLY", "reason": "ok-b"},
            ],
        }
        decisions = arbiter._parse_arbiter_response(ai_response, proposed, [])
        assert [d.decision for d in decisions] == [
            ArbiterDecision.APPLY,
            ArbiterDecision.APPLY,
        ]

    def test_default_apply_is_gone_when_index_missing(self):
        """The legacy ``No conflict / default apply`` path must NOT resurrect."""
        arbiter = Arbiter(ai_service=object())
        proposed = [_proposed("a"), _proposed("b")]
        ai_response = {"reasoning": "none", "decisions": []}
        with pytest.raises(ArbiterParseError):
            arbiter._parse_arbiter_response(ai_response, proposed, [])


# ---------------------------------------------------------------------------
# arbitrate()-level fail-closed + debug event emission
# ---------------------------------------------------------------------------


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro) if False else asyncio.run(coro)


class TestArbitrateFailClosedBatch:
    def test_invalid_decision_batches_discard_all_and_emit_event(self):
        """When parsing fails the whole batch is DISCARDED and an event is logged."""
        captured: List[Dict[str, Any]] = []

        async def _capture(event_type: str, payload: Dict[str, Any]) -> None:
            captured.append({"event_type": event_type, "payload": payload})

        arbiter = Arbiter(
            ai_service=object(),
            debug_event_callback=_capture,
        )

        # Stub internal helpers so we land at the parser fail-closed path.
        arbiter._filter_stale_edits = lambda edits, content: (edits, [], [])
        arbiter._detect_conflicts = lambda edits, history: []
        arbiter._classify_distribution = (
            lambda edits, qa_model_count, conflicts: __import__(
                "arbiter"
            ).EditDistribution.CONSENSUS
        )
        arbiter._select_model_for_distribution = (
            lambda distribution, gran_sabio_model: ("fake-model", False)
        )

        async def _fake_resolve(context, conflicts, distribution, selected_model):
            return {
                "reasoning": "Legacy merge emitted.",
                "decisions": [
                    {"edit_index": 0, "decision": "merge", "reason": "legacy"},
                    {"edit_index": 1, "decision": "APPLY", "reason": "keep"},
                ],
            }

        arbiter._resolve_with_ai = _fake_resolve

        context = _two_edit_context()
        result = asyncio.run(arbiter.arbitrate(context))

        # Every edit discarded, none applied
        assert result.edits_to_apply == []
        assert len(result.edits_discarded) == 2
        assert all(
            d.decision == ArbiterDecision.DISCARD for d in result.edit_decisions
        )

        # arbiter_parse_error event captured
        event_types = [item["event_type"] for item in captured]
        assert "arbiter_parse_error" in event_types
        parse_event = next(
            item for item in captured if item["event_type"] == "arbiter_parse_error"
        )
        payload = parse_event["payload"]
        assert payload["total_edits"] == 2
        assert any(
            entry.get("decision") == "merge"
            for entry in payload.get("invalid_decisions", [])
        )

    def test_missing_index_batches_discard_all_and_emit_event(self):
        captured: List[Dict[str, Any]] = []

        async def _capture(event_type: str, payload: Dict[str, Any]) -> None:
            captured.append({"event_type": event_type, "payload": payload})

        arbiter = Arbiter(ai_service=object(), debug_event_callback=_capture)
        arbiter._filter_stale_edits = lambda edits, content: (edits, [], [])
        arbiter._detect_conflicts = lambda edits, history: []
        arbiter._classify_distribution = (
            lambda edits, qa_model_count, conflicts: __import__(
                "arbiter"
            ).EditDistribution.CONSENSUS
        )
        arbiter._select_model_for_distribution = (
            lambda distribution, gran_sabio_model: ("fake-model", False)
        )

        async def _fake_resolve(context, conflicts, distribution, selected_model):
            return {
                "reasoning": "Only decision 0 emitted.",
                "decisions": [
                    {"edit_index": 0, "decision": "APPLY", "reason": "ok"},
                ],
            }

        arbiter._resolve_with_ai = _fake_resolve
        context = _two_edit_context()

        result = asyncio.run(arbiter.arbitrate(context))
        assert result.edits_to_apply == []
        assert all(
            d.decision == ArbiterDecision.DISCARD for d in result.edit_decisions
        )

        parse_events = [
            item for item in captured if item["event_type"] == "arbiter_parse_error"
        ]
        assert len(parse_events) == 1
        assert 1 in parse_events[0]["payload"]["missing_indices"]
