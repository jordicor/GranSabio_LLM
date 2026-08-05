"""Security regression tests for the QA package.

Covers QA-01 (weakened quorum lets unanimous deal-breakers through),
QA-02 (NaN/Infinity scores clamped to 10.0), QA-03 (gather leaves orphaned
sibling tasks), QA-04 (dead "Gran Sabio approved with modifications" branch),
QA-05 (retry budget shared across failure categories), QA-06 (hardcoded 8.0
cumulative repetition threshold) and QA-07 (full raw preflight output in logs).
"""

import asyncio
import logging
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

# The venv's libmagic DLL crashes the interpreter when loaded (pre-existing
# environment issue); stub the module before importing project modules that
# pull it in transitively (preflight_validator -> attachment_validation).
sys.modules.setdefault("magic", MagicMock())

import json_utils as json
import preflight_validator
from core.qa_decision_engine import (
    _check_majority_deal_breakers,
    _evaluate_layer_based_approval,
)
from deterministic_validation import evaluate_cumulative_repetition_check
from models import PhraseFrequencyConfig, PhraseFrequencyRule, QAEvaluation, QALayer
from qa_engine import QAEngine
from qa_evaluation_service import QAEvaluationService, QAResponseParseError
from qa_result_utils import resolve_effective_required_valid
from qa_scheduler import (
    QAScheduler,
    QASchedulerPolicy,
    QASchedulerSlot,
    QASchedulerTechnicalFailure,
    QASchedulerUnavailableError,
)

# ============================================================================
# Shared helpers
# ============================================================================

def _layer(name: str = "Accuracy", min_score: float = 7.0, order: int = 1) -> QALayer:
    return QALayer(
        name=name,
        description=f"{name} description",
        criteria=f"{name} criteria",
        min_score=min_score,
        order=order,
    )


def _eval(
    model: str,
    *,
    layer: str = "Accuracy",
    score: float = 9.0,
    deal_breaker: bool = False,
) -> QAEvaluation:
    return QAEvaluation(
        model=model,
        layer=layer,
        score=score,
        feedback="ok" if not deal_breaker else "blocked",
        deal_breaker=deal_breaker,
        deal_breaker_reason="fabricated quote" if deal_breaker else None,
        passes_score=score >= 7.0,
    )


def _request(**overrides):
    base = {
        "qa_layers": [_layer()],
        "qa_models": ["qa-1", "qa-2", "qa-3"],
        "max_iterations": 3,
        "min_global_score": 8.0,
        "gran_sabio_fallback": False,
        "min_valid_qa_models": None,
        "min_valid_qa_model_ratio": None,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _consensus(score: float = 9.0, layer_name: str = "Accuracy"):
    return SimpleNamespace(average_score=score, layer_averages={layer_name: score})


def _slots(count: int):
    return [
        QASchedulerSlot(
            index=i,
            model=f"model-{i}",
            model_name=f"model-{i}",
            result_key=f"model-{i}",
            evaluator_label=f"QA-{i}",
            timeout_seconds=1.0,
            slot_id=f"qa:{i}",
        )
        for i in range(count)
    ]


def _slot_eval(slot: QASchedulerSlot, *, deal_breaker: bool = False, score: float = 8.0) -> QAEvaluation:
    return QAEvaluation(
        model=slot.model_name,
        slot_id=slot.slot_id,
        evaluator_alias=slot.evaluator_label,
        layer="Accuracy",
        score=score,
        feedback="ok" if not deal_breaker else "bad",
        deal_breaker=deal_breaker,
        deal_breaker_reason="fabricated" if deal_breaker else None,
        passes_score=score >= 7.0,
    )


# ============================================================================
# QA-01: weakened quorum must not let unanimous deal-breakers through
# ============================================================================

class TestQA01WeakenedQuorumDealBreakers:
    """A user-set min_valid_qa_models=1 admits a layer with a single valid
    evaluator; a unanimous (1/1) semantic deal-breaker must still block."""

    def test_decision_engine_blocks_unanimous_deal_breaker_with_weakened_quorum(self):
        request = _request(min_valid_qa_models=1)
        qa_results = {"Accuracy": {"qa-1": _eval("qa-1", deal_breaker=True)}}

        result = _evaluate_layer_based_approval(
            qa_results=qa_results,
            consensus_result=_consensus(),
            request=request,
            session_id="sess-qa01",
            iteration=1,
            evaluated_layers=request.qa_layers,
            qa_summary={"summary": {}},
        )

        assert result["approved"] is False
        assert result["deal_breaker_type"] == "majority_consensus_retry"

    def test_decision_engine_final_rejection_at_max_iterations_with_weakened_quorum(self):
        request = _request(min_valid_qa_models=1, max_iterations=2)
        qa_results = {"Accuracy": {"qa-1": _eval("qa-1", deal_breaker=True)}}

        result = _evaluate_layer_based_approval(
            qa_results=qa_results,
            consensus_result=_consensus(),
            request=request,
            session_id="sess-qa01-max",
            iteration=2,
            evaluated_layers=request.qa_layers,
            qa_summary={"summary": {}},
        )

        assert result["approved"] is False
        assert result["final_rejection"] is True
        assert result["deal_breaker_type"] == "majority_consensus"

    def test_majority_gate_uses_propagated_quorum(self):
        qa_results = {"Accuracy": {"qa-1": _eval("qa-1", deal_breaker=True)}}
        qa_models = ["qa-1", "qa-2", "qa-3"]

        # Old behavior: default quorum (2 of 3) silently discards the 1/1 DB.
        default_quorum = _check_majority_deal_breakers(qa_results, qa_models, required_valid=2)
        # Effective quorum propagated from the scheduler: 1/1 is a majority.
        effective_quorum = _check_majority_deal_breakers(qa_results, qa_models, required_valid=1)

        assert default_quorum["has_majority_deal_breakers"] is True  # fail-closed
        assert effective_quorum["has_majority_deal_breakers"] is True
        assert effective_quorum["layers"][0]["qa_quorum"]["required_valid"] == 1

    def test_fail_closed_when_valid_below_default_quorum(self):
        """Without an explicit quorum override, a valid semantic deal-breaker
        found below the default quorum blocks instead of being approved."""
        qa_results = {"Accuracy": {"qa-1": _eval("qa-1", deal_breaker=True)}}

        result = _check_majority_deal_breakers(qa_results, ["qa-1", "qa-2", "qa-3"])

        assert result["has_majority_deal_breakers"] is True

    def test_weakened_quorum_without_deal_breakers_still_approves(self):
        """Control: the weakened quorum alone must not block clean content."""
        request = _request(min_valid_qa_models=1)
        qa_results = {"Accuracy": {"qa-1": _eval("qa-1", deal_breaker=False)}}

        result = _evaluate_layer_based_approval(
            qa_results=qa_results,
            consensus_result=_consensus(),
            request=request,
            session_id="sess-qa01-clean",
            iteration=1,
            evaluated_layers=request.qa_layers,
            qa_summary={"summary": {}},
        )

        assert result["approved"] is True

    def test_resolve_effective_required_valid_matches_scheduler_resolution(self):
        assert resolve_effective_required_valid(_request(min_valid_qa_models=1), 3) == 1
        assert resolve_effective_required_valid(_request(), 3) == 2
        assert resolve_effective_required_valid(_request(min_valid_qa_model_ratio=0.5), 4) == 2
        assert resolve_effective_required_valid(None, 3) == 2


# ============================================================================
# QA-02: NaN/Infinity scores must be rejected, not clamped to 10.0
# ============================================================================

class TestQA02NonFiniteScoresRejected:
    def _parse(self, payload: str):
        service = QAEvaluationService(MagicMock())
        return service._parse_qa_json_response(
            response=payload,
            model="fake-model",
            layer_name="Accuracy",
        )

    def test_nan_string_score_rejected(self):
        payload = json.dumps({"score": "NaN", "feedback": "ok"})
        with pytest.raises(QAResponseParseError):
            self._parse(payload)

    def test_infinity_string_score_rejected(self):
        for raw in ("Infinity", "-Infinity", "inf"):
            payload = json.dumps({"score": raw, "feedback": "ok"})
            with pytest.raises(QAResponseParseError):
                self._parse(payload)

    def test_overflowed_numeric_score_rejected(self):
        # 1e400 overflows to inf when parsed as a JSON number.
        with pytest.raises(QAResponseParseError):
            self._parse('{"score": 1e400, "feedback": "ok"}')

    def test_valid_string_score_still_accepted(self):
        result = self._parse(json.dumps({"score": "8.5", "feedback": "ok"}))
        assert result["score"] == 8.5

    def test_out_of_range_score_still_clamped(self):
        result = self._parse(json.dumps({"score": 12, "feedback": "ok"}))
        assert result["score"] == 10.0


# ============================================================================
# QA-03: gather must cancel sibling tasks on any failure
# ============================================================================

class TestQA03GatherCancelsSiblings:
    @pytest.mark.asyncio
    async def test_scheduler_batch_cancels_siblings_on_policy_abort(self):
        cancelled = []

        async def evaluate(slot, attempt):
            if slot.index == 0:
                raise QASchedulerTechnicalFailure("api_failure", "provider down", retryable=False)
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                cancelled.append(slot.model_name)
                raise
            return _slot_eval(slot)

        scheduler = QAScheduler(
            QASchedulerPolicy(execution_mode="parallel", on_model_unavailable="fail")
        )
        with pytest.raises(QASchedulerUnavailableError):
            await scheduler.evaluate_layer(
                layer_name="Accuracy",
                has_deal_breaker_criteria=False,
                slots=_slots(3),
                evaluate_slot=evaluate,
            )

        assert sorted(cancelled) == ["model-1", "model-2"]

    @pytest.mark.asyncio
    async def test_final_verification_cancels_sibling_layers(self):
        engine = QAEngine(ai_service=MagicMock(), bypass_engine=MagicMock())
        cancelled = []

        async def fake_single_layer(content, layer, **kwargs):
            if layer.name == "Failing":
                raise QASchedulerUnavailableError("QA quorum impossible")
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                cancelled.append(layer.name)
                raise
            return {}, None

        engine._evaluate_single_semantic_layer = fake_single_layer

        with pytest.raises(QASchedulerUnavailableError):
            await engine.evaluate_final_verification(
                content="final candidate text",
                layers=[_layer("Slow", order=1), _layer("Failing", order=2)],
                qa_models=["qa-1"],
                strategy="full_parallel",
            )

        assert cancelled == ["Slow"]


# ============================================================================
# QA-04: dead "Gran Sabio approved with modifications" branch removed
# ============================================================================

def _no_bypass_engine() -> MagicMock:
    """Bypass engine mock that never triggers the algorithmic bypass."""
    bypass = MagicMock()
    bypass.should_bypass_qa_layer = Mock(return_value=False)
    bypass.bypass_layer_evaluation = Mock(return_value={})
    bypass.should_skip_incremental_repair = Mock(return_value=False)
    return bypass


class TestQA04GranSabioModifiedBranchRemoved:
    def test_modified_result_contract_removed(self):
        assert not hasattr(QAEngine, "_create_gran_sabio_modified_result")

    @pytest.mark.asyncio
    async def test_approval_with_modifications_continues_as_false_positive(self, sample_qa_layer):
        """Gran Sabio approving a minority DB with modifications must no longer
        emit the dead gran_sabio_modified result shape; evaluation continues
        with the original content as a false positive."""
        engine = QAEngine(ai_service=MagicMock(), bypass_engine=_no_bypass_engine())

        call_count = 0

        async def mock_evaluate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return QAEvaluation(
                    model="gpt-4o", layer="Test Quality", score=4.0,
                    feedback="Minor error", deal_breaker=True,
                    deal_breaker_reason="Small issue", passes_score=False,
                )
            return QAEvaluation(
                model="other", layer="Test Quality", score=8.0,
                feedback="Good", deal_breaker=False, passes_score=True,
            )

        engine.qa_evaluator = MagicMock()
        engine.qa_evaluator.evaluate_content = AsyncMock(side_effect=mock_evaluate)

        gran_sabio = MagicMock()
        gran_sabio.review_minority_deal_breakers = AsyncMock(return_value=MagicMock(
            approved=True,
            error=None,
            reason="Fixed the issue",
            final_score=9.0,
            final_content="Corrected content here",
            modifications_made=True,
        ))

        with patch("qa_engine.calculate_qa_timeout_for_model", return_value=60):
            with patch("deal_breaker_tracker.get_tracker") as mock_tracker:
                mock_tracker.return_value.record_escalation = Mock(return_value="esc-123")
                mock_tracker.return_value.complete_escalation = Mock()

                results = await engine.evaluate_all_layers_with_progress(
                    content="Test content",
                    layers=[sample_qa_layer],
                    qa_models=["gpt-4o", "claude-sonnet-4", "gemini-pro"],
                    gran_sabio_engine=gran_sabio,
                )

        assert results.get("gran_sabio_modified_content") is None
        assert results.get("summary", {}).get("gran_sabio_modified") is not True
        assert "Test Quality" in results


# ============================================================================
# QA-05: retry budgets are tracked per failure category
# ============================================================================

class TestQA05PerCategoryRetryBudgets:
    @pytest.mark.asyncio
    async def test_timeout_does_not_consume_parse_error_budget(self):
        """With defaults (timeout_retries=2, parse_error_retries=1), a timeout
        followed by a parse error must still leave the parse retry available."""
        attempts = []

        async def evaluate(slot, attempt):
            attempts.append(attempt)
            if attempt == 1:
                raise QASchedulerTechnicalFailure("timeout", "evaluator timed out")
            if attempt == 2:
                raise QASchedulerTechnicalFailure("parse_error", "invalid JSON")
            return _slot_eval(slot)

        scheduler = QAScheduler(QASchedulerPolicy(execution_mode="parallel"))
        result = await scheduler.evaluate_layer(
            layer_name="Accuracy",
            has_deal_breaker_criteria=False,
            slots=_slots(1),
            evaluate_slot=evaluate,
        )

        assert attempts == [1, 2, 3]
        assert result.counts["valid"] == 1

    @pytest.mark.asyncio
    async def test_category_budgets_still_exhaust(self):
        """Each category keeps its own cap: two parse errors with
        parse_error_retries=1 exhaust the budget and fail technically."""
        attempts = []

        async def evaluate(slot, attempt):
            attempts.append(attempt)
            raise QASchedulerTechnicalFailure("parse_error", "invalid JSON")

        scheduler = QAScheduler(
            QASchedulerPolicy(execution_mode="parallel", on_model_unavailable="skip_if_quorum")
        )
        with pytest.raises(QASchedulerUnavailableError):
            await scheduler.evaluate_layer(
                layer_name="Accuracy",
                has_deal_breaker_criteria=False,
                slots=_slots(1),
                evaluate_slot=evaluate,
            )

        assert attempts == [1, 2]


# ============================================================================
# QA-06: cumulative repetition check uses the configured layer min_score
# ============================================================================

class TestQA06CumulativeRepetitionThreshold:
    def _request(self, min_score: float):
        config = PhraseFrequencyConfig(
            enabled=True,
            min_score=min_score,
            rules=[
                PhraseFrequencyRule(name="bigrams", min_length=2, max_repetitions=1),
            ],
        )
        return SimpleNamespace(
            phrase_frequency=config,
            cumulative_text="previous chapter text",
            cumulative_word_count=None,
        )

    def _analysis_with_one_warning(self):
        issue = SimpleNamespace(
            phrase="foo bar",
            n=2,
            count=3,
            limit=1,
            severity="warn",
            rule_label="bigrams",
            guidance="Vary the phrasing.",
            repeat_ratio_tokens=0.1,
        )
        return SimpleNamespace(issues=[issue], analyzer_output="diag")

    def test_min_score_from_layer_config_is_used(self):
        # One warning issue -> score 7.5. With min_score=7.0 the guard must
        # pass; the old hardcoded 8.0 threshold would fail it.
        with patch(
            "deterministic_validation.analyze_phrase_frequency",
            return_value=self._analysis_with_one_warning(),
        ):
            result = evaluate_cumulative_repetition_check("draft text", self._request(7.0))

        assert result is not None
        assert result.score == 7.5
        assert result.pass_threshold == 7.0
        assert result.passed is True

    def test_min_score_above_score_still_fails(self):
        with patch(
            "deterministic_validation.analyze_phrase_frequency",
            return_value=self._analysis_with_one_warning(),
        ):
            result = evaluate_cumulative_repetition_check("draft text", self._request(9.0))

        assert result is not None
        assert result.pass_threshold == 9.0
        assert result.passed is False


# ============================================================================
# QA-07: preflight raw output is truncated in logs
# ============================================================================

class TestQA07PreflightLogTruncation:
    @pytest.mark.asyncio
    async def test_raw_output_truncated_in_info_and_warning_logs(self, caplog):
        raw_output = "X" * 5000
        ai_service = MagicMock()
        ai_service.generate_content = AsyncMock(return_value=raw_output)

        fake_config = SimpleNamespace(
            PREFLIGHT_SYSTEM_PROMPT="system",
            REQUEST_TIMEOUTS={},
            REQUEST_TIMEOUT=120,
            resolve_output_max_tokens=lambda *a, **k: {"max_tokens": 100},
        )
        request = SimpleNamespace(_llm_routing=None)

        with caplog.at_level(logging.DEBUG, logger="preflight_validator"):
            with patch.object(preflight_validator, "_should_bypass_llm_preflight", return_value=False), \
                 patch.object(preflight_validator, "resolve_preflight_model", return_value="fake-model"), \
                 patch.object(preflight_validator, "resolve_call", return_value=SimpleNamespace(params={})), \
                 patch.object(preflight_validator, "resolve_temperature", return_value=0.0), \
                 patch.object(preflight_validator, "config", fake_config), \
                 patch.object(preflight_validator, "_validate_evidence_grounding_config", return_value=None), \
                 patch.object(preflight_validator, "_build_validation_payload", return_value={}), \
                 patch.object(preflight_validator, "_build_validator_prompt", return_value="prompt"), \
                 patch.object(preflight_validator, "_build_prompt_safety_parts", return_value=[]), \
                 patch.object(preflight_validator, "resolve_request_timeout", return_value=60.0):
                result = await preflight_validator.run_preflight_validation(
                    ai_service,
                    request,
                )

        assert result.decision == "reject"
        raw_info = [r for r in caplog.records if "Preflight validator raw output" in r.getMessage()]
        unparseable = [r for r in caplog.records if "unparseable output" in r.getMessage()]
        assert raw_info, "expected the raw-output INFO log"
        assert unparseable, "expected the unparseable-output WARNING log"
        for record in raw_info + unparseable:
            message = record.getMessage()
            assert "X" * 500 in message
            assert "X" * 501 not in message
