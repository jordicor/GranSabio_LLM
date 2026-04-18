import asyncio

import pytest

from models import QAEvaluation, ContentRequest
from qa_result_utils import required_valid_qa_models
from qa_scheduler import (
    QAScheduler,
    QASchedulerPolicy,
    QASchedulerSlot,
    QASchedulerTechnicalFailure,
    QASchedulerUnavailableError,
)


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


def _eval(slot, *, deal_breaker=False, score=8.0):
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


def test_required_valid_defaults_are_strict_majority():
    assert [required_valid_qa_models(n) for n in range(1, 7)] == [1, 2, 2, 3, 3, 4]


def test_content_request_rejects_ambiguous_quorum_config():
    with pytest.raises(ValueError, match="Configure only one"):
        ContentRequest(
            prompt="Write a valid article prompt",
            min_valid_qa_models=2,
            min_valid_qa_model_ratio=0.5,
        )


@pytest.mark.asyncio
async def test_progressive_quorum_cuts_after_initial_majority():
    called = []

    async def evaluate(slot, attempt):
        called.append(slot.model_name)
        return _eval(slot, deal_breaker=True, score=2.0)

    scheduler = QAScheduler(QASchedulerPolicy(execution_mode="progressive_quorum"))
    result = await scheduler.evaluate_layer(
        layer_name="Accuracy",
        has_deal_breaker_criteria=True,
        slots=_slots(6),
        evaluate_slot=evaluate,
    )

    assert len(called) == 4
    assert result.majority_deal_breaker is not None
    assert result.counts["semantic_deal_breakers"] == 4
    assert result.counts["skipped"] == 2


@pytest.mark.asyncio
async def test_progressive_quorum_launches_only_needed_extra_model():
    called = []
    outcomes = [True, True, True, False, True, False]

    async def evaluate(slot, attempt):
        called.append(slot.model_name)
        deal_breaker = outcomes[slot.index]
        return _eval(slot, deal_breaker=deal_breaker, score=2.0 if deal_breaker else 8.0)

    scheduler = QAScheduler(QASchedulerPolicy(execution_mode="progressive_quorum"))
    result = await scheduler.evaluate_layer(
        layer_name="Accuracy",
        has_deal_breaker_criteria=True,
        slots=_slots(6),
        evaluate_slot=evaluate,
    )

    assert called == ["model-0", "model-1", "model-2", "model-3", "model-4"]
    assert result.majority_deal_breaker is not None
    assert result.counts["semantic_deal_breakers"] == 4
    assert result.counts["skipped"] == 1


@pytest.mark.asyncio
async def test_parallel_without_deal_breakers_respects_concurrency_limit():
    active = 0
    max_active = 0
    called = []

    async def evaluate(slot, attempt):
        nonlocal active, max_active
        called.append(slot.model_name)
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.01)
        active -= 1
        return _eval(slot)

    scheduler = QAScheduler(
        QASchedulerPolicy(execution_mode="auto", max_concurrency=2)
    )
    result = await scheduler.evaluate_layer(
        layer_name="Style",
        has_deal_breaker_criteria=False,
        slots=_slots(5),
        evaluate_slot=evaluate,
    )

    assert len(called) == 5
    assert max_active <= 2
    assert result.counts["valid"] == 5


@pytest.mark.asyncio
async def test_technical_failures_do_not_count_as_deal_breakers():
    async def evaluate(slot, attempt):
        if slot.index == 0:
            raise QASchedulerTechnicalFailure("api_failure", "provider down")
        return _eval(slot, deal_breaker=slot.index in {1, 2}, score=2.0 if slot.index in {1, 2} else 8.0)

    scheduler = QAScheduler(QASchedulerPolicy(on_model_unavailable="skip_if_quorum"))
    result = await scheduler.evaluate_layer(
        layer_name="Accuracy",
        has_deal_breaker_criteria=True,
        slots=_slots(4),
        evaluate_slot=evaluate,
    )

    assert result.counts["technical_failed"] == 1
    assert result.counts["semantic_deal_breakers"] == 2
    assert result.majority_deal_breaker is not None


@pytest.mark.asyncio
async def test_six_configured_two_technical_three_of_four_valid_is_majority():
    async def evaluate(slot, attempt):
        if slot.index in {0, 1}:
            raise QASchedulerTechnicalFailure("api_failure", "provider down")
        deal_breaker = slot.index in {2, 3, 4}
        return _eval(slot, deal_breaker=deal_breaker, score=2.0 if deal_breaker else 8.0)

    scheduler = QAScheduler(QASchedulerPolicy(on_model_unavailable="skip_if_quorum"))
    result = await scheduler.evaluate_layer(
        layer_name="Accuracy",
        has_deal_breaker_criteria=True,
        slots=_slots(6),
        evaluate_slot=evaluate,
    )

    assert result.counts["configured"] == 6
    assert result.counts["technical_failed"] == 2
    assert result.counts["valid"] == 4
    assert result.counts["semantic_deal_breakers"] == 3
    assert result.majority_deal_breaker is not None


@pytest.mark.asyncio
async def test_skip_if_quorum_fails_when_valid_quorum_is_impossible():
    async def evaluate(slot, attempt):
        if slot.index in {0, 1}:
            raise QASchedulerTechnicalFailure("api_failure", "provider down")
        return _eval(slot)

    scheduler = QAScheduler(QASchedulerPolicy(on_model_unavailable="skip_if_quorum"))

    with pytest.raises(QASchedulerUnavailableError, match="quorum impossible"):
        await scheduler.evaluate_layer(
            layer_name="Style",
            has_deal_breaker_criteria=False,
            slots=_slots(3),
            evaluate_slot=evaluate,
        )


@pytest.mark.asyncio
async def test_fail_policy_stops_on_unavailable_model():
    async def evaluate(slot, attempt):
        raise QASchedulerTechnicalFailure("api_failure", "provider down")

    scheduler = QAScheduler(QASchedulerPolicy(on_model_unavailable="fail"))

    with pytest.raises(QASchedulerUnavailableError, match="unavailable"):
        await scheduler.evaluate_layer(
            layer_name="Style",
            has_deal_breaker_criteria=False,
            slots=_slots(3),
            evaluate_slot=evaluate,
        )


@pytest.mark.asyncio
async def test_timeout_retries_then_skips_if_quorum_remains():
    attempts = {}

    async def evaluate(slot, attempt):
        attempts[slot.model_name] = attempts.get(slot.model_name, 0) + 1
        if slot.index == 0:
            raise QASchedulerTechnicalFailure("timeout", "timeout", retryable=True)
        return _eval(slot)

    scheduler = QAScheduler(
        QASchedulerPolicy(
            on_timeout="retry_then_skip_if_quorum",
            timeout_retries=1,
        )
    )
    result = await scheduler.evaluate_layer(
        layer_name="Style",
        has_deal_breaker_criteria=False,
        slots=_slots(3),
        evaluate_slot=evaluate,
    )

    assert attempts["model-0"] == 2
    assert result.counts["technical_failed"] == 1
    assert result.counts["valid"] == 2
