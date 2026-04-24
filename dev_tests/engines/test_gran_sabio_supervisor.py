from pathlib import Path

from deal_breaker_tracker import DealBreakerTracker
from gran_sabio_supervisor import GranSabioSupervisor
from models import QAEvaluation, ContentRequest


def _deal_breaker(model: str, layer: str = "Accuracy", slot_id: str = "qa:0"):
    return QAEvaluation(
        model=model,
        slot_id=slot_id,
        evaluator_alias=slot_id,
        layer=layer,
        score=2.0,
        feedback="bad",
        deal_breaker=True,
        deal_breaker_reason="fabricated",
        passes_score=False,
    )


def _tracker(tmp_path: Path):
    return DealBreakerTracker(persistence_path=str(tmp_path))


def test_tracker_completes_escalation_for_each_evaluator(tmp_path):
    tracker = _tracker(tmp_path)
    escalation_id = tracker.record_escalation(
        session_id="session-1",
        iteration=1,
        layer_name="Accuracy",
        trigger_type="minority_deal_breaker",
        triggering_model="model-a",
        deal_breaker_reason="fabricated",
        total_models=3,
        deal_breaker_count=2,
        gran_sabio_model="gran-sabio",
        deal_breaker_evaluations=[
            _deal_breaker("model-a", slot_id="qa:0"),
            _deal_breaker("model-b", slot_id="qa:1"),
        ],
    )

    tracker.complete_escalation(
        escalation_id=escalation_id,
        decision="false_positive",
        reasoning="bad judge call",
        was_real=False,
    )

    events = tracker.get_session_events("session-1")
    assert len(events) == 2
    assert {event["real_model"] for event in events} == {"model-a", "model-b"}
    assert all(event["was_real_deal_breaker"] is False for event in events)
    assert tracker.get_model_stats("model-a").confirmed_false_positive == 1
    assert tracker.get_model_stats("model-b").confirmed_false_positive == 1


def test_tracker_clear_session_releases_in_memory_session_state(tmp_path):
    tracker = _tracker(tmp_path)
    escalation_id = tracker.record_escalation(
        session_id="session-1",
        iteration=1,
        layer_name="Accuracy",
        trigger_type="minority_deal_breaker",
        triggering_model="model-a",
        deal_breaker_reason="fabricated",
        total_models=3,
        deal_breaker_count=1,
        gran_sabio_model="gran-sabio",
        deal_breaker_evaluations=[_deal_breaker("model-a", slot_id="qa:0")],
    )
    tracker.record_model_benched("session-1", "Accuracy", "model-a")
    tracker.record_model_benched("session-2", "Accuracy", "model-b")

    tracker.clear_session("session-1")

    assert tracker.get_session_events("session-1") == []
    assert tracker.get_session_escalations("session-1") == []
    assert escalation_id not in tracker.escalation_event_links
    assert tracker.get_session_events("session-2")


def test_supervisor_flags_two_false_positives_same_layer(tmp_path):
    tracker = _tracker(tmp_path)
    for i in range(2):
        escalation_id = tracker.record_escalation(
            session_id="session-1",
            iteration=i + 1,
            layer_name="Accuracy",
            trigger_type="minority_deal_breaker",
            triggering_model="model-a",
            deal_breaker_reason="fabricated",
            total_models=3,
            deal_breaker_count=1,
            gran_sabio_model="gran-sabio",
            deal_breaker_evaluations=[_deal_breaker("model-a", slot_id="qa:0")],
        )
        tracker.complete_escalation(
            escalation_id=escalation_id,
            decision="false_positive",
            reasoning="false positive",
            was_real=False,
        )

    result = GranSabioSupervisor().review_session(
        session_id="session-1",
        tracker=tracker,
        request=ContentRequest(prompt="Write a valid article prompt"),
        session={},
    )

    assert result["health_state"] == "ok"
    assert result["supervisor_flags"][0]["type"] == "suspect_false_positive_pattern"
    assert result["bench_recommendations"] == []


def test_supervisor_recommends_bench_after_three_false_positives_without_replacement(tmp_path):
    tracker = _tracker(tmp_path)
    for i in range(3):
        escalation_id = tracker.record_escalation(
            session_id="session-1",
            iteration=i + 1,
            layer_name="Accuracy",
            trigger_type="minority_deal_breaker",
            triggering_model="model-a",
            deal_breaker_reason="fabricated",
            total_models=3,
            deal_breaker_count=1,
            gran_sabio_model="gran-sabio",
            deal_breaker_evaluations=[_deal_breaker("model-a", slot_id="qa:0")],
        )
        tracker.complete_escalation(
            escalation_id=escalation_id,
            decision="false_positive",
            reasoning="false positive",
            was_real=False,
        )

    result = GranSabioSupervisor().review_session(
        session_id="session-1",
        tracker=tracker,
        request=ContentRequest(prompt="Write a valid article prompt"),
        session={},
    )

    assert result["health_state"] == "attention"
    assert result["recommended_action"] == "review_bench_recommendations"
    assert result["bench_recommendations"][0]["replacement"] is None
    assert result["bench_actions"] == []


def test_supervisor_tracks_technical_failures_separately(tmp_path):
    tracker = _tracker(tmp_path)
    tracker.record_technical_failure(
        session_id="session-1",
        layer_name="Accuracy",
        model_name="model-a",
        slot_id="qa:0",
        error_type="api_failure",
    )
    tracker.record_technical_failure(
        session_id="session-1",
        layer_name="Accuracy",
        model_name="model-a",
        slot_id="qa:0",
        error_type="timeout",
    )

    result = GranSabioSupervisor().review_session(
        session_id="session-1",
        tracker=tracker,
        request=ContentRequest(prompt="Write a valid article prompt"),
        session={},
    )

    assert result["health_state"] == "degraded"
    assert result["supervisor_flags"][0]["type"] == "technical_failure_pattern"
    assert result["bench_recommendations"] == []
