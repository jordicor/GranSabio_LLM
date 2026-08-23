"""Focused offline propagation checks for local-only dispatch context."""

from __future__ import annotations

import sys
import types

import pytest

from evidence_grounding.budget_scorer import BudgetScorer

# ``feedback_memory`` only needs this cancellation handle at import time.  The
# production ``core`` package imports the app state (which imports feedback
# memory), so isolate this focused unit test from that unrelated import cycle.
_previous_core = sys.modules.get("core")
_previous_cancellation = sys.modules.get("core.cancellation")
if _previous_core is None:
    _core = types.ModuleType("core")
    _core.__path__ = []
    sys.modules["core"] = _core
if _previous_cancellation is None:
    _cancellation = types.ModuleType("core.cancellation")
    _cancellation.ProviderCallHandle = object
    sys.modules["core.cancellation"] = _cancellation

try:
    import feedback_memory as feedback_memory_module
    from feedback_memory import FeedbackProcessor
finally:
    if _previous_cancellation is None:
        sys.modules.pop("core.cancellation", None)
    if _previous_core is None:
        sys.modules.pop("core", None)

from smart_edit.operations import SmartTextEditor


class _RecordingAI:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def generate_with_logprobs(self, **kwargs):
        self.calls.append(kwargs)
        return "YES", [{"token": "YES", "logprob": 0.0, "top_logprobs": []}]

    async def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return "edited text"


@pytest.mark.asyncio
async def test_evidence_logprobs_forwards_exact_local_dispatch_context() -> None:
    ai = _RecordingAI()
    runtime = object()

    await BudgetScorer(ai)._get_yes_probability(
        prompt="Does the evidence support this?",
        model="local-model",
        local_execution=runtime,
    )

    assert ai.calls[0]["local_execution"] is runtime
    assert ai.calls[0]["dispatch_call_id"] == "evidence.score_logprobs"
    assert ai.calls[0]["dispatch_route_ordinal"] == 0


@pytest.mark.asyncio
async def test_smart_edit_forwards_local_dispatch_context() -> None:
    ai = _RecordingAI()
    runtime = object()

    result = await SmartTextEditor(ai_service=ai).apply_edit(
        content="Rewrite this paragraph.",
        target="Rewrite this paragraph.",
        instruction="Make the wording more direct.",
        model="local-model",
        endpoint_id="spark-a",
        local_execution=runtime,
        dispatch_call_id="generation.smart_edit",
    )

    assert result.success is True
    assert ai.calls[0]["local_execution"] is runtime
    assert ai.calls[0]["dispatch_call_id"] == "generation.smart_edit"
    assert ai.calls[0]["dispatch_route_ordinal"] == 0
    assert ai.calls[0]["endpoint_id"] == "spark-a"


@pytest.mark.asyncio
async def test_feedback_memory_rejects_local_runtime_before_embedding_http() -> None:
    processor = FeedbackProcessor.__new__(FeedbackProcessor)
    processor.ai_service = _RecordingAI()

    with pytest.raises(RuntimeError, match="local_feedback_memory_unsupported"):
        await processor.get_embeddings(["feedback"], local_execution=object())

    assert processor.ai_service.calls == []


@pytest.mark.asyncio
async def test_feedback_routes_forward_endpoint_only_when_model_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FeedbackAI(_RecordingAI):
        async def generate_content(self, **kwargs):
            self.calls.append(kwargs)
            if kwargs.get("dispatch_call_id") == "feedback.analyze":
                return '{"tiered_summaries": {}, "issues": [], "next_iteration_hint": "Fix it."}'
            return '{"rules": []}'

    route = types.SimpleNamespace(
        model="feedback-local",
        endpoint_id="spark-a",
        params={},
        provider_options={},
    )
    monkeypatch.setattr(feedback_memory_module, "resolve_call", lambda *_args, **_kwargs: route)
    monkeypatch.setattr(feedback_memory_module, "resolve_temperature", lambda *_args, **_kwargs: 0.2)
    monkeypatch.setattr(
        feedback_memory_module,
        "app_config",
        types.SimpleNamespace(resolve_output_max_tokens=lambda *_args, **_kwargs: {"max_tokens": 64}),
    )
    ai = _FeedbackAI()
    processor = FeedbackProcessor.__new__(FeedbackProcessor)
    processor.ai_service = ai
    processor.config = types.SimpleNamespace(
        analysis_model=None,
        analysis_temperature=0.2,
        max_rules=5,
    )

    await processor.extract_feedback_analysis("Needs better chronology.")
    await processor.synthesize_normative_rules(
        [{"canonical_label": "chronology", "occurrences": 2, "actions": ["Add dates."]}]
    )

    assert [call["endpoint_id"] for call in ai.calls] == ["spark-a", "spark-a"]

    processor.config.analysis_model = "override-local"
    await processor.extract_feedback_analysis("Needs better chronology.")

    assert "endpoint_id" not in ai.calls[-1]
