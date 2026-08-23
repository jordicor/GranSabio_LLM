"""Focused endpoint identity propagation coverage for routed auxiliary calls."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import evidence_grounding.budget_scorer as budget_scorer_module
import evidence_grounding.claim_extractor as claim_extractor_module
import evidence_grounding.evidence_matcher as evidence_matcher_module
import smart_edit.analyzer as analyzer_module
import smart_edit.operations as operations_module
from evidence_grounding.route_identity import endpoint_id_for_effective_model
from smart_edit.analyzer import TextAnalyzer
from smart_edit.operations import SmartTextEditor


class _RecordingAI:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def generate_content(self, **kwargs: object) -> str:
        self.calls.append(dict(kwargs))
        return '{"issues": []}'


@pytest.mark.asyncio
async def test_smart_analyzer_forwards_route_endpoint_only_for_the_route_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = SimpleNamespace(
        model="route-model",
        endpoint_id="spark-a",
        params={},
        provider_options={},
    )
    monkeypatch.setattr(analyzer_module, "resolve_call", lambda *_args, **_kwargs: route)
    monkeypatch.setattr(
        type(analyzer_module.config),
        "resolve_output_max_tokens",
        lambda *_args, **_kwargs: {"max_tokens": 64},
    )
    ai = _RecordingAI()
    analyzer = TextAnalyzer(ai)

    await analyzer.analyze("Text to inspect.", llm_routing={})
    await analyzer.analyze("Text to inspect.", model="override-model", llm_routing={})

    assert ai.calls[0]["endpoint_id"] == "spark-a"
    assert "endpoint_id" not in ai.calls[1]


@pytest.mark.asyncio
async def test_smart_editor_inherits_endpoint_only_for_its_route_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = SimpleNamespace(
        model="route-model",
        endpoint_id="spark-a",
        params={},
        provider_options={},
    )
    monkeypatch.setattr(operations_module, "resolve_call", lambda *_args, **_kwargs: route)
    monkeypatch.setattr(
        type(operations_module.config),
        "resolve_output_max_tokens",
        lambda *_args, **_kwargs: {"max_tokens": 64},
    )
    ai = _RecordingAI()
    editor = SmartTextEditor(ai_service=ai)

    await editor._call_ai("Rewrite this.", llm_routing={})
    await editor._call_ai("Rewrite this.", model="override-model", llm_routing={})
    await editor._call_ai(
        "Rewrite this.",
        model="override-model",
        llm_routing={},
        endpoint_id="spark-explicit",
    )

    assert ai.calls[0]["endpoint_id"] == "spark-a"
    assert "endpoint_id" not in ai.calls[1]
    assert ai.calls[2]["endpoint_id"] == "spark-explicit"


@pytest.mark.parametrize(
    "module",
    [claim_extractor_module, evidence_matcher_module, budget_scorer_module],
)
def test_all_evidence_calls_share_the_safe_route_endpoint_rule(module: object) -> None:
    route = SimpleNamespace(model="route-model", endpoint_id="spark-a")

    assert getattr(module, "endpoint_id_for_effective_model") is endpoint_id_for_effective_model
    assert endpoint_id_for_effective_model(route, "route-model") == "spark-a"
    assert endpoint_id_for_effective_model(route, "override-model") is None
