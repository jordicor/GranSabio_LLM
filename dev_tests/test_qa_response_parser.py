from types import SimpleNamespace

import pytest

from qa_evaluation_service import QAEvaluationService, QAResponseParseError


class _FakeAIService:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def test_parse_qa_json_response_extracts_fenced_json_after_prose():
    service = QAEvaluationService(ai_service=None)

    parsed = service._parse_qa_json_response(
        "Analysis first.\n```json\n"
        '{"score": 8.5, "feedback": "Passed", "deal_breaker": false, '
        '"deal_breaker_reason": null}\n```',
        "claude-sonnet-4-6",
        "AI Pattern Detection",
    )

    assert parsed["score"] == 8.5
    assert parsed["feedback"] == "Passed"
    assert parsed["deal_breaker"] is False


def test_parse_qa_json_response_normalizes_unknown_edit_strategy():
    service = QAEvaluationService(ai_service=None)

    parsed = service._parse_qa_json_response(
        '{"score": 7, "feedback": "Needs work", "deal_breaker": false, '
        '"deal_breaker_reason": null, "editable": true, '
        '"edit_strategy": "word_replace", "edit_groups": []}',
        "gpt-5.2",
        "Style",
    )

    assert parsed["edit_strategy"] is None


def test_parse_qa_json_response_raises_specific_error_when_no_json():
    service = QAEvaluationService(ai_service=None)

    with pytest.raises(QAResponseParseError):
        service._parse_qa_json_response(
            "I will analyze this in prose only.",
            "claude-sonnet-4-6",
            "AI Pattern Detection",
        )


@pytest.mark.asyncio
async def test_evaluate_content_retries_once_without_schema_on_schema_rejection():
    fake_ai = _FakeAIService([
        ValueError("Schema must have a 'type', 'anyOf', 'oneOf', or 'allOf' field."),
        '{"score": 9, "feedback": "Passed", "deal_breaker": false, "deal_breaker_reason": null}',
    ])
    service = QAEvaluationService(fake_ai)

    result = await service.evaluate_content(
        content="Sample content.",
        criteria="Check quality.",
        model="fake-model",
        layer_name="Quality",
        min_score=8.0,
        original_request=SimpleNamespace(content_type="analysis", prompt="Analyze this."),
        request_edit_info=False,
    )

    assert result.score == 9.0
    assert result.passes_score is True
    assert len(fake_ai.calls) == 2
    assert fake_ai.calls[0]["json_output"] is True
    assert fake_ai.calls[0]["json_schema"] is not None
    assert fake_ai.calls[1]["json_output"] is True
    assert fake_ai.calls[1]["json_schema"] is None


@pytest.mark.asyncio
async def test_evaluate_content_technical_error_is_not_deal_breaker():
    fake_ai = _FakeAIService([RuntimeError("provider transport failed")])
    service = QAEvaluationService(fake_ai)

    result = await service.evaluate_content(
        content="Sample content.",
        criteria="Check quality.",
        model="fake-model",
        layer_name="Quality",
        min_score=8.0,
        original_request=SimpleNamespace(content_type="analysis", prompt="Analyze this."),
        request_edit_info=False,
    )

    assert result.score == 0.0
    assert result.passes_score is False
    assert result.deal_breaker is False
    assert result.deal_breaker_reason is None
    assert result.reason == "Technical error during evaluation"
