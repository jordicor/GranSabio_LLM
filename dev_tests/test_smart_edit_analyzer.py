"""
Tests for the Smart Edit Analyzer module (Phase 6).

Tests cover:
- Explicit analysis failures and full admitted text scope
- AI response parsing
- Action conversion
- Statistics calculation
- Integration with mock AI service
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


class TestCalculateStats:
    """Tests for the calculate_stats function."""

    def test_calculates_total_actions(self):
        """Should count total actions correctly."""
        from smart_edit.analyzer import calculate_stats

        actions = [
            {"ai_required": False, "estimated_ms": 10, "category": "redundancy"},
            {"ai_required": True, "estimated_ms": 2500, "category": "style"},
            {"ai_required": False, "estimated_ms": 10, "category": "redundancy"},
        ]

        stats = calculate_stats(actions)

        assert stats["total_actions"] == 3

    def test_separates_direct_and_ai_actions(self):
        """Should correctly count direct vs AI actions."""
        from smart_edit.analyzer import calculate_stats

        actions = [
            {"ai_required": False, "estimated_ms": 10, "category": "redundancy"},
            {"ai_required": True, "estimated_ms": 2500, "category": "style"},
            {"ai_required": False, "estimated_ms": 10, "category": "grammar"},
        ]

        stats = calculate_stats(actions)

        assert stats["direct_actions"] == 2
        assert stats["ai_actions"] == 1

    def test_calculates_estimated_time(self):
        """Should sum estimated execution times."""
        from smart_edit.analyzer import calculate_stats

        actions = [
            {"ai_required": False, "estimated_ms": 10, "category": "a"},
            {"ai_required": True, "estimated_ms": 2500, "category": "b"},
            {"ai_required": False, "estimated_ms": 15, "category": "c"},
        ]

        stats = calculate_stats(actions)

        assert stats["estimated_total_ms"] == 2525

    def test_counts_categories(self):
        """Should count actions by category."""
        from smart_edit.analyzer import calculate_stats

        actions = [
            {"ai_required": False, "estimated_ms": 10, "category": "redundancy"},
            {"ai_required": True, "estimated_ms": 2500, "category": "style"},
            {"ai_required": False, "estimated_ms": 10, "category": "redundancy"},
            {"ai_required": False, "estimated_ms": 10, "category": "grammar"},
        ]

        stats = calculate_stats(actions)

        assert stats["categories"]["redundancy"] == 2
        assert stats["categories"]["style"] == 1
        assert stats["categories"]["grammar"] == 1

    def test_empty_actions_returns_zero_stats(self):
        """Should return zero stats for empty actions list."""
        from smart_edit.analyzer import calculate_stats

        stats = calculate_stats([])

        assert stats["total_actions"] == 0
        assert stats["direct_actions"] == 0
        assert stats["ai_actions"] == 0
        assert stats["estimated_total_ms"] == 0


class TestParseResponse:
    """Tests for the _parse_response method."""

    def test_parses_valid_json(self):
        """Should parse valid JSON response."""
        from smart_edit.analyzer import TextAnalyzer

        analyzer = TextAnalyzer.__new__(TextAnalyzer)
        response = '{"issues": [{"target": "test", "type": "delete"}]}'

        issues = analyzer._parse_response(response)

        assert len(issues) == 1
        assert issues[0]["target"] == "test"

    def test_parses_json_with_markdown_fence(self):
        """Should handle JSON wrapped in markdown code blocks."""
        from smart_edit.analyzer import TextAnalyzer

        analyzer = TextAnalyzer.__new__(TextAnalyzer)
        response = '```json\n{"issues": [{"target": "test", "type": "delete"}]}\n```'

        issues = analyzer._parse_response(response)

        assert len(issues) == 1
        assert issues[0]["target"] == "test"

    @pytest.mark.parametrize("response", ["This is not valid JSON", '{}', '{"issues": null}'])
    def test_rejects_invalid_analysis_response(self, response):
        """Invalid or missing analysis is distinct from an empty issues list."""
        from smart_edit.analyzer import TextAnalysisError, TextAnalyzer

        analyzer = TextAnalyzer.__new__(TextAnalyzer)

        with pytest.raises(TextAnalysisError):
            analyzer._parse_response(response)
        assert analyzer._parse_response('{"issues": []}') == []

    def test_handles_list_format(self):
        """Should handle response that is just a list."""
        from smart_edit.analyzer import TextAnalyzer

        analyzer = TextAnalyzer.__new__(TextAnalyzer)
        response = '[{"target": "test", "type": "replace"}]'

        issues = analyzer._parse_response(response)

        assert len(issues) == 1

    def test_extracts_json_from_text(self):
        """Should extract JSON embedded in other text."""
        from smart_edit.analyzer import TextAnalyzer

        analyzer = TextAnalyzer.__new__(TextAnalyzer)
        response = 'Here is the analysis:\n{"issues": [{"target": "x", "type": "delete"}]}\nThank you!'

        issues = analyzer._parse_response(response)

        assert len(issues) == 1


class TestIssuesToActions:
    """Tests for the _issues_to_actions method."""

    def test_converts_delete_issue(self):
        """Should convert delete issue to action."""
        from smart_edit.analyzer import TextAnalyzer

        analyzer = TextAnalyzer.__new__(TextAnalyzer)
        issues = [{"target": "very very", "type": "delete", "description": "Remove duplicate", "category": "redundancy"}]
        text = "This is very very nice."

        actions = analyzer._issues_to_actions(issues, text)

        assert len(actions) == 1
        assert actions[0]["type"] == "delete"
        assert actions[0]["ai_required"] is False
        assert actions[0]["estimated_ms"] == 10

    def test_converts_replace_issue(self):
        """Should convert replace issue to action with content."""
        from smart_edit.analyzer import TextAnalyzer

        analyzer = TextAnalyzer.__new__(TextAnalyzer)
        issues = [{
            "target": "in the year",
            "type": "replace",
            "replacement": "in",
            "description": "Simplify",
            "category": "style"
        }]
        text = "Born in the year 1985."

        actions = analyzer._issues_to_actions(issues, text)

        assert len(actions) == 1
        assert actions[0]["type"] == "replace"
        assert actions[0]["content"] == "in"
        assert actions[0]["ai_required"] is False

    def test_converts_format_issue(self):
        """Should convert format issue with format_type."""
        from smart_edit.analyzer import TextAnalyzer

        analyzer = TextAnalyzer.__new__(TextAnalyzer)
        issues = [{
            "target": "Important",
            "type": "format",
            "format_type": "bold",
            "description": "Emphasize",
            "category": "formatting"
        }]
        text = "This is Important information."

        actions = analyzer._issues_to_actions(issues, text)

        assert len(actions) == 1
        assert actions[0]["type"] == "format"
        assert actions[0]["metadata"]["format_type"] == "bold"
        assert actions[0]["ai_required"] is False

    def test_converts_rephrase_issue(self):
        """Should convert rephrase issue with AI required."""
        from smart_edit.analyzer import TextAnalyzer

        analyzer = TextAnalyzer.__new__(TextAnalyzer)
        issues = [{
            "target": "awkward sentence here",
            "type": "rephrase",
            "description": "Make clearer",
            "category": "style"
        }]
        text = "This is an awkward sentence here."

        actions = analyzer._issues_to_actions(issues, text)

        assert len(actions) == 1
        assert actions[0]["type"] == "rephrase"
        assert actions[0]["ai_required"] is True
        assert actions[0]["estimated_ms"] == 2500

    @pytest.mark.parametrize("target", ["nonexistent text", ""])
    def test_rejects_target_not_found_in_text(self, target):
        """An unverifiable issue cannot become a successful empty analysis."""
        from smart_edit.analyzer import TextAnalysisError, TextAnalyzer

        analyzer = TextAnalyzer.__new__(TextAnalyzer)
        issues = [{"target": target, "type": "delete", "description": "Remove", "category": "redundancy"}]
        text = "This is completely different text."

        with pytest.raises(TextAnalysisError):
            analyzer._issues_to_actions(issues, text)

    def test_generates_sequential_ids(self):
        """Should generate sequential action IDs."""
        from smart_edit.analyzer import TextAnalyzer

        analyzer = TextAnalyzer.__new__(TextAnalyzer)
        issues = [
            {"target": "a", "type": "delete", "description": "1", "category": "x"},
            {"target": "b", "type": "delete", "description": "2", "category": "x"},
            {"target": "c", "type": "delete", "description": "3", "category": "x"},
        ]
        text = "a b c"

        actions = analyzer._issues_to_actions(issues, text)

        assert actions[0]["id"] == "act-001"
        assert actions[1]["id"] == "act-002"
        assert actions[2]["id"] == "act-003"


class TestTextAnalyzerIntegration:
    """Integration tests with mock AI service."""

    @pytest.mark.asyncio
    async def test_analyze_with_mock_ai_service(self):
        """Should work with mock AI service."""
        from smart_edit.analyzer import TextAnalyzer

        # Create mock AI service
        mock_response = MagicMock()
        mock_response.content = '{"issues": [{"target": "very very", "type": "delete", "description": "Remove duplicate", "category": "redundancy"}]}'

        mock_ai_service = MagicMock()
        mock_ai_service.generate = AsyncMock(return_value=mock_response)

        analyzer = TextAnalyzer(mock_ai_service)
        text = "This is very very nice."

        actions = await analyzer.analyze(text, model="gpt-4o-mini")

        assert len(actions) >= 1
        mock_ai_service.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyze_filters_by_categories(self):
        """Should filter actions by categories."""
        from smart_edit.analyzer import TextAnalyzer

        mock_response = MagicMock()
        mock_response.content = '''{"issues": [
            {"target": "a", "type": "delete", "description": "1", "category": "redundancy"},
            {"target": "b", "type": "replace", "replacement": "x", "description": "2", "category": "style"},
            {"target": "c", "type": "delete", "description": "3", "category": "grammar"}
        ]}'''

        mock_ai_service = MagicMock()
        mock_ai_service.generate = AsyncMock(return_value=mock_response)

        analyzer = TextAnalyzer(mock_ai_service)

        actions = await analyzer.analyze(
            "a b c",
            model="gpt-4o-mini",
            categories=["redundancy"]
        )

        assert len(actions) == 1
        assert actions[0]["category"] == "redundancy"

    @pytest.mark.asyncio
    async def test_analyze_respects_max_issues(self):
        """Should limit number of returned issues."""
        from smart_edit.analyzer import TextAnalyzer

        mock_response = MagicMock()
        mock_response.content = '''{"issues": [
            {"target": "a", "type": "delete", "description": "1", "category": "x"},
            {"target": "b", "type": "delete", "description": "2", "category": "x"},
            {"target": "c", "type": "delete", "description": "3", "category": "x"},
            {"target": "d", "type": "delete", "description": "4", "category": "x"},
            {"target": "e", "type": "delete", "description": "5", "category": "x"}
        ]}'''

        mock_ai_service = MagicMock()
        mock_ai_service.generate = AsyncMock(return_value=mock_response)

        analyzer = TextAnalyzer(mock_ai_service)

        actions = await analyzer.analyze("a b c d e", max_issues=3)

        assert len(actions) <= 3


class TestAnalysisPrompt:
    """Tests for the analysis prompt template."""

    def test_prompt_includes_text(self):
        """Should include the text in the prompt."""
        from smart_edit.analyzer import ANALYSIS_PROMPT

        text = "Sample text to analyze"
        prompt = ANALYSIS_PROMPT.format(text=text, max_issues=10)

        assert text in prompt

    def test_prompt_includes_max_issues(self):
        """Should include max issues limit in prompt."""
        from smart_edit.analyzer import ANALYSIS_PROMPT

        prompt = ANALYSIS_PROMPT.format(text="test", max_issues=15)

        assert "15" in prompt

    def test_prompt_requests_json_format(self):
        """Should request JSON format response."""
        from smart_edit.analyzer import ANALYSIS_PROMPT

        prompt = ANALYSIS_PROMPT.format(text="test", max_issues=10)

        assert "JSON" in prompt
        assert '"issues"' in prompt


@pytest.fixture
def analysis_api(monkeypatch):
    """Exercise the real router with only the AI service boundary simulated."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    import ai_service
    from smart_edit.router import router

    generate = AsyncMock(return_value='{"issues": []}')
    service = SimpleNamespace(generate_content=generate)
    monkeypatch.setattr(ai_service, "get_ai_service", lambda: service)
    app = FastAPI()
    app.include_router(router, prefix="/smart-edit")
    with TestClient(app) as client:
        yield client, generate


@pytest.mark.parametrize("failure", ["provider", "response", "replacement", "capacity"])
def test_analysis_error_is_not_success(analysis_api, failure):
    """The HTTP analysis path keeps provider, parser and capacity failures visible."""
    from context_budget import ContextCapacityUnknown

    client, generate = analysis_api
    if failure == "provider":
        generate.side_effect = RuntimeError("private provider details")
        status, error_code = 502, "smart_edit_analysis_failed"
    elif failure == "response":
        generate.return_value = "No valid JSON was supplied"
        status, error_code = 502, "smart_edit_analysis_invalid"
    elif failure == "replacement":
        generate.return_value = '{"issues": [{"target": "very very cold", "type": "replace"}]}'
        status, error_code = 502, "smart_edit_analysis_invalid"
    else:
        generate.side_effect = ContextCapacityUnknown(call_id="smart_edit.analyze")
        status, error_code = 400, "context_capacity_unknown"

    response = client.post("/smart-edit/analyze", json={"text": "very very cold"})

    assert response.status_code == status
    assert response.json()["detail"]["error_code"] == error_code
    assert "actions" not in response.json()
    assert "private provider details" not in response.text
    generate.assert_awaited_once()


def test_analysis_processes_full_accepted_scope(analysis_api):
    """The 50000-character admission boundary is not truncated in the AI prompt."""
    import json_utils as json

    client, generate = analysis_api
    tail = "TAIL TO EDIT"
    text = "x" * (50000 - len(tail)) + tail
    generate.return_value = json.dumps({"issues": [{
        "target": tail, "type": "replace", "replacement": "fixed", "category": "grammar",
    }]})

    response = client.post("/smart-edit/analyze", json={"text": text})

    assert response.status_code == 200
    assert text in generate.call_args.kwargs["prompt"]
    assert response.json()["actions"][0]["target"]["value"] == tail
    rejected = client.post("/smart-edit/analyze", json={"text": text + "x"})
    assert rejected.status_code == 422
    generate.assert_awaited_once()


@pytest.fixture
def repeated_edit_analysis():
    """One repeated fragment, disambiguated by its surrounding text."""
    return {
        "text": "very very quiet. Later, very very busy. Banana.",
        "issues": [
            {"target": "very very quiet.", "type": "replace", "replacement": "quiet."},
            {"target": "very very busy.", "type": "replace", "replacement": "busy."},
        ],
    }


@pytest.mark.parametrize("target", ["very very", "VERY VERY", "ana", "ANA"])
def test_analysis_rejects_repeated_fragment(analysis_api, repeated_edit_analysis, target):
    """Neither exact nor casing recovery may silently choose the first match."""
    import json_utils as json

    client, generate = analysis_api
    generate.return_value = json.dumps({"issues": [{"target": target, "type": "delete"}]})

    response = client.post("/smart-edit/analyze", json={"text": repeated_edit_analysis["text"]})

    assert response.status_code == 502
    assert "ambiguous" in response.json()["detail"]["message"]
    assert "actions" not in response.json()


def test_generated_actions_revalidate_current_text(analysis_api, repeated_edit_analysis):
    """Serial edits locate the current text and fail when an earlier edit makes it ambiguous."""
    import json_utils as json

    client, generate = analysis_api
    generate.return_value = json.dumps({"issues": repeated_edit_analysis["issues"]})
    text = repeated_edit_analysis["text"]
    analysis = client.post("/smart-edit/analyze", json={"text": text})
    assert analysis.status_code == 200
    actions = analysis.json()["actions"]

    applied = client.post("/smart-edit/execute-batch", json={"text": text, "actions": actions})

    assert applied.status_code == 200
    assert applied.json()["success"] is True
    assert applied.json()["text_final"] == "quiet. Later, busy. Banana."

    actions[0]["content"] = actions[1]["target"]["value"]
    ambiguous = client.post("/smart-edit/execute-batch", json={"text": text, "actions": actions})

    assert ambiguous.status_code == 200
    result = ambiguous.json()
    assert result["success"] is False
    assert result["text_final"] == "very very busy. Later, very very busy. Banana."
    assert result["results"][0]["success"] is True
    assert result["results"][1]["success"] is False
    assert "ambiguous" in result["results"][1]["error"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
