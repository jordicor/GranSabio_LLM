"""
Tests for the Smart Edit Analyzer module (Phase 6).

Tests cover:
- Basic pattern matching analysis (no AI)
- AI response parsing
- Action conversion
- Statistics calculation
- Integration with mock AI service
"""

import pytest
from unittest.mock import AsyncMock, MagicMock


class TestAnalyzeBasic:
    """Tests for the analyze_basic method (pattern matching, no AI)."""

    def test_detects_double_words(self):
        """Should detect repeated words like 'very very'."""
        from smart_edit.analyzer import TextAnalyzer

        analyzer = TextAnalyzer.__new__(TextAnalyzer)
        text = "The weather is very very nice today."

        actions = analyzer.analyze_basic(text)

        assert len(actions) >= 1
        assert actions[0]["type"] == "delete"
        assert "very" in actions[0]["target"]["value"].lower()
        assert actions[0]["category"] == "redundancy"
        assert actions[0]["ai_required"] is False

    def test_detects_intensifier_duplications(self):
        """Should detect duplicate intensifiers."""
        from smart_edit.analyzer import TextAnalyzer

        analyzer = TextAnalyzer.__new__(TextAnalyzer)
        text = "This is really really important and quite quite obvious."

        actions = analyzer.analyze_basic(text)

        assert len(actions) >= 2
        categories = [a["category"] for a in actions]
        assert all(c == "redundancy" for c in categories)

    def test_detects_redundant_phrases(self):
        """Should detect common redundant phrases."""
        from smart_edit.analyzer import TextAnalyzer

        analyzer = TextAnalyzer.__new__(TextAnalyzer)
        text = "She was born in the year 1985. Due to the fact that she worked hard, she succeeded."

        actions = analyzer.analyze_basic(text)

        # Should find "in the year" and "due to the fact that"
        assert len(actions) >= 2
        descriptions = [a["description"].lower() for a in actions]
        assert any("year" in d or "simplify" in d for d in descriptions)

    def test_respects_max_issues_limit(self):
        """Should respect the max_issues parameter."""
        from smart_edit.analyzer import TextAnalyzer

        analyzer = TextAnalyzer.__new__(TextAnalyzer)
        text = "very very very very word word text text test test"

        actions = analyzer.analyze_basic(text, max_issues=2)

        assert len(actions) <= 2

    def test_empty_text_returns_empty_list(self):
        """Should return empty list for empty text."""
        from smart_edit.analyzer import TextAnalyzer

        analyzer = TextAnalyzer.__new__(TextAnalyzer)

        actions = analyzer.analyze_basic("")

        assert actions == []

    def test_clean_text_returns_empty_list(self):
        """Should return empty list for text without issues."""
        from smart_edit.analyzer import TextAnalyzer

        analyzer = TextAnalyzer.__new__(TextAnalyzer)
        text = "This is a clean sentence without any issues."

        actions = analyzer.analyze_basic(text)

        assert len(actions) == 0


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

    def test_returns_empty_for_invalid_json(self):
        """Should return empty list for invalid JSON."""
        from smart_edit.analyzer import TextAnalyzer

        analyzer = TextAnalyzer.__new__(TextAnalyzer)
        response = "This is not valid JSON"

        issues = analyzer._parse_response(response)

        assert issues == []

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
        issues = [{"target": "very ", "type": "delete", "description": "Remove duplicate", "category": "redundancy"}]
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

    def test_skips_target_not_found_in_text(self):
        """Should skip issues where target not found in text."""
        from smart_edit.analyzer import TextAnalyzer

        analyzer = TextAnalyzer.__new__(TextAnalyzer)
        issues = [{"target": "nonexistent text", "type": "delete", "description": "Remove", "category": "redundancy"}]
        text = "This is completely different text."

        actions = analyzer._issues_to_actions(issues, text)

        assert len(actions) == 0

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
