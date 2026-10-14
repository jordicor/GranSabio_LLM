"""
Regression tests for security package AUX-2 (smart_edit).

Each test reproduces the scenario of its finding and would have failed
before the corresponding fix:
- AUX-02: ReDoS via user-supplied regex (validation + off-thread timeout +
  batch action cap)
- AUX-05: locate_by_markers must fail (None) when the start marker is not
  unique instead of editing the first occurrence
- AUX-08: case-insensitive validated actions must carry the exact fragment
  from the original text so they execute
- AUX-09: replace(..., count=N) with N>1 must replace N occurrences
- AUX-15: case-insensitive offsets must stay aligned with the original text
  for Unicode characters whose lowercase form expands (e.g. 'İ')
- AUX-17 (smart_edit items): execute must not leak raw internal errors
"""

import pytest
from pydantic import ValidationError

import smart_edit.router as se_router
from smart_edit.analyzer import TextAnalyzer
from smart_edit.locators import locate_by_markers
from smart_edit.models import TargetMode, TextTarget
from smart_edit.operations import SmartTextEditor
from smart_edit.router import (
    ActionRequest,
    BatchExecuteRequest,
    ExecuteRequest,
    TextTargetRequest,
    execute_single_action,
)

# U+0130 'İ' lowercases to two characters ('i' + combining dot above),
# shifting str.lower() offsets by one for everything after it.
EXPANSIVE_TEXT = "İstanbul loves cats"


# =============================================================================
# AUX-02: ReDoS via user-supplied regex
# =============================================================================


class TestAux02RegexValidation:
    """Nested variable-width quantifiers are rejected at target construction."""

    @pytest.mark.parametrize(
        "pattern",
        [
            r"(a+)+$",
            r"(\w+\s?)+$",
            r"(a{2,4})+",
            r"((ab|c)*d)+",
            r"(\d+)*",
        ],
    )
    def test_nested_quantifiers_rejected(self, pattern):
        with pytest.raises(ValueError, match="catastrophic backtracking"):
            TextTarget(mode=TargetMode.REGEX, value=pattern)

    @pytest.mark.parametrize(
        "pattern",
        [
            r"[A-Z]-\d+",  # no groups at all
            r"(\d{4})+",  # fixed-width inner repeat is safe
            r"(a|b)+",  # alternation without inner quantifier
            r"\d{4}-\d{2}-\d{2}",
            r"\bword\b",
        ],
    )
    def test_safe_patterns_accepted(self, pattern):
        target = TextTarget(mode=TargetMode.REGEX, value=pattern)
        assert target.value == pattern

    def test_api_request_rejects_nested_quantifiers(self):
        """The /smart-edit/execute request model rejects ReDoS patterns (422)."""
        with pytest.raises(ValidationError):
            TextTargetRequest(mode="regex", value="(a+)+$")

    @pytest.mark.asyncio
    async def test_regex_matching_runs_off_loop_with_timeout(self, monkeypatch):
        """Overlapping-alternation patterns are guarded by a hard timeout.

        The timeout is set to zero so it fires before the worker thread can
        deliver its result (a thread result always needs at least one extra
        event-loop iteration), making the guard deterministic without leaving
        a runaway thread behind.
        """
        monkeypatch.setattr(se_router, "REGEX_MATCH_TIMEOUT_SECONDS", 0.0)
        request = ExecuteRequest(
            text="a" * 5000,
            action=ActionRequest(
                type="delete",
                target=TextTargetRequest(mode="regex", value="z+", occurrence=1),
            ),
        )

        response = await execute_single_action(request)

        assert response.success is False
        assert response.error is not None
        assert "timed out" in response.error
        assert response.text_after == request.text

    def test_batch_actions_capped(self):
        """/execute-batch rejects unbounded action lists (AUX-02/AUX-17)."""
        action = ActionRequest(
            type="delete",
            target=TextTargetRequest(mode="exact", value="x"),
        )
        BatchExecuteRequest(
            text="x", actions=[action] * se_router.MAX_BATCH_ACTIONS
        )
        with pytest.raises(ValidationError):
            BatchExecuteRequest(
                text="x",
                actions=[action] * (se_router.MAX_BATCH_ACTIONS + 1),
            )


# =============================================================================
# AUX-05: non-unique start marker must fail localization
# =============================================================================


class TestAux05AmbiguousMarkers:
    def test_non_unique_start_marker_returns_none(self):
        text = (
            "The cat sat on the mat. The dog ran fast.\n\n"
            "The cat sat on the mat again near the door."
        )
        start = {"1": "The", "2": "cat", "3": "sat"}  # appears twice
        end = {"1": "the", "2": "mat."}

        assert locate_by_markers(text, start, end, expected_phrase_length=3) is None

    def test_unique_start_marker_still_locates(self):
        text = "Alpha beta gamma. The cat sat on the mat."
        start = {"1": "The", "2": "cat", "3": "sat"}
        end = {"1": "the", "2": "mat."}

        result = locate_by_markers(text, start, end, expected_phrase_length=3)

        assert result is not None
        start_pos, end_pos = result
        assert text[start_pos:end_pos] == "The cat sat on the mat."


# =============================================================================
# AUX-08: case-insensitive fallback keeps the exact original fragment
# =============================================================================


class TestAux08CaseInsensitiveFallback:
    def test_action_adopts_exact_fragment_from_original(self):
        analyzer = TextAnalyzer.__new__(TextAnalyzer)
        issues = [
            {
                "target": "maria was born",
                "type": "replace",
                "replacement": "Maria was raised",
                "description": "Fix fact",
                "category": "style",
            }
        ]
        text = "Maria was born in Barcelona."

        actions = analyzer._issues_to_actions(issues, text)

        assert len(actions) == 1
        assert actions[0]["target"]["value"] == "Maria was born"
        assert actions[0]["target"]["case_sensitive"] is True

    def test_generated_action_actually_executes(self):
        """Before the fix, the action kept the LLM casing and /execute failed
        with 'Target not found' despite having passed validation."""
        analyzer = TextAnalyzer.__new__(TextAnalyzer)
        issues = [
            {
                "target": "maria was born",
                "type": "delete",
                "description": "Remove",
                "category": "style",
            }
        ]
        text = "Maria was born in Barcelona."

        actions = analyzer._issues_to_actions(issues, text)
        assert len(actions) == 1

        result = SmartTextEditor().delete(text, actions[0]["target"]["value"])
        assert result.success is True
        assert result.content_after == " in Barcelona."


# =============================================================================
# AUX-09: replace(count=N) with N>1 replaces N occurrences
# =============================================================================


class TestAux09ReplaceCount:
    def test_count_two_replaces_two_occurrences(self):
        result = SmartTextEditor().replace("cat cat cat", "cat", "dog", count=2)
        assert result.success is True
        assert result.content_after == "dog dog cat"

    def test_count_one_keeps_previous_behavior(self):
        result = SmartTextEditor().replace("cat cat cat", "cat", "dog", count=1)
        assert result.success is True
        assert result.content_after == "dog cat cat"

    def test_count_minus_one_replaces_all(self):
        result = SmartTextEditor().replace("cat cat cat", "cat", "dog")
        assert result.success is True
        assert result.content_after == "dog dog dog"


# =============================================================================
# AUX-15: case-insensitive offsets aligned with Unicode expansion
# =============================================================================


class TestAux15UnicodeOffsets:
    def test_locate_all_case_insensitive_after_expansive_char(self):
        editor = SmartTextEditor()
        target = TextTarget(
            mode=TargetMode.EXACT,
            value="CATS",
            occurrence=-1,
            case_sensitive=False,
        )

        assert editor.locate_all(EXPANSIVE_TEXT, target) == [(15, 19)]

    def test_delete_case_insensitive_after_expansive_char(self):
        """Before the fix the lowered-text offset cut the edit one character
        off, leaving a mangled document."""
        editor = SmartTextEditor()
        target = TextTarget(
            mode=TargetMode.EXACT,
            value="CATS",
            case_sensitive=False,
        )

        result = editor.delete(EXPANSIVE_TEXT, target)

        assert result.success is True
        assert result.content_after == "İstanbul loves "

    def test_locate_nth_occurrence_case_insensitive(self):
        editor = SmartTextEditor()
        target = TextTarget(
            mode=TargetMode.EXACT,
            value="cats",
            occurrence=2,
            case_sensitive=False,
        )

        assert editor.locate("İstanbul cats loves cats", target) == (20, 24)

    def test_locate_by_markers_after_expansive_char(self):
        text = "İstanbul intro line.\n\nThe cat sat on the mat."
        start = {"1": "the", "2": "cat", "3": "sat"}
        end = {"1": "the", "2": "mat."}

        result = locate_by_markers(text, start, end, expected_phrase_length=3)

        assert result is not None
        start_pos, end_pos = result
        assert text[start_pos:end_pos] == "The cat sat on the mat."


# =============================================================================
# AUX-17 (smart_edit items): no raw internal errors leaked by /execute
# =============================================================================


class TestAux17ErrorHandling:
    @pytest.mark.asyncio
    async def test_internal_errors_return_generic_message(self, monkeypatch):
        def boom(*args, **kwargs):
            raise RuntimeError("secret internal path C:/app/keys.pem")

        monkeypatch.setattr(SmartTextEditor, "delete", boom)
        request = ExecuteRequest(
            text="hello",
            action=ActionRequest(
                type="delete",
                target=TextTargetRequest(mode="exact", value="hello"),
            ),
        )

        response = await execute_single_action(request)

        assert response.success is False
        assert response.error == "Internal error while executing the action"
        assert "keys.pem" not in (response.error or "")

    @pytest.mark.asyncio
    async def test_validation_errors_stay_descriptive(self):
        request = ExecuteRequest(
            text="hello",
            action=ActionRequest(
                type="bogus",
                target=TextTargetRequest(mode="exact", value="hello"),
            ),
        )

        response = await execute_single_action(request)

        assert response.success is False
        assert response.error == "Unknown action type: bogus"
