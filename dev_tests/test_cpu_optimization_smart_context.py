"""Regression checks for CPU optimizations that must preserve edit and count results."""

import math
import re

import pytest

import context_budget as cb
from smart_edit.models import TargetMode, TextTarget
from smart_edit.operations import SmartTextEditor
from smart_edit.segment_map import SegmentMap, TextNode, build_segment_map
from word_count_utils import count_words


def test_prompt_map_preserves_node_order_escaping_and_empty_paragraphs():
    nodes = [
        TextNode("p2s2", "sentence", 0, 0, " Second & last. ", "p2"),
        TextNode("p2", "paragraph", 0, 0, "unused paragraph text"),
        TextNode("p1", "paragraph", 0, 0, " <fallback> "),
        TextNode("p2s1", "sentence", 0, 0, " First <item>. ", "p2"),
        TextNode("orphan", "sentence", 0, 0, "Do not render.", "missing"),
    ]
    segment_map = SegmentMap("", nodes)

    assert segment_map.render_for_prompt() == (
        '<draft_map>\n'
        '  <p id="p2">\n'
        '    <s id="p2s2">Second &amp; last.</s>\n'
        '    <s id="p2s1">First &lt;item&gt;.</s>\n'
        '  </p>\n'
        '  <p id="p1">\n'
        '    &lt;fallback&gt;\n'
        '  </p>\n'
        '</draft_map>'
    )
    segment_map.nodes.append(TextNode("p1s1", "sentence", 0, 0, "Added.", "p1"))
    assert '    <s id="p1s1">Added.</s>' in segment_map.render_for_prompt()
    assert "&lt;fallback&gt;" not in segment_map.render_for_prompt()


def test_repeated_sentence_id_still_resolves_the_requested_paragraph():
    text = "The same phrase.\n\nThe same phrase."
    segment_map = build_segment_map(text)
    resolution = segment_map.resolve_target_ids(["p2s1"], "The same phrase.")

    assert resolution.quote_match
    assert not resolution.invalid_ids
    assert len(resolution.spans) == 1
    assert resolution.spans[0].start == text.rfind("The same phrase.")
    assert segment_map.resolve_target_ids(["p2s1"], "Unrelated quote.").rejected_reason


@pytest.mark.parametrize(
    ("content", "pattern", "occurrence", "case_sensitive", "expected"),
    [
        ("cat cat cat", "cat", 1, True, (0, 3)),
        ("cat cat cat", "cat", 2, True, (4, 7)),
        ("cat cat cat", "cat", -1, True, (0, 3)),
        ("cat cat cat", "cat", 4, True, None),
        ("no match", "cat", 1, True, None),
        ("ab", "(?=.)", 2, True, (1, 1)),
        ("\u0130 cat CAT", "cat", 2, False, (6, 9)),
        ("abc", "$", 1, True, (3, 3)),
    ],
)
def test_regex_selection_retains_requested_occurrence(
    content, pattern, occurrence, case_sensitive, expected
):
    target = TextTarget(
        TargetMode.REGEX, pattern, occurrence=occurrence, case_sensitive=case_sensitive
    )
    assert SmartTextEditor().locate(content, target) == expected


def test_all_regex_matches_and_counted_replacement_remain_complete():
    editor = SmartTextEditor()
    target = TextTarget(TargetMode.REGEX, "cat", occurrence=-1)

    assert editor.locate_all("cat cat cat", target) == [(0, 3), (4, 7), (8, 11)]
    result = editor.replace("cat cat cat", target, "dog", count=2)
    assert result.success
    assert result.content_after == "dog dog cat"
    zero_width = TextTarget(TargetMode.REGEX, "(?=.)", occurrence=-1)
    assert editor.locate_all("abc", zero_width) == [(0, 0), (1, 1), (2, 2)]


@pytest.mark.parametrize(
    ("content", "occurrence"), [("cat", 1.0), ("cat cat", 1.5)]
)
def test_noninteger_occurrence_retains_list_index_error(content, occurrence):
    target = TextTarget(TargetMode.REGEX, "cat", occurrence=occurrence)
    with pytest.raises(TypeError) as error:
        SmartTextEditor().locate(content, target)
    assert str(error.value) == "list indices must be integers or slices, not float"


@pytest.mark.parametrize(
    ("content", "occurrence", "expected"),
    [("absent", 1.0, None), ("cat", 1.5, None), ("cat", 2.0, None), ("cat", -1.0, (0, 3))],
)
def test_noninteger_occurrence_absence_and_all_sentinel(content, occurrence, expected):
    target = TextTarget(TargetMode.REGEX, "cat", occurrence=occurrence)
    assert SmartTextEditor().locate(content, target) == expected


def test_invalid_regex_still_fails_before_occurrence_selection():
    target = TextTarget(TargetMode.REGEX, "[", occurrence=1.0)
    with pytest.raises(re.error):
        SmartTextEditor().locate("", target)


@pytest.mark.parametrize(
    ("start", "end"),
    [(0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x3040, 0x30FF),
     (0xAC00, 0xD7AF), (0x1100, 0x11FF), (0x20000, 0x2A6DF)],
)
def test_cjk_estimation_keeps_exact_unicode_range_boundaries(start, end):
    assert cb._cjk_fraction(chr(start)) == 1.0
    assert cb._cjk_fraction(chr(end)) == 1.0
    assert cb._cjk_fraction(chr(start - 1)) == 0.0
    assert cb._cjk_fraction(chr(end + 1)) == 0.0


def test_cjk_density_excludes_unicode_whitespace_and_keeps_threshold():
    text = "\u6c34 a\u2003b\tc\nd"
    assert cb._cjk_fraction(text) == 0.2
    assert cb.estimate_tokens(text) == math.ceil(len(text) / cb.RATIO_PROSE)
    assert cb.estimate_tokens(text, "json") == math.ceil(len(text) / cb.RATIO_JSON)
    dense = text + "\u6c34"
    assert cb._cjk_fraction(dense) == 2 / 6
    assert cb.estimate_tokens(dense) == math.ceil(len(dense) / cb.RATIO_CJK)
    assert cb._cjk_fraction(" \t\n\u001c\u0085\u00a0\u2003\u202f\u3000") == 0.0
    assert cb.estimate_tokens("") == 0
    with pytest.raises(TypeError):
        cb.estimate_tokens(None)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (None, 0),
        ("", 0),
        (False, 0),
        (0, 0),
        (" \t\n\u001c\u0085\u00a0\u2003", 0),
        ("Hello, world! 42 _snake_case", 4),
        ("caf\u00e9 cafe\u0301 \u4e2d\u6587 \u0130stanbul", 4),
        ("can't re-enter \U0001f642", 4),
    ],
)
def test_word_count_keeps_unicode_words_and_empty_values(text, expected):
    assert count_words(text) == expected
