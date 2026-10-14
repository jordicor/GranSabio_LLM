"""Regression coverage for shared lexical preparation and optional position maps."""

from copy import deepcopy
from dataclasses import asdict, replace
from unittest.mock import Mock

import pytest

from tools import lexical_diversity as lexical
from tools import lexical_diversity_utils as lexical_utils
from tools import repetition_analyzer as repetition


@pytest.mark.parametrize(
    "word_count, explicit, decision_labels, expected_windows",
    [
        (1199, False, (), False),
        (1200, False, (), True),
        (1201, False, (), True),
        (12, True, (), True),
        (12, False, ("red",), True),
        (12, False, (), False),
    ],
)
def test_lexical_window_policy_reuses_preparation(
    monkeypatch, word_count, explicit, decision_labels, expected_windows
):
    text = " ".join(["echo"] * word_count)
    settings = lexical_utils.LexicalDiversitySettings(
        enabled=True,
        include_positions=True,
        window_policy=lexical_utils.LexicalDiversityWindowPolicy(
            analyze_windows=explicit,
            auto_window_on_decision=decision_labels,
            window_size=200,
            window_step=100,
            window_include_positions=True,
        ),
    )
    original_settings = deepcopy(settings)
    tokenize = Mock(wraps=lexical.tokenize_with_spans)
    monkeypatch.setattr(lexical, "tokenize_with_spans", tokenize)

    result = lexical_utils.analyze_text_lexical_diversity(text, settings)

    assert tokenize.call_count == 1
    assert settings == original_settings
    assert result.analysis["settings"]["analyze_windows"] is expected_windows
    assert bool(result.analysis["windows"]) is expected_windows
    assert result.analysis["meta"]["total_word_tokens"] == word_count
    explicit_config = replace(settings.build_lexdiv_config(), analyze_windows=expected_windows)
    assert result.analysis == lexical.analyze_lexical_diversity(text, explicit_config)


def test_lexical_windows_preserve_raw_decision_before_metric_overrides():
    settings = lexical_utils.LexicalDiversitySettings(
        enabled=True,
        metrics="c,yulek",
        decision_policy=lexical_utils.LexicalDiversityDecisionPolicy(
            custom_metric_thresholds={
                "c": {"green_min": 0},
                "yulek": {"green_max": 10000},
            },
        ),
    )
    result = lexical_utils.analyze_text_lexical_diversity("echo " * 12, settings)

    assert result.decision_label == "RED"
    assert result.analysis["grades"] == {"c": "RED", "yulek": "RED"}
    assert result.adjusted_grades == {"c": "GREEN", "yulek": "GREEN"}
    assert result.analysis["meta"]["windows_count"] == 1
    assert result.deal_breaker is True
    assert result.score == 3.0


def test_lexical_paragraph_windows_preserve_thresholds_metadata_and_positions():
    text = "Caf\u00e9 alpha alpha.\n\nBeta caf\u00e9 beta.\n\nGamma alpha gamma."
    settings = lexical_utils.LexicalDiversitySettings(
        enabled=True,
        metrics="all",
        language="en",
        strip_accents=True,
        include_positions=True,
        thresholds_overrides={"herdan_green_min": 0.0, "yulek_green_max": 10000.0},
        window_policy=lexical_utils.LexicalDiversityWindowPolicy(
            analyze_windows=True,
            window_mode="paragraphs",
            include_window_metrics=True,
            window_include_positions=True,
            window_preview_chars=8,
        ),
    )
    original_settings = deepcopy(settings)
    expected = lexical.analyze_lexical_diversity(
        text, replace(settings.build_lexdiv_config(), analyze_windows=True)
    )

    result = lexical_utils.analyze_text_lexical_diversity(text, settings)

    assert result.analysis == expected
    assert settings == original_settings
    assert result.analysis["meta"]["windows_count"] == 3
    assert result.analysis["meta"]["stop_words_filtered"] is True
    assert all(window["metrics"] for window in result.analysis["windows"])
    assert all("positions" in word for word in result.analysis["top_words"])


def test_lexical_direct_window_decision_does_not_mutate_or_disable_config():
    text = "alpha beta alpha."
    cfg = lexical.LexDivConfig(top_words_k=2)
    original = asdict(cfg)
    decision = Mock(return_value=True)

    result = lexical.analyze_lexical_diversity(text, cfg, window_decision=decision)

    decision.assert_called_once_with(3, result["decision"]["label"])
    assert asdict(cfg) == original
    assert result["settings"]["analyze_windows"] is True
    explicit = replace(cfg, analyze_windows=True)
    assert lexical.analyze_lexical_diversity(
        text, explicit, window_decision=lambda count, label: False
    ) == result


@pytest.mark.parametrize("respect_sentences", [False, True])
def test_repetition_position_metrics_preserve_boundaries_and_optional_details(
    monkeypatch, respect_sentences
):
    text = "alpha beta. gamma alpha beta.\n\nalpha beta.\n\n\nalpha beta."
    cfg = repetition.AnalysisConfig(
        min_n=2,
        max_n=2,
        algo_mode="counter",
        respect_sentences=respect_sentences,
        details="all",
        diagnostics="full",
        enable_position_metrics=True,
        enable_clusters=True,
        enable_windows=True,
    )
    sentence_segments = Mock(wraps=repetition.compute_sentence_segments)
    monkeypatch.setattr(repetition, "compute_sentence_segments", sentence_segments)

    positioned = repetition.analyze_text(text, cfg)

    assert sentence_segments.call_count == 1
    detail = positioned["phrases"][2][0]
    assert detail["text"] == "alpha beta"
    assert detail["count"] == 4
    assert detail["occurrence_positions"]["starts_tokens_preview"] == [0, 4, 7, 10]
    bias = detail["position_bias"]
    assert bias["sentence"]["start_count"] == 3.0
    assert bias["sentence"]["end_count"] == 4.0
    assert bias["paragraph"]["start_count"] == 3.0
    assert bias["block"]["start_count"] == 2.0

    map_builder = Mock(side_effect=AssertionError("Position maps should remain unused"))
    monkeypatch.setattr(repetition, "token_to_segment_index_map", map_builder)
    monkeypatch.setattr(repetition, "compute_token_segments_from_blanklines", map_builder)
    without_positions = repetition.analyze_text(text, replace(cfg, enable_position_metrics=False))
    assert map_builder.call_count == 0
    assert without_positions["summary"] == positioned["summary"]
    assert without_positions["diagnostics"] == positioned["diagnostics"]
    assert without_positions["meta"] == positioned["meta"]
    assert without_positions["phrases"][2][0] == {
        key: value for key, value in detail.items() if key != "position_bias"
    }


def test_repetition_occurrence_helper_keeps_character_spans():
    text = "alpha beta alpha"
    tokens, spans = repetition.tokenize_with_spans(text)
    occurrences = repetition.collect_occurrences_for_phrases(
        tokens, spans, [(0, len(tokens))], {1: {("alpha",)}}
    )
    occurrence = occurrences[1][("alpha",)]

    assert isinstance(occurrence, repetition.OccurrenceData)
    assert occurrence.starts_tokens == [0, 2]
    assert occurrence.char_starts == [0, 11]
    assert occurrence.char_ends == [5, 16]
