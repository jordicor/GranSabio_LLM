"""Tests for the segment budgets contract (segment-budgets/v1).

Covers the deterministic ``evaluate_segment_budget_check`` wired into
``validate_generation_candidate``, the post-generation gate helper
``evaluate_segment_budget_gate`` and the synthetic deal-breaker evaluations
that feed the iteration feedback prompt when the gate rejects a candidate.
No provider calls: everything is local and deterministic.

Note: ``core.feedback_formatter`` is loaded by file path instead of as a
package import because ``core/__init__.py`` pulls every route module
(including the native libmagic attachment chain), which is unnecessary for
these tests and broken in some local environments.
"""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from deterministic_validation import (
    SEGMENT_BUDGET_CHECK_NAME,
    build_segment_budget_gate_evaluations,
    evaluate_segment_budget_gate,
    has_active_generation_validators,
    validate_generation_candidate,
)
from models import ContentRequest, QAModelConfig, SegmentBudgetsContract
from word_count_utils import count_words


def _load_feedback_formatter():
    module_path = Path(__file__).resolve().parent.parent / "core" / "feedback_formatter.py"
    spec = importlib.util.spec_from_file_location("gs_feedback_formatter", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_FEEDBACK_FORMATTER = _load_feedback_formatter()


def _contract_dict(**overrides):
    base = {
        "enabled": True,
        "schema_version": "segment-budgets/v1",
        "structure_field": "composition",
        "base_heading_level": 3,
        "severity": "deal_breaker",
    }
    base.update(overrides)
    return base


def _request(segment_budgets=None, **overrides):
    base = {
        "json_output": True,
        "content_type": "json",
        "json_schema": None,
        "json_expectations": None,
        "target_field": "text",
        "min_words": None,
        "max_words": None,
        "word_count_enforcement": None,
        "phrase_frequency": None,
        "lexical_diversity": None,
        "cumulative_text": None,
        "include_stylistic_metrics": False,
        "segment_budgets": segment_budgets,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _words(count):
    return " ".join(f"w{i}" for i in range(count))


def _heading(level, title):
    return f"{'#' * level} {title}"


def _section_text(level, title, body_words):
    return f"{_heading(level, title)}\n\n{_words(body_words)}"


def _budget(min_words, target_words, max_words):
    return {
        "min_words": min_words,
        "target_words": target_words,
        "max_words": max_words,
    }


def _section(title, min_words, target_words, max_words, subsection_id=None, parent=None):
    return {
        "title": title,
        "subsection_id": subsection_id,
        "parent_subsection_id": parent,
        "word_budget": _budget(min_words, target_words, max_words),
    }


def _payload(text, document_budget, sections, structure_field="composition", **extra):
    structure = {"word_budget": document_budget, "sections": sections}
    structure.update(extra)
    return json.dumps({"text": text, structure_field: structure}, ensure_ascii=False)


def _segment_check(report):
    return report.checks.get("segment_budgets")


def _issue_codes(report):
    return [issue["code"] for issue in report.issues]


# ---------------------------------------------------------------------------
# Opt-in behavior: absent/disabled contract keeps today's behavior
# ---------------------------------------------------------------------------


def test_absent_field_keeps_current_behavior():
    request = _request(segment_budgets=None)

    report = validate_generation_candidate(
        json.dumps({"text": "Uno dos tres."}),
        request,
    )

    assert report.approved is True
    assert "segment_budgets" not in report.checks
    assert evaluate_segment_budget_gate(json.dumps({"text": "Uno dos tres."}), request) is None


def test_disabled_contract_keeps_current_behavior():
    request = _request(segment_budgets=_contract_dict(enabled=False))

    report = validate_generation_candidate(
        json.dumps({"text": "Uno dos tres."}),
        request,
    )

    assert report.approved is True
    assert "segment_budgets" not in report.checks


def test_invalid_json_candidate_is_skipped():
    request = _request(segment_budgets=_contract_dict())

    report = validate_generation_candidate(
        '{"text": "abc", "composition": {',
        request,
    )

    assert report.checks["json_output"]["passed"] is False
    assert "segment_budgets" not in report.checks


def test_segment_budgets_model_instance_is_supported():
    request = _request(segment_budgets=SegmentBudgetsContract())
    text = _words(25)

    report = validate_generation_candidate(
        _payload(text, _budget(10, 20, 40), []),
        request,
    )

    assert _segment_check(report)["passed"] is True
    assert report.approved is True


# ---------------------------------------------------------------------------
# Structure validation (segment_budget_structure_invalid)
# ---------------------------------------------------------------------------


def test_missing_structure_field_fails_hard():
    request = _request(segment_budgets=_contract_dict())

    report = validate_generation_candidate(
        json.dumps({"text": _words(20)}),
        request,
    )

    assert report.approved is False
    assert report.hard_failed is True
    assert _issue_codes(report) == ["segment_budget_structure_invalid"]
    assert _segment_check(report)["deal_breaker"] is True


def test_sections_must_be_a_list():
    request = _request(segment_budgets=_contract_dict())
    content = json.dumps({
        "text": _words(20),
        "composition": {"word_budget": _budget(10, 20, 40), "sections": "nope"},
    })

    report = validate_generation_candidate(content, request)

    assert "segment_budget_structure_invalid" in _issue_codes(report)


def test_document_budget_requires_integer_triple():
    request = _request(segment_budgets=_contract_dict())
    content = json.dumps({
        "text": _words(20),
        "composition": {
            "word_budget": {"min_words": 10, "max_words": 40},
            "sections": [],
        },
    })

    report = validate_generation_candidate(content, request)

    assert "segment_budget_structure_invalid" in _issue_codes(report)


def test_document_budget_rejects_incoherent_bounds():
    request = _request(segment_budgets=_contract_dict())

    report = validate_generation_candidate(
        _payload(_words(20), _budget(50, 20, 40), []),
        request,
    )

    assert "segment_budget_structure_invalid" in _issue_codes(report)


def test_section_title_must_be_trimmed_single_line():
    request = _request(segment_budgets=_contract_dict())

    report = validate_generation_candidate(
        _payload(
            _section_text(3, "Seccion Uno", 10),
            _budget(5, 12, 30),
            [_section("Seccion Uno ", 5, 12, 30)],
        ),
        request,
    )

    assert "segment_budget_structure_invalid" in _issue_codes(report)


def test_duplicate_subsection_id_fails():
    request = _request(segment_budgets=_contract_dict())

    report = validate_generation_candidate(
        _payload(
            _words(20),
            _budget(10, 20, 40),
            [
                _section("Seccion Uno", 5, 10, 20, subsection_id="dup"),
                _section("Seccion Dos", 5, 10, 20, subsection_id="dup"),
            ],
        ),
        request,
    )

    assert "segment_budget_structure_invalid" in _issue_codes(report)


def test_parent_must_reference_preceding_top_level_section():
    request = _request(segment_budgets=_contract_dict())

    report = validate_generation_candidate(
        _payload(
            _words(20),
            _budget(10, 20, 40),
            [
                _section("Hija", 5, 10, 20, subsection_id="h1", parent="fantasma"),
                _section("Seccion Uno", 5, 10, 20, subsection_id="s1"),
            ],
        ),
        request,
    )

    assert "segment_budget_structure_invalid" in _issue_codes(report)


def test_depth_beyond_two_levels_fails():
    request = _request(segment_budgets=_contract_dict())

    report = validate_generation_candidate(
        _payload(
            _words(30),
            _budget(10, 30, 60),
            [
                _section("Padre", 5, 20, 40, subsection_id="p"),
                _section("Hija", 2, 10, 20, subsection_id="h", parent="p"),
                _section("Nieta", 1, 10, 20, subsection_id="n", parent="h"),
            ],
        ),
        request,
    )

    assert "segment_budget_structure_invalid" in _issue_codes(report)


def test_children_must_be_contiguous_to_parent():
    request = _request(segment_budgets=_contract_dict())

    report = validate_generation_candidate(
        _payload(
            _words(30),
            _budget(10, 30, 60),
            [
                _section("Padre", 5, 20, 40, subsection_id="p"),
                _section("Otra", 5, 10, 20, subsection_id="o"),
                _section("Hija tardia", 5, 10, 20, subsection_id="h", parent="p"),
            ],
        ),
        request,
    )

    assert "segment_budget_structure_invalid" in _issue_codes(report)


# ---------------------------------------------------------------------------
# Distribution rules (segment_budget_distribution_invalid)
# ---------------------------------------------------------------------------


def test_top_level_targets_must_sum_document_target():
    request = _request(segment_budgets=_contract_dict())

    report = validate_generation_candidate(
        _payload(
            _words(30),
            _budget(10, 30, 60),
            [
                _section("Seccion Uno", 5, 10, 20),
                _section("Seccion Dos", 5, 10, 20),
            ],
        ),
        request,
    )

    assert _issue_codes(report) == ["segment_budget_distribution_invalid"]


def test_top_level_maximums_must_fit_document_maximum():
    request = _request(segment_budgets=_contract_dict())

    report = validate_generation_candidate(
        _payload(
            _words(30),
            _budget(10, 30, 45),
            [
                _section("Seccion Uno", 5, 15, 30),
                _section("Seccion Dos", 5, 15, 30),
            ],
        ),
        request,
    )

    assert _issue_codes(report) == ["segment_budget_distribution_invalid"]


def test_child_targets_must_sum_parent_target():
    request = _request(segment_budgets=_contract_dict())

    report = validate_generation_candidate(
        _payload(
            _words(30),
            _budget(10, 30, 90),
            [
                _section("Padre", 10, 30, 90, subsection_id="p"),
                _section("Hija", 5, 10, 40, subsection_id="h", parent="p"),
            ],
        ),
        request,
    )

    assert _issue_codes(report) == ["segment_budget_distribution_invalid"]


# ---------------------------------------------------------------------------
# Anti-elusion: declared document budget must stay inside request bounds
# ---------------------------------------------------------------------------


def test_document_budget_below_request_min_fails():
    request = _request(segment_budgets=_contract_dict(), min_words=100)

    report = validate_generation_candidate(
        _payload(_words(120), _budget(50, 100, 150), []),
        request,
    )

    assert "segment_budget_document_exceeds_request" in _issue_codes(report)


def test_document_budget_above_request_max_fails():
    request = _request(segment_budgets=_contract_dict(), max_words=200)

    report = validate_generation_candidate(
        _payload(_words(120), _budget(50, 100, 300), []),
        request,
    )

    assert "segment_budget_document_exceeds_request" in _issue_codes(report)


def test_document_budget_inside_request_bounds_passes_anti_elusion():
    request = _request(segment_budgets=_contract_dict(), min_words=50, max_words=200)

    report = validate_generation_candidate(
        _payload(_words(120), _budget(50, 100, 200), []),
        request,
    )

    assert "segment_budget_document_exceeds_request" not in _issue_codes(report)
    assert _segment_check(report)["passed"] is True


# ---------------------------------------------------------------------------
# Heading sequence validation (segment_budget_headings_mismatch)
# ---------------------------------------------------------------------------


def _two_section_request():
    return _request(segment_budgets=_contract_dict())


def _two_section_payload(text):
    return _payload(
        text,
        _budget(20, 40, 90),
        [
            _section("Seccion Uno", 10, 20, 45),
            _section("Seccion Dos", 10, 20, 45),
        ],
    )


def test_headings_wrong_order_fails_with_expected_list():
    text = (
        _section_text(3, "Seccion Dos", 15)
        + "\n\n"
        + _section_text(3, "Seccion Uno", 15)
    )

    report = validate_generation_candidate(_two_section_payload(text), _two_section_request())

    assert _issue_codes(report) == ["segment_budget_headings_mismatch"]
    # The expected ordered (level, title) list must reach the feedback.
    assert "(H3) Seccion Uno" in report.feedback
    assert "(H3) Seccion Dos" in report.feedback
    assert report.feedback.index("(H3) Seccion Uno") < report.feedback.index("(H3) Seccion Dos")


def test_headings_wrong_level_fails():
    text = (
        _section_text(4, "Seccion Uno", 15)
        + "\n\n"
        + _section_text(3, "Seccion Dos", 15)
    )

    report = validate_generation_candidate(_two_section_payload(text), _two_section_request())

    assert _issue_codes(report) == ["segment_budget_headings_mismatch"]


def test_renamed_title_fails():
    text = (
        _section_text(3, "Seccion Renombrada", 15)
        + "\n\n"
        + _section_text(3, "Seccion Dos", 15)
    )

    report = validate_generation_candidate(_two_section_payload(text), _two_section_request())

    assert _issue_codes(report) == ["segment_budget_headings_mismatch"]


def test_extra_heading_fails():
    text = (
        _section_text(3, "Seccion Uno", 15)
        + "\n\n"
        + _section_text(3, "Seccion Dos", 15)
        + "\n\n"
        + _section_text(3, "Seccion Extra", 5)
    )

    report = validate_generation_candidate(_two_section_payload(text), _two_section_request())

    assert _issue_codes(report) == ["segment_budget_headings_mismatch"]


def test_continuous_document_must_have_no_headings():
    request = _request(segment_budgets=_contract_dict())
    text = _section_text(3, "Inesperado", 20)

    report = validate_generation_candidate(
        _payload(text, _budget(10, 20, 40), []),
        request,
    )

    assert _issue_codes(report) == ["segment_budget_headings_mismatch"]
    assert "no headings (continuous document)" in report.feedback


# ---------------------------------------------------------------------------
# Document and section bounds
# ---------------------------------------------------------------------------


def test_document_out_of_bounds_reports_actual_min_max():
    request = _request(segment_budgets=_contract_dict())
    content = _payload(_words(60), _budget(10, 20, 40), [])

    report = validate_generation_candidate(content, request)

    assert _issue_codes(report) == ["segment_budget_document_out_of_bounds"]
    gate = evaluate_segment_budget_gate(content, request)
    document = gate.metadata["document"]
    assert document == {"actual": 60, "min_words": 10, "max_words": 40, "within_budget": False}


def test_only_the_violated_section_fails():
    request = _request(segment_budgets=_contract_dict())
    # Realistic case: sections 129 in 90-180, 175 in 100-200, 149 in 70-140;
    # chapter 453 inside 280-520. Only the third section (149 > 140) fails.
    text = (
        _section_text(3, "Seccion Uno", 127)
        + "\n\n"
        + _section_text(3, "Seccion Dos", 173)
        + "\n\n"
        + _section_text(3, "Seccion Tres", 147)
    )
    content = _payload(
        text,
        _budget(280, 400, 520),
        [
            _section("Seccion Uno", 90, 120, 180),
            _section("Seccion Dos", 100, 150, 200),
            _section("Seccion Tres", 70, 130, 140),
        ],
    )

    report = validate_generation_candidate(content, request)

    assert _segment_check(report)["passed"] is False
    assert _issue_codes(report) == ["segment_budget_section_out_of_bounds"]
    gate = evaluate_segment_budget_gate(content, request)
    sections = gate.metadata["sections"]
    assert [s["within_budget"] for s in sections] == [True, True, False]
    failing = sections[2]
    assert failing["title"] == "Seccion Tres"
    assert failing["actual"] == 149
    assert failing["min_words"] == 70
    assert failing["max_words"] == 140
    assert gate.metadata["document"]["within_budget"] is True
    assert gate.metadata["document"]["actual"] == 453


def test_sections_inside_bounds_with_chapter_inside_pass():
    request = _request(segment_budgets=_contract_dict())
    # Counter-case: sections 164 and 156 within budget, chapter 320 within.
    text = (
        _section_text(3, "Seccion Uno", 162)
        + "\n\n"
        + _section_text(3, "Seccion Dos", 154)
    )
    content = _payload(
        text,
        _budget(200, 270, 520),
        [
            _section("Seccion Uno", 90, 120, 180),
            _section("Seccion Dos", 100, 150, 200),
        ],
    )

    report = validate_generation_candidate(content, request)

    assert _segment_check(report)["passed"] is True
    assert report.approved is True


# ---------------------------------------------------------------------------
# Valid structures
# ---------------------------------------------------------------------------


def test_valid_nested_sections_pass():
    request = _request(segment_budgets=_contract_dict())
    text = (
        _section_text(3, "Seccion Alfa", 10)
        + "\n\n"
        + _section_text(4, "Subseccion Uno", 58)
        + "\n\n"
        + _section_text(4, "Subseccion Dos", 78)
        + "\n\n"
        + _section_text(3, "Seccion Beta", 118)
    )
    content = _payload(
        text,
        _budget(200, 400, 600),
        [
            _section("Seccion Alfa", 80, 240, 360, subsection_id="sec-a"),
            _section("Subseccion Uno", 30, 100, 160, subsection_id="a-1", parent="sec-a"),
            _section("Subseccion Dos", 30, 140, 180, subsection_id="a-2", parent="sec-a"),
            _section("Seccion Beta", 60, 160, 220, subsection_id="sec-b"),
        ],
    )

    report = validate_generation_candidate(content, request)

    assert _segment_check(report)["passed"] is True
    assert report.approved is True
    # Parent length includes its heading and all child sections.
    gate = evaluate_segment_budget_gate(content, request)
    alfa = gate.metadata["sections"][0]
    assert alfa["actual"] == 152


def test_valid_continuous_document_passes():
    request = _request(segment_budgets=_contract_dict())
    content = _payload(_words(25), _budget(10, 20, 40), [])

    report = validate_generation_candidate(content, request)

    assert _segment_check(report)["passed"] is True
    assert report.metrics["segment_budgets"]["sections_checked"] == 0
    assert report.approved is True


def test_extra_fields_in_structure_are_ignored():
    request = _request(segment_budgets=_contract_dict())
    section = _section("Seccion Uno", 5, 12, 30, subsection_id="s1")
    section["block_ids"] = ["b1", "b2"]
    section["future_field"] = {"anything": True}

    report = validate_generation_candidate(
        _payload(
            _section_text(3, "Seccion Uno", 10),
            _budget(5, 12, 30),
            [section],
            schema_version="chapter_composition.v2",
            chapter_block_ids=[],
            structure_mode="subsections",
        ),
        request,
    )

    assert _segment_check(report)["passed"] is True


def test_custom_structure_field_and_base_heading_level():
    request = _request(
        segment_budgets=_contract_dict(structure_field="plan", base_heading_level=2),
    )
    text = _section_text(2, "Seccion Uno", 10)

    report = validate_generation_candidate(
        _payload(text, _budget(5, 12, 30), [_section("Seccion Uno", 5, 12, 30)], structure_field="plan"),
        request,
    )

    assert _segment_check(report)["passed"] is True


# ---------------------------------------------------------------------------
# Word count parity with Biography (\w+ over the whole text, headings included)
# ---------------------------------------------------------------------------


def test_word_count_parity_unicode_punctuation_and_markdown():
    # Raw criterion parity with Biography: \w+ runs over the whole text,
    # headings and markdown emphasis included, punctuation ignored.
    assert count_words("### Título Niño\n\nCafé, té; über... **fin**!") == 6

    text = "Café, té; über... **fin**!"
    assert count_words(text) == 4

    request = _request(segment_budgets=_contract_dict())
    passing = validate_generation_candidate(
        _payload(text, _budget(4, 4, 4), []),
        request,
    )
    assert _segment_check(passing)["passed"] is True

    failing_content = _payload(text, _budget(1, 2, 3), [])
    failing = validate_generation_candidate(failing_content, request)
    assert _segment_check(failing)["passed"] is False
    assert _issue_codes(failing) == ["segment_budget_document_out_of_bounds"]
    gate = evaluate_segment_budget_gate(failing_content, request)
    assert gate.metadata["document"]["actual"] == 4


def test_heading_words_count_towards_section_and_document():
    # A two-word title contributes 2 words to both the section and the
    # document measurement; budgets that ignore headings would misfire here.
    request = _request(segment_budgets=_contract_dict())
    text = _section_text(3, "Seccion Uno", 8)
    content = _payload(
        text,
        _budget(10, 10, 10),
        [_section("Seccion Uno", 10, 10, 10)],
    )

    report = validate_generation_candidate(content, request)

    assert _segment_check(report)["passed"] is True
    gate = evaluate_segment_budget_gate(content, request)
    assert gate.metadata["sections"][0]["actual"] == 10
    assert gate.metadata["document"]["actual"] == 10


# ---------------------------------------------------------------------------
# Post-generation gate helpers
# ---------------------------------------------------------------------------


def _violating_content():
    text = _section_text(3, "Seccion Uno", 100)
    return _payload(
        text,
        _budget(5, 12, 120),
        [_section("Seccion Uno", 5, 12, 30)],
    )


def test_gate_returns_failed_check_for_violating_candidate():
    request = _request(segment_budgets=_contract_dict())

    gate = evaluate_segment_budget_gate(_violating_content(), request)

    assert gate is not None
    assert gate.passed is False
    assert gate.deal_breaker is True
    assert gate.name == SEGMENT_BUDGET_CHECK_NAME
    assert all(issue.hard for issue in gate.issues)


def test_gate_rechecks_the_current_text_after_an_edit():
    request = ContentRequest.model_validate({
        "prompt": "Generate a short report about the supplied facts.",
        "json_output": True,
        "target_field": "text",
        "segment_budgets": {},
    })
    declared = _budget(5, 12, 120)
    sections = [_section("Overview", 5, 12, 30)]
    original = _payload(_section_text(3, "Overview", 10), declared, sections)
    edited = _payload(_section_text(3, "Overview", 100), declared, sections)
    assert evaluate_segment_budget_gate(original, request).passed
    assert not evaluate_segment_budget_gate(edited, request).passed


def test_gate_rejects_unparseable_current_json():
    request = ContentRequest.model_validate({
        "prompt": "Generate a short report about the supplied facts.",
        "json_output": True,
        "target_field": "text",
        "segment_budgets": {},
    })
    gate = evaluate_segment_budget_gate('{"text":', request)
    assert gate is not None and not gate.passed and gate.deal_breaker


def test_gate_synthetic_evaluations_feed_iteration_feedback_prompt():
    request = _request(segment_budgets=_contract_dict(), qa_models=["qa-a", "qa-b"])

    gate = evaluate_segment_budget_gate(_violating_content(), request)
    evaluations = build_segment_budget_gate_evaluations(gate, request.qa_models)

    assert set(evaluations) == {"qa-a", "qa-b"}
    evaluation = evaluations["qa-a"]
    assert evaluation.model == "Arithmos-Prime (Algorithmic)"
    assert evaluation.layer == SEGMENT_BUDGET_CHECK_NAME
    assert evaluation.deal_breaker is True
    assert evaluation.passes_score is False
    assert evaluation.score == 0.0
    assert "segment-budgets/v1" in evaluation.feedback

    # The synthetic qa_results must surface through the standard deal-breaker
    # extraction used by build_iteration_feedback_prompt (no LLM involved).
    qa_results = {SEGMENT_BUDGET_CHECK_NAME: evaluations}
    details = _FEEDBACK_FORMATTER._extract_deal_breaker_details(qa_results)

    assert f"[{SEGMENT_BUDGET_CHECK_NAME}]" in details
    assert "Segment budget validation failed" in details
    assert "Critical Issue:" in details

    actionable = _FEEDBACK_FORMATTER._fallback_actionable_feedback(qa_results)
    assert any(SEGMENT_BUDGET_CHECK_NAME in item for item in actionable)


# ---------------------------------------------------------------------------
# ContentRequest integration / mixed-version behavior
# ---------------------------------------------------------------------------


def test_content_request_accepts_segment_budgets_with_defaults():
    request = ContentRequest.model_validate({
        "prompt": "Genera un texto de prueba.",
        "json_output": True,
        "target_field": "text",
        "segment_budgets": {"enabled": True},
    })

    assert isinstance(request.segment_budgets, SegmentBudgetsContract)
    assert request.segment_budgets.schema_version == "segment-budgets/v1"
    assert request.segment_budgets.structure_field == "composition"
    assert request.segment_budgets.base_heading_level == 3
    assert request.segment_budgets.severity == "deal_breaker"
    assert has_active_generation_validators(request)
    assert not validate_generation_candidate(_violating_content(), request).approved


def test_real_qa_model_configs_are_valid_gate_keys():
    request = ContentRequest.model_validate({
        "prompt": "Genera un texto de prueba.",
        "json_output": True,
        "target_field": "text",
        "segment_budgets": {},
        "qa_models": [{"model": "qa-a"}, {"model": "qa-b"}],
    })
    gate = evaluate_segment_budget_gate(_violating_content(), request)
    evaluations = build_segment_budget_gate_evaluations(gate, request.qa_models)
    assert set(evaluations) == {"qa-a", "qa-b"}
    assert all(isinstance(model, QAModelConfig) for model in request.qa_models)


@pytest.mark.parametrize("extra", [
    {},
    {"json_output": True},
    {"json_output": True, "target_field": "composition"},
    {"json_output": True, "target_field": ["text", "notes"]},
    {"json_output": True, "target_field": "text", "segment_budgets": {"base_heading_level": 6}},
])
def test_active_contract_rejects_invalid_request_configuration(extra):
    with pytest.raises(ValueError):
        ContentRequest.model_validate({
            "prompt": "Genera un texto de prueba.",
            "segment_budgets": {},
            **extra,
        })


def test_content_request_defaults_segment_budgets_to_none():
    request = ContentRequest.model_validate({"prompt": "Genera un texto de prueba."})

    assert request.segment_budgets is None


def test_unknown_fields_are_ignored_like_an_old_server():
    # An old server without the field silently drops it (extra="ignore" is the
    # pydantic default; ContentRequest does not override it), so a new client
    # talking to an old server gets today's behavior with no validation.
    request = ContentRequest.model_validate({
        "prompt": "Genera un texto de prueba.",
        "segment_budgets_v9_future": {"enabled": True},
    })

    assert not hasattr(request, "segment_budgets_v9_future")
    assert request.segment_budgets is None
