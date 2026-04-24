from types import SimpleNamespace

from deterministic_validation import prepare_validation_context, validate_generation_candidate
from core.generation_processor import _locate_edit_segment
from smart_edit import build_segment_map
from smart_edit.models import TextEditRange
from smart_edit.qa_integration import parse_qa_edit_groups
from tools.ai_json_cleanroom import make_loose_json_validate_options
from tool_loop_models import PayloadScope
from word_count_utils import create_word_count_qa_layer


def _json_request(**overrides):
    base = {
        "json_output": True,
        "content_type": "json",
        "json_schema": None,
        "json_expectations": None,
        "target_field": None,
        "min_words": None,
        "max_words": None,
        "word_count_enforcement": None,
        "phrase_frequency": None,
        "lexical_diversity": None,
        "cumulative_text": None,
        "include_stylistic_metrics": False,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_prepare_validation_context_extracts_target_field_without_json_validation():
    request = SimpleNamespace(
        json_output=True,
        json_schema=None,
        json_expectations=None,
        target_field="content",
    )

    context = prepare_validation_context(
        '{"content": "Uno dos tres.", "notes": "ignorar"}',
        request,
        include_json_validation=False,
    )

    assert context.text_for_validation == "Uno dos tres."
    assert context.target_field_paths == ["content"]


def test_json_object_without_string_fields_passes_when_no_text_validators():
    request = _json_request()

    report = validate_generation_candidate(
        '{"ids": [1, 2, 3], "meta": {"ok": true}}',
        request,
        json_options=make_loose_json_validate_options(),
    )

    assert report.approved is True
    assert report.word_count == 0
    assert report.checks["json_output"]["passed"] is True
    payload = report.build_visible_payload(PayloadScope.MEASUREMENT_ONLY)
    assert "word_count" not in payload
    assert "word_count" not in payload["metrics"]
    assert "target_field_paths" not in payload["metrics"]["json_output"]


def test_json_array_without_string_fields_passes_when_no_text_validators():
    request = _json_request()

    report = validate_generation_candidate(
        '[{"id": 1}, {"id": 2}]',
        request,
        json_options=make_loose_json_validate_options(),
    )

    assert report.approved is True
    assert report.word_count == 0
    assert report.checks["json_output"]["passed"] is True


def test_json_without_text_fields_fails_only_when_text_validation_is_required():
    request = _json_request(min_words=1)

    report = validate_generation_candidate(
        '{"ids": [1, 2, 3]}',
        request,
        json_options=make_loose_json_validate_options(),
    )

    assert report.approved is False
    assert report.checks["json_output"]["passed"] is False
    assert report.word_count == 0


def test_json_target_field_missing_still_fails():
    request = _json_request(target_field="content")

    report = validate_generation_candidate(
        '{"other": "Uno dos tres."}',
        request,
        json_options=make_loose_json_validate_options(),
    )

    assert report.approved is False
    assert report.checks["json_output"]["passed"] is False


def test_json_target_field_text_can_satisfy_word_count():
    request = _json_request(target_field="content", min_words=3)

    report = validate_generation_candidate(
        '{"content": "Uno dos tres cuatro."}',
        request,
        json_options=make_loose_json_validate_options(),
    )

    assert report.approved is True
    assert report.word_count == 4
    assert report.checks["json_output"]["passed"] is True
    assert report.checks["word_count"]["passed"] is True
    payload = report.build_visible_payload(PayloadScope.MEASUREMENT_ONLY)
    assert payload["word_count"] == 4
    assert payload["metrics"]["word_count"] == 4
    assert payload["metrics"]["target_field_paths"] == ["content"]


def test_content_type_json_alias_activates_json_validation():
    request = _json_request(json_output=False, content_type="json")

    report = validate_generation_candidate(
        '{"ids": [1, 2, 3]}',
        request,
        json_options=make_loose_json_validate_options(),
    )

    assert report.approved is True
    assert "json_output" in report.checks


def test_loose_json_options_extract_embedded_json_like_default_options():
    request = _json_request()
    content = 'prefix {"ids": [1, 2, 3]}'

    default_report = validate_generation_candidate(content, request)
    loose_report = validate_generation_candidate(
        content,
        request,
        json_options=make_loose_json_validate_options(),
    )

    assert default_report.approved is True
    assert loose_report.approved is True
    assert loose_report.checks["json_output"]["passed"] is True


def test_parse_qa_edit_groups_ids_mode_preserves_target_ids_and_quote():
    edit_groups = [
        {
            "target_ids": ["p1s1", "p1s2"],
            "evidence_quote": "Primera frase. Segunda frase.",
            "operation_type": "replace",
            "instruction": "Reescribe con más precisión.",
            "exact_fragment": "Primera frase. Segunda frase.",
            "suggested_text": "Texto revisado.",
            "severity": "major",
        }
    ]

    ranges = parse_qa_edit_groups(edit_groups, marker_mode="ids")

    assert ranges is not None
    assert len(ranges) == 1
    assert ranges[0].marker_mode == "ids"
    assert ranges[0].target_ids == ["p1s1", "p1s2"]
    assert ranges[0].evidence_quote == "Primera frase. Segunda frase."
    assert ranges[0].can_use_direct is True


def test_segment_map_resolves_contiguous_sentence_ids_with_evidence_quote():
    text = "Primera frase. Segunda frase.\n\nTercer bloque."

    segment_map = build_segment_map(text)
    resolution = segment_map.resolve_target_ids(
        ["p1s1", "p1s2"],
        evidence_quote="Primera frase. Segunda frase.",
    )

    assert resolution.invalid_ids == []
    assert resolution.quote_match is True
    assert len(resolution.spans) == 1
    assert resolution.spans[0].node_ids == ["p1s1", "p1s2"]
    assert "Primera frase. Segunda frase." in resolution.spans[0].text


def test_locate_edit_segment_rejects_partial_invalid_target_ids():
    text = "Primera frase. Segunda frase.\n\nTercer bloque."
    edit = TextEditRange(
        marker_mode="ids",
        target_ids=["p1s1", "p9s9"],
        evidence_quote="Primera frase.",
    )

    span = _locate_edit_segment(text, edit, None, None)

    assert span is None


def test_validate_generation_candidate_no_longer_uses_semantic_regex_scaffolding_check():
    request = SimpleNamespace(
        json_output=False,
        target_field=None,
        min_words=700,
        max_words=900,
        word_count_enforcement=None,
        phrase_frequency=None,
        lexical_diversity=None,
        cumulative_text=None,
    )

    report = validate_generation_candidate(
        "El relato actual tiene 693 palabras y está por debajo del mínimo requerido. "
        "Procederé a expandir el texto para cumplir con el requisito de conteo de palabras.\n\n"
        "Después empieza el relato real.",
        request,
    )

    assert report.approved is False
    assert "word_count_scaffolding" not in report.checks


def test_word_count_qa_layer_mentions_meta_compliance_narration():
    layer = create_word_count_qa_layer(
        10,
        100,
        {
            "enabled": True,
            "flexibility_percent": 0,
            "direction": "both",
            "severity": "important",
        },
    )

    assert "word-count compliance" in layer.criteria
    assert "draft" in layer.criteria
    assert "prompt" in layer.criteria
