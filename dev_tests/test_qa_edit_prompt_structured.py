from smart_edit import build_qa_edit_prompt
from smart_edit.qa_integration import parse_qa_edit_groups


def test_structured_phrase_prompt_uses_plain_string_markers():
    prompt = build_qa_edit_prompt(
        marker_mode="phrase",
        phrase_length=5,
        include_edit_info=True,
        structured_output=True,
        min_score=8.0,
    )

    assert '"paragraph_start": "<plain 5+ word marker copied from the paragraph start>"' in prompt
    assert '"paragraph_start": {"1":' not in prompt
    assert "paragraph_start and paragraph_end MUST be strings, not objects" in prompt


def test_legacy_phrase_prompt_keeps_counted_marker_contract():
    prompt = build_qa_edit_prompt(
        marker_mode="phrase",
        phrase_length=5,
        include_edit_info=True,
        structured_output=False,
        min_score=8.0,
    )

    assert '"paragraph_start": {"1":' in prompt
    assert "MUST be JSON objects" in prompt


def test_parse_qa_edit_groups_skips_unknown_operation_type():
    ranges = parse_qa_edit_groups(
        [
            {
                "paragraph_start": "one two three",
                "paragraph_end": "four five six",
                "operation_type": "typo_replace",
                "instruction": "Fix wording",
                "severity": "minor",
                "exact_fragment": "one two three",
                "suggested_text": "one two done",
            }
        ],
        marker_mode="phrase",
        marker_length=3,
        model_name="test-model",
    )

    assert ranges is None
