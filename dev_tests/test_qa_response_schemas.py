import pytest

from qa_response_schemas import QA_SCHEMA_EDITABLE, QA_SCHEMA_SIMPLE
from schema_utils import json_schema_to_pydantic


def test_qa_schema_simple_pydantic_conversion():
    model = json_schema_to_pydantic(QA_SCHEMA_SIMPLE, model_name="QASimpleModel")

    valid = {
        "score": 8.5,
        "feedback": "Good overall",
        "deal_breaker": False,
        "deal_breaker_reason": None,
    }
    assert model.model_validate(valid) is not None

    with pytest.raises(Exception):
        model.model_validate({
            "score": 8.5,
            "feedback": "x",
            "deal_breaker": False,
        })


def test_qa_schema_editable_pydantic_conversion():
    model = json_schema_to_pydantic(QA_SCHEMA_EDITABLE, model_name="QAEditableModel")

    payload = {
        "score": 8.5,
        "feedback": "Needs minor edits",
        "deal_breaker": False,
        "deal_breaker_reason": None,
        "editable": True,
        "edit_strategy": "incremental",
        "edit_groups": [
            {
                "target_ids": None,
                "target_id": None,
                "evidence_quote": None,
                "paragraph_start": "The opening paragraph starts",
                "paragraph_end": "with a concrete marker",
                "start_word_index": None,
                "end_word_index": None,
                "operation_type": "replace",
                "instruction": "Tighten phrasing",
                "severity": "minor",
                "exact_fragment": "The opening paragraph starts with a concrete marker",
                "suggested_text": "The opening is more direct",
            },
            {
                "target_ids": ["p2s1"],
                "target_id": None,
                "evidence_quote": "Sentence copied from p2s1",
                "paragraph_start": None,
                "paragraph_end": None,
                "start_word_index": None,
                "end_word_index": None,
                "operation_type": "replace",
                "instruction": "Fix local wording",
                "severity": "minor",
                "exact_fragment": "Sentence copied from p2s1",
                "suggested_text": "Improved sentence copied from p2s1",
            },
            {
                "target_ids": None,
                "target_id": None,
                "evidence_quote": None,
                "paragraph_start": None,
                "paragraph_end": None,
                "start_word_index": 42,
                "end_word_index": 57,
                "operation_type": "rephrase",
                "instruction": "Rewrite this word-index span for clarity",
                "severity": "major",
                "exact_fragment": None,
                "suggested_text": None,
            },
        ],
    }

    assert model.model_validate(payload) is not None


def test_qa_schema_editable_rejects_whole_string_edit_groups():
    model = json_schema_to_pydantic(QA_SCHEMA_EDITABLE, model_name="QAEditableModel")

    with pytest.raises(Exception):
        model.model_validate({
            "score": 8.5,
            "feedback": "x",
            "deal_breaker": False,
            "deal_breaker_reason": None,
            "editable": True,
            "edit_strategy": "incremental",
            "edit_groups": ["legacy whole-string edit group"],
        })


def test_qa_schema_editable_rejects_unknown_operation_type():
    model = json_schema_to_pydantic(QA_SCHEMA_EDITABLE, model_name="QAEditableModel")

    with pytest.raises(Exception):
        model.model_validate({
            "score": 8.5,
            "feedback": "x",
            "deal_breaker": False,
            "deal_breaker_reason": None,
            "editable": True,
            "edit_strategy": "incremental",
            "edit_groups": [
                {
                    "target_ids": None,
                    "target_id": None,
                    "evidence_quote": None,
                    "paragraph_start": "one two three",
                    "paragraph_end": "four five six",
                    "start_word_index": None,
                    "end_word_index": None,
                    "operation_type": "typo_replace",
                    "instruction": "Fix wording",
                    "severity": "minor",
                    "exact_fragment": "one two three",
                    "suggested_text": "one two done",
                }
            ],
        })
