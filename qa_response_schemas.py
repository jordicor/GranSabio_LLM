"""
JSON schemas for QA evaluator responses.

There are two contracts:
- QA_SCHEMA_SIMPLE: minimal fields for non-editable content types.
- QA_SCHEMA_EDITABLE: includes smart-edit fields for editable content types.

Provider-facing schemas intentionally avoid provider-fragile validation keywords
such as numeric min/max. Runtime code clamps and validates values after parsing.
"""

QA_SCHEMA_SIMPLE = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "score": {"type": "number"},
        "feedback": {"type": "string"},
        "deal_breaker": {"type": "boolean"},
        "deal_breaker_reason": {"type": ["string", "null"]},
    },
    "required": [
        "score",
        "feedback",
        "deal_breaker",
        "deal_breaker_reason",
    ],
}


QA_SCHEMA_EDITABLE = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "score": {"type": "number"},
        "feedback": {"type": "string"},
        "deal_breaker": {"type": "boolean"},
        "deal_breaker_reason": {"type": ["string", "null"]},
        "editable": {"type": ["boolean", "null"]},
        "edit_strategy": {
            "type": ["string", "null"],
            "enum": ["incremental", "regenerate", None],
            "description": "Controlled values: incremental, regenerate, or null.",
        },
        "edit_groups": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "target_ids": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                    },
                    "target_id": {"type": ["string", "null"]},
                    "evidence_quote": {"type": ["string", "null"]},
                    "paragraph_start": {"type": ["string", "null"]},
                    "paragraph_end": {"type": ["string", "null"]},
                    "start_word_index": {"type": ["integer", "null"]},
                    "end_word_index": {"type": ["integer", "null"]},
                    "operation_type": {
                        "type": ["string", "null"],
                        "enum": [
                            "delete",
                            "remove",
                            "replace",
                            "rephrase",
                            "add_before",
                            "add_after",
                            "insert_before",
                            "insert_after",
                            "fix_grammar",
                            "fix_style",
                            "improve",
                            "expand",
                            "condense",
                            None,
                        ],
                    },
                    "instruction": {"type": ["string", "null"]},
                    "severity": {"type": ["string", "null"]},
                    "exact_fragment": {"type": ["string", "null"]},
                    "suggested_text": {"type": ["string", "null"]},
                },
                "required": [
                    "target_ids",
                    "target_id",
                    "evidence_quote",
                    "paragraph_start",
                    "paragraph_end",
                    "start_word_index",
                    "end_word_index",
                    "operation_type",
                    "instruction",
                    "severity",
                    "exact_fragment",
                    "suggested_text",
                ],
            },
        },
    },
    "required": [
        "score",
        "feedback",
        "deal_breaker",
        "deal_breaker_reason",
        "editable",
        "edit_strategy",
        "edit_groups",
    ],
}
