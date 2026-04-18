from ai_service import AIService
from qa_response_schemas import QA_SCHEMA_EDITABLE, QA_SCHEMA_SIMPLE


def _walk(node):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk(value)


def test_qa_schemas_survive_gemini_provider_transforms():
    for schema in (QA_SCHEMA_SIMPLE, QA_SCHEMA_EDITABLE):
        transformed = AIService._strip_additional_properties(schema)
        transformed = AIService._convert_nullable_to_gemini_format(transformed)

        assert transformed["type"] == "object"
        assert transformed["properties"]
        assert transformed["required"]

        for node in _walk(transformed):
            assert "additionalProperties" not in node
            node_type = node.get("type")
            if isinstance(node_type, list):
                assert "null" not in node_type
