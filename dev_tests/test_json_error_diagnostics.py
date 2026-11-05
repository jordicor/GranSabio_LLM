"""Safe local JSON diagnostics; offline, no provider calls or credentials."""

import json

import pytest

from provider_errors import ProviderErrorKind, classify_provider_exception
from tool_loop_models import JsonContractError, parse_json_with_markdown_fences
from tools.ai_json_cleanroom import ErrorCode, ValidationIssue, ValidationResult, validate_ai_json


@pytest.mark.parametrize("content,schema,code,category,truncated", [
    ('{"x": @}', None, "parse_error", "syntax", False),
    ('{"x": "unfinished', None, "parse_error", "syntax", True),
    ('{"x": 1, "x": 2}', None, "duplicate_key", "syntax", False),
    ('{}', {"required": ["private_name"]}, "missing_required", "schema", False),
    ('{"x": "private_value"}', {"properties": {"x": {"type": "integer"}}}, "type_mismatch", "schema", False),
    ('{"x": "private_value"}', {"properties": {"x": {"enum": ["private_option"]}}}, "enum_mismatch", "schema", False),
])
def test_json_failure_has_safe_reason_without_changing_rejection(content, schema, code, category, truncated):
    with pytest.raises(JsonContractError) as raised:
        parse_json_with_markdown_fences(content, schema=schema)
    error = raised.value
    diagnostics = error.json_validation
    assert diagnostics["issues"][0]["code"] == code
    assert diagnostics["issues"][0]["category"] == category
    assert diagnostics["likely_truncated"] is truncated
    assert diagnostics["error_count"] == 1
    assert diagnostics["issues_omitted"] == 0
    serialized = json.dumps(diagnostics) + str(error)
    assert "private_" not in serialized
    assert "unfinished" not in serialized
    # Existing identity and retry/downgrade semantics survive classification.
    error.request_id = "synthetic-request-id"
    failure = classify_provider_exception(error, provider="openai", model_id="test", operation="generation",
                                         attempt=2, max_attempts=3, attempted_feature="response_format")
    assert failure.kind is ProviderErrorKind.UNKNOWN
    assert failure.retryable is False and failure.downgradable is False
    assert failure.raw_exception_class == "JsonContractError"
    assert failure.request_id == "synthetic-request-id"
    assert (failure.attempt, failure.max_attempts) == (2, 3)
    assert failure.json_validation == diagnostics


def test_diagnostics_mask_schema_and_dynamic_keys_and_bound_output():
    secret = "private_network_unavailable"
    schema = {"type": "object", "additionalProperties": {"type": "array", "items": {
        "properties": {secret: {"const": "private_expected"}},
    }}}
    result = validate_ai_json({secret: [{secret: "private_value"}] * 20}, schema=schema)
    diagnostics = result.public_diagnostics()
    assert diagnostics["issues"][0]["path"] == "$.*[*].*"
    assert diagnostics["issues"][0]["path_redacted"] is True
    assert diagnostics["error_count"] == 20 and diagnostics["issues_omitted"] == 12
    assert len(diagnostics["issues"]) == 8
    assert "private_" not in json.dumps(diagnostics)


@pytest.mark.parametrize("path", ["$.private[123456].name", '$["private key"]', "$." + "private." * 30, "private", "$." + "x" * 2000])
def test_untrusted_issue_text_never_enters_public_diagnostics(path):
    result = ValidationResult(False, False, errors=[
        ValidationIssue(ErrorCode.ENUM_MISMATCH, path, "private message", {"value": "private detail"}),
        ValidationIssue("private_code", path, "private message"),
    ], data={"private": "data"}, info={"private": "info"})
    diagnostics = result.public_diagnostics()
    assert diagnostics["issues"][1]["code"] == "unknown"
    assert "private" not in json.dumps(diagnostics)
    assert "123456" not in json.dumps(diagnostics)


def test_success_and_legacy_errors_have_no_new_failure_behavior():
    assert parse_json_with_markdown_fences('```json\n{"score": 9}\n```') == {"score": 9}
    assert str(JsonContractError("legacy message")) == "legacy message"
    assert JsonContractError("legacy message").json_validation is None
    failure = classify_provider_exception(RuntimeError("old failure"), provider="test", model_id="test", operation="test")
    assert failure.json_validation is None


@pytest.mark.parametrize("wrapped", [False, True])
def test_qa_failure_metadata_retains_safe_diagnostics(wrapped):
    from ai_service import AIRequestError
    from core.generation_processor import _attach_qa_provider_failure
    from qa_result_utils import normalize_qa_technical_failure_metadata

    cause = JsonContractError("JSON contract failed.", validation=validate_ai_json('{"private": @}'))
    error = AIRequestError("test", "test", 1, 1, cause) if wrapped else cause
    metadata = normalize_qa_technical_failure_metadata(
        error_type="model_unavailable", original_exception=error,
    )
    assert metadata["json_validation"] == cause.json_validation
    assert metadata["retryable"] is False
    assert "private" not in json.dumps(metadata)
    outer = RuntimeError("QA model unavailable")
    outer.__cause__ = error
    session, final = {}, {}
    _attach_qa_provider_failure(session, final, outer)
    for payload in (session, final):
        assert (payload["provider_error"] if wrapped else payload)["json_validation"] == cause.json_validation
