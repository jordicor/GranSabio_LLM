"""
Tests for preflight_validator.py - Preflight validation utilities.

Sub-Phase 3.4: Tests for the preflight validation system that runs
feasibility analysis before content generation.

Functions tested:
- _normalize_qa_models(): Convert QAModelConfig to strings
- _build_validation_payload(): Build validation request payload
- _extract_json_blob(): Extract JSON from raw output
- _normalise_issues(): Convert raw issues to PreflightIssue
- _analyze_word_count_conflicts(): Detect word count conflicts
- run_preflight_validation(): Main validation function
"""

import pytest
from unittest.mock import Mock, MagicMock, AsyncMock, patch
from typing import List, Dict, Any

# Import functions under test
from preflight_validator import (
    _normalize_qa_models,
    _build_validation_payload,
    _build_validator_prompt,
    _extract_json_blob,
    _normalise_issues,
    _parse_validator_response,
    _analyze_word_count_conflicts,
    run_preflight_validation,
)
from models import ContentRequest, QALayer, PreflightIssue, PreflightResult


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def minimal_request():
    """Minimal ContentRequest for testing."""
    return ContentRequest(
        prompt="Write a test article about software testing",
        content_type="article",
        generator_model="gpt-4o",
        qa_layers=[],
        qa_models=[],
    )


@pytest.fixture
def request_with_qa_layers():
    """ContentRequest with QA layers."""
    return ContentRequest(
        prompt="Write a biography about a scientist",
        content_type="biography",
        generator_model="gpt-4o",
        qa_layers=[
            QALayer(
                name="Accuracy",
                description="Check factual accuracy",
                criteria="All facts must be verifiable",
                min_score=8.0,
                order=1,
            ),
            QALayer(
                name="Style",
                description="Check writing style",
                criteria="Writing should be engaging",
                min_score=7.0,
                order=2,
            ),
        ],
        qa_models=["gpt-4o", "claude-sonnet-4-20250514"],
    )


@pytest.fixture
def request_with_word_count():
    """ContentRequest with word count enforcement."""
    return ContentRequest(
        prompt="Write a short article",
        content_type="article",
        generator_model="gpt-4o",
        min_words=500,
        max_words=1000,
        word_count_enforcement={
            "enabled": True,
            "flexibility_percent": 10,
            "direction": "both",
            "severity": "important",
        },
        qa_layers=[
            QALayer(
                name="Word Count Check",
                description="Verify word count is within limits",
                criteria="Content must be between 500-1000 words",
                min_score=8.0,
                order=1,
            ),
        ],
        qa_models=["gpt-4o"],
    )


@pytest.fixture
def mock_ai_service():
    """Mocked AI service for testing."""
    service = MagicMock()
    service.generate_content = AsyncMock(return_value='{"decision": "proceed", "summary": "OK"}')
    service.generate_content_stream = AsyncMock()
    return service


# ============================================================================
# Tests: _normalize_qa_models
# ============================================================================

class TestNormalizeQaModels:
    """Tests for _normalize_qa_models() function."""

    def test_empty_list_returns_empty(self):
        """Given: Empty list, Then: Returns empty list."""
        result = _normalize_qa_models([])
        assert result == []

    def test_none_returns_empty(self):
        """Given: None input, Then: Returns empty list."""
        result = _normalize_qa_models(None)
        assert result == []

    def test_string_list_returns_same(self):
        """Given: List of strings, Then: Returns same strings."""
        models = ["gpt-4o", "claude-sonnet-4-20250514"]
        result = _normalize_qa_models(models)
        assert result == models

    def test_qa_model_config_extracts_model_name(self):
        """Given: QAModelConfig objects, Then: Extracts model names."""
        mock_config = Mock()
        mock_config.model = "gpt-4o"
        result = _normalize_qa_models([mock_config])
        assert result == ["gpt-4o"]

    def test_mixed_types_handles_all(self):
        """Given: Mixed string and objects, Then: Handles all correctly."""
        mock_config = Mock()
        mock_config.model = "claude-sonnet-4-20250514"
        models = ["gpt-4o", mock_config]
        result = _normalize_qa_models(models)
        assert result == ["gpt-4o", "claude-sonnet-4-20250514"]

    def test_unknown_type_converts_to_string(self):
        """Given: Unknown type without model attr, Then: Converts to string."""
        result = _normalize_qa_models([123, 456])
        assert result == ["123", "456"]


# ============================================================================
# Tests: _build_validation_payload (8 tests)
# ============================================================================

class TestBuildValidationPayload:
    """Tests for _build_validation_payload() function."""

    def test_minimal_request_includes_required_fields(self, minimal_request):
        """Given: Minimal request, Then: Payload has required fields."""
        payload = _build_validation_payload(minimal_request)

        assert "prompt" in payload
        assert "content_type" in payload
        assert "generator_model" in payload
        assert "qa_models" in payload
        assert "qa_layers" in payload
        assert payload["prompt"] == minimal_request.prompt
        assert payload["content_type"] == "article"
        assert payload["generator_model"] == "gpt-4o"

    def test_qa_layers_serialized_correctly(self, request_with_qa_layers):
        """Given: Request with QA layers, Then: Layers serialized properly."""
        payload = _build_validation_payload(request_with_qa_layers)

        assert len(payload["qa_layers"]) == 2
        assert payload["qa_layers"][0]["name"] == "Accuracy"
        assert payload["qa_layers"][0]["criteria"] == "All facts must be verifiable"
        assert payload["qa_layers"][1]["name"] == "Style"

    def test_word_count_info_included(self, request_with_word_count):
        """Given: Request with word limits, Then: word_count in payload."""
        payload = _build_validation_payload(request_with_word_count)

        assert "word_count" in payload
        assert payload["word_count"]["min_words"] == 500
        assert payload["word_count"]["max_words"] == 1000
        assert "word_count_enforcement" in payload["word_count"]

    def test_context_documents_included(self, minimal_request):
        """Given: Context documents, Then: Included in payload."""
        context_docs = [
            {"filename": "doc1.txt", "size_bytes": 1000},
            {"filename": "doc2.txt", "size_bytes": 2000},
        ]
        payload = _build_validation_payload(minimal_request, context_documents=context_docs)

        assert "context_documents" in payload
        assert len(payload["context_documents"]) == 2
        assert payload["context_documents_total_bytes"] == 3000

    def test_image_info_included(self, minimal_request):
        """Given: Image info, Then: Included in payload."""
        image_info = {
            "count": 2,
            "total_estimated_tokens": 1000,
            "filenames": ["img1.png", "img2.jpg"],
            "generator_supports_vision": True,
        }
        payload = _build_validation_payload(minimal_request, image_info=image_info)

        assert "images" in payload
        assert payload["images"]["count"] == 2
        assert payload["images"]["generator_supports_vision"] is True

    def test_source_text_included_when_present(self):
        """Given: Request with source_text, Then: Included in payload."""
        request = ContentRequest(
            prompt="Summarize this text",
            content_type="article",
            generator_model="gpt-4o",
            source_text="This is the source text to summarize.",
            qa_layers=[],
            qa_models=[],
        )
        payload = _build_validation_payload(request)

        assert "source_text" in payload
        assert payload["source_text"] == "This is the source text to summarize."

    def test_qa_layer_dict_handled(self, minimal_request):
        """Given: QA layer as dict, Then: Serialized correctly."""
        minimal_request.qa_layers = [
            {"name": "Test Layer", "description": "Test", "criteria": "Test criteria", "min_score": 7.0}
        ]
        payload = _build_validation_payload(minimal_request)

        assert len(payload["qa_layers"]) == 1
        assert payload["qa_layers"][0]["name"] == "Test Layer"

    def test_empty_context_not_included(self, minimal_request):
        """Given: No context documents, Then: Not in payload."""
        payload = _build_validation_payload(minimal_request)

        assert "context_documents" not in payload
        assert "images" not in payload


# ============================================================================
# Tests: _extract_json_blob (8 tests)
# ============================================================================

class TestExtractJsonBlob:
    """Tests for _extract_json_blob() function."""

    def test_valid_json_object_returned(self):
        """Given: Valid JSON object, Then: Returns the JSON string."""
        raw = '{"decision": "proceed", "summary": "OK"}'
        result = _extract_json_blob(raw)
        assert result == raw

    def test_json_with_markdown_fences_extracted(self):
        """Given: JSON wrapped in markdown fences, Then: Extracts JSON."""
        raw = '```json\n{"decision": "proceed"}\n```'
        result = _extract_json_blob(raw)
        assert result == '{"decision": "proceed"}'

    def test_json_with_generic_fences_extracted(self):
        """Given: JSON wrapped in generic fences, Then: Extracts JSON."""
        raw = '```\n{"decision": "reject"}\n```'
        result = _extract_json_blob(raw)
        assert result == '{"decision": "reject"}'

    def test_json_with_surrounding_text_extracted(self):
        """Given: JSON with text before/after, Then: Extracts JSON."""
        raw = 'Here is my response: {"decision": "proceed"} Hope this helps!'
        result = _extract_json_blob(raw)
        assert result == '{"decision": "proceed"}'

    def test_nested_json_extracted(self):
        """Given: Nested JSON object, Then: Extracts full object."""
        raw = '{"outer": {"inner": "value"}, "array": [1, 2, 3]}'
        result = _extract_json_blob(raw)
        assert result == raw

    def test_empty_string_returns_none(self):
        """Given: Empty string, Then: Returns None."""
        result = _extract_json_blob("")
        assert result is None

    def test_none_input_returns_none(self):
        """Given: None input, Then: Returns None."""
        result = _extract_json_blob(None)
        assert result is None

    def test_invalid_json_returns_none(self):
        """Given: Invalid JSON, Then: Returns None."""
        raw = "This is not JSON at all"
        result = _extract_json_blob(raw)
        assert result is None


# ============================================================================
# Tests: _normalise_issues (6 tests)
# ============================================================================

class TestNormaliseIssues:
    """Tests for _normalise_issues() function."""

    def test_valid_issue_converted(self):
        """Given: Valid issue dict, Then: Creates PreflightIssue."""
        raw_issues = [
            {
                "code": "contradiction",
                "severity": "critical",
                "message": "Conflicting requirements detected",
                "blockers": True,
                "related_requirements": ["Accuracy", "Fiction"],
            }
        ]
        result = _normalise_issues(raw_issues)

        assert len(result) == 1
        assert isinstance(result[0], PreflightIssue)
        assert result[0].code == "contradiction"
        assert result[0].severity == "critical"
        assert result[0].message == "Conflicting requirements detected"
        assert result[0].blockers is True
        assert result[0].related_requirements == ["Accuracy", "Fiction"]

    def test_alternative_field_names_handled(self):
        """Given: Alternative field names, Then: Parsed correctly."""
        raw_issues = [
            {
                "type": "warning_issue",
                "severity": "warning",
                "details": "Minor issue found",
                "requirements": ["Style"],
            }
        ]
        result = _normalise_issues(raw_issues)

        assert len(result) == 1
        assert result[0].code == "warning_issue"
        assert result[0].message == "Minor issue found"
        assert result[0].related_requirements == ["Style"]

    def test_missing_message_skips_issue(self):
        """Given: Issue without message, Then: Skipped."""
        raw_issues = [
            {"code": "no_message", "severity": "info"}
        ]
        result = _normalise_issues(raw_issues)

        assert len(result) == 0

    def test_invalid_severity_defaults_to_critical(self):
        """Given: Invalid severity, Then: Defaults based on blockers."""
        raw_issues = [
            {
                "code": "test",
                "severity": "invalid_severity",
                "message": "Test message",
                "blockers": True,
            }
        ]
        result = _normalise_issues(raw_issues)

        assert result[0].severity == "critical"

    def test_non_list_input_returns_empty(self):
        """Given: Non-list input, Then: Returns empty list."""
        result = _normalise_issues("not a list")
        assert result == []

        result = _normalise_issues(None)
        assert result == []

    def test_related_requirements_normalizes_types(self):
        """Given: Mixed types in related_requirements, Then: All converted to strings."""
        raw_issues = [
            {
                "message": "Test",
                "related_requirements": ["Layer1", 123, 4.5],
            }
        ]
        result = _normalise_issues(raw_issues)

        assert result[0].related_requirements == ["Layer1", "123", "4.5"]


# ============================================================================
# Tests: _analyze_word_count_conflicts (5 tests)
# ============================================================================

class TestAnalyzeWordCountConflicts:
    """Tests for _analyze_word_count_conflicts() function."""

    def test_no_word_limits_returns_no_conflicts(self, minimal_request):
        """Given: No word limits, Then: No conflicts detected."""
        result = _analyze_word_count_conflicts(minimal_request)

        assert result["enable_algorithmic_word_count"] is False
        assert result["duplicate_layers_to_remove"] == []
        assert "No word count enforcement" in result["analysis_reason"]

    def test_word_limits_without_enforcement_no_conflicts(self):
        """Given: Word limits but no enforcement, Then: No conflicts."""
        request = ContentRequest(
            prompt="Test prompt with enough characters for validation",
            content_type="article",
            generator_model="gpt-4o",
            min_words=500,
            max_words=1000,
            word_count_enforcement={"enabled": False},
            qa_layers=[],
            qa_models=[],
        )
        result = _analyze_word_count_conflicts(request)

        assert result["enable_algorithmic_word_count"] is False

    def test_detects_word_count_related_layers(self, request_with_word_count):
        """Given: Word count layer with enforcement, Then: Conflict detected."""
        result = _analyze_word_count_conflicts(request_with_word_count)

        assert result["enable_algorithmic_word_count"] is True
        assert "Word Count Check" in result["duplicate_layers_to_remove"]
        assert len(result["conflicting_layers_found"]) == 1

    def test_detects_length_related_keywords(self):
        """Given: Layer with length-related criteria, Then: Detected."""
        request = ContentRequest(
            prompt="Test prompt with enough characters for validation",
            content_type="article",
            generator_model="gpt-4o",
            min_words=500,
            max_words=1000,
            word_count_enforcement={"enabled": True, "flexibility_percent": 10},
            qa_layers=[
                QALayer(
                    name="Brevity Check",
                    description="Ensure content is not too long",
                    criteria="Keep the length under control",
                    min_score=7.0,
                    order=1,
                )
            ],
            qa_models=["gpt-4o"],
        )
        result = _analyze_word_count_conflicts(request)

        assert result["enable_algorithmic_word_count"] is True
        assert "Brevity Check" in result["duplicate_layers_to_remove"]

    def test_non_word_count_layers_not_removed(self):
        """Given: Non-word-count layers, Then: Not flagged for removal."""
        request = ContentRequest(
            prompt="Test prompt with enough characters for validation",
            content_type="article",
            generator_model="gpt-4o",
            min_words=500,
            max_words=1000,
            word_count_enforcement={"enabled": True, "flexibility_percent": 10},
            qa_layers=[
                QALayer(
                    name="Accuracy",
                    description="Check factual accuracy",
                    criteria="All facts must be correct",
                    min_score=8.0,
                    order=1,
                )
            ],
            qa_models=["gpt-4o"],
        )
        result = _analyze_word_count_conflicts(request)

        assert result["enable_algorithmic_word_count"] is False
        assert result["duplicate_layers_to_remove"] == []


# ============================================================================
# Tests: _parse_validator_response
# ============================================================================

class TestParseValidatorResponse:
    """Tests for _parse_validator_response() function."""

    def test_valid_json_parsed(self):
        """Given: Valid JSON response, Then: Returns parsed dict."""
        raw = '{"decision": "proceed", "summary": "OK"}'
        result = _parse_validator_response(raw)

        assert result == {"decision": "proceed", "summary": "OK"}

    def test_invalid_json_returns_none(self):
        """Given: Invalid JSON, Then: Returns None."""
        result = _parse_validator_response("not valid json")
        assert result is None

    def test_empty_returns_none(self):
        """Given: Empty string, Then: Returns None."""
        result = _parse_validator_response("")
        assert result is None


# ============================================================================
# Tests: run_preflight_validation (3 tests)
# ============================================================================

class TestRunPreflightValidation:
    """Tests for run_preflight_validation() async function."""

    @pytest.mark.asyncio
    async def test_no_model_configured_returns_proceed(self, minimal_request, mock_ai_service):
        """Given: No preflight model configured, Then: Returns proceed."""
        with patch("preflight_validator.config") as mock_config:
            mock_config.PREFLIGHT_VALIDATION_MODEL = None

            result = await run_preflight_validation(
                ai_service=mock_ai_service,
                request=minimal_request,
            )

            assert result.decision == "proceed"
            assert "disabled" in result.user_feedback.lower() or "skipped" in result.summary.lower()

    @pytest.mark.asyncio
    async def test_successful_validation_returns_result(self, minimal_request, mock_ai_service):
        """Given: Successful AI validation, Then: Returns parsed result."""
        mock_ai_service.generate_content = AsyncMock(
            return_value='{"decision": "proceed", "summary": "Request validated", "user_feedback": "Approved", "confidence": 0.95}'
        )

        with patch("preflight_validator.config") as mock_config:
            mock_config.PREFLIGHT_VALIDATION_MODEL = "gpt-4o"
            mock_config.PREFLIGHT_SYSTEM_PROMPT = "You are a validator."

            result = await run_preflight_validation(
                ai_service=mock_ai_service,
                request=minimal_request,
            )

            assert result.decision == "proceed"
            assert result.user_feedback == "Approved"
            assert result.confidence == 0.95

    @pytest.mark.asyncio
    async def test_ai_error_returns_proceed_fallback(self, minimal_request, mock_ai_service):
        """Given: AI service error, Then: Returns proceed as fallback."""
        mock_ai_service.generate_content = AsyncMock(
            side_effect=Exception("API Error")
        )

        with patch("preflight_validator.config") as mock_config:
            mock_config.PREFLIGHT_VALIDATION_MODEL = "gpt-4o"
            mock_config.PREFLIGHT_SYSTEM_PROMPT = "You are a validator."

            result = await run_preflight_validation(
                ai_service=mock_ai_service,
                request=minimal_request,
            )

            assert result.decision == "proceed"
            assert "unavailable" in result.user_feedback.lower()

    @pytest.mark.asyncio
    async def test_unparseable_response_returns_proceed(self, minimal_request, mock_ai_service):
        """Given: Unparseable AI response, Then: Returns proceed as fallback."""
        mock_ai_service.generate_content = AsyncMock(
            return_value="This is not valid JSON at all"
        )

        with patch("preflight_validator.config") as mock_config:
            mock_config.PREFLIGHT_VALIDATION_MODEL = "gpt-4o"
            mock_config.PREFLIGHT_SYSTEM_PROMPT = "You are a validator."

            result = await run_preflight_validation(
                ai_service=mock_ai_service,
                request=minimal_request,
            )

            assert result.decision == "proceed"
            assert "unexpected" in result.user_feedback.lower()

    @pytest.mark.asyncio
    async def test_reject_decision_parsed_correctly(self, minimal_request, mock_ai_service):
        """Given: AI returns reject, Then: Result has reject decision."""
        mock_ai_service.generate_content = AsyncMock(
            return_value='{"decision": "reject", "summary": "Contradictions found", "user_feedback": "Cannot proceed", "issues": [{"code": "contradiction", "severity": "critical", "message": "Conflicting requirements"}]}'
        )

        with patch("preflight_validator.config") as mock_config:
            mock_config.PREFLIGHT_VALIDATION_MODEL = "gpt-4o"
            mock_config.PREFLIGHT_SYSTEM_PROMPT = "You are a validator."

            result = await run_preflight_validation(
                ai_service=mock_ai_service,
                request=minimal_request,
            )

            assert result.decision == "reject"
            assert result.user_feedback == "Cannot proceed"
            assert len(result.issues) == 1
            assert result.issues[0].code == "contradiction"

    @pytest.mark.asyncio
    async def test_word_count_analysis_parsed(self, minimal_request, mock_ai_service):
        """Given: AI returns word count analysis, Then: Parsed correctly."""
        mock_ai_service.generate_content = AsyncMock(
            return_value='{"decision": "proceed", "summary": "OK", "user_feedback": "Approved", "word_count_analysis": {"conflicting_layers": ["Word Count"], "recommended_removals": ["Word Count"], "analysis_reason": "Redundant layer"}}'
        )

        with patch("preflight_validator.config") as mock_config:
            mock_config.PREFLIGHT_VALIDATION_MODEL = "gpt-4o"
            mock_config.PREFLIGHT_SYSTEM_PROMPT = "You are a validator."

            result = await run_preflight_validation(
                ai_service=mock_ai_service,
                request=minimal_request,
            )

            assert result.word_count_analysis is not None
            assert result.word_count_analysis.conflicting_layers == ["Word Count"]
            assert result.word_count_analysis.recommended_removals == ["Word Count"]


# ============================================================================
# Tests: _build_validator_prompt
# ============================================================================

class TestBuildValidatorPrompt:
    """Tests for _build_validator_prompt() function."""

    def test_includes_instructions(self):
        """Given: Any payload, Then: Prompt includes instructions."""
        payload = {"prompt": "Test", "content_type": "article"}
        result = _build_validator_prompt(payload)

        assert "Instructions:" in result
        assert "decision" in result.lower()
        assert "proceed" in result.lower() or "reject" in result.lower()

    def test_includes_schema_hint(self):
        """Given: Any payload, Then: Prompt includes schema hint."""
        payload = {"prompt": "Test"}
        result = _build_validator_prompt(payload)

        assert "Output schema" in result
        assert "decision" in result
        assert "summary" in result
        assert "issues" in result

    def test_includes_payload_json(self):
        """Given: Payload with data, Then: Payload serialized in prompt."""
        payload = {"prompt": "Write an article", "content_type": "article"}
        result = _build_validator_prompt(payload)

        assert "Request payload:" in result
        assert "Write an article" in result
        assert '"content_type"' in result
