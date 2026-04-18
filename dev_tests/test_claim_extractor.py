"""
Tests for evidence_grounding/claim_extractor.py - Claim Extraction Module.

Phase 2 of Strawberry Integration: Extracts atomic, verifiable claims from
generated content for evidence grounding verification.

Tests use mocked AI responses to avoid API calls.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import sys
from pathlib import Path

# Add parent directory to path for json_utils import
sys.path.insert(0, str(Path(__file__).parent.parent))
import json_utils as json

# Import module under test
from evidence_grounding.claim_extractor import (
    ClaimExtractor,
    extract_claims,
    _build_extraction_prompt,
    _extract_json_from_response,
    _filter_claims,
    _parse_claims_response,
    CLAIM_EXTRACTION_SCHEMA,
    CLAIM_EXTRACTION_SYSTEM_PROMPT,
)
from models import ExtractedClaim


class TestBuildExtractionPrompt:
    """Tests for _build_extraction_prompt helper function."""

    def test_includes_content_and_context(self):
        """
        Given: Content and context strings
        When: _build_extraction_prompt is called
        Then: Both are included in the prompt
        """
        content = "Generated text about Einstein"
        context = "Evidence about Einstein's life"
        result = _build_extraction_prompt(content, context, max_claims=10)

        assert content in result
        assert context in result

    def test_includes_max_claims_instruction(self):
        """
        Given: max_claims parameter
        When: _build_extraction_prompt is called
        Then: max_claims limit is mentioned in prompt
        """
        result = _build_extraction_prompt("content", "context", max_claims=25)
        assert "25" in result

    def test_returns_string(self):
        """
        Given: Valid inputs
        When: _build_extraction_prompt is called
        Then: Returns a string
        """
        result = _build_extraction_prompt("content", "context", max_claims=10)
        assert isinstance(result, str)
        assert len(result) > 0


class TestExtractJsonFromResponse:
    """Tests for _extract_json_from_response helper function."""

    def test_parses_direct_json(self):
        """
        Given: Direct JSON string
        When: _extract_json_from_response is called
        Then: Returns parsed dictionary
        """
        json_str = '{"claims": []}'
        result = _extract_json_from_response(json_str)
        assert result == {"claims": []}

    def test_extracts_from_markdown_code_block(self):
        """
        Given: JSON wrapped in markdown code block
        When: _extract_json_from_response is called
        Then: Extracts and parses the JSON
        """
        response = '''Here is the extraction:
```json
{"claims": [{"idx": 0, "claim": "test"}]}
```'''
        result = _extract_json_from_response(response)
        assert "claims" in result
        assert result["claims"][0]["claim"] == "test"

    def test_extracts_from_plain_code_block(self):
        """
        Given: JSON in code block without language specifier
        When: _extract_json_from_response is called
        Then: Extracts and parses the JSON
        """
        response = '''```
{"claims": []}
```'''
        result = _extract_json_from_response(response)
        assert result == {"claims": []}

    def test_finds_json_object_in_text(self):
        """
        Given: JSON embedded in text without code block
        When: _extract_json_from_response is called
        Then: Finds and parses the JSON object
        """
        response = 'The result is {"claims": []} and that is all.'
        result = _extract_json_from_response(response)
        assert result == {"claims": []}

    def test_raises_on_invalid_json(self):
        """
        Given: String without valid JSON
        When: _extract_json_from_response is called
        Then: Raises JSONDecodeError
        """
        with pytest.raises(json.JSONDecodeError):
            _extract_json_from_response("no json here at all")


class TestFilterClaims:
    """Tests for _filter_claims helper function."""

    def test_filters_trivial_claims_when_enabled(self):
        """
        Given: Claims including trivial ones and filter_trivial=True
        When: _filter_claims is called
        Then: Trivial claims are removed
        """
        claims = [
            {"idx": 0, "kind": "factual", "importance": 0.8},
            {"idx": 1, "kind": "trivial", "importance": 0.9},
            {"idx": 2, "kind": "inference", "importance": 0.7},
        ]
        result = _filter_claims(claims, filter_trivial=True, min_importance=0.0)

        assert len(result) == 2
        assert all(c["kind"] != "trivial" for c in result)

    def test_keeps_trivial_claims_when_disabled(self):
        """
        Given: Claims including trivial ones and filter_trivial=False
        When: _filter_claims is called
        Then: Trivial claims are kept
        """
        claims = [
            {"idx": 0, "kind": "factual", "importance": 0.8},
            {"idx": 1, "kind": "trivial", "importance": 0.9},
        ]
        result = _filter_claims(claims, filter_trivial=False, min_importance=0.0)

        assert len(result) == 2

    def test_filters_by_importance_threshold(self):
        """
        Given: Claims with varying importance scores
        When: _filter_claims is called with min_importance=0.6
        Then: Only claims with importance >= 0.6 are kept
        """
        claims = [
            {"idx": 0, "kind": "factual", "importance": 0.8},
            {"idx": 1, "kind": "factual", "importance": 0.5},
            {"idx": 2, "kind": "factual", "importance": 0.3},
        ]
        result = _filter_claims(claims, filter_trivial=False, min_importance=0.6)

        assert len(result) == 1
        assert result[0]["importance"] == 0.8

    def test_combines_both_filters(self):
        """
        Given: Various claims
        When: _filter_claims with both filters active
        Then: Both filters are applied
        """
        claims = [
            {"idx": 0, "kind": "factual", "importance": 0.8},  # Keep
            {"idx": 1, "kind": "trivial", "importance": 0.9},  # Filter: trivial
            {"idx": 2, "kind": "factual", "importance": 0.3},  # Filter: importance
            {"idx": 3, "kind": "inference", "importance": 0.7},  # Keep
        ]
        result = _filter_claims(claims, filter_trivial=True, min_importance=0.6)

        assert len(result) == 2
        assert result[0]["idx"] == 0
        assert result[1]["idx"] == 3

    def test_returns_empty_list_when_all_filtered(self):
        """
        Given: Claims that all get filtered
        When: _filter_claims is called
        Then: Returns empty list
        """
        claims = [
            {"idx": 0, "kind": "trivial", "importance": 0.1},
        ]
        result = _filter_claims(claims, filter_trivial=True, min_importance=0.5)
        assert result == []


class TestParseClaimsResponse:
    """Tests for _parse_claims_response function."""

    def test_parses_valid_response(self):
        """
        Given: Valid JSON response with claims
        When: _parse_claims_response is called
        Then: Returns list of ExtractedClaim objects
        """
        response = json.dumps({
            "claims": [
                {
                    "idx": 0,
                    "claim": "Einstein was born in 1879",
                    "kind": "factual",
                    "importance": 0.9,
                    "cited_spans": ["born in 1879"],
                    "source_text": "Albert Einstein was born in 1879"
                }
            ]
        })
        result = _parse_claims_response(response, filter_trivial=False, min_importance=0.0)

        assert len(result) == 1
        assert isinstance(result[0], ExtractedClaim)
        assert result[0].claim == "Einstein was born in 1879"
        assert result[0].kind == "factual"
        assert result[0].importance == 0.9

    def test_applies_filters(self):
        """
        Given: Response with claims of varying importance
        When: _parse_claims_response with filters
        Then: Filters are applied
        """
        response = json.dumps({
            "claims": [
                {"idx": 0, "claim": "Important", "kind": "factual", "importance": 0.9},
                {"idx": 1, "claim": "Unimportant", "kind": "factual", "importance": 0.3},
            ]
        })
        result = _parse_claims_response(response, filter_trivial=True, min_importance=0.6)

        assert len(result) == 1
        assert result[0].claim == "Important"

    def test_handles_missing_optional_fields(self):
        """
        Given: Response with minimal claim data
        When: _parse_claims_response is called
        Then: Uses defaults for missing optional fields
        """
        response = json.dumps({
            "claims": [
                {"idx": 0, "claim": "Test claim", "kind": "factual", "importance": 0.7}
            ]
        })
        result = _parse_claims_response(response, filter_trivial=False, min_importance=0.0)

        assert len(result) == 1
        assert result[0].cited_spans == []
        assert result[0].source_text == ""

    def test_raises_on_invalid_json(self):
        """
        Given: Invalid JSON response
        When: _parse_claims_response is called
        Then: Raises ValueError
        """
        with pytest.raises(ValueError) as exc_info:
            _parse_claims_response("not valid json", filter_trivial=False, min_importance=0.0)
        assert "Invalid JSON" in str(exc_info.value)

    def test_raises_on_invalid_claims_field(self):
        """
        Given: JSON without valid claims list
        When: _parse_claims_response is called
        Then: Raises ValueError
        """
        response = json.dumps({"claims": "not a list"})
        with pytest.raises(ValueError) as exc_info:
            _parse_claims_response(response, filter_trivial=False, min_importance=0.0)
        assert "not a list" in str(exc_info.value)


class TestClaimExtractor:
    """Tests for ClaimExtractor class."""

    @pytest.fixture
    def mock_ai_service(self):
        """Create a mock AI service."""
        service = MagicMock()
        service.generate_content = AsyncMock()
        return service

    @pytest.fixture
    def extractor(self, mock_ai_service):
        """Create ClaimExtractor with mocked AI service."""
        return ClaimExtractor(mock_ai_service)

    @pytest.mark.asyncio
    async def test_extracts_claims_successfully(self, extractor, mock_ai_service):
        """
        Given: Valid content and context
        When: extract_claims is called
        Then: Returns extracted claims
        """
        mock_ai_service.generate_content.return_value = json.dumps({
            "claims": [
                {"idx": 0, "claim": "Test claim", "kind": "factual", "importance": 0.8}
            ]
        })

        result = await extractor.extract_claims(
            content="Generated content",
            context="Source evidence"
        )

        assert len(result) == 1
        assert result[0].claim == "Test claim"
        mock_ai_service.generate_content.assert_called_once()

    @pytest.mark.asyncio
    async def test_uses_config_default_model(self, extractor, mock_ai_service):
        """
        Given: No model specified
        When: extract_claims is called
        Then: Uses config.EVIDENCE_GROUNDING_MODEL
        """
        mock_ai_service.generate_content.return_value = '{"claims": []}'

        with patch(
            "evidence_grounding.claim_extractor.config.EVIDENCE_GROUNDING_EXTRACTION_MODEL",
            "gpt-5-nano",
        ):
            await extractor.extract_claims(
                content="Content",
                context="Context",
                model=None
            )

        call_kwargs = mock_ai_service.generate_content.call_args[1]
        assert call_kwargs["model"] == "gpt-5-nano"

    @pytest.mark.asyncio
    async def test_uses_specified_model(self, extractor, mock_ai_service):
        """
        Given: Specific model provided
        When: extract_claims is called
        Then: Uses the specified model
        """
        mock_ai_service.generate_content.return_value = '{"claims": []}'

        await extractor.extract_claims(
            content="Content",
            context="Context",
            model="custom-model"
        )

        call_kwargs = mock_ai_service.generate_content.call_args[1]
        assert call_kwargs["model"] == "custom-model"

    @pytest.mark.asyncio
    async def test_passes_json_schema(self, extractor, mock_ai_service):
        """
        Given: Valid request
        When: extract_claims is called
        Then: Passes JSON schema for structured output
        """
        mock_ai_service.generate_content.return_value = '{"claims": []}'

        await extractor.extract_claims(content="Content", context="Context")

        call_kwargs = mock_ai_service.generate_content.call_args[1]
        assert call_kwargs["json_output"] is True
        assert call_kwargs["json_schema"] == CLAIM_EXTRACTION_SCHEMA

    @pytest.mark.asyncio
    async def test_raises_on_empty_content(self, extractor):
        """
        Given: Empty content
        When: extract_claims is called
        Then: Raises ValueError
        """
        with pytest.raises(ValueError) as exc_info:
            await extractor.extract_claims(content="", context="Context")
        assert "Content cannot be empty" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_raises_on_empty_context(self, extractor):
        """
        Given: Empty context
        When: extract_claims is called
        Then: Raises ValueError
        """
        with pytest.raises(ValueError) as exc_info:
            await extractor.extract_claims(content="Content", context="")
        assert "Context/evidence cannot be empty" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_applies_filters(self, extractor, mock_ai_service):
        """
        Given: Response with claims of varying types
        When: extract_claims with filter_trivial=True
        Then: Trivial claims are filtered
        """
        mock_ai_service.generate_content.return_value = json.dumps({
            "claims": [
                {"idx": 0, "claim": "Important", "kind": "factual", "importance": 0.9},
                {"idx": 1, "claim": "Trivial", "kind": "trivial", "importance": 0.9},
            ]
        })

        result = await extractor.extract_claims(
            content="Content",
            context="Context",
            filter_trivial=True
        )

        assert len(result) == 1
        assert result[0].claim == "Important"

    @pytest.mark.asyncio
    async def test_uses_low_temperature(self, extractor, mock_ai_service):
        """
        Given: Any request
        When: extract_claims is called
        Then: Uses low temperature for consistent extraction
        """
        mock_ai_service.generate_content.return_value = '{"claims": []}'

        await extractor.extract_claims(content="Content", context="Context")

        call_kwargs = mock_ai_service.generate_content.call_args[1]
        assert call_kwargs["temperature"] == 0.3

    @pytest.mark.asyncio
    async def test_passes_usage_callback(self, extractor, mock_ai_service):
        """
        Given: Usage callback provided
        When: extract_claims is called
        Then: Callback is passed to AI service
        """
        mock_ai_service.generate_content.return_value = '{"claims": []}'
        callback = MagicMock()

        await extractor.extract_claims(
            content="Content",
            context="Context",
            usage_callback=callback
        )

        call_kwargs = mock_ai_service.generate_content.call_args[1]
        assert call_kwargs["usage_callback"] == callback


class TestExtractClaimsFunction:
    """Tests for the standalone extract_claims convenience function."""

    @pytest.mark.asyncio
    async def test_uses_shared_ai_service(self):
        """
        Given: Call to standalone extract_claims function
        When: Function is invoked
        Then: Uses get_ai_service() to get shared instance
        """
        # Patch at the source module since import is local inside the function
        with patch("ai_service.get_ai_service") as mock_get:
            mock_service = MagicMock()
            mock_service.generate_content = AsyncMock(return_value='{"claims": []}')
            mock_get.return_value = mock_service

            await extract_claims(content="Content", context="Context")

            mock_get.assert_called_once()
            mock_service.generate_content.assert_called_once()


class TestClaimExtractionSchema:
    """Tests for the CLAIM_EXTRACTION_SCHEMA constant."""

    def test_schema_is_valid_json_schema(self):
        """
        Given: CLAIM_EXTRACTION_SCHEMA constant
        When: Inspecting the schema
        Then: Has required JSON Schema structure
        """
        assert "type" in CLAIM_EXTRACTION_SCHEMA
        assert CLAIM_EXTRACTION_SCHEMA["type"] == "object"
        assert "properties" in CLAIM_EXTRACTION_SCHEMA
        assert "claims" in CLAIM_EXTRACTION_SCHEMA["properties"]

    def test_claims_array_has_item_schema(self):
        """
        Given: CLAIM_EXTRACTION_SCHEMA
        When: Inspecting claims array definition
        Then: Has proper item schema with required fields
        """
        claims_schema = CLAIM_EXTRACTION_SCHEMA["properties"]["claims"]
        assert claims_schema["type"] == "array"
        assert "items" in claims_schema

        item_schema = claims_schema["items"]
        assert "idx" in item_schema["properties"]
        assert "claim" in item_schema["properties"]
        assert "kind" in item_schema["properties"]
        assert "importance" in item_schema["properties"]

    def test_kind_enum_values(self):
        """
        Given: CLAIM_EXTRACTION_SCHEMA
        When: Inspecting kind field
        Then: Has correct enum values
        """
        kind_schema = CLAIM_EXTRACTION_SCHEMA["properties"]["claims"]["items"]["properties"]["kind"]
        assert kind_schema["enum"] == ["factual", "inference", "opinion", "trivial"]


class TestClaimExtractionSystemPrompt:
    """Tests for the CLAIM_EXTRACTION_SYSTEM_PROMPT constant."""

    def test_prompt_mentions_claim_types(self):
        """
        Given: CLAIM_EXTRACTION_SYSTEM_PROMPT
        When: Inspecting content
        Then: Mentions all claim types
        """
        assert "factual" in CLAIM_EXTRACTION_SYSTEM_PROMPT
        assert "inference" in CLAIM_EXTRACTION_SYSTEM_PROMPT
        assert "opinion" in CLAIM_EXTRACTION_SYSTEM_PROMPT
        assert "trivial" in CLAIM_EXTRACTION_SYSTEM_PROMPT

    def test_prompt_mentions_json_output(self):
        """
        Given: CLAIM_EXTRACTION_SYSTEM_PROMPT
        When: Inspecting content
        Then: Specifies JSON-only output
        """
        assert "JSON" in CLAIM_EXTRACTION_SYSTEM_PROMPT
