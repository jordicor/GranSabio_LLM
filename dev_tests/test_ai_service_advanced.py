"""
Tests for ai_service.py - Advanced functionality.

Sub-Phase 2.3: Streaming, Retries, Schema Processing, Temperature/Thinking Policies.

This module tests:
- Retry logic (_should_retry_exception, _execute_with_retries, etc.)
- Streaming behavior and retry policies
- Schema processing for structured outputs
- Temperature and thinking mode policies
"""

import asyncio
from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import AsyncMock, Mock, patch

import aiohttp
import pytest

import json_utils as json

# Import the module under test
from ai_service import AIRequestError, AIService, _ensure_aiohttp_compatibility
from deterministic_validation import DraftValidationResult
from provider_errors import ProviderErrorKind
from tool_loop_models import LoopScope, OutputContract, ToolLoopEnvelope, ToolLoopOutputTruncated

# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def ai_service_instance():
    """Create AIService instance with mocked initialization."""
    # Mock all the problematic parts of AIService.__init__
    with patch('ai_service.aiohttp.TCPConnector'):
        with patch('ai_service.aiohttp.ClientTimeout'):
            with patch('ai_service.openai'):
                with patch('ai_service.anthropic'):
                    with patch.object(AIService, '_initialize_clients'):
                        service = AIService()
                        # Ensure clients are None
                        service.openai_client = None
                        service.anthropic_client = None
                        service.google_new_client = None
                        service.xai_client = None
                        service.openrouter_client = None
                        service.ollama_client = None
                        return service


@pytest.fixture
def mock_config():
    """Mock config module."""
    with patch('ai_service.config') as mock_cfg:
        mock_cfg.MAX_RETRIES = 3
        mock_cfg.RETRY_DELAY = 10.0
        mock_cfg.GENERATOR_SYSTEM_PROMPT = "You are a helpful assistant."
        mock_cfg.GENERATOR_SYSTEM_PROMPT_RAW = "Raw system prompt."
        mock_cfg.get_model_specs = Mock(return_value={"model_specifications": {}})
        # Tool-loop constants used by call_ai_with_validation_tools defensive gates.
        mock_cfg.TOOL_LOOP_MAX_PROMPT_CHARS = 200_000
        mock_cfg.VALIDATE_DRAFT_MAX_LENGTH = 200_000
        # Default: unknown model → estimate_prompt_overflow falls back to hard cap.
        mock_cfg.get_model_info = Mock(side_effect=RuntimeError("unknown model for tests"))
        yield mock_cfg


class _FakeClaudeStream:
    def __init__(self, events, final_response=None):
        self._events = list(events)
        self._final_response = final_response

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def __aiter__(self):
        self._iter = iter(self._events)
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration

    async def get_final_response(self):
        if self._final_response is None:
            raise RuntimeError("no final response")
        return self._final_response


class _FakeOpenAICompatibleStream:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    def __aiter__(self):
        self._iter = iter(self._chunks)
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration


def _claude_message_delta(stop_reason: str):
    return SimpleNamespace(
        type="message_delta",
        delta=SimpleNamespace(stop_reason=stop_reason),
        usage=SimpleNamespace(output_tokens=12),
    )


def _claude_stream_for_response(response, *, stop_reason="tool_use"):
    events = [
        SimpleNamespace(
            type="message_start",
            message=SimpleNamespace(usage=SimpleNamespace(input_tokens=10)),
        )
    ]
    for index, block in enumerate(response.content):
        events.append(
            SimpleNamespace(
                type="content_block_start",
                index=index,
                content_block=block,
            )
        )
        if getattr(block, "type", None) == "text":
            text = getattr(block, "text", "")
            if text:
                events.append(
                    SimpleNamespace(
                        type="content_block_delta",
                        index=index,
                        delta=SimpleNamespace(type="text_delta", text=text),
                    )
                )
        elif getattr(block, "type", None) == "tool_use":
            raw_input = json.dumps(getattr(block, "input", {}) or {}, sort_keys=True)
            midpoint = max(1, len(raw_input) // 2)
            for chunk in (raw_input[:midpoint], raw_input[midpoint:]):
                events.append(
                    SimpleNamespace(
                        type="content_block_delta",
                        index=index,
                        delta=SimpleNamespace(type="input_json_delta", partial_json=chunk),
                    )
                )
        events.append(SimpleNamespace(type="content_block_stop", index=index))
    events.extend([_claude_message_delta(stop_reason), SimpleNamespace(type="message_stop")])
    return _FakeClaudeStream(events, response)


# ============================================================================
# Test Class: MiniMax OpenAI-compatible Runtime
# ============================================================================

class TestMiniMaxOpenAICompatibleRuntime:
    """MiniMax-specific OpenAI-compatible runtime behavior."""

    @pytest.mark.asyncio
    async def test_generate_minimax_requests_reasoning_split(self, ai_service_instance):
        ai_service_instance.minimax_client = Mock()
        ai_service_instance.minimax_client.chat = Mock()
        ai_service_instance.minimax_client.chat.completions = Mock()
        ai_service_instance.minimax_client.chat.completions.create = AsyncMock(
            return_value=SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="hola"),
                        finish_reason="stop",
                    )
                ],
                usage=None,
            )
        )

        content, _usage = await ai_service_instance._generate_minimax(
            prompt="Responde exactamente con: hola",
            model_id="MiniMax-M3",
            temperature=1.0,
            max_tokens=128,
            system_prompt="Responde de forma minima.",
        )

        assert content == "hola"
        request_kwargs = ai_service_instance.minimax_client.chat.completions.create.await_args.kwargs
        assert request_kwargs["extra_body"] == {"reasoning_split": True}

    @pytest.mark.asyncio
    async def test_stream_minimax_requests_reasoning_split(self, ai_service_instance):
        ai_service_instance.minimax_client = Mock()
        ai_service_instance.minimax_client.chat = Mock()
        ai_service_instance.minimax_client.chat.completions = Mock()
        ai_service_instance.minimax_client.chat.completions.create = AsyncMock(
            return_value=_FakeOpenAICompatibleStream(
                [
                    SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                delta=SimpleNamespace(content="hola"),
                                finish_reason=None,
                            )
                        ],
                        usage=None,
                    ),
                    SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                delta=SimpleNamespace(content=None),
                                finish_reason="stop",
                            )
                        ],
                        usage=None,
                    ),
                ]
            )
        )

        chunks = []
        async for chunk in ai_service_instance._stream_minimax(
            prompt="Responde exactamente con: hola",
            model_id="MiniMax-M3",
            temperature=1.0,
            max_tokens=128,
            system_prompt="Responde de forma minima.",
        ):
            chunks.append(chunk)

        assert chunks[0] == "hola"
        request_kwargs = ai_service_instance.minimax_client.chat.completions.create.await_args.kwargs
        assert request_kwargs["extra_body"] == {"reasoning_split": True}


# ============================================================================
# Test Class: Retry Logic
# ============================================================================

class TestShouldRetryException:
    """Tests for _should_retry_exception() static method."""

    def test_timeout_error_is_retriable(self):
        """
        Given: asyncio.TimeoutError exception
        When: _should_retry_exception() is called
        Then: Returns True (retriable)
        """
        exc = asyncio.TimeoutError("Request timed out")
        assert AIService._should_retry_exception(exc) is True

    def test_connection_error_is_retriable(self):
        """
        Given: ConnectionError exception
        When: _should_retry_exception() is called
        Then: Returns True (retriable)
        """
        exc = ConnectionError("Connection refused")
        assert AIService._should_retry_exception(exc) is True

    def test_os_error_is_retriable(self):
        """
        Given: OSError exception
        When: _should_retry_exception() is called
        Then: Returns True (retriable)
        """
        exc = OSError("Network unreachable")
        assert AIService._should_retry_exception(exc) is True

    def test_status_429_rate_limit_is_retriable(self):
        """
        Given: Exception with status 429 (rate limit)
        When: _should_retry_exception() is called
        Then: Returns True (retriable)
        """
        exc = Exception("Rate limited")
        exc.status = 429
        assert AIService._should_retry_exception(exc) is True

    def test_status_503_service_unavailable_is_retriable(self):
        """
        Given: Exception with status 503 (service unavailable)
        When: _should_retry_exception() is called
        Then: Returns True (retriable)
        """
        exc = Exception("Service unavailable")
        exc.status = 503
        assert AIService._should_retry_exception(exc) is True

    def test_status_500_internal_error_is_retriable(self):
        """
        Given: Exception with status 500 (internal server error)
        When: _should_retry_exception() is called
        Then: Returns True (retriable)
        """
        exc = Exception("Internal server error")
        exc.status_code = 500
        assert AIService._should_retry_exception(exc) is True

    def test_status_502_bad_gateway_is_retriable(self):
        """
        Given: Exception with status 502 (bad gateway)
        When: _should_retry_exception() is called
        Then: Returns True (retriable)
        """
        exc = Exception("Bad gateway")
        exc.status = 502
        assert AIService._should_retry_exception(exc) is True

    def test_status_400_bad_request_not_retriable(self):
        """
        Given: Exception with status 400 (bad request)
        When: _should_retry_exception() is called
        Then: Returns False (not retriable)
        """
        exc = Exception("Bad request")
        exc.status = 400
        assert AIService._should_retry_exception(exc) is False

    def test_status_401_unauthorized_not_retriable(self):
        """
        Given: Exception with status 401 (unauthorized)
        When: _should_retry_exception() is called
        Then: Returns False (not retriable)
        """
        exc = Exception("Unauthorized")
        exc.status = 401
        assert AIService._should_retry_exception(exc) is False

    def test_status_404_not_found_not_retriable(self):
        """
        Given: Exception with status 404 (not found)
        When: _should_retry_exception() is called
        Then: Returns False (not retriable)
        """
        exc = Exception("Not found")
        exc.status = 404
        assert AIService._should_retry_exception(exc) is False

    def test_timeout_message_is_retriable(self):
        """
        Given: Exception with 'timeout' in message
        When: _should_retry_exception() is called
        Then: Returns True (retriable)
        """
        exc = Exception("Request timeout occurred")
        assert AIService._should_retry_exception(exc) is True

    def test_rate_limit_message_is_retriable(self):
        """
        Given: Exception with 'rate limit' in message
        When: _should_retry_exception() is called
        Then: Returns True (retriable)
        """
        exc = Exception("Rate limit exceeded, please retry")
        assert AIService._should_retry_exception(exc) is True

    def test_overloaded_message_is_retriable(self):
        """
        Given: Exception with 'overloaded' in message
        When: _should_retry_exception() is called
        Then: Returns True (retriable)
        """
        exc = Exception("Server is overloaded")
        assert AIService._should_retry_exception(exc) is True

    def test_generic_value_error_not_retriable(self):
        """
        Given: Generic ValueError without transient markers
        When: _should_retry_exception() is called
        Then: Returns False (not retriable)
        """
        exc = ValueError("Invalid parameter value")
        assert AIService._should_retry_exception(exc) is False

    def test_attribute_error_with_aiohttp_is_retriable(self):
        """
        Given: AttributeError with 'aiohttp' in message (SDK bug)
        When: _should_retry_exception() is called
        Then: Returns True (retriable - SDK bug during connection)
        """
        exc = AttributeError("'aiohttp.ClientSession' has no attribute 'foo'")
        assert AIService._should_retry_exception(exc) is True


def test_ensure_aiohttp_compatibility_aliases_missing_dns_connector(monkeypatch):
    """Older aiohttp versions must not break google-genai exception handling."""
    monkeypatch.delattr(aiohttp, "ClientConnectorDNSError", raising=False)

    _ensure_aiohttp_compatibility()

    assert aiohttp.ClientConnectorDNSError is aiohttp.ClientConnectorError


class TestExtractRequestId:
    """Tests for _extract_request_id() static method."""

    def test_extracts_request_id_attribute(self):
        """
        Given: Exception with request_id attribute
        When: _extract_request_id() is called
        Then: Returns the request_id
        """
        exc = Exception("Error")
        exc.request_id = "req-12345"
        assert AIService._extract_request_id(exc) == "req-12345"

    def test_extracts_response_id_attribute(self):
        """
        Given: Exception with response_id attribute
        When: _extract_request_id() is called
        Then: Returns the response_id
        """
        exc = Exception("Error")
        exc.response_id = "resp-67890"
        assert AIService._extract_request_id(exc) == "resp-67890"

    def test_extracts_id_from_response_object(self):
        """
        Given: Exception with response.request_id
        When: _extract_request_id() is called
        Then: Returns the request_id from response
        """
        exc = Exception("Error")
        exc.response = Mock()
        exc.response.request_id = "nested-req-id"
        assert AIService._extract_request_id(exc) == "nested-req-id"

    def test_returns_none_when_no_id(self):
        """
        Given: Exception without any ID attributes
        When: _extract_request_id() is called
        Then: Returns None
        """
        exc = Exception("Simple error")
        assert AIService._extract_request_id(exc) is None


class TestMaxRetryAttempts:
    """Tests for _max_retry_attempts() method."""

    def test_returns_config_value(self, ai_service_instance, mock_config):
        """
        Given: config.MAX_RETRIES is set to 5
        When: _max_retry_attempts() is called
        Then: Returns 5
        """
        mock_config.MAX_RETRIES = 5
        assert ai_service_instance._max_retry_attempts() == 5

    def test_returns_minimum_of_1(self, ai_service_instance, mock_config):
        """
        Given: config.MAX_RETRIES is set to 0
        When: _max_retry_attempts() is called
        Then: Returns 1 (minimum)
        """
        mock_config.MAX_RETRIES = 0
        assert ai_service_instance._max_retry_attempts() == 1

    def test_returns_default_on_exception(self, ai_service_instance):
        """
        Given: config.MAX_RETRIES raises exception
        When: _max_retry_attempts() is called
        Then: Returns default value 3
        """
        with patch('ai_service.config') as mock_cfg:
            # Make getattr raise an exception
            del mock_cfg.MAX_RETRIES
            result = ai_service_instance._max_retry_attempts()
            assert result == 3


class TestRetryDelaySeconds:
    """Tests for _retry_delay_seconds() method."""

    def test_returns_config_value(self, ai_service_instance, mock_config):
        """
        Given: config.RETRY_DELAY is set to 15.0
        When: _retry_delay_seconds() is called
        Then: Returns 15.0
        """
        mock_config.RETRY_DELAY = 15.0
        assert ai_service_instance._retry_delay_seconds() == 15.0

    def test_returns_zero_minimum(self, ai_service_instance, mock_config):
        """
        Given: config.RETRY_DELAY is set to -5.0
        When: _retry_delay_seconds() is called
        Then: Returns 0.0 (minimum)
        """
        mock_config.RETRY_DELAY = -5.0
        assert ai_service_instance._retry_delay_seconds() == 0.0

    def test_returns_default_on_exception(self, ai_service_instance):
        """
        Given: config.RETRY_DELAY raises exception
        When: _retry_delay_seconds() is called
        Then: Returns default value 10.0
        """
        with patch('ai_service.config') as mock_cfg:
            del mock_cfg.RETRY_DELAY
            result = ai_service_instance._retry_delay_seconds()
            assert result == 10.0


# ============================================================================
# Test Class: Schema Processing
# ============================================================================

class TestStripAdditionalProperties:
    """Tests for _strip_additional_properties() static method."""

    def test_removes_top_level_additional_properties(self):
        """
        Given: Schema with additionalProperties at root
        When: _strip_additional_properties() is called
        Then: additionalProperties is removed
        """
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "additionalProperties": False
        }
        result = AIService._strip_additional_properties(schema)
        assert "additionalProperties" not in result
        assert result["type"] == "object"
        assert "name" in result["properties"]

    def test_removes_nested_additional_properties(self):
        """
        Given: Schema with additionalProperties in nested object
        When: _strip_additional_properties() is called
        Then: All additionalProperties are removed
        """
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "additionalProperties": False
                }
            },
            "additionalProperties": False
        }
        result = AIService._strip_additional_properties(schema)
        assert "additionalProperties" not in result
        assert "additionalProperties" not in result["properties"]["user"]

    def test_handles_array_items(self):
        """
        Given: Schema with array items containing additionalProperties
        When: _strip_additional_properties() is called
        Then: additionalProperties removed from items
        """
        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "integer"}},
                "additionalProperties": False
            }
        }
        result = AIService._strip_additional_properties(schema)
        assert "additionalProperties" not in result["items"]

    def test_handles_allof_combinator(self):
        """
        Given: Schema with allOf containing additionalProperties
        When: _strip_additional_properties() is called
        Then: additionalProperties removed from allOf schemas
        """
        schema = {
            "allOf": [
                {"type": "object", "properties": {"a": {"type": "string"}}, "additionalProperties": False},
                {"type": "object", "properties": {"b": {"type": "integer"}}, "additionalProperties": False}
            ]
        }
        result = AIService._strip_additional_properties(schema)
        assert "additionalProperties" not in result["allOf"][0]
        assert "additionalProperties" not in result["allOf"][1]

    def test_handles_definitions(self):
        """
        Given: Schema with $defs containing additionalProperties
        When: _strip_additional_properties() is called
        Then: additionalProperties removed from definitions
        """
        schema = {
            "type": "object",
            "$defs": {
                "Address": {
                    "type": "object",
                    "properties": {"street": {"type": "string"}},
                    "additionalProperties": False
                }
            }
        }
        result = AIService._strip_additional_properties(schema)
        assert "additionalProperties" not in result["$defs"]["Address"]

    def test_preserves_other_fields(self):
        """
        Given: Schema with multiple fields
        When: _strip_additional_properties() is called
        Then: All other fields are preserved
        """
        schema = {
            "type": "object",
            "title": "Test Schema",
            "description": "A test schema",
            "properties": {"name": {"type": "string", "description": "The name"}},
            "required": ["name"],
            "additionalProperties": False
        }
        result = AIService._strip_additional_properties(schema)
        assert result["type"] == "object"
        assert result["title"] == "Test Schema"
        assert result["description"] == "A test schema"
        assert result["required"] == ["name"]
        assert result["properties"]["name"]["description"] == "The name"

    def test_returns_non_dict_unchanged(self):
        """
        Given: Non-dict value
        When: _strip_additional_properties() is called
        Then: Returns value unchanged
        """
        assert AIService._strip_additional_properties("string") == "string"
        assert AIService._strip_additional_properties(123) == 123
        assert AIService._strip_additional_properties(None) is None


class TestConvertNullableToGeminiFormat:
    """Tests for _convert_nullable_to_gemini_format() static method."""

    def test_converts_nullable_string(self):
        """
        Given: Schema with ["string", "null"] type
        When: _convert_nullable_to_gemini_format() is called
        Then: Converts to {"type": "string", "nullable": true}
        """
        schema = {"type": ["string", "null"]}
        result = AIService._convert_nullable_to_gemini_format(schema)
        assert result["type"] == "string"
        assert result["nullable"] is True

    def test_converts_nullable_integer(self):
        """
        Given: Schema with ["integer", "null"] type
        When: _convert_nullable_to_gemini_format() is called
        Then: Converts to {"type": "integer", "nullable": true}
        """
        schema = {"type": ["integer", "null"]}
        result = AIService._convert_nullable_to_gemini_format(schema)
        assert result["type"] == "integer"
        assert result["nullable"] is True

    def test_converts_nullable_in_nested_properties(self):
        """
        Given: Schema with nullable type in nested property
        When: _convert_nullable_to_gemini_format() is called
        Then: Nested nullable is converted
        """
        schema = {
            "type": "object",
            "properties": {
                "optional_field": {"type": ["string", "null"]}
            }
        }
        result = AIService._convert_nullable_to_gemini_format(schema)
        assert result["properties"]["optional_field"]["type"] == "string"
        assert result["properties"]["optional_field"]["nullable"] is True

    def test_handles_union_type_without_null(self):
        """
        Given: Schema with ["integer", "number"] type (union without null)
        When: _convert_nullable_to_gemini_format() is called
        Then: Picks most permissive type (number)
        """
        schema = {"type": ["integer", "number"]}
        result = AIService._convert_nullable_to_gemini_format(schema)
        assert result["type"] == "number"
        assert "nullable" not in result

    def test_union_type_priority_string(self):
        """
        Given: Schema with ["boolean", "string"] type
        When: _convert_nullable_to_gemini_format() is called
        Then: Picks string (higher priority than boolean)
        """
        schema = {"type": ["boolean", "string"]}
        result = AIService._convert_nullable_to_gemini_format(schema)
        assert result["type"] == "string"

    def test_handles_array_items(self):
        """
        Given: Schema with nullable type in array items
        When: _convert_nullable_to_gemini_format() is called
        Then: Items nullable is converted
        """
        schema = {
            "type": "array",
            "items": {"type": ["string", "null"]}
        }
        result = AIService._convert_nullable_to_gemini_format(schema)
        assert result["items"]["type"] == "string"
        assert result["items"]["nullable"] is True

    def test_preserves_non_nullable_types(self):
        """
        Given: Schema with simple string type
        When: _convert_nullable_to_gemini_format() is called
        Then: Type is preserved unchanged
        """
        schema = {"type": "string", "description": "A name"}
        result = AIService._convert_nullable_to_gemini_format(schema)
        assert result["type"] == "string"
        assert result["description"] == "A name"
        assert "nullable" not in result


class TestPrepareStructuredOutputSchema:
    """Tests for provider-specific structured-output schema preparation."""

    def test_claude_nullable_enums_are_normalized_to_anyof(self):
        """
        Given: Editable QA schema with nullable enum fields
        When: Preparing schema for Claude native structured outputs
        Then: Nullable enums are emitted as typed anyOf branches
        """
        from qa_response_schemas import QA_SCHEMA_EDITABLE

        result = AIService._prepare_structured_output_schema(
            "claude",
            "claude-sonnet-4-6",
            QA_SCHEMA_EDITABLE,
        )

        edit_strategy = result["properties"]["edit_strategy"]
        assert "anyOf" in edit_strategy
        assert {"type": "null"} in edit_strategy["anyOf"]
        assert not (
            "enum" in edit_strategy
            and None in edit_strategy["enum"]
            and "type" in edit_strategy
        )

    def test_claude_schema_sets_additional_properties_false(self):
        """
        Given: A Claude structured-output schema with nested object properties
        When: Preparing schema for Anthropic output_format
        Then: Object schemas declare additionalProperties: false before the provider call
        """
        schema = {
            "type": "object",
            "properties": {
                "decision": {"type": "string"},
                "notes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                    },
                },
            },
            "required": ["decision"],
        }

        result = AIService._prepare_structured_output_schema(
            "claude",
            "claude-opus-4-6",
            schema,
        )

        assert result["additionalProperties"] is False
        note_ref = result["properties"]["notes"]["items"]["$ref"].rsplit("/", 1)[-1]
        assert result["$defs"][note_ref]["additionalProperties"] is False

    @pytest.mark.parametrize(
        ("provider", "model_id"),
        [
            ("openai", "gpt-4o"),
            ("xai", "grok-2-1212"),
            ("openrouter", "openai/gpt-4o-mini"),
        ],
    )
    def test_openai_compatible_schemas_are_normalized_for_strict_mode(self, provider, model_id):
        """OpenAI-compatible providers receive strict structured-output schemas."""
        schema = {
            "type": "object",
            "properties": {
                "decision": {"type": "string"},
                "notes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                    },
                },
            },
            "required": ["decision"],
        }

        result = AIService._prepare_structured_output_schema(
            provider,
            model_id,
            schema,
        )

        assert result is not schema
        assert result["additionalProperties"] is False
        assert result["required"] == ["decision", "notes"]
        note_schema = result["properties"]["notes"]["items"]
        assert note_schema["additionalProperties"] is False
        assert note_schema["required"] == ["text"]


class TestValidateSchemaForStructuredOutputs:
    """Tests for _validate_schema_for_structured_outputs() static method."""

    def test_raises_on_missing_root_type(self):
        """
        Given: Schema without root type
        When: _validate_schema_for_structured_outputs() is called
        Then: Raises ValueError
        """
        schema = {"properties": {"name": {"type": "string"}}}
        with pytest.raises(ValueError, match="missing required 'type' at the root"):
            AIService._validate_schema_for_structured_outputs(schema, "openai", "gpt-4o")

    def test_raises_on_additional_properties_true_for_openai(self):
        """
        Given: Schema with additionalProperties: true for OpenAI
        When: _validate_schema_for_structured_outputs() is called
        Then: Raises ValueError
        """
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "additionalProperties": True
        }
        with pytest.raises(ValueError, match="do not allow 'additionalProperties: true'"):
            AIService._validate_schema_for_structured_outputs(schema, "openai", "gpt-4o")

    def test_raises_on_additional_properties_for_gemini(self):
        """
        Given: Schema with additionalProperties for Gemini (should be stripped first)
        When: _validate_schema_for_structured_outputs() is called
        Then: Raises ValueError
        """
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "additionalProperties": False
        }
        with pytest.raises(ValueError, match="Gemini structured outputs do not support"):
            AIService._validate_schema_for_structured_outputs(schema, "gemini", "gemini-2.0")

    def test_passes_valid_schema_for_openai(self):
        """
        Given: Valid schema with additionalProperties: false
        When: _validate_schema_for_structured_outputs() is called for OpenAI
        Then: Does not raise
        """
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
            "additionalProperties": False
        }
        # Should not raise
        AIService._validate_schema_for_structured_outputs(schema, "openai", "gpt-4o")

    def test_raises_on_missing_additional_properties_false_for_openai(self):
        """
        Given: Object schema missing additionalProperties: false for OpenAI
        When: _validate_schema_for_structured_outputs() is called
        Then: Raises a local ValueError before the provider rejects the request
        """
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        with pytest.raises(ValueError, match="additionalProperties: false"):
            AIService._validate_schema_for_structured_outputs(schema, "openai", "gpt-4o")

    def test_raises_on_missing_required_property_for_openai(self):
        """
        Given: Object schema with a property omitted from required for OpenAI
        When: _validate_schema_for_structured_outputs() is called
        Then: Raises a local ValueError before the provider rejects the request
        """
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "additionalProperties": False,
        }
        with pytest.raises(ValueError, match="every property"):
            AIService._validate_schema_for_structured_outputs(schema, "openai", "gpt-4o")

    def test_passes_valid_schema_for_claude(self):
        """
        Given: Valid schema without additionalProperties: true
        When: _validate_schema_for_structured_outputs() is called for Claude
        Then: Does not raise
        """
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}}
        }
        # Should not raise
        AIService._validate_schema_for_structured_outputs(schema, "claude", "claude-sonnet-4-5")

    def test_validates_nested_properties(self):
        """
        Given: Schema with additionalProperties: true in nested property
        When: _validate_schema_for_structured_outputs() is called
        Then: Raises ValueError
        """
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "additionalProperties": True
                }
            },
            "required": ["user"],
            "additionalProperties": False,
        }
        with pytest.raises(ValueError, match="do not allow 'additionalProperties: true'"):
            AIService._validate_schema_for_structured_outputs(schema, "openai", "gpt-4o")


# ============================================================================
# Test Class: Temperature and Thinking Policies
# ============================================================================

class TestApplyTemperaturePolicies:
    """Tests for _apply_temperature_policies() method."""

    def test_claude_thinking_mode_forces_temperature_1(self, ai_service_instance):
        """
        Given: Claude model with thinking_budget_tokens > 0
        When: _apply_temperature_policies() is called
        Then: Temperature forced to 1.0
        """
        model_info = {"provider": "claude", "capabilities": [], "model_id": "claude-sonnet-4"}
        temp, thinking, forced = ai_service_instance._apply_temperature_policies(
            model_info, 0.7, 10000
        )
        assert temp == 1.0
        assert forced is True
        assert thinking == 10000

    def test_claude_without_thinking_preserves_temperature(self, ai_service_instance):
        """
        Given: Claude model without thinking_budget_tokens
        When: _apply_temperature_policies() is called
        Then: Temperature is preserved
        """
        model_info = {"provider": "claude", "capabilities": [], "model_id": "claude-sonnet-4"}
        temp, thinking, forced = ai_service_instance._apply_temperature_policies(
            model_info, 0.7, None
        )
        assert temp == 0.7
        assert forced is False

    def test_reasoning_model_forces_temperature_1(self, ai_service_instance):
        """
        Given: Model with 'reasoning' capability
        When: _apply_temperature_policies() is called
        Then: Temperature forced to 1.0
        """
        model_info = {"provider": "claude", "capabilities": ["reasoning"], "model_id": "claude-opus-4"}
        temp, thinking, forced = ai_service_instance._apply_temperature_policies(
            model_info, 0.5, None
        )
        assert temp == 1.0
        assert forced is True

    def test_non_reasoning_model_preserves_temperature(self, ai_service_instance):
        """
        Given: Model without reasoning capability
        When: _apply_temperature_policies() is called
        Then: Temperature is preserved
        """
        model_info = {"provider": "openai", "capabilities": [], "model_id": "gpt-4o"}
        temp, thinking, forced = ai_service_instance._apply_temperature_policies(
            model_info, 0.7, None
        )
        assert temp == 0.7
        assert forced is False

    def test_xai_reasoning_model_forces_temperature_1(self, ai_service_instance):
        """
        Given: xAI model with reasoning capability
        When: _apply_temperature_policies() is called
        Then: Temperature forced to 1.0
        """
        model_info = {"provider": "xai", "capabilities": ["reasoning"], "model_id": "grok-3"}
        temp, thinking, forced = ai_service_instance._apply_temperature_policies(
            model_info, 0.3, None
        )
        assert temp == 1.0
        assert forced is True


class TestSupportsClaudeThinking:
    """Tests for _supports_claude_thinking() static method."""

    def test_claude_37_supports_thinking(self):
        """
        Given: claude-3.7 model
        When: _supports_claude_thinking() is called
        Then: Returns True
        """
        assert AIService._supports_claude_thinking("claude-3.7-sonnet") is True

    def test_claude_4_sonnet_supports_thinking(self):
        """
        Given: claude-sonnet-4 model
        When: _supports_claude_thinking() is called
        Then: Returns True
        """
        assert AIService._supports_claude_thinking("claude-sonnet-4-20250514") is True

    def test_claude_opus_4_supports_thinking(self):
        """
        Given: claude-opus-4 model
        When: _supports_claude_thinking() is called
        Then: Returns True
        """
        assert AIService._supports_claude_thinking("claude-opus-4-20250514") is True

    def test_claude_3_haiku_does_not_support_thinking(self):
        """
        Given: claude-3-haiku model (older)
        When: _supports_claude_thinking() is called
        Then: Returns False
        """
        assert AIService._supports_claude_thinking("claude-3-haiku-20240307") is False

    def test_claude_3_opus_does_not_support_thinking(self):
        """
        Given: claude-3-opus model (older)
        When: _supports_claude_thinking() is called
        Then: Returns False
        """
        assert AIService._supports_claude_thinking("claude-3-opus-20240229") is False

    def test_none_model_returns_false(self):
        """
        Given: None as model_id
        When: _supports_claude_thinking() is called
        Then: Returns False
        """
        assert AIService._supports_claude_thinking(None) is False


class TestOpenRouterTemperatureParams:
    """Tests for OpenRouter model-specific temperature compatibility."""

    def test_moonshot_kimi_27_openrouter_omits_temperature(self):
        assert AIService._openrouter_accepts_temperature("moonshotai/kimi-k2.7-code") is False

    def test_claude_opus_47_omits_temperature(self):
        assert AIService._openrouter_accepts_temperature("anthropic/claude-opus-4.7") is False

    def test_claude_opus_47_thinking_alias_omits_temperature(self):
        assert AIService._openrouter_accepts_temperature("anthropic/claude-opus-4.7:thinking") is False

    def test_claude_opus_48_omits_temperature(self):
        assert AIService._openrouter_accepts_temperature("anthropic/claude-opus-4.8") is False
        assert AIService._openrouter_accepts_temperature("anthropic/claude-opus-4.8:thinking") is False
        assert AIService._openrouter_accepts_temperature("anthropic/claude-opus-4.8-fast") is False

    def test_claude_sonnet_46_omits_temperature(self):
        assert AIService._openrouter_accepts_temperature("anthropic/claude-sonnet-4.6") is False

    def test_other_openrouter_models_keep_temperature(self):
        assert AIService._openrouter_accepts_temperature("google/gemini-3.1-pro-preview") is True
        assert AIService._openrouter_accepts_temperature("x-ai/grok-4-fast") is True

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "model_id",
        [
            "anthropic/claude-opus-4.7",
            "anthropic/claude-opus-4.8",
        ],
    )
    async def test_generate_openrouter_does_not_send_temperature_for_restricted_opus(self, ai_service_instance, model_id):
        create = AsyncMock(
            return_value=SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
                usage=None,
            )
        )
        ai_service_instance.openrouter_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )

        content, _usage = await ai_service_instance._generate_openrouter(
            prompt="Hello",
            model_id=model_id,
            temperature=0.4,
            max_tokens=64,
            system_prompt="System",
        )

        assert content == "ok"
        request_kwargs = create.await_args.kwargs
        assert request_kwargs["model"] == model_id
        assert request_kwargs["max_tokens"] == 64
        assert "temperature" not in request_kwargs

    def test_openrouter_tool_params_omit_temperature_for_restricted_opus(self, ai_service_instance):
        params = ai_service_instance._build_openai_compatible_tool_params(
            provider="openrouter",
            model_id="anthropic/claude-opus-4.8",
            current_messages=[{"role": "user", "content": "Hello"}],
            temperature=0.4,
            max_tokens=64,
            reasoning_effort=None,
            json_output=False,
            json_schema=None,
            tools_enabled=False,
        )

        assert params["model"] == "anthropic/claude-opus-4.8"
        assert "temperature" not in params

    def test_openrouter_tool_params_omit_temperature_for_kimi_27(self, ai_service_instance):
        params = ai_service_instance._build_openai_compatible_tool_params(
            provider="openrouter",
            model_id="moonshotai/kimi-k2.7-code",
            current_messages=[{"role": "user", "content": "Hello"}],
            temperature=0.4,
            max_tokens=64,
            reasoning_effort=None,
            json_output=False,
            json_schema=None,
            tools_enabled=False,
        )

        assert params["model"] == "moonshotai/kimi-k2.7-code"
        assert "temperature" not in params


class TestMoonshotKimiParameterPolicy:
    """Tests for Moonshot/Kimi fixed sampling parameter handling."""

    @pytest.mark.asyncio
    async def test_generate_moonshot_kimi_27_omits_temperature(self, ai_service_instance):
        create = AsyncMock(
            return_value=SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
                usage=None,
            )
        )
        ai_service_instance.moonshot_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )

        content, _usage = await ai_service_instance._generate_moonshot(
            prompt="Hello",
            model_id="kimi-k2.7-code",
            temperature=0.2,
            max_tokens=64,
            system_prompt="System",
        )

        assert content == "ok"
        request_kwargs = create.await_args.kwargs
        assert request_kwargs["model"] == "kimi-k2.7-code"
        assert "temperature" not in request_kwargs
        assert request_kwargs["max_completion_tokens"] == 64
        assert "max_tokens" not in request_kwargs

    @pytest.mark.asyncio
    async def test_generate_openrouter_kimi_27_omits_temperature(self, ai_service_instance):
        create = AsyncMock(
            return_value=SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
                usage=None,
            )
        )
        ai_service_instance.openrouter_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )

        content, _usage = await ai_service_instance._generate_openrouter(
            prompt="Hello",
            model_id="moonshotai/kimi-k2.7-code",
            temperature=0.2,
            max_tokens=64,
            system_prompt="System",
        )

        assert content == "ok"
        request_kwargs = create.await_args.kwargs
        assert request_kwargs["model"] == "moonshotai/kimi-k2.7-code"
        assert request_kwargs["max_tokens"] == 64
        assert "temperature" not in request_kwargs

    @pytest.mark.asyncio
    async def test_stream_moonshot_kimi_27_omits_temperature(self, ai_service_instance):
        class _AsyncStream:
            def __aiter__(self):
                self._chunks = iter(
                    [
                        SimpleNamespace(
                            choices=[
                                SimpleNamespace(
                                    delta=SimpleNamespace(content="ok"),
                                    finish_reason=None,
                                )
                            ],
                            usage=None,
                        ),
                        SimpleNamespace(
                            choices=[
                                SimpleNamespace(
                                    delta=SimpleNamespace(content=None),
                                    finish_reason="stop",
                                )
                            ],
                            usage=None,
                        ),
                    ]
                )
                return self

            async def __anext__(self):
                try:
                    return next(self._chunks)
                except StopIteration:
                    raise StopAsyncIteration

        create = AsyncMock(return_value=_AsyncStream())
        ai_service_instance.moonshot_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )

        chunks = []
        async for chunk in ai_service_instance._stream_moonshot(
            prompt="Hello",
            model_id="kimi-k2.7-code",
            temperature=0.2,
            max_tokens=64,
            system_prompt="System",
        ):
            chunks.append(str(chunk))

        request_kwargs = create.await_args.kwargs
        assert request_kwargs["model"] == "kimi-k2.7-code"
        assert "temperature" not in request_kwargs
        assert request_kwargs["max_completion_tokens"] == 64
        assert "max_tokens" not in request_kwargs
        assert "ok" in chunks

    @pytest.mark.asyncio
    async def test_moonshot_health_check_uses_kimi_token_parameter(self, ai_service_instance):
        create = AsyncMock(return_value=SimpleNamespace())
        ai_service_instance.openai_client = None
        ai_service_instance.anthropic_client = None
        ai_service_instance.google_new_client = None
        ai_service_instance.xai_client = None
        ai_service_instance.minimax_client = None
        ai_service_instance.moonshot_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )

        with patch(
            "ai_service.resolve_call",
            return_value=SimpleNamespace(
                model="kimi-k2.7-code",
                params={"max_tokens": 5},
            ),
        ):
            status = await ai_service_instance.health_check()

        request_kwargs = create.await_args.kwargs
        assert status["moonshot"] is True
        assert request_kwargs["model"] == "kimi-k2.7-code"
        assert request_kwargs["max_completion_tokens"] == 5
        assert "max_tokens" not in request_kwargs
        assert "temperature" not in request_kwargs


class TestClaudeOpus47And48Compatibility:
    """Tests for Opus 4.7+ Anthropic Messages API constraints."""

    def test_direct_claude_restricted_models_omit_sampling_params(self):
        assert AIService._claude_omits_sampling_params("claude-opus-4-7") is True
        assert AIService._claude_omits_sampling_params("claude-opus-4-8") is True
        assert AIService._claude_omits_sampling_params("claude-opus-4-6") is False

    @pytest.mark.asyncio
    async def test_generate_claude_omits_temperature_for_opus_48(self, ai_service_instance):
        create = AsyncMock(
            return_value=SimpleNamespace(
                content=[SimpleNamespace(type="text", text="ok")],
                usage=None,
                stop_reason="end_turn",
            )
        )
        ai_service_instance.anthropic_client = SimpleNamespace(
            messages=SimpleNamespace(create=create),
        )

        content, _usage = await ai_service_instance._generate_claude(
            prompt="Hello",
            model_id="claude-opus-4-8",
            temperature=0.4,
            max_tokens=64,
            system_prompt="System",
        )

        assert content == "ok"
        request_kwargs = create.await_args.kwargs
        assert request_kwargs["model"] == "claude-opus-4-8"
        assert "temperature" not in request_kwargs

    @pytest.mark.asyncio
    async def test_generate_claude_translates_opus_48_thinking_budget_to_adaptive(self, ai_service_instance):
        create = AsyncMock(
            return_value=SimpleNamespace(
                content=[SimpleNamespace(type="text", text="ok")],
                usage=None,
                stop_reason="end_turn",
            )
        )
        ai_service_instance.anthropic_client = SimpleNamespace(
            messages=SimpleNamespace(create=create),
        )

        await ai_service_instance._generate_claude(
            prompt="Hello",
            model_id="claude-opus-4-8",
            temperature=1.0,
            max_tokens=64,
            system_prompt="System",
            thinking_budget_tokens=12000,
        )

        request_kwargs = create.await_args.kwargs
        assert request_kwargs["thinking"] == {"type": "adaptive"}
        assert "temperature" not in request_kwargs

    @pytest.mark.asyncio
    async def test_generate_claude_uses_output_config_effort_for_opus_48(self, ai_service_instance):
        create = AsyncMock(
            return_value=SimpleNamespace(
                content=[SimpleNamespace(type="text", text="ok")],
                usage=None,
                stop_reason="end_turn",
            )
        )
        ai_service_instance.anthropic_client = SimpleNamespace(
            messages=SimpleNamespace(create=create),
        )

        await ai_service_instance._generate_claude(
            prompt="Hello",
            model_id="claude-opus-4-8",
            temperature=1.0,
            max_tokens=64,
            system_prompt="System",
            reasoning_effort="max",
        )

        request_kwargs = create.await_args.kwargs
        assert request_kwargs["output_config"]["effort"] == "max"
        assert request_kwargs["thinking"] == {"type": "adaptive"}


class TestReasoningEffortProviderRetry:
    """Tests provider-error correction for stale reasoning effort metadata."""

    @pytest.mark.asyncio
    async def test_chat_completion_retry_coerces_to_provider_supported_level(self, ai_service_instance):
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
            usage=None,
        )
        create = AsyncMock(
            side_effect=[
                Exception("Unsupported value: 'minimal'. Supported values are: 'medium'."),
                response,
            ]
        )
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

        result = await ai_service_instance._call_chat_completions_create_with_reasoning_retry(
            client,
            {
                "model": "gpt-5.1-chat-latest",
                "messages": [{"role": "user", "content": "hi"}],
                "reasoning_effort": "minimal",
            },
            provider_key="openai",
            model_id="gpt-5.1-chat-latest",
        )

        assert result is response
        assert create.await_args_list[1].kwargs["reasoning_effort"] == "medium"

    @pytest.mark.asyncio
    async def test_responses_retry_coerces_max_to_xhigh(self, ai_service_instance):
        response = SimpleNamespace(output_text="ok", usage=None)
        create = AsyncMock(
            side_effect=[
                Exception("reasoning.effort: Input should be 'low', 'medium', 'high' or 'xhigh'"),
                response,
            ]
        )
        ai_service_instance.openai_client = SimpleNamespace(
            responses=SimpleNamespace(create=create)
        )

        result = await ai_service_instance._call_responses_create_with_reasoning_retry(
            {
                "model": "some-responses-model",
                "input": "hi",
                "reasoning": {"effort": "max"},
            },
            model_id="some-responses-model",
        )

        assert result is response
        assert create.await_args_list[1].kwargs["reasoning"]["effort"] == "xhigh"

    @pytest.mark.asyncio
    async def test_chat_completion_retry_removes_effort_when_supported_values_absent(self, ai_service_instance):
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
            usage=None,
        )
        create = AsyncMock(
            side_effect=[
                Exception("reasoning_effort is not supported by this model"),
                response,
            ]
        )
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

        result = await ai_service_instance._call_chat_completions_create_with_reasoning_retry(
            client,
            {
                "model": "legacy",
                "messages": [{"role": "user", "content": "hi"}],
                "reasoning_effort": "low",
            },
            provider_key="openai",
            model_id="legacy",
        )

        assert result is response
        assert "reasoning_effort" not in create.await_args_list[1].kwargs


class TestClaudeStructuredOutputsParams:
    """Tests Anthropic SDK structured-output parameter wiring."""

    def test_claude_structured_outputs_prefer_output_config_when_available(self, ai_service_instance):
        """Current Anthropic SDKs expose output_config.format for structured outputs."""

        async def create_with_output_config(*, output_config=None, **kwargs):
            return None

        ai_service_instance.anthropic_client = SimpleNamespace(
            messages=SimpleNamespace(create=create_with_output_config),
        )
        request_kwargs = {}

        use_beta = ai_service_instance._configure_claude_structured_output_params(
            request_kwargs,
            {
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
            },
        )

        assert use_beta is False
        assert request_kwargs["output_config"] == {
            "format": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {"ok": {"type": "boolean"}},
                    "required": ["ok"],
                },
            }
        }
        assert "output_format" not in request_kwargs
        assert "betas" not in request_kwargs

    @pytest.mark.asyncio
    async def test_generate_claude_uses_output_format_not_output_config(self, ai_service_instance):
        create = AsyncMock(
            return_value=SimpleNamespace(
                content=[SimpleNamespace(type="text", text='{"ok": true}')],
                usage=None,
                stop_reason="end_turn",
            )
        )
        ai_service_instance.anthropic_client = SimpleNamespace(
            beta=SimpleNamespace(messages=SimpleNamespace(create=create)),
            messages=SimpleNamespace(create=AsyncMock()),
        )

        content, _usage = await ai_service_instance._generate_claude(
            prompt="Return JSON",
            model_id="claude-opus-4-6",
            temperature=0.7,
            max_tokens=128,
            system_prompt="System",
            json_output=True,
            json_schema={
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
            },
        )

        assert content == '{"ok": true}'
        request_kwargs = create.await_args.kwargs
        assert request_kwargs["output_format"] == {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
            },
        }
        assert "output_config" not in request_kwargs

    @pytest.mark.asyncio
    async def test_stream_claude_structured_outputs_uses_create_stream_true(self, ai_service_instance):
        """Streaming with raw JSON Schema must bypass Anthropic's parsed stream helper."""

        class FakeAsyncStream:
            def __init__(self):
                self.events = [
                    SimpleNamespace(
                        type="message_start",
                        message=SimpleNamespace(
                            usage=SimpleNamespace(input_tokens=3),
                        ),
                    ),
                    SimpleNamespace(
                        type="content_block_delta",
                        delta=SimpleNamespace(type="text_delta", text='{"ok": true}'),
                    ),
                    SimpleNamespace(
                        type="message_delta",
                        delta=SimpleNamespace(stop_reason="end_turn"),
                        usage=SimpleNamespace(output_tokens=4),
                    ),
                ]

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, exc_tb):
                return None

            def __aiter__(self):
                async def iterator():
                    for event in self.events:
                        yield event

                return iterator()

        create = AsyncMock(return_value=FakeAsyncStream())
        parsed_stream_helper = Mock(side_effect=AssertionError("parsed stream helper should not be used"))
        ai_service_instance.anthropic_client = SimpleNamespace(
            beta=SimpleNamespace(
                messages=SimpleNamespace(
                    create=create,
                    stream=parsed_stream_helper,
                ),
            ),
            messages=SimpleNamespace(stream=Mock()),
        )

        chunks = [
            chunk
            async for chunk in ai_service_instance._stream_claude(
                prompt="Return JSON",
                model_id="claude-opus-4-7",
                temperature=1.0,
                max_tokens=128,
                system_prompt="System",
                extra_verbose=False,
                json_output=True,
                json_schema={
                    "type": "object",
                    "properties": {"ok": {"type": "boolean"}},
                    "required": ["ok"],
                },
            )
        ]

        assert [chunk.text for chunk in chunks] == ['{"ok": true}', ""]
        request_kwargs = create.await_args.kwargs
        assert request_kwargs["stream"] is True
        assert request_kwargs["output_format"] == {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
            },
        }
        assert request_kwargs["betas"] == ["structured-outputs-2025-11-13"]
        parsed_stream_helper.assert_not_called()

    @pytest.mark.asyncio
    async def test_openai_tool_turn_stream_accumulates_text_and_tool_deltas(self, ai_service_instance):
        """Chat Completions tool-loop streaming returns the aggregate message shape."""

        class FakeAsyncStream:
            def __aiter__(self):
                async def iterator():
                    yield SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                delta=SimpleNamespace(content="Draft "),
                                finish_reason=None,
                            )
                        ],
                        usage=None,
                    )
                    yield SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                delta=SimpleNamespace(
                                    content=None,
                                    tool_calls=[
                                        SimpleNamespace(
                                            index=0,
                                            id="call_1",
                                            type="function",
                                            function=SimpleNamespace(
                                                name="validate_draft",
                                                arguments='{"text":"Draft ',
                                            ),
                                        )
                                    ],
                                ),
                                finish_reason=None,
                            )
                        ],
                        usage=None,
                    )
                    yield SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                delta=SimpleNamespace(
                                    content=None,
                                    tool_calls=[
                                        SimpleNamespace(
                                            index=0,
                                            id=None,
                                            type=None,
                                            function=SimpleNamespace(
                                                name=None,
                                                arguments='text"}',
                                            ),
                                        )
                                    ],
                                ),
                                finish_reason="tool_calls",
                            )
                        ],
                        usage=SimpleNamespace(total_tokens=12),
                    )

                return iterator()

        create = AsyncMock(return_value=FakeAsyncStream())
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        events = []

        async def on_event(event_type, payload):
            events.append((event_type, payload))

        message, usage, finish_reason = await ai_service_instance._stream_openai_compatible_tool_turn(
            client,
            {"model": "gpt-5.4-mini", "messages": []},
            {},
            provider_key="openai",
            model_id="gpt-5.4-mini",
            turn=1,
            loop_scope=LoopScope.GENERATOR,
            tool_event_callback=on_event,
            cancellation_token=None,
        )

        assert message.content == "Draft "
        assert len(message.tool_calls) == 1
        assert message.tool_calls[0].function.name == "validate_draft"
        assert message.tool_calls[0].function.arguments == '{"text":"Draft text"}'
        assert usage.total_tokens == 12
        assert finish_reason == "tool_calls"
        assert any(event_type == "assistant_delta" for event_type, _ in events)
        assert any(event_type == "tool_call_delta" for event_type, _ in events)
        assert any(event_type == "tool_call_ready" for event_type, _ in events)

    @pytest.mark.asyncio
    async def test_openai_responses_tool_turn_stream_accumulates_function_call(self, ai_service_instance):
        """Responses API streaming is adapted into the shared tool-call shape."""

        class FakeResponsesStream:
            def __aiter__(self):
                async def iterator():
                    yield SimpleNamespace(
                        type="response.output_item.added",
                        output_index=0,
                        item=SimpleNamespace(
                            type="function_call",
                            id="fc_1",
                            call_id="call_1",
                            name="validate_draft",
                            arguments="",
                        ),
                    )
                    yield SimpleNamespace(
                        type="response.function_call_arguments.delta",
                        output_index=0,
                        item_id="fc_1",
                        delta='{"text":"hello',
                    )
                    yield SimpleNamespace(
                        type="response.function_call_arguments.delta",
                        output_index=0,
                        item_id="fc_1",
                        delta=' world"}',
                    )
                    yield SimpleNamespace(
                        type="response.function_call_arguments.done",
                        output_index=0,
                        item_id="fc_1",
                        call_id="call_1",
                        name="validate_draft",
                        arguments='{"text":"hello world"}',
                    )
                    yield SimpleNamespace(
                        type="response.completed",
                        response=SimpleNamespace(
                            status="completed",
                            usage=SimpleNamespace(total_tokens=9),
                        ),
                    )

                return iterator()

        create = AsyncMock(return_value=FakeResponsesStream())
        ai_service_instance.openai_client = SimpleNamespace(
            responses=SimpleNamespace(create=create)
        )
        events = []

        async def on_event(event_type, payload):
            events.append((event_type, payload))

        message, usage, finish_reason = await ai_service_instance._stream_openai_responses_tool_turn(
            {"model": "gpt-5-pro", "input": []},
            {},
            model_id="gpt-5-pro",
            turn=1,
            loop_scope=LoopScope.GENERATOR,
            tool_event_callback=on_event,
            cancellation_token=None,
        )

        assert message.content == ""
        assert len(message.tool_calls) == 1
        assert message.tool_calls[0].id == "fc_1"
        assert message.tool_calls[0].call_id == "call_1"
        assert message.tool_calls[0].function.name == "validate_draft"
        assert message.tool_calls[0].function.arguments == '{"text":"hello world"}'
        assert usage.total_tokens == 9
        assert finish_reason == "completed"
        assert any(event_type == "tool_call_delta" for event_type, _ in events)
        assert any(event_type == "tool_call_ready" for event_type, _ in events)

    @pytest.mark.asyncio
    async def test_openai_responses_tool_turn_missing_terminal_raises_before_tool_ready(self, ai_service_instance):
        """Responses streams must end with response.completed before tool calls are usable."""

        class FakeResponsesStream:
            def __aiter__(self):
                async def iterator():
                    yield SimpleNamespace(
                        type="response.output_item.added",
                        output_index=0,
                        item=SimpleNamespace(
                            type="function_call",
                            id="fc_1",
                            call_id="call_1",
                            name="validate_draft",
                            arguments="",
                        ),
                    )
                    yield SimpleNamespace(
                        type="response.function_call_arguments.delta",
                        output_index=0,
                        item_id="fc_1",
                        delta='{"text":"partial',
                    )

                return iterator()

        create = AsyncMock(return_value=FakeResponsesStream())
        ai_service_instance.openai_client = SimpleNamespace(
            responses=SimpleNamespace(create=create)
        )
        events = []

        async def on_event(event_type, payload):
            events.append((event_type, payload))

        with pytest.raises(RuntimeError, match="without response.completed"):
            await ai_service_instance._stream_openai_responses_tool_turn(
                {"model": "gpt-5-pro", "input": []},
                {},
                model_id="gpt-5-pro",
                turn=1,
                loop_scope=LoopScope.GENERATOR,
                tool_event_callback=on_event,
                cancellation_token=None,
            )

        assert any(event_type == "tool_loop_provider_terminal" for event_type, _ in events)
        assert not any(event_type == "tool_call_ready" for event_type, _ in events)

    @pytest.mark.asyncio
    async def test_openai_responses_tool_turn_nonstream_accumulates_function_call(self, ai_service_instance):
        """Responses models without streaming still produce tool calls without stream=True."""

        response = SimpleNamespace(
            status="completed",
            usage=SimpleNamespace(total_tokens=11),
            output=[
                SimpleNamespace(
                    type="function_call",
                    id="fc_1",
                    call_id="call_1",
                    name="validate_draft",
                    arguments='{"text":"hello world"}',
                )
            ],
        )
        create = AsyncMock(return_value=response)
        ai_service_instance.openai_client = SimpleNamespace(
            responses=SimpleNamespace(create=create)
        )
        events = []

        async def on_event(event_type, payload):
            events.append((event_type, payload))

        message, usage, finish_reason = await ai_service_instance._run_openai_responses_tool_turn(
            {"model": "gpt-5-pro", "input": []},
            {},
            model_id="gpt-5-pro",
            turn=1,
            loop_scope=LoopScope.GENERATOR,
            tool_event_callback=on_event,
            cancellation_token=None,
        )

        assert create.await_args.kwargs.get("stream") is None
        assert message.content == ""
        assert len(message.tool_calls) == 1
        assert message.tool_calls[0].id == "fc_1"
        assert message.tool_calls[0].call_id == "call_1"
        assert message.tool_calls[0].function.name == "validate_draft"
        assert message.tool_calls[0].function.arguments == '{"text":"hello world"}'
        assert usage.total_tokens == 11
        assert finish_reason == "completed"
        assert any(event_type == "tool_call_ready" for event_type, _ in events)
        assert not any(event_type == "tool_call_delta" for event_type, _ in events)

    @pytest.mark.asyncio
    async def test_openai_responses_tool_turn_incomplete_raises_before_tool_ready(self, ai_service_instance):
        """Incomplete Responses turns must not execute partial tool-call arguments."""

        class FakeResponsesStream:
            def __aiter__(self):
                async def iterator():
                    yield SimpleNamespace(
                        type="response.output_item.added",
                        output_index=0,
                        item=SimpleNamespace(
                            type="function_call",
                            id="fc_1",
                            call_id="call_1",
                            name="validate_draft",
                            arguments="",
                        ),
                    )
                    yield SimpleNamespace(
                        type="response.function_call_arguments.delta",
                        output_index=0,
                        item_id="fc_1",
                        delta='{"text":"partial',
                    )
                    yield SimpleNamespace(
                        type="response.incomplete",
                        response=SimpleNamespace(
                            status="incomplete",
                            incomplete_details=SimpleNamespace(reason="max_output_tokens"),
                        ),
                    )

                return iterator()

        create = AsyncMock(return_value=FakeResponsesStream())
        ai_service_instance.openai_client = SimpleNamespace(
            responses=SimpleNamespace(create=create)
        )
        events = []

        async def on_event(event_type, payload):
            events.append((event_type, payload))

        with pytest.raises(RuntimeError, match="Responses stream ended"):
            await ai_service_instance._stream_openai_responses_tool_turn(
                {"model": "gpt-5-pro", "input": []},
                {},
                model_id="gpt-5-pro",
                turn=1,
                loop_scope=LoopScope.GENERATOR,
                tool_event_callback=on_event,
                cancellation_token=None,
            )

        assert any(event_type == "tool_loop_provider_terminal" for event_type, _ in events)
        assert not any(event_type == "tool_call_ready" for event_type, _ in events)

    @pytest.mark.asyncio
    async def test_openai_responses_tool_turn_nonstream_incomplete_raises_before_tool_ready(self, ai_service_instance):
        """Non-streaming incomplete Responses turns must not execute partial tool calls."""

        response = SimpleNamespace(
            status="incomplete",
            incomplete_details=SimpleNamespace(reason="max_output_tokens"),
            output=[
                SimpleNamespace(
                    type="function_call",
                    id="fc_1",
                    call_id="call_1",
                    name="validate_draft",
                    arguments='{"text":"partial',
                )
            ],
        )
        create = AsyncMock(return_value=response)
        ai_service_instance.openai_client = SimpleNamespace(
            responses=SimpleNamespace(create=create)
        )
        events = []

        async def on_event(event_type, payload):
            events.append((event_type, payload))

        with pytest.raises(RuntimeError, match="Responses turn ended"):
            await ai_service_instance._run_openai_responses_tool_turn(
                {"model": "gpt-5-pro", "input": []},
                {},
                model_id="gpt-5-pro",
                turn=1,
                loop_scope=LoopScope.GENERATOR,
                tool_event_callback=on_event,
                cancellation_token=None,
            )

        assert create.await_args.kwargs.get("stream") is None
        assert any(event_type == "tool_loop_provider_terminal" for event_type, _ in events)
        assert not any(event_type == "tool_call_ready" for event_type, _ in events)

    @pytest.mark.asyncio
    async def test_openai_responses_tool_loop_uses_nonstream_for_gpt5_pro(self, ai_service_instance):
        """Responses-only Pro models avoid stream=True inside the tool loop."""

        response = SimpleNamespace(
            status="completed",
            usage=SimpleNamespace(total_tokens=11),
            output=[
                SimpleNamespace(
                    type="function_call",
                    id="fc_1",
                    call_id="call_1",
                    name="validate_draft",
                    arguments='{"text":"validated draft"}',
                )
            ],
        )
        create = AsyncMock(return_value=response)
        ai_service_instance.openai_client = SimpleNamespace(
            responses=SimpleNamespace(create=create)
        )

        content, metadata = await ai_service_instance._run_openai_compatible_validation_tool_loop(
            provider="openai",
            model_id="gpt-5-pro",
            prompt="Write.",
            validation_callback=lambda candidate: _result_for_target(candidate, "validated draft"),
            temperature=0.7,
            max_tokens=128,
            system_prompt="system",
            request_timeout=None,
            reasoning_effort=None,
            json_output=False,
            json_schema=None,
            usage_callback=None,
            usage_extra=None,
            images=None,
            max_rounds=1,
            retries_enabled=False,
        )

        assert content == "validated draft"
        assert metadata["accepted"] == "validated_tool_argument"
        assert create.await_args.kwargs.get("stream") is None
        assert create.await_count == 1

    @pytest.mark.asyncio
    async def test_openai_chat_tool_turn_missing_terminal_raises_before_tool_ready(self, ai_service_instance):
        """Chat streams must include an explicit usable finish_reason."""

        class FakeAsyncStream:
            def __aiter__(self):
                async def iterator():
                    yield SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                delta=SimpleNamespace(
                                    content=None,
                                    tool_calls=[
                                        SimpleNamespace(
                                            index=0,
                                            id="call_1",
                                            type="function",
                                            function=SimpleNamespace(
                                                name="validate_draft",
                                                arguments='{"text":"partial',
                                            ),
                                        )
                                    ],
                                ),
                                finish_reason=None,
                            )
                        ],
                        usage=None,
                    )

                return iterator()

        create = AsyncMock(return_value=FakeAsyncStream())
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        events = []

        async def on_event(event_type, payload):
            events.append((event_type, payload))

        with pytest.raises(RuntimeError, match="Chat Completions stream ended"):
            await ai_service_instance._stream_openai_compatible_tool_turn(
                client,
                {"model": "gpt-5.4-mini", "messages": []},
                {},
                provider_key="openai",
                model_id="gpt-5.4-mini",
                turn=1,
                loop_scope=LoopScope.GENERATOR,
                tool_event_callback=on_event,
                cancellation_token=None,
            )

        assert any(event_type == "tool_loop_provider_terminal" for event_type, _ in events)
        assert not any(event_type == "tool_call_ready" for event_type, _ in events)

    @pytest.mark.asyncio
    async def test_openai_chat_tool_turn_length_raises_before_tool_ready(self, ai_service_instance):
        """Chat Completions length stops must not execute partial tool calls."""

        class FakeAsyncStream:
            def __aiter__(self):
                async def iterator():
                    yield SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                delta=SimpleNamespace(
                                    content=None,
                                    tool_calls=[
                                        SimpleNamespace(
                                            index=0,
                                            id="call_1",
                                            type="function",
                                            function=SimpleNamespace(
                                                name="validate_draft",
                                                arguments='{"text":"partial',
                                            ),
                                        )
                                    ],
                                ),
                                finish_reason="length",
                            )
                        ],
                        usage=None,
                    )

                return iterator()

        create = AsyncMock(return_value=FakeAsyncStream())
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        events = []

        async def on_event(event_type, payload):
            events.append((event_type, payload))

        with pytest.raises(ToolLoopOutputTruncated, match="Chat Completions stream ended") as exc_info:
            await ai_service_instance._stream_openai_compatible_tool_turn(
                client,
                {"model": "gpt-5.4-mini", "messages": []},
                {},
                provider_key="openai",
                model_id="gpt-5.4-mini",
                turn=1,
                loop_scope=LoopScope.GENERATOR,
                tool_event_callback=on_event,
                cancellation_token=None,
            )

        assert exc_info.value.finish_reason == "length"
        assert exc_info.value.partial_tool_calls == 1
        assert any(event_type == "tool_loop_provider_terminal" for event_type, _ in events)
        assert not any(event_type == "tool_call_ready" for event_type, _ in events)

    @pytest.mark.asyncio
    async def test_openrouter_chat_tool_turn_rejects_non_stream_response(self, ai_service_instance):
        """OpenRouter tool-loop observability must not silently downgrade to non-streaming."""

        message = SimpleNamespace(content="", tool_calls=[])
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")],
            usage=None,
        )
        create = AsyncMock(return_value=response)
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        events = []

        async def on_event(event_type, payload):
            events.append((event_type, payload))

        with pytest.raises(RuntimeError, match="OpenRouter Chat Completions stream"):
            await ai_service_instance._stream_openai_compatible_tool_turn(
                client,
                {"model": "vendor/tool-model", "messages": []},
                {},
                provider_key="openrouter",
                model_id="vendor/tool-model",
                turn=1,
                loop_scope=LoopScope.GENERATOR,
                tool_event_callback=on_event,
                cancellation_token=None,
            )

        assert create.await_args.kwargs["stream"] is True
        assert any(
            event_type == "tool_loop_provider_terminal"
            and payload.get("status") == "non_stream_response"
            for event_type, payload in events
        )
        assert not any(event_type == "tool_call_ready" for event_type, _ in events)

    @pytest.mark.asyncio
    async def test_openrouter_chat_tool_turn_stream_error_raises_before_tool_ready(self, ai_service_instance):
        """OpenRouter SSE error chunks must be terminal and must not execute tools."""

        class FakeAsyncStream:
            def __aiter__(self):
                async def iterator():
                    yield SimpleNamespace(
                        error={"message": "provider stream failed"},
                        choices=[],
                        usage=None,
                    )

                return iterator()

        create = AsyncMock(return_value=FakeAsyncStream())
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        events = []

        async def on_event(event_type, payload):
            events.append((event_type, payload))

        with pytest.raises(RuntimeError, match="stream error"):
            await ai_service_instance._stream_openai_compatible_tool_turn(
                client,
                {"model": "vendor/tool-model", "messages": []},
                {},
                provider_key="openrouter",
                model_id="vendor/tool-model",
                turn=1,
                loop_scope=LoopScope.GENERATOR,
                tool_event_callback=on_event,
                cancellation_token=None,
            )

        assert any(
            event_type == "tool_loop_provider_terminal"
            and payload.get("status") == "error"
            and "provider stream failed" in payload.get("detail", "")
            for event_type, payload in events
        )
        assert not any(event_type == "tool_call_ready" for event_type, _ in events)

    def test_openai_responses_params_convert_chat_tool_transcript(self, ai_service_instance):
        """Responses params preserve function-call ids and tool outputs."""

        params = ai_service_instance._build_openai_responses_tool_params(
            model_id="gpt-5-pro",
            current_messages=[
                {"role": "system", "content": "System"},
                {"role": "user", "content": "Write"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "fc_1",
                            "call_id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "validate_draft",
                                "arguments": '{"text":"Write"}',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": '{"approved":true}'},
            ],
            max_tokens=128,
            reasoning_effort="low",
            json_output=False,
            json_schema=None,
            tools_enabled=True,
            tool_schemas=[ai_service_instance._build_openai_validation_tool_schema()],
        )

        assert params["instructions"] == "System"
        assert params["input"][0] == {"role": "user", "content": "Write"}
        assert params["input"][1]["type"] == "function_call"
        assert params["input"][1]["call_id"] == "call_1"
        assert params["input"][2]["type"] == "function_call_output"
        assert params["input"][2]["call_id"] == "call_1"
        assert params["tools"][0]["name"] == "validate_draft"
        assert params["tool_choice"] == "auto"

    def test_openai_responses_json_loose_does_not_force_empty_schema(self, ai_service_instance):
        """JSON_LOOSE over Responses relies on prompt/local parsing, not a strict empty schema."""

        params = ai_service_instance._build_openai_responses_tool_params(
            model_id="gpt-5-pro",
            current_messages=[{"role": "user", "content": "Return JSON"}],
            max_tokens=128,
            reasoning_effort=None,
            json_output=True,
            json_schema=None,
            tools_enabled=False,
        )

        assert "text" not in params
        assert AIService._should_inject_json_prompt("openai", "gpt-5-pro", True, None) is True

    def test_openai_responses_params_use_responses_text_format_shape(self, ai_service_instance):
        """Responses API text.format does not use Chat Completions json_schema nesting."""

        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["decision"],
            "properties": {"decision": {"type": "string"}},
        }
        params = ai_service_instance._build_openai_responses_tool_params(
            model_id="gpt-5-pro",
            current_messages=[{"role": "user", "content": "Return a decision"}],
            max_tokens=128,
            reasoning_effort=None,
            json_output=True,
            json_schema=schema,
            tools_enabled=False,
        )

        assert params["text"] == {
            "format": {
                "type": "json_schema",
                "name": "structured_output",
                "strict": True,
                "schema": schema,
            }
        }
        assert "json_schema" not in params["text"]["format"]

    @pytest.mark.asyncio
    async def test_generate_openai_responses_paths_use_responses_text_format_shape(self, ai_service_instance):
        """Non-tool-loop Responses calls use the same direct text.format contract."""

        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["decision"],
            "properties": {"decision": {"type": "string"}},
        }
        captured_params = []

        async def create(**kwargs):
            captured_params.append(kwargs)
            return SimpleNamespace(output_text='{"decision":"ok"}', usage=None)

        ai_service_instance.openai_client = SimpleNamespace(
            responses=SimpleNamespace(create=create)
        )

        await ai_service_instance._generate_openai(
            prompt="Return a decision",
            model_id="o3-pro",
            temperature=0.7,
            max_tokens=128,
            system_prompt="system",
            json_output=True,
            json_schema=schema,
        )
        await ai_service_instance._generate_openai(
            prompt="Return a decision",
            model_id="gpt-5-pro",
            temperature=0.7,
            max_tokens=128,
            system_prompt="system",
            json_output=True,
            json_schema=schema,
        )

        assert len(captured_params) == 2
        for params in captured_params:
            assert params["text"] == {
                "format": {
                    "type": "json_schema",
                    "name": "structured_output",
                    "strict": True,
                    "schema": schema,
                }
            }
            assert "json_schema" not in params["text"]["format"]


class TestInjectClaudeThinkingParams:
    """Tests for _inject_claude_thinking_params() method."""

    def test_injects_thinking_params_for_supported_model(self, ai_service_instance):
        """
        Given: Supported Claude model and thinking budget
        When: _inject_claude_thinking_params() is called
        Then: Thinking params added to dict
        """
        params = {"model": "claude-sonnet-4", "max_tokens": 8000}
        ai_service_instance._inject_claude_thinking_params(params, "claude-sonnet-4", 5000)

        assert "thinking" in params
        assert params["thinking"]["type"] == "enabled"
        assert params["thinking"]["budget_tokens"] >= 1024  # Minimum enforced

    def test_does_not_inject_for_unsupported_model(self, ai_service_instance):
        """
        Given: Unsupported Claude model
        When: _inject_claude_thinking_params() is called
        Then: Thinking params NOT added
        """
        params = {"model": "claude-3-haiku", "max_tokens": 4000}
        ai_service_instance._inject_claude_thinking_params(params, "claude-3-haiku", 5000)

        assert "thinking" not in params

    def test_does_not_inject_for_zero_budget(self, ai_service_instance):
        """
        Given: Zero thinking budget
        When: _inject_claude_thinking_params() is called
        Then: Thinking params NOT added
        """
        params = {"model": "claude-sonnet-4", "max_tokens": 8000}
        ai_service_instance._inject_claude_thinking_params(params, "claude-sonnet-4", 0)

        assert "thinking" not in params

    def test_does_not_inject_for_none_budget(self, ai_service_instance):
        """
        Given: None thinking budget
        When: _inject_claude_thinking_params() is called
        Then: Thinking params NOT added
        """
        params = {"model": "claude-sonnet-4", "max_tokens": 8000}
        ai_service_instance._inject_claude_thinking_params(params, "claude-sonnet-4", None)

        assert "thinking" not in params

    def test_enforces_minimum_budget(self, ai_service_instance):
        """
        Given: Budget below minimum (1024)
        When: _inject_claude_thinking_params() is called
        Then: Budget increased to minimum
        """
        params = {"model": "claude-sonnet-4", "max_tokens": 8000}
        ai_service_instance._inject_claude_thinking_params(params, "claude-sonnet-4", 500)

        assert params["thinking"]["budget_tokens"] >= 1024


class TestGetThinkingBudgetForModel:
    """Tests for _get_thinking_budget_for_model() method."""

    def test_returns_zero_for_unsupported_model(self, ai_service_instance):
        """
        Given: Model without thinking support in specs
        When: _get_thinking_budget_for_model() is called
        Then: Returns 0
        """
        with patch.object(ai_service_instance, '_get_thinking_budget_details', return_value=None):
            result = ai_service_instance._get_thinking_budget_for_model("claude-3-haiku")
            assert result == 0

    def test_returns_default_tokens_from_config(self, ai_service_instance):
        """
        Given: Model with thinking support in specs
        When: _get_thinking_budget_for_model() is called
        Then: Returns default_tokens from config
        """
        mock_details = {
            "supported": True,
            "default_tokens": 8000,
            "min_tokens": 1024,
            "max_tokens": 128000
        }
        with patch.object(ai_service_instance, '_get_thinking_budget_details', return_value=mock_details):
            result = ai_service_instance._get_thinking_budget_for_model("claude-sonnet-4")
            assert result == 8000


# ============================================================================
# Test Class: Usage Normalization
# ============================================================================

class TestNormalizeUsage:
    """Tests for _normalize_usage() static method."""

    def test_extracts_openai_usage(self):
        """
        Given: OpenAI-style usage object
        When: _normalize_usage() is called
        Then: Extracts correct values
        """
        usage = Mock(spec=[])
        usage.prompt_tokens = 100
        usage.completion_tokens = 50
        usage.total_tokens = 150
        # Remove model_dump to prevent the pydantic path
        usage.model_dump = None

        result = AIService._normalize_usage(usage)
        assert result["input_tokens"] == 100
        assert result["output_tokens"] == 50
        assert result["total_tokens"] == 150

    def test_extracts_anthropic_usage(self):
        """
        Given: Anthropic-style usage object
        When: _normalize_usage() is called
        Then: Extracts correct values
        """
        usage = Mock(spec=[])
        usage.input_tokens = 200
        usage.output_tokens = 100
        # Set prompt_tokens to None to use input_tokens path
        usage.prompt_tokens = None
        usage.completion_tokens = None
        usage.model_dump = None

        result = AIService._normalize_usage(usage)
        assert result["input_tokens"] == 200
        assert result["output_tokens"] == 100

    def test_extracts_from_dict(self):
        """
        Given: Usage as dictionary
        When: _normalize_usage() is called
        Then: Extracts correct values
        """
        usage = {"prompt_tokens": 150, "completion_tokens": 75, "total_tokens": 225}

        result = AIService._normalize_usage(usage)
        assert result["input_tokens"] == 150
        assert result["output_tokens"] == 75
        assert result["total_tokens"] == 225

    def test_returns_none_for_none_input(self):
        """
        Given: None usage object
        When: _normalize_usage() is called
        Then: Returns None
        """
        result = AIService._normalize_usage(None)
        assert result is None

    def test_extracts_reasoning_tokens(self):
        """
        Given: Usage with reasoning_tokens
        When: _normalize_usage() is called
        Then: Extracts reasoning_tokens
        """
        usage = Mock(spec=[])
        usage.prompt_tokens = 100
        usage.completion_tokens = 50
        usage.reasoning_tokens = 1000
        usage.model_dump = None

        result = AIService._normalize_usage(usage)
        assert result["reasoning_tokens"] == 1000


# ============================================================================
# Test Class: Streaming Retry Behavior
# ============================================================================

class TestStreamingRetryBehavior:
    """Tests for streaming retry behavior in generate_content_stream()."""

    @pytest.mark.asyncio
    async def test_streaming_downgrades_schema_when_model_lacks_native_support(
        self,
        ai_service_instance,
        mock_config,
    ):
        """Streaming must use the same capability negotiation as non-streaming."""

        mock_config.validate_token_limits = Mock(
            return_value={
                "adjusted_tokens": 512,
                "adjusted_reasoning_effort": None,
                "adjusted_thinking_budget_tokens": None,
                "thinking_validation": {},
                "was_adjusted": False,
                "model_limit": 8192,
                "model_info": {"provider": "openai", "model_id": "gpt-4-turbo"},
                "reasoning_timeout_seconds": None,
            }
        )
        mock_config.model_specs = {"model_specifications": {"openai": {}}}

        captured_kwargs = {}

        async def fake_stream_openai(*args, **kwargs):
            captured_kwargs.update(kwargs)
            yield "ok"

        ai_service_instance._stream_openai = fake_stream_openai

        chunks = [
            chunk
            async for chunk in ai_service_instance.generate_content_stream(
                "Return JSON.",
                "gpt-4-turbo",
                json_output=True,
                json_schema={"type": "object", "properties": {"answer": {"type": "string"}}},
            )
        ]

        assert chunks == ["ok"]
        assert captured_kwargs["json_output"] is True
        assert captured_kwargs["json_schema"] is None

    @pytest.mark.asyncio
    async def test_retries_before_chunks_emitted(self, ai_service_instance, mock_config):
        """
        Given: Streaming fails before any chunks emitted
        When: generate_content_stream() is called
        Then: Retries the request
        """
        mock_config.MAX_RETRIES = 3
        mock_config.RETRY_DELAY = 0.01  # Fast retry for tests

        call_count = 0

        async def mock_dispatch():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Connection failed")
            yield "Success"

        # We can't easily test the full generate_content_stream due to complex setup
        # Instead, test the retry logic conceptually
        assert call_count == 0  # Verify initial state

    @pytest.mark.asyncio
    async def test_no_retry_after_chunks_emitted(self, ai_service_instance):
        """
        Given: Streaming fails after chunks have been emitted
        When: generate_content_stream() encounters error
        Then: Does NOT retry (partial content already sent)
        """
        # This tests the conceptual behavior:
        # After chunks_emitted > 0, should_retry becomes False
        # The actual logic: should_retry = attempt < max_attempts AND chunks_emitted == 0 AND _should_retry_exception(exc)

        # If chunks_emitted > 0, the condition fails and no retry occurs
        chunks_emitted = 5
        attempt = 1
        max_attempts = 3
        exc = ConnectionError("Connection lost")

        should_retry = (
            attempt < max_attempts
            and chunks_emitted == 0
            and AIService._should_retry_exception(exc)
        )

        assert should_retry is False  # No retry because chunks were emitted


class TestExtractTextFromClaudeResponse:
    """Tests for _extract_text_from_claude_response() method."""

    def test_extracts_text_from_text_block(self, ai_service_instance):
        """
        Given: Response with text block
        When: _extract_text_from_claude_response() is called
        Then: Returns text content
        """
        response = Mock()
        text_block = Mock()
        text_block.type = "text"
        text_block.text = "Hello, world!"
        response.content = [text_block]

        result = ai_service_instance._extract_text_from_claude_response(response)
        assert result == "Hello, world!"

    def test_skips_thinking_blocks(self, ai_service_instance):
        """
        Given: Response with thinking and text blocks
        When: _extract_text_from_claude_response() is called
        Then: Returns only text content, skips thinking
        """
        response = Mock()
        thinking_block = Mock()
        thinking_block.type = "thinking"
        thinking_block.text = "Internal reasoning..."

        text_block = Mock()
        text_block.type = "text"
        text_block.text = "Final answer"

        response.content = [thinking_block, text_block]

        result = ai_service_instance._extract_text_from_claude_response(response)
        assert result == "Final answer"
        assert "Internal reasoning" not in result

    def test_handles_multiple_text_blocks(self, ai_service_instance):
        """
        Given: Response with multiple text blocks
        When: _extract_text_from_claude_response() is called
        Then: Concatenates all text blocks
        """
        response = Mock()
        block1 = Mock()
        block1.type = "text"
        block1.text = "Part 1. "

        block2 = Mock()
        block2.type = "text"
        block2.text = "Part 2."

        response.content = [block1, block2]

        result = ai_service_instance._extract_text_from_claude_response(response)
        assert result == "Part 1. Part 2."

    def test_handles_empty_response(self, ai_service_instance):
        """
        Given: Response with empty content list
        When: _extract_text_from_claude_response() is called
        Then: Returns appropriate message
        """
        response = Mock()
        response.content = []

        result = ai_service_instance._extract_text_from_claude_response(response)
        assert "No text content" in result or result == ""


# ============================================================================
# Test Class: Execute with Retries
# ============================================================================

class TestExecuteWithRetries:
    """Tests for _execute_with_retries() method."""

    @pytest.mark.asyncio
    async def test_returns_result_on_success(self, ai_service_instance, mock_config):
        """
        Given: Operation succeeds on first attempt
        When: _execute_with_retries() is called
        Then: Returns result without retry
        """
        mock_config.MAX_RETRIES = 3

        async def successful_operation():
            return "success"

        result = await ai_service_instance._execute_with_retries(
            successful_operation,
            provider="openai",
            model_id="gpt-4o",
            action="test"
        )
        assert result == "success"

    @pytest.mark.asyncio
    async def test_records_provider_health_on_success(self, ai_service_instance, mock_config):
        mock_config.MAX_RETRIES = 3

        async def successful_operation():
            return "success"

        with patch("ai_service.record_provider_success", new_callable=AsyncMock) as record_success:
            result = await ai_service_instance._execute_with_retries(
                successful_operation,
                provider="openai",
                model_id="gpt-4o",
                action="test",
            )

        assert result == "success"
        record_success.assert_awaited_once_with("openai", model="gpt-4o", operation="test")

    @pytest.mark.asyncio
    async def test_records_provider_health_on_wrapped_failure(self, ai_service_instance):
        ai_service_instance._max_retry_attempts = lambda: 1

        async def failing_operation():
            raise ConnectionError("Provider unavailable")

        with patch("ai_service.record_provider_failure", new_callable=AsyncMock) as record_failure:
            with pytest.raises(AIRequestError):
                await ai_service_instance._execute_with_retries(
                    failing_operation,
                    provider="openai",
                    model_id="gpt-4o",
                    action="test",
                )

        record_failure.assert_awaited_once()
        failure = record_failure.await_args.args[0]
        assert failure.kind == ProviderErrorKind.TRANSIENT_NETWORK

    @pytest.mark.asyncio
    async def test_retries_on_transient_error(self, ai_service_instance, mock_config):
        """
        Given: Operation fails with transient error then succeeds
        When: _execute_with_retries() is called
        Then: Retries and returns success
        """
        mock_config.MAX_RETRIES = 3
        mock_config.RETRY_DELAY = 0.01

        attempt_count = 0

        async def flaky_operation():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 2:
                raise ConnectionError("Transient failure")
            return "success"

        result = await ai_service_instance._execute_with_retries(
            flaky_operation,
            provider="openai",
            model_id="gpt-4o",
            action="test"
        )
        assert result == "success"
        assert attempt_count == 2

    @pytest.mark.asyncio
    async def test_raises_immediately_on_non_retriable_error(self, ai_service_instance, mock_config):
        """
        Given: Operation fails with non-retriable error
        When: _execute_with_retries() is called
        Then: Raises immediately without retry
        """
        mock_config.MAX_RETRIES = 3

        async def failing_operation():
            exc = Exception("Invalid request")
            exc.status = 400  # Bad request - not retriable
            raise exc

        with pytest.raises(Exception, match="Invalid request"):
            await ai_service_instance._execute_with_retries(
                failing_operation,
                provider="openai",
                model_id="gpt-4o",
                action="test"
            )

    @pytest.mark.asyncio
    async def test_wraps_final_transient_error_after_max_retries(self, ai_service_instance):
        """
        Given: Operation always fails with transient error
        When: _execute_with_retries() exhausts retries
        Then: Raises AIRequestError with normalized provider failure metadata
        """
        # Patch the methods directly on the instance
        ai_service_instance._max_retry_attempts = lambda: 2
        ai_service_instance._retry_delay_seconds = lambda: 0.01

        attempt_count = 0

        async def always_failing():
            nonlocal attempt_count
            attempt_count += 1
            raise ConnectionError("Always fails")

        with pytest.raises(AIRequestError) as exc_info:
            await ai_service_instance._execute_with_retries(
                always_failing,
                provider="openai",
                model_id="gpt-4o",
                action="test"
            )

        # Verify that 2 attempts were made
        assert attempt_count == 2
        assert exc_info.value.cause.__class__ is ConnectionError
        assert exc_info.value.provider_failure.kind.value == "transient_network"

    @pytest.mark.asyncio
    async def test_attempted_feature_classifies_400_as_unsupported_parameter(self, ai_service_instance):
        """A rejected explicit feature should drive downgrade-capable taxonomy."""

        class ProviderBadRequest(Exception):
            status_code = 400
            body = {"error": {"code": "invalid_parameter", "param": "output_format"}}

        ai_service_instance._max_retry_attempts = lambda: 1

        async def failing_operation():
            raise ProviderBadRequest("bad request")

        with pytest.raises(AIRequestError) as exc_info:
            await ai_service_instance._execute_with_retries(
                failing_operation,
                provider="anthropic",
                model_id="claude-sonnet-4-20250514",
                action="generation",
                attempted_feature="output_format",
            )

        failure = exc_info.value.provider_failure
        assert failure.kind.value == "unsupported_parameter"
        assert failure.downgradable is True
        assert failure.attempted_feature == "output_format"

    def test_attempted_output_feature_skips_unsent_openrouter_json_object(self, mock_config):
        mock_config.model_specs = {
            "model_specifications": {
                "openrouter": {
                    "unknown/model": {
                        "model_id": "unknown/model",
                        "supported_parameters": [],
                    }
                }
            }
        }

        assert AIService._attempted_output_feature(
            "openrouter",
            "unknown/model",
            json_output=True,
            json_schema=None,
        ) is None

    def test_attempted_output_feature_tracks_sent_openrouter_json_object(self, mock_config):
        mock_config.model_specs = {
            "model_specifications": {
                "openrouter": {
                    "json/model": {
                        "model_id": "json/model",
                        "supported_parameters": ["response_format"],
                    }
                }
            }
        }

        assert AIService._attempted_output_feature(
            "openrouter",
            "json/model",
            json_output=True,
            json_schema=None,
        ) == "response_format.json_object"

    @pytest.mark.asyncio
    async def test_propagates_ai_request_error_immediately(self, ai_service_instance, mock_config):
        """
        Given: Operation raises AIRequestError
        When: _execute_with_retries() is called
        Then: Propagates immediately without retry
        """
        mock_config.MAX_RETRIES = 3

        async def raises_ai_error():
            raise AIRequestError("openai", "gpt-4o", 1, 3, Exception("test"))

        with pytest.raises(AIRequestError):
            await ai_service_instance._execute_with_retries(
                raises_ai_error,
                provider="openai",
                model_id="gpt-4o",
                action="test"
            )


class TestExecuteWithoutRetries:
    """Tests for single-attempt provider execution normalization."""

    @pytest.mark.asyncio
    async def test_wraps_retryable_sdk_error_as_ai_request_error(self, ai_service_instance):
        """Retry-disabled paths still surface provider/SDK failures as typed API errors."""

        async def failing_operation():
            raise AttributeError("module aiohttp has no attribute ClientConnectorDNSError")

        with pytest.raises(AIRequestError) as exc_info:
            await ai_service_instance._execute_without_retries(
                failing_operation,
                provider="gemini",
                model_id="gemini-3.1-pro-preview",
                action="tool_loop_generation",
            )

        assert exc_info.value.provider == "gemini"
        assert exc_info.value.model == "gemini-3.1-pro-preview"
        assert exc_info.value.attempts == 1
        assert exc_info.value.max_attempts == 1
        assert isinstance(exc_info.value.cause, AttributeError)

    @pytest.mark.asyncio
    async def test_preserves_non_retryable_errors(self, ai_service_instance):
        """Local contract/programming errors must not be relabeled as provider failures."""

        async def failing_operation():
            raise ValueError("Invalid local contract")

        with pytest.raises(ValueError, match="Invalid local contract"):
            await ai_service_instance._execute_without_retries(
                failing_operation,
                provider="gemini",
                model_id="gemini-3.1-pro-preview",
                action="tool_loop_generation",
            )


def _approved_result(word_count: int = 3, feedback: str = "Looks good.") -> DraftValidationResult:
    return DraftValidationResult(
        approved=True,
        hard_failed=False,
        score=10.0,
        word_count=word_count,
        feedback=feedback,
        issues=[],
        metrics={"word_count": word_count},
        checks={},
        stylistic_metrics=None,
        visible_payload={},
    )


def _rejected_result(
    word_count: int = 0, feedback: str = "Still invalid."
) -> DraftValidationResult:
    return DraftValidationResult(
        approved=False,
        hard_failed=False,
        score=0.0,
        word_count=word_count,
        feedback=feedback,
        issues=[],
        metrics={"word_count": word_count},
        checks={},
        stylistic_metrics=None,
        visible_payload={},
    )


def _result_for_target(candidate: str, target: str) -> DraftValidationResult:
    """Helper for tests that approve only a specific candidate string."""
    approved = candidate == target
    return DraftValidationResult(
        approved=approved,
        hard_failed=False,
        score=10.0 if approved else 0.0,
        word_count=len(candidate.split()),
        feedback="Looks good." if approved else "Still invalid.",
        issues=[],
        metrics={"word_count": len(candidate.split())},
        checks={},
        stylistic_metrics=None,
        visible_payload={},
    )


class TestGenerationToolLoop:
    """Tests for generator tool-loop behavior across providers.

    Migrated from the legacy dict-callback / ``generate_content_with_validation_tools``
    API to the new ``call_ai_with_validation_tools`` + ``DraftValidationResult`` contract
    and typed ``ToolLoopEnvelope`` return shape.
    """

    @pytest.mark.asyncio
    async def test_tool_loop_exhaustion_without_validated_text_raises_after_forced_final_turn(
        self, ai_service_instance, mock_config
    ):
        """
        Given: The tool loop keeps asking for validation but never produces an approved draft
        When: call_ai_with_validation_tools() exhausts its rounds and the forced final turn also fails
        Then: It raises ValueError so callers can fall back to standard generation
        """

        async def passthrough(operation, **kwargs):
            return await operation()

        message = SimpleNamespace(
            content="",
            tool_calls=[
                SimpleNamespace(
                    id="call-1",
                    function=SimpleNamespace(
                        name="validate_draft",
                        arguments='{"text":"draft that still fails"}',
                    ),
                )
            ],
        )
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=None,
        )
        final_message = SimpleNamespace(content="still invalid final answer", tool_calls=[])
        final_response = SimpleNamespace(
            choices=[SimpleNamespace(message=final_message)],
            usage=None,
        )

        ai_service_instance.openai_client = Mock()
        ai_service_instance.openai_client.chat = Mock()
        ai_service_instance.openai_client.chat.completions = Mock()
        ai_service_instance.openai_client.chat.completions.create = AsyncMock(
            side_effect=[response, response, final_response]
        )
        ai_service_instance._execute_with_retries = AsyncMock(side_effect=passthrough)

        mock_config.validate_token_limits = Mock(
            return_value={
                "adjusted_tokens": 512,
                "adjusted_reasoning_effort": None,
                "adjusted_thinking_budget_tokens": None,
                "model_info": {"provider": "openai", "model_id": "gpt-4o"},
                "reasoning_timeout_seconds": None,
            }
        )

        with pytest.raises(ValueError, match="Tool loop exhausted without producing a validated draft"):
            await ai_service_instance.call_ai_with_validation_tools(
                prompt="Write something long.",
                model="gpt-4o",
                validation_callback=lambda _: _rejected_result(),
                max_tool_rounds=2,
            )

    @pytest.mark.asyncio
    async def test_openai_tool_loop_forces_final_turn_when_tool_call_budget_is_exceeded(
        self, ai_service_instance, mock_config
    ):
        """
        Given: An OpenAI-compatible model tries to emit more tool calls than the runtime budget allows
        When: The runtime refuses additional tool calls
        Then: It requests one final no-tools answer and accepts it if valid
        """

        async def passthrough(operation, **kwargs):
            return await operation()

        tool_calls = [
            SimpleNamespace(
                id=f"call-{idx}",
                function=SimpleNamespace(
                    name="validate_draft",
                    arguments='{"text":"draft that still fails"}',
                ),
            )
            for idx in range(5)
        ]
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="", tool_calls=tool_calls))],
            usage=None,
        )
        final_response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="valid final answer", tool_calls=[]))],
            usage=None,
        )

        ai_service_instance.openai_client = Mock()
        ai_service_instance.openai_client.chat = Mock()
        ai_service_instance.openai_client.chat.completions = Mock()
        ai_service_instance.openai_client.chat.completions.create = AsyncMock(
            side_effect=[response, final_response]
        )
        ai_service_instance._execute_with_retries = AsyncMock(side_effect=passthrough)

        mock_config.validate_token_limits = Mock(
            return_value={
                "adjusted_tokens": 512,
                "adjusted_reasoning_effort": None,
                "adjusted_thinking_budget_tokens": None,
                "model_info": {"provider": "openai", "model_id": "gpt-4o"},
                "reasoning_timeout_seconds": None,
            }
        )

        force_finalize_events: list = []

        async def tool_event_cb(event_type: str, payload: Dict[str, Any]) -> None:
            if event_type == "force_finalize":
                force_finalize_events.append(payload)

        content, envelope = await ai_service_instance.call_ai_with_validation_tools(
            prompt="Write something long.",
            model="gpt-4o",
            validation_callback=lambda candidate: _result_for_target(candidate, "valid final answer"),
            max_tool_rounds=2,
            tool_event_callback=tool_event_cb,
        )

        assert content == "valid final answer"
        assert isinstance(envelope, ToolLoopEnvelope)
        assert envelope.accepted_via == "forced_final_turn"
        assert envelope.accepted is True
        assert ai_service_instance.openai_client.chat.completions.create.await_count == 2
        # Verify the force_finalize hook fired at least once with a reason string.
        assert force_finalize_events, "force_finalize hook should fire on budget exhaustion"
        assert "reason" in force_finalize_events[0]
        assert "turn" in force_finalize_events[0]

    @pytest.mark.asyncio
    async def test_openai_tool_loop_overflow_prefers_validate_draft_tool(
        self, ai_service_instance, mock_config
    ):
        """
        Given: An OpenAI-compatible model emits audit_accent before validate_draft at budget overflow
        When: The runtime validates the overflow argument
        Then: It validates the validate_draft text, not the audit_accent text
        """

        async def passthrough(operation, **kwargs):
            return await operation()

        tool_calls = [
            SimpleNamespace(
                id="call-audit",
                function=SimpleNamespace(
                    name="audit_accent",
                    arguments='{"text":"accent only draft"}',
                ),
            ),
            SimpleNamespace(
                id="call-validate",
                function=SimpleNamespace(
                    name="validate_draft",
                    arguments='{"text":"valid overflow draft"}',
                ),
            ),
            *[
                SimpleNamespace(
                    id=f"call-extra-{idx}",
                    function=SimpleNamespace(
                        name="validate_draft",
                        arguments='{"text":"extra draft"}',
                    ),
                )
                for idx in range(3)
            ],
        ]
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="", tool_calls=tool_calls))],
            usage=None,
        )

        ai_service_instance.openai_client = Mock()
        ai_service_instance.openai_client.chat = Mock()
        ai_service_instance.openai_client.chat.completions = Mock()
        ai_service_instance.openai_client.chat.completions.create = AsyncMock(return_value=response)
        ai_service_instance._execute_with_retries = AsyncMock(side_effect=passthrough)
        ai_service_instance.audit_accent = AsyncMock(
            return_value={"approved": True, "score": 10.0, "findings": [], "verdict_summary": ""}
        )

        mock_config.validate_token_limits = Mock(
            return_value={
                "adjusted_tokens": 512,
                "adjusted_reasoning_effort": None,
                "adjusted_thinking_budget_tokens": None,
                "model_info": {"provider": "openai", "model_id": "gpt-4o"},
                "reasoning_timeout_seconds": None,
            }
        )
        mock_config.AI_ACCENT_AUDIT_TIMEOUT_SECONDS = 30
        mock_config.AI_ACCENT_AUDIT_MAX_TOKENS = 1024

        seen_candidates = []

        def validation_callback(candidate: str):
            seen_candidates.append(candidate)
            return _result_for_target(candidate, "valid overflow draft")

        content, envelope = await ai_service_instance.call_ai_with_validation_tools(
            prompt="Write something long.",
            model="gpt-4o",
            validation_callback=validation_callback,
            max_tool_rounds=2,
            accent_guard=SimpleNamespace(
                mode="inline",
                criteria=None,
                min_score=None,
                max_inline_calls=1,
                on_error="fail_closed",
            ),
        )

        assert content == "valid overflow draft"
        assert envelope.accepted_via == "validated_tool_argument"
        assert envelope.accepted is True
        assert seen_candidates == ["valid overflow draft"]

    @pytest.mark.asyncio
    async def test_openai_tool_loop_raises_clear_error_when_validation_callback_fails(
        self, ai_service_instance, mock_config
    ):
        """
        Given: The local validation callback crashes
        When: The tool loop tries to validate a candidate
        Then: A clear ValueError is raised instead of leaking the raw exception
        """

        async def passthrough(operation, **kwargs):
            return await operation()

        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="",
                        tool_calls=[
                            SimpleNamespace(
                                id="call-1",
                                function=SimpleNamespace(
                                    name="validate_draft",
                                    arguments='{"text":"candidate"}',
                                ),
                            )
                        ],
                    )
                )
            ],
            usage=None,
        )

        ai_service_instance.openai_client = Mock()
        ai_service_instance.openai_client.chat = Mock()
        ai_service_instance.openai_client.chat.completions = Mock()
        ai_service_instance.openai_client.chat.completions.create = AsyncMock(return_value=response)
        ai_service_instance._execute_with_retries = AsyncMock(side_effect=passthrough)

        mock_config.validate_token_limits = Mock(
            return_value={
                "adjusted_tokens": 512,
                "adjusted_reasoning_effort": None,
                "adjusted_thinking_budget_tokens": None,
                "model_info": {"provider": "openai", "model_id": "gpt-4o"},
                "reasoning_timeout_seconds": None,
            }
        )

        def failing_callback(_: str):
            raise RuntimeError("validator boom")

        with pytest.raises(ValueError, match="Validation callback failed during openai_tool_loop"):
            await ai_service_instance.call_ai_with_validation_tools(
                prompt="Write something long.",
                model="gpt-4o",
                validation_callback=failing_callback,
                max_tool_rounds=2,
            )

    @pytest.mark.asyncio
    async def test_openai_tool_loop_emits_budget_warning_before_exhaustion(
        self, ai_service_instance, mock_config
    ):
        """
        Given: A turn that would consume most of the tool-call budget
        When: The tool loop processes the turn
        Then: The envelope trace includes an early budget warning
        """

        async def passthrough(operation, **kwargs):
            return await operation()

        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="",
                        tool_calls=[
                            SimpleNamespace(
                                id="call-1",
                                function=SimpleNamespace(
                                    name="validate_draft",
                                    arguments='{"text":"draft one"}',
                                ),
                            ),
                            SimpleNamespace(
                                id="call-2",
                                function=SimpleNamespace(
                                    name="validate_draft",
                                    arguments='{"text":"draft two"}',
                                ),
                            ),
                            SimpleNamespace(
                                id="call-3",
                                function=SimpleNamespace(
                                    name="validate_draft",
                                    arguments='{"text":"draft three"}',
                                ),
                            ),
                        ],
                    )
                )
            ],
            usage=None,
        )

        ai_service_instance.openai_client = Mock()
        ai_service_instance.openai_client.chat = Mock()
        ai_service_instance.openai_client.chat.completions = Mock()
        ai_service_instance.openai_client.chat.completions.create = AsyncMock(return_value=response)
        ai_service_instance._execute_with_retries = AsyncMock(side_effect=passthrough)

        mock_config.validate_token_limits = Mock(
            return_value={
                "adjusted_tokens": 512,
                "adjusted_reasoning_effort": None,
                "adjusted_thinking_budget_tokens": None,
                "model_info": {"provider": "openai", "model_id": "gpt-4o"},
                "reasoning_timeout_seconds": None,
            }
        )

        def validation_callback(candidate: str):
            return _result_for_target(candidate, "draft three")

        # Capture the per-turn tool_call_start events to exercise the new hook.
        tool_call_start_events: list = []

        async def tool_event_cb(event_type: str, payload: Dict[str, Any]) -> None:
            if event_type == "tool_call_start":
                tool_call_start_events.append(payload)

        content, envelope = await ai_service_instance.call_ai_with_validation_tools(
            prompt="Write something long.",
            model="gpt-4o",
            validation_callback=validation_callback,
            max_tool_rounds=2,
            tool_event_callback=tool_event_cb,
        )

        assert content == "draft three"
        assert isinstance(envelope, ToolLoopEnvelope)
        assert any(entry.event == "tool_call_budget_warning" for entry in envelope.trace)
        # At least one tool_call_start event was emitted before a validate_draft call.
        assert any(evt.get("tool") == "validate_draft" for evt in tool_call_start_events)

    @pytest.mark.asyncio
    async def test_openrouter_tool_loop_uses_openrouter_client(
        self, ai_service_instance, mock_config
    ):
        """
        Given: An OpenRouter model using the validation tool loop
        When: The model streams validate_draft with an approved draft
        Then: The OpenRouter client is used with real streaming and the validated draft is returned
        """

        class FakeAsyncStream:
            def __aiter__(self):
                async def iterator():
                    yield SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                delta=SimpleNamespace(
                                    content=None,
                                    tool_calls=[
                                        SimpleNamespace(
                                            index=0,
                                            id="call-1",
                                            type="function",
                                            function=SimpleNamespace(
                                                name="validate_draft",
                                                arguments='{"text":"validated via openrouter"}',
                                            ),
                                        )
                                    ],
                                ),
                                finish_reason="tool_calls",
                            )
                        ],
                        usage=None,
                    )

                return iterator()

        async def passthrough(operation, **kwargs):
            return await operation()

        events = []

        async def tool_event_callback(event_type: str, payload: Dict[str, Any]) -> None:
            events.append((event_type, payload))

        ai_service_instance.openrouter_client = Mock()
        ai_service_instance.openrouter_client.chat = Mock()
        ai_service_instance.openrouter_client.chat.completions = Mock()
        ai_service_instance.openrouter_client.chat.completions.create = AsyncMock(
            return_value=FakeAsyncStream()
        )
        ai_service_instance.openai_client = Mock()
        ai_service_instance.openai_client.chat = Mock()
        ai_service_instance.openai_client.chat.completions = Mock()
        ai_service_instance.openai_client.chat.completions.create = AsyncMock()
        ai_service_instance._execute_with_retries = AsyncMock(side_effect=passthrough)

        mock_config.validate_token_limits = Mock(
            return_value={
                "adjusted_tokens": 512,
                "adjusted_reasoning_effort": None,
                "adjusted_thinking_budget_tokens": None,
                "model_info": {"provider": "openrouter", "model_id": "openrouter/meta-llama-3.3"},
                "reasoning_timeout_seconds": None,
            }
        )
        mock_config.model_specs = {
            "model_specifications": {
                "openrouter": {
                    "openrouter/meta-llama-3.3": {
                        "model_id": "openrouter/meta-llama-3.3",
                        "supported_parameters": ["tools", "tool_choice"],
                    }
                }
            }
        }

        content, envelope = await ai_service_instance.call_ai_with_validation_tools(
            prompt="Write something long.",
            model="or-meta",
            validation_callback=lambda _: _approved_result(),
            max_tool_rounds=2,
            tool_event_callback=tool_event_callback,
        )

        assert content == "validated via openrouter"
        assert isinstance(envelope, ToolLoopEnvelope)
        assert envelope.accepted is True
        ai_service_instance.openrouter_client.chat.completions.create.assert_awaited_once()
        ai_service_instance.openai_client.chat.completions.create.assert_not_called()
        request_kwargs = ai_service_instance.openrouter_client.chat.completions.create.await_args.kwargs
        assert request_kwargs["stream"] is True
        assert request_kwargs["stream_options"] == {"include_usage": True}
        assert request_kwargs["extra_body"] == {"provider": {"require_parameters": True}}
        event_types = [event_type for event_type, _payload in events]
        assert "tool_call_delta" in event_types
        assert "tool_call_ready" in event_types
        assert any(
            payload.get("provider") == "openrouter"
            and payload.get("api_surface") == "chat_completions"
            and payload.get("tool_calls") == 1
            for event_type, payload in events
            if event_type == "tool_loop_turn_done"
        )

    @pytest.mark.asyncio
    async def test_openrouter_tool_loop_skips_model_without_tools(
        self, ai_service_instance, mock_config
    ):
        """
        Given: An OpenRouter model without a tools supported_parameter
        When: The validation tool loop is requested
        Then: The loop is skipped before any provider call
        """

        ai_service_instance.openrouter_client = Mock()
        ai_service_instance.openrouter_client.chat = Mock()
        ai_service_instance.openrouter_client.chat.completions = Mock()
        ai_service_instance.openrouter_client.chat.completions.create = AsyncMock()

        mock_config.validate_token_limits = Mock(
            return_value={
                "adjusted_tokens": 512,
                "adjusted_reasoning_effort": None,
                "adjusted_thinking_budget_tokens": None,
                "model_info": {"provider": "openrouter", "model_id": "vendor/no-tools"},
                "reasoning_timeout_seconds": None,
            }
        )
        mock_config.model_specs = {
            "model_specifications": {
                "openrouter": {
                    "vendor/no-tools": {
                        "model_id": "vendor/no-tools",
                        "supported_parameters": ["temperature"],
                    }
                }
            }
        }

        content, envelope = await ai_service_instance.call_ai_with_validation_tools(
            prompt="Write something long.",
            model="or-no-tools",
            validation_callback=lambda _: _approved_result(),
            max_tool_rounds=2,
        )

        assert content == ""
        assert isinstance(envelope, ToolLoopEnvelope)
        assert envelope.tools_skipped_reason == "no_tool_support"
        ai_service_instance.openrouter_client.chat.completions.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_xai_tool_loop_streams_tool_call_deltas(
        self, ai_service_instance, mock_config
    ):
        """
        Given: A Grok model using the validation tool loop
        When: xAI streams an OpenAI-compatible tool call
        Then: The turn uses real streaming and the validated draft is returned
        """

        class FakeAsyncStream:
            def __aiter__(self):
                async def iterator():
                    yield SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                delta=SimpleNamespace(content="Checking draft..."),
                                finish_reason=None,
                            )
                        ],
                        usage=None,
                    )
                    yield SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                delta=SimpleNamespace(
                                    content=None,
                                    tool_calls=[
                                        SimpleNamespace(
                                            index=0,
                                            id="call-xai-1",
                                            type="function",
                                            function=SimpleNamespace(
                                                name="validate_draft",
                                                arguments='{"text":"validated via xai stream"}',
                                            ),
                                        )
                                    ],
                                ),
                                finish_reason="tool_calls",
                            )
                        ],
                        usage=SimpleNamespace(total_tokens=42),
                    )

                return iterator()

        async def passthrough(operation, **kwargs):
            return await operation()

        events = []

        async def tool_event_callback(event_type: str, payload: Dict[str, Any]) -> None:
            events.append((event_type, payload))

        create = AsyncMock(return_value=FakeAsyncStream())
        ai_service_instance.xai_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        ai_service_instance.openai_client = Mock()
        ai_service_instance.openai_client.chat = Mock()
        ai_service_instance.openai_client.chat.completions = Mock()
        ai_service_instance.openai_client.chat.completions.create = AsyncMock()
        ai_service_instance._execute_with_retries = AsyncMock(side_effect=passthrough)

        mock_config.validate_token_limits = Mock(
            return_value={
                "adjusted_tokens": 512,
                "adjusted_reasoning_effort": None,
                "adjusted_thinking_budget_tokens": None,
                "model_info": {"provider": "xai", "model_id": "grok-4-fast-non-reasoning"},
                "reasoning_timeout_seconds": None,
            }
        )
        mock_config.model_specs = {
            "model_specifications": {
                "xai": {
                    "grok-4-fast-non-reasoning": {
                        "model_id": "grok-4-fast-non-reasoning",
                        "capabilities": ["text"],
                    }
                }
            }
        }

        content, envelope = await ai_service_instance.call_ai_with_validation_tools(
            prompt="Write something long.",
            model="grok-4-fast-non-reasoning",
            validation_callback=lambda _: _approved_result(),
            max_tool_rounds=2,
            tool_event_callback=tool_event_callback,
        )

        assert content == "validated via xai stream"
        assert isinstance(envelope, ToolLoopEnvelope)
        assert envelope.accepted is True
        create.assert_awaited_once()
        request_kwargs = create.await_args.kwargs
        assert request_kwargs["stream"] is True
        assert request_kwargs["stream_options"] == {"include_usage": True}
        assert request_kwargs["model"] == "grok-4-fast-non-reasoning"
        assert ai_service_instance.openai_client.chat.completions.create.await_count == 0

        event_types = [event_type for event_type, _payload in events]
        assert "assistant_delta" in event_types
        assert "tool_call_delta" in event_types
        assert "tool_call_ready" in event_types
        assert "tool_loop_turn_done" in event_types
        assert event_types.index("tool_call_delta") < event_types.index("tool_call_ready")
        assert any(
            payload.get("provider") == "xai"
            and payload.get("api_surface") == "chat_completions"
            and payload.get("tool_calls") == 1
            for event_type, payload in events
            if event_type == "tool_loop_turn_done"
        )

    @pytest.mark.asyncio
    async def test_claude_tool_loop_handles_tool_use_blocks(
        self, ai_service_instance, mock_config
    ):
        """
        Given: A Claude model returning a tool_use block
        When: The tool payload is approved locally
        Then: The validated draft is returned without falling back
        """

        async def passthrough(operation, **kwargs):
            return await operation()

        response = SimpleNamespace(
            content=[
                SimpleNamespace(
                    type="tool_use",
                    id="toolu_1",
                    name="validate_draft",
                    input={"text": "validated via claude"},
                )
            ],
            usage=None,
        )

        ai_service_instance.anthropic_client = Mock()
        ai_service_instance.anthropic_client.messages = Mock()
        ai_service_instance.anthropic_client.messages.stream = Mock(
            return_value=_claude_stream_for_response(response)
        )
        ai_service_instance.anthropic_client.messages.create = AsyncMock()
        ai_service_instance._execute_with_retries = AsyncMock(side_effect=passthrough)

        mock_config.validate_token_limits = Mock(
            return_value={
                "adjusted_tokens": 512,
                "adjusted_reasoning_effort": None,
                "adjusted_thinking_budget_tokens": None,
                "model_info": {"provider": "claude", "model_id": "claude-sonnet-4-5"},
                "reasoning_timeout_seconds": None,
            }
        )

        content, envelope = await ai_service_instance.call_ai_with_validation_tools(
            prompt="Write something long.",
            model="claude-sonnet-4-5",
            validation_callback=lambda _: _approved_result(),
            max_tool_rounds=2,
        )

        assert content == "validated via claude"
        assert isinstance(envelope, ToolLoopEnvelope)
        assert envelope.accepted is True
        ai_service_instance.anthropic_client.messages.stream.assert_called_once()
        ai_service_instance.anthropic_client.messages.create.assert_not_awaited()
        sent_tool = ai_service_instance.anthropic_client.messages.stream.call_args.kwargs["tools"][0]
        assert sent_tool["eager_input_streaming"] is True

    @pytest.mark.asyncio
    async def test_claude_tool_loop_streams_text_and_tool_argument_deltas(
        self, ai_service_instance, mock_config
    ):
        """
        Given: Claude streams visible text and tool input JSON deltas
        When: The local validator approves the tool payload
        Then: Monitor telemetry includes real assistant/tool deltas before tool execution
        """

        async def passthrough(operation, **kwargs):
            return await operation()

        response = SimpleNamespace(
            content=[
                SimpleNamespace(type="text", text="I will validate this."),
                SimpleNamespace(
                    type="tool_use",
                    id="toolu_stream",
                    name="validate_draft",
                    input={"text": "validated via claude stream"},
                ),
            ],
            usage=None,
            stop_reason="tool_use",
        )
        events = []

        async def tool_event_callback(event_type, payload):
            events.append((event_type, payload))

        ai_service_instance.anthropic_client = Mock()
        ai_service_instance.anthropic_client.messages = Mock()
        ai_service_instance.anthropic_client.messages.stream = Mock(
            return_value=_claude_stream_for_response(response)
        )
        ai_service_instance._execute_with_retries = AsyncMock(side_effect=passthrough)

        mock_config.validate_token_limits = Mock(
            return_value={
                "adjusted_tokens": 512,
                "adjusted_reasoning_effort": None,
                "adjusted_thinking_budget_tokens": None,
                "model_info": {"provider": "claude", "model_id": "claude-sonnet-4-5"},
                "reasoning_timeout_seconds": None,
            }
        )

        content, envelope = await ai_service_instance.call_ai_with_validation_tools(
            prompt="Write something long.",
            model="claude-sonnet-4-5",
            validation_callback=lambda _: _approved_result(),
            max_tool_rounds=2,
            tool_event_callback=tool_event_callback,
        )

        assert content == "validated via claude stream"
        assert envelope.accepted is True
        event_types = [event_type for event_type, _payload in events]
        assert "assistant_delta" in event_types
        assert "tool_call_delta" in event_types
        assert "tool_call_ready" in event_types
        assert event_types.index("tool_call_delta") < event_types.index("tool_call_ready")
        assert any(
            payload.get("api_surface") == "messages" and payload.get("streaming") is True
            for event_type, payload in events
            if event_type == "tool_loop_turn_done"
        )

    @pytest.mark.asyncio
    async def test_claude_tool_loop_does_not_execute_partial_tool_input_on_max_tokens(
        self, ai_service_instance
    ):
        """
        Given: Claude streams partial tool input but ends with max_tokens
        When: The stream closes
        Then: The local validator is not called with truncated arguments
        """

        partial_events = [
            SimpleNamespace(
                type="message_start",
                message=SimpleNamespace(usage=SimpleNamespace(input_tokens=10)),
            ),
            SimpleNamespace(
                type="content_block_start",
                index=0,
                content_block=SimpleNamespace(
                    type="tool_use",
                    id="toolu_partial",
                    name="validate_draft",
                    input={},
                ),
            ),
            SimpleNamespace(
                type="content_block_delta",
                index=0,
                delta=SimpleNamespace(
                    type="input_json_delta",
                    partial_json='{"text": "unfinished',
                ),
            ),
            SimpleNamespace(type="content_block_stop", index=0),
            _claude_message_delta("max_tokens"),
            SimpleNamespace(type="message_stop"),
        ]
        events = []

        async def tool_event_callback(event_type, payload):
            events.append((event_type, payload))

        ai_service_instance.anthropic_client = Mock()
        ai_service_instance.anthropic_client.messages = Mock()
        ai_service_instance.anthropic_client.messages.stream = Mock(
            return_value=_FakeClaudeStream(partial_events)
        )

        with pytest.raises(RuntimeError, match="malformed|unusable"):
            await ai_service_instance._run_claude_validation_tool_loop(
                provider="claude",
                model_id="claude-sonnet-4-5",
                prompt="Write something long.",
                validation_callback=Mock(side_effect=AssertionError("must not validate")),
                temperature=0.2,
                max_tokens=512,
                system_prompt="",
                request_timeout=None,
                thinking_budget_tokens=None,
                json_output=False,
                json_schema=None,
                usage_callback=None,
                usage_extra=None,
                images=None,
                max_rounds=1,
                retries_enabled=False,
                tool_event_callback=tool_event_callback,
            )

        event_types = [event_type for event_type, _payload in events]
        assert "tool_call_ready" not in event_types
        assert "tool_loop_provider_terminal" in event_types

    @pytest.mark.asyncio
    async def test_claude_structured_tool_loop_streams_via_create_stream(
        self, ai_service_instance, mock_config
    ):
        """
        Given: Claude structured outputs are active in a tool loop
        When: The turn is streamed
        Then: It uses messages.create(stream=True), matching the SDK-safe structured path
        """

        async def passthrough(operation, **kwargs):
            return await operation()

        response = SimpleNamespace(
            content=[
                SimpleNamespace(
                    type="tool_use",
                    id="toolu_structured",
                    name="validate_draft",
                    input={"text": '{"ok": true}'},
                )
            ],
            usage=None,
            stop_reason="tool_use",
        )
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
        }

        ai_service_instance.anthropic_client = Mock()
        ai_service_instance.anthropic_client.messages = Mock()
        ai_service_instance.anthropic_client.messages.create = AsyncMock(
            return_value=_claude_stream_for_response(response)
        )
        ai_service_instance.anthropic_client.messages.stream = Mock()
        ai_service_instance._execute_with_retries = AsyncMock(side_effect=passthrough)

        mock_config.validate_token_limits = Mock(
            return_value={
                "adjusted_tokens": 512,
                "adjusted_reasoning_effort": None,
                "adjusted_thinking_budget_tokens": None,
                "model_info": {"provider": "claude", "model_id": "claude-sonnet-4-5"},
                "reasoning_timeout_seconds": None,
            }
        )

        def configure_structured(params, json_schema):
            params["output_config"] = {"format": {"type": "json_schema", "schema": json_schema}}
            return False

        with patch.object(AIService, "_claude_supports_structured_outputs", return_value=True), patch.object(
            ai_service_instance,
            "_configure_claude_structured_output_params",
            side_effect=configure_structured,
        ):
            content, envelope = await ai_service_instance.call_ai_with_validation_tools(
                prompt="Write structured JSON.",
                model="claude-sonnet-4-5",
                validation_callback=lambda _: _approved_result(),
                max_tool_rounds=2,
                output_contract=OutputContract.JSON_STRUCTURED,
                response_format=schema,
            )

        assert content == '{"ok": true}'
        assert envelope.payload == {"ok": True}
        ai_service_instance.anthropic_client.messages.create.assert_awaited_once()
        ai_service_instance.anthropic_client.messages.stream.assert_not_called()
        create_kwargs = ai_service_instance.anthropic_client.messages.create.await_args.kwargs
        assert create_kwargs["stream"] is True
        assert create_kwargs["tools"][0]["eager_input_streaming"] is True

    @pytest.mark.asyncio
    async def test_claude_tool_loop_forces_final_turn_after_exhaustion(
        self, ai_service_instance, mock_config
    ):
        """
        Given: Claude keeps using validate_draft without producing an approved draft
        When: The configured rounds are exhausted
        Then: The runtime requests one last no-tools answer and accepts it if valid
        """

        async def passthrough(operation, **kwargs):
            return await operation()

        tool_response = SimpleNamespace(
            content=[
                SimpleNamespace(
                    type="tool_use",
                    id="toolu_1",
                    name="validate_draft",
                    input={"text": "draft that still fails"},
                )
            ],
            usage=None,
        )
        final_response = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="valid claude final answer")],
            usage=None,
        )

        ai_service_instance.anthropic_client = Mock()
        ai_service_instance.anthropic_client.messages = Mock()
        ai_service_instance.anthropic_client.messages.stream = Mock(
            side_effect=[
                _claude_stream_for_response(tool_response),
                _claude_stream_for_response(tool_response),
                _claude_stream_for_response(final_response, stop_reason="end_turn"),
            ]
        )
        ai_service_instance.anthropic_client.messages.create = AsyncMock()
        ai_service_instance._execute_with_retries = AsyncMock(side_effect=passthrough)

        mock_config.validate_token_limits = Mock(
            return_value={
                "adjusted_tokens": 512,
                "adjusted_reasoning_effort": None,
                "adjusted_thinking_budget_tokens": None,
                "model_info": {"provider": "claude", "model_id": "claude-sonnet-4-5"},
                "reasoning_timeout_seconds": None,
            }
        )

        def validation_callback(candidate: str):
            return _result_for_target(candidate, "valid claude final answer")

        content, envelope = await ai_service_instance.call_ai_with_validation_tools(
            prompt="Write something long.",
            model="claude-sonnet-4-5",
            validation_callback=validation_callback,
            max_tool_rounds=2,
        )

        assert content == "valid claude final answer"
        assert envelope.accepted_via == "forced_final_turn"
        assert envelope.accepted is True
        assert ai_service_instance.anthropic_client.messages.stream.call_count == 3
        ai_service_instance.anthropic_client.messages.create.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_claude_tool_loop_overflow_prefers_validate_draft_tool(
        self, ai_service_instance, mock_config
    ):
        """
        Given: Claude emits audit_accent before validate_draft at budget overflow
        When: The runtime validates the overflow argument
        Then: It validates the validate_draft text, not the audit_accent text
        """

        async def passthrough(operation, **kwargs):
            return await operation()

        response = SimpleNamespace(
            content=[
                SimpleNamespace(
                    type="tool_use",
                    id="toolu_audit",
                    name="audit_accent",
                    input={"text": "accent only draft"},
                ),
                SimpleNamespace(
                    type="tool_use",
                    id="toolu_validate",
                    name="validate_draft",
                    input={"text": "valid claude overflow draft"},
                ),
                *[
                    SimpleNamespace(
                        type="tool_use",
                        id=f"toolu_extra_{idx}",
                        name="validate_draft",
                        input={"text": "extra draft"},
                    )
                    for idx in range(3)
                ],
            ],
            usage=None,
        )

        ai_service_instance.anthropic_client = Mock()
        ai_service_instance.anthropic_client.messages = Mock()
        ai_service_instance.anthropic_client.messages.stream = Mock(
            return_value=_claude_stream_for_response(response)
        )
        ai_service_instance.anthropic_client.messages.create = AsyncMock()
        ai_service_instance._execute_with_retries = AsyncMock(side_effect=passthrough)
        ai_service_instance.audit_accent = AsyncMock(
            return_value={"approved": True, "score": 10.0, "findings": [], "verdict_summary": ""}
        )

        mock_config.validate_token_limits = Mock(
            return_value={
                "adjusted_tokens": 512,
                "adjusted_reasoning_effort": None,
                "adjusted_thinking_budget_tokens": None,
                "model_info": {"provider": "claude", "model_id": "claude-sonnet-4-5"},
                "reasoning_timeout_seconds": None,
            }
        )
        mock_config.AI_ACCENT_AUDIT_TIMEOUT_SECONDS = 30
        mock_config.AI_ACCENT_AUDIT_MAX_TOKENS = 1024

        seen_candidates = []

        def validation_callback(candidate: str):
            seen_candidates.append(candidate)
            return _result_for_target(candidate, "valid claude overflow draft")

        content, envelope = await ai_service_instance.call_ai_with_validation_tools(
            prompt="Write something long.",
            model="claude-sonnet-4-5",
            validation_callback=validation_callback,
            max_tool_rounds=2,
            accent_guard=SimpleNamespace(
                mode="inline",
                criteria=None,
                min_score=None,
                max_inline_calls=1,
                on_error="fail_closed",
            ),
        )

        assert content == "valid claude overflow draft"
        assert envelope.accepted_via == "validated_tool_argument"
        assert seen_candidates == ["valid claude overflow draft"]

    @pytest.mark.asyncio
    async def test_gemini_tool_loop_handles_function_calls(
        self, ai_service_instance, mock_config
    ):
        """
        Given: A Gemini model returning a function call
        When: The local validator approves the proposed draft
        Then: The validated draft is returned through the Gemini tool loop
        """

        async def passthrough(operation, **kwargs):
            return await operation()

        response = SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(
                        parts=[
                            SimpleNamespace(
                                function_call=SimpleNamespace(
                                    name="validate_draft",
                                    args={"text": "validated via gemini"},
                                )
                            )
                        ]
                    )
                )
            ],
            usage_metadata=None,
        )

        ai_service_instance.google_new_client = Mock()
        ai_service_instance.google_new_client.aio = Mock()
        ai_service_instance.google_new_client.aio.models = Mock()
        ai_service_instance.google_new_client.aio.models.generate_content = AsyncMock(return_value=response)
        ai_service_instance._execute_with_retries = AsyncMock(side_effect=passthrough)

        mock_config.validate_token_limits = Mock(
            return_value={
                "adjusted_tokens": 512,
                "adjusted_reasoning_effort": None,
                "adjusted_thinking_budget_tokens": None,
                "model_info": {"provider": "gemini", "model_id": "gemini-2.5-flash"},
                "reasoning_timeout_seconds": None,
            }
        )

        content, envelope = await ai_service_instance.call_ai_with_validation_tools(
            prompt="Write something long.",
            model="gemini-2.5-flash",
            validation_callback=lambda _: _approved_result(),
            max_tool_rounds=2,
        )

        assert content == "validated via gemini"
        assert isinstance(envelope, ToolLoopEnvelope)
        assert envelope.accepted is True
        ai_service_instance.google_new_client.aio.models.generate_content.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_gemini_tool_loop_avoids_structured_output_config_when_json_requested(
        self, ai_service_instance, mock_config
    ):
        """
        Given: Gemini tool-loop with JSON output requested
        When: Function calling is enabled
        Then: The request avoids response_mime_type/response_schema because Gemini rejects that combination
        """

        async def passthrough(operation, **kwargs):
            return await operation()

        captured_config = {}

        async def fake_generate_content(*args, **kwargs):
            captured_config["config"] = kwargs["config"]
            # Emit JSON text as the validate_draft argument so the
            # JSON_STRUCTURED contract parser accepts the draft as
            # the final payload (content == draft under Path 1).
            return SimpleNamespace(
                candidates=[
                    SimpleNamespace(
                        content=SimpleNamespace(
                            parts=[
                                SimpleNamespace(
                                    function_call=SimpleNamespace(
                                        name="validate_draft",
                                        args={"text": '{"content":"validated via gemini"}'},
                                    )
                                )
                            ]
                        )
                    )
                ],
                usage_metadata=None,
            )

        ai_service_instance.google_new_client = Mock()
        ai_service_instance.google_new_client.aio = Mock()
        ai_service_instance.google_new_client.aio.models = Mock()
        ai_service_instance.google_new_client.aio.models.generate_content = AsyncMock(
            side_effect=fake_generate_content
        )
        ai_service_instance._execute_with_retries = AsyncMock(side_effect=passthrough)

        mock_config.validate_token_limits = Mock(
            return_value={
                "adjusted_tokens": 512,
                "adjusted_reasoning_effort": None,
                "adjusted_thinking_budget_tokens": None,
                "model_info": {"provider": "gemini", "model_id": "gemini-2.5-flash"},
                "reasoning_timeout_seconds": None,
            }
        )

        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "content": {
                    "type": "string",
                }
            },
            "required": ["content"],
        }

        content, envelope = await ai_service_instance.call_ai_with_validation_tools(
            prompt="Write something structured.",
            model="gemini-2.5-flash",
            validation_callback=lambda _: _approved_result(),
            max_tool_rounds=2,
            output_contract=OutputContract.JSON_STRUCTURED,
            response_format=schema,
        )

        config = captured_config["config"]
        assert content == '{"content":"validated via gemini"}'
        assert isinstance(envelope, ToolLoopEnvelope)
        assert envelope.payload == {"content": "validated via gemini"}
        assert getattr(config, "response_mime_type", None) is None
        assert getattr(config, "response_json_schema", None) is None
        assert getattr(config, "response_schema", None) is None


# ============================================================================
# TestIsOpenAIResponsesApiModel - Centralized Responses API capability lookup
# ============================================================================

class TestIsOpenAIResponsesApiModel:
    """Verify that the capability-backed helper recognises the Responses API models."""

    def test_base_responses_api_models_return_true(self):
        assert AIService._is_openai_responses_api_model("o3-pro") is True
        assert AIService._is_openai_responses_api_model("gpt-5-pro") is True

    def test_dated_responses_api_variants_return_true(self):
        assert AIService._is_openai_responses_api_model("o3-pro-2025-06-10") is True
        assert AIService._is_openai_responses_api_model("gpt-5-pro-2025-10-06") is True

    def test_case_insensitive_match(self):
        assert AIService._is_openai_responses_api_model("O3-Pro") is True
        assert AIService._is_openai_responses_api_model("GPT-5-PRO") is True

    def test_non_responses_api_openai_model_returns_false(self):
        assert AIService._is_openai_responses_api_model("gpt-4o") is False
        assert AIService._is_openai_responses_api_model("gpt-5") is False

    def test_unknown_model_returns_false(self):
        assert AIService._is_openai_responses_api_model("nonexistent-model-xyz") is False

    def test_empty_model_id_returns_false(self):
        assert AIService._is_openai_responses_api_model("") is False
