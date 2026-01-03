"""
AI Service Module for Gran Sabio LLM Engine
============================================

Handles communication with multiple AI providers (OpenAI, Anthropic, Google).
Provides unified interface for content generation across different models.
"""

import asyncio
import aiohttp
import time
import logging
import threading
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple, Callable, TYPE_CHECKING, TypeVar, Awaitable

if TYPE_CHECKING:
    from logging_utils import PhaseLogger
import openai
import anthropic
try:
    from google import genai as google_genai
    GOOGLE_NEW_SDK = True
except ImportError:
    google_genai = None
    GOOGLE_NEW_SDK = False

# Legacy SDK as fallback
try:
    import google.generativeai as genai
    from google.ai.generativelanguage_v1beta.types import content
except ImportError:
    genai = None

# Use optimized JSON (3.6x faster than standard json)
import json_utils as json

from config import config, get_model_parameter_requirements
from models import QAEvaluation
from schema_utils import json_schema_to_pydantic


logger = logging.getLogger(__name__)


class AIRequestError(RuntimeError):
    """Raised when an AI provider keeps failing after retry attempts."""

    def __init__(self, provider: str, model: str, attempts: int, max_attempts: int, cause: Exception):
        message = (
            f"AI request failed for {model} via {provider} after "
            f"{attempts}/{max_attempts} attempts: {cause}"
        )
        super().__init__(message)
        self.provider = provider
        self.model = model
        self.attempts = attempts
        self.max_attempts = max_attempts
        self.cause = cause


class StreamChunk:
    """
    Wrapper for streaming chunks that distinguishes thinking from content.

    Used to stream Claude's thinking process in real-time while keeping it
    separate from the final accumulated content.

    Attributes:
        text: The actual text content of the chunk
        is_thinking: True if this chunk is from Claude's thinking process

    Note: Currently thinking chunks are passed through directly without visual
    markers. In the future, consider adding tags like [THINKING]...[/THINKING]
    or a structured format to help frontends differentiate and style them.
    """
    __slots__ = ('text', 'is_thinking')

    def __init__(self, text: str, is_thinking: bool = False):
        self.text = text
        self.is_thinking = is_thinking

    def __str__(self) -> str:
        return self.text


T = TypeVar("T")


def _build_openai_params(model_id: str, messages: list, temperature: float, max_tokens: int, reasoning_effort: Optional[str] = None) -> dict:
    """Build correct parameters for OpenAI API calls based on model requirements."""
    requirements = get_model_parameter_requirements(model_id)

    params = {
        "model": model_id,
        "messages": messages
    }

    # Apply temperature constraints
    if requirements["supports_temperature"]:
        params["temperature"] = temperature
    elif requirements.get("forced_temperature") is not None:
        params["temperature"] = requirements["forced_temperature"]

    # Apply correct token parameter
    token_param = requirements["max_tokens_param"]
    params[token_param] = max_tokens

    # Apply reasoning effort if supported
    if requirements["supports_reasoning_effort"] and reasoning_effort:
        params["reasoning_effort"] = reasoning_effort

    return params


_shared_ai_service: Optional["AIService"] = None
_ai_service_init_lock = threading.Lock()


def get_ai_service() -> "AIService":
    """Return the shared AIService instance, creating it on first use."""
    global _shared_ai_service
    if _shared_ai_service is None:
        with _ai_service_init_lock:
            if _shared_ai_service is None:
                _shared_ai_service = AIService()
    return _shared_ai_service


class AIService:
    """Unified AI service for multiple providers"""
    
    def __init__(self):
        """Initialize AI service with API clients"""
        self.openai_client = None
        self.anthropic_client = None
        self.genai_client = None
        self.google_new_client = None
        self.xai_client = None
        self.openrouter_client = None
        self.ollama_client = None
        
        # Configure optimized HTTP connector for better performance
        self.http_connector = aiohttp.TCPConnector(
            limit=200,              # Total connection pool size
            limit_per_host=50,      # Max connections per API host (OpenAI/Claude/Google)
            keepalive_timeout=60,   # Keep connections alive for 60 seconds
            enable_cleanup_closed=True,  # Clean up closed connections
            ttl_dns_cache=3600,     # Cache DNS lookups for 1 hour
            use_dns_cache=True
        )
        
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize API clients for each provider"""
        # Initialize OpenAI client with optimized HTTP settings
        if config.OPENAI_API_KEY:
            self.openai_client = openai.AsyncOpenAI(
                api_key=config.OPENAI_API_KEY,
                timeout=180.0,  # Increased timeout for better stability
                max_retries=3  # Retry failed requests twice
            )
            # Initialize sync OpenAI client for O3-pro (Responses API requirement)
            self.openai_sync_client = openai.OpenAI(
                api_key=config.OPENAI_API_KEY,
                timeout=180.0,  # Increased timeout for better stability
                max_retries=3
            )
        else:
            logger.warning("OpenAI API key not found")
        
        # Initialize Anthropic client with updated version and beta features
        if config.ANTHROPIC_API_KEY:
            # Add beta headers for extended thinking and other Claude 4 features
            default_headers = {
                "anthropic-beta": "interleaved-thinking-2025-05-14"
            }

            self.anthropic_client = anthropic.AsyncAnthropic(
                api_key=config.ANTHROPIC_API_KEY,
                timeout=180.0,    # Increased timeout for better stability
                max_retries=3,   # Retry failed requests twice
                default_headers=default_headers
            )
            logger.info("Anthropic client initialized with updated SDK and beta headers for thinking mode")
        else:
            logger.warning("Anthropic API key not found")
        
        # Initialize Google AI client
        if config.GOOGLE_API_KEY:
            if GOOGLE_NEW_SDK and google_genai:
                try:
                    self.google_new_client = google_genai.Client(api_key=config.GOOGLE_API_KEY)
                    self.genai_client = None
                    logger.info("Using new Google GenAI SDK")
                except Exception as e:
                    logger.error(f"Failed to initialize new Google GenAI SDK: {e}")
                    # Fallback to legacy SDK
                    if genai:
                        try:
                            genai.configure(api_key=config.GOOGLE_API_KEY)
                            self.genai_client = genai
                            logger.info("Fell back to legacy Google GenerativeAI SDK")
                        except Exception as e2:
                            logger.error(f"Failed to initialize legacy SDK: {e2}")
            elif genai:
                try:
                    genai.configure(api_key=config.GOOGLE_API_KEY)
                    self.genai_client = genai
                    logger.info("Using legacy Google GenerativeAI SDK")
                except Exception as e:
                    logger.error(f"Failed to initialize Google GenerativeAI SDK: {e}")
            else:
                logger.error("No Google SDK available")
        else:
            logger.warning("Google AI API key not found")
        
        # Initialize xAI client (using OpenAI SDK with custom base_url)
        if config.XAI_API_KEY:
            self.xai_client = openai.AsyncOpenAI(
                api_key=config.XAI_API_KEY,
                base_url="https://api.x.ai/v1",
                timeout=130.0,  # Increased timeout for better stability
                max_retries=3
            )
        else:
            logger.warning("xAI API key not found")

        # Initialize OpenRouter client (unified API for additional models)
        if config.OPENROUTER_API_KEY:
            self.openrouter_client = openai.AsyncOpenAI(
                api_key=config.OPENROUTER_API_KEY,
                base_url="https://openrouter.ai/api/v1",
                timeout=180.0,  # Increased timeout for better stability
                max_retries=3,
                default_headers={
                    "HTTP-Referer": "https://gransabio-llm.local",
                    "X-Title": "Gran Sabio LLM Engine"
                }
            )
            logger.info("OpenRouter client initialized for unified model access")
        else:
            logger.warning("OpenRouter API key not found")

        # Initialize Ollama client (local models, OpenAI-compatible API)
        if config.OLLAMA_HOST:
            ollama_base_url = config.OLLAMA_HOST.rstrip("/")
            # Add http:// if no protocol specified
            if not ollama_base_url.startswith(("http://", "https://")):
                ollama_base_url = f"http://{ollama_base_url}"
            if not ollama_base_url.endswith("/v1"):
                ollama_base_url = f"{ollama_base_url}/v1"
            self.ollama_client = openai.AsyncOpenAI(
                api_key="ollama",  # Dummy key, Ollama doesn't require authentication
                base_url=ollama_base_url,
                timeout=300.0,  # Longer timeout for local models (can be slow)
                max_retries=2
            )
            logger.info(f"Ollama client initialized at {ollama_base_url}")
        else:
            logger.info("Ollama not configured (OLLAMA_HOST not set)")

    @staticmethod
    def _strip_additional_properties(schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively remove 'additionalProperties' from a JSON schema.

        Gemini structured outputs do not support 'additionalProperties' at all (neither
        true nor false), while OpenAI requires it to be false. This function creates
        a clean copy of the schema without the field for Gemini compatibility.

        Args:
            schema: The original JSON schema

        Returns:
            A deep copy of the schema with all 'additionalProperties' fields removed
        """
        if not isinstance(schema, dict):
            return schema

        result = {}
        for key, value in schema.items():
            if key == "additionalProperties":
                continue  # Skip this field entirely
            elif key == "properties" and isinstance(value, dict):
                result[key] = {
                    k: AIService._strip_additional_properties(v)
                    for k, v in value.items()
                }
            elif key == "items":
                if isinstance(value, dict):
                    result[key] = AIService._strip_additional_properties(value)
                elif isinstance(value, list):
                    result[key] = [AIService._strip_additional_properties(item) for item in value]
                else:
                    result[key] = value
            elif key in ("allOf", "anyOf", "oneOf") and isinstance(value, list):
                result[key] = [AIService._strip_additional_properties(item) for item in value]
            elif key in ("definitions", "$defs") and isinstance(value, dict):
                result[key] = {
                    k: AIService._strip_additional_properties(v)
                    for k, v in value.items()
                }
            elif isinstance(value, dict):
                result[key] = AIService._strip_additional_properties(value)
            elif isinstance(value, list):
                result[key] = [
                    AIService._strip_additional_properties(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                result[key] = value

        return result

    @staticmethod
    def _convert_nullable_to_gemini_format(schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert JSON Schema type arrays to Gemini-compatible format.

        Gemini's SDK does not accept array-based types. This function handles two cases:

        1. Nullable types: ["type", "null"] → {"type": "type", "nullable": true}
        2. Union types: ["integer", "number"] → {"type": "number"} (picks most permissive)

        Transforms:
            {"type": ["string", "null"]} → {"type": "string", "nullable": true}
            {"type": ["number", "null"]} → {"type": "number", "nullable": true}
            {"type": ["integer", "null"]} → {"type": "integer", "nullable": true}
            {"type": ["boolean", "null"]} → {"type": "boolean", "nullable": true}
            {"type": ["array", "null"]} → {"type": "array", "nullable": true}
            {"type": ["object", "null"]} → {"type": "object", "nullable": true}
            {"type": ["integer", "number"]} → {"type": "number"} (union without null)

        For union types without null, priority is: number > integer > string > boolean > first

        Args:
            schema: The original JSON schema

        Returns:
            A deep copy of the schema with type arrays converted to Gemini format
        """
        if not isinstance(schema, dict):
            return schema

        result = {}
        for key, value in schema.items():
            if key == "type" and isinstance(value, list):
                # Check if this is a nullable type pattern: [actual_type, "null"]
                if len(value) == 2 and "null" in value:
                    # Extract the non-null type
                    non_null_type = [t for t in value if t != "null"][0]
                    result["type"] = non_null_type
                    result["nullable"] = True
                    continue
                elif "null" not in value and len(value) >= 1:
                    # Union type without null (e.g., ["integer", "number"])
                    # Gemini doesn't support type arrays, pick the most permissive type
                    # Priority: number > integer > string > boolean > first type
                    if "number" in value:
                        selected_type = "number"
                    elif "integer" in value:
                        selected_type = "integer"
                    elif "string" in value:
                        selected_type = "string"
                    elif "boolean" in value:
                        selected_type = "boolean"
                    else:
                        selected_type = value[0]
                    result["type"] = selected_type
                    logger.debug(
                        f"Gemini schema conversion: Simplified union type {value} to '{selected_type}'"
                    )
                    continue
                else:
                    # Other patterns (e.g., more than 2 types with null), keep as is
                    result[key] = value
            elif key == "properties" and isinstance(value, dict):
                result[key] = {
                    k: AIService._convert_nullable_to_gemini_format(v)
                    for k, v in value.items()
                }
            elif key == "items":
                if isinstance(value, dict):
                    result[key] = AIService._convert_nullable_to_gemini_format(value)
                elif isinstance(value, list):
                    result[key] = [AIService._convert_nullable_to_gemini_format(item) for item in value]
                else:
                    result[key] = value
            elif key in ("allOf", "anyOf", "oneOf") and isinstance(value, list):
                result[key] = [AIService._convert_nullable_to_gemini_format(item) for item in value]
            elif key in ("definitions", "$defs") and isinstance(value, dict):
                result[key] = {
                    k: AIService._convert_nullable_to_gemini_format(v)
                    for k, v in value.items()
                }
            elif isinstance(value, dict):
                result[key] = AIService._convert_nullable_to_gemini_format(value)
            elif isinstance(value, list):
                result[key] = [
                    AIService._convert_nullable_to_gemini_format(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                result[key] = value

        return result

    @staticmethod
    def _validate_schema_for_structured_outputs(
        schema: Dict[str, Any],
        provider: str,
        model_id: str
    ) -> None:
        """
        Validate JSON schema compatibility with structured output providers.

        Note: For Gemini, schemas should be pre-processed before validation:
        1. _strip_additional_properties() - removes 'additionalProperties' (not supported)
        2. _convert_nullable_to_gemini_format() - converts ["type", "null"] to nullable: true

        Raises:
            ValueError: If the schema contains unsupported features (e.g., additionalProperties: true)
        """
        provider_key = (provider or "").lower()

        def _check(node: Any, path: str) -> None:
            if not isinstance(node, dict):
                return

            schema_type = node.get("type")

            if path == "root" and schema_type is None:
                raise ValueError(
                    f"Schema validation error at '{path}': missing required 'type' at the root. "
                    f"Structured outputs for {model_id} need an explicit type (e.g., 'object'). "
                    "See https://docs.anthropic.com/en/docs/build-with-claude/structured-outputs"
                )

            # For Gemini, additionalProperties should already be stripped before validation
            # This check is now a safety net in case someone forgets to strip
            if provider_key in {"gemini", "google"}:
                if "additionalProperties" in node:
                    raise ValueError(
                        f"Schema validation error at '{path}': Gemini structured outputs "
                        f"do not support 'additionalProperties'. Use _strip_additional_properties() first."
                    )
            else:
                if node.get("additionalProperties") is True:
                    raise ValueError(
                        f"Schema validation error at '{path}': structured outputs for {model_id} "
                        f"do not allow 'additionalProperties: true'. Set it to false or remove it. "
                        "See https://docs.anthropic.com/en/docs/build-with-claude/structured-outputs"
                    )

            properties = node.get("properties")
            if isinstance(properties, dict):
                for prop_name, prop_schema in properties.items():
                    _check(prop_schema, f"{path}.properties.{prop_name}")

            items = node.get("items")
            if isinstance(items, dict):
                _check(items, f"{path}.items")
            elif isinstance(items, list):
                for index, item_schema in enumerate(items):
                    _check(item_schema, f"{path}.items[{index}]")

            # Handle common combinators
            for combinator in ("allOf", "anyOf", "oneOf"):
                if combinator in node and isinstance(node[combinator], list):
                    for idx, subschema in enumerate(node[combinator]):
                        _check(subschema, f"{path}.{combinator}[{idx}]")

            # Handle definitions / $defs
            for defs_key in ("definitions", "$defs"):
                definitions = node.get(defs_key)
                if isinstance(definitions, dict):
                    for def_name, def_schema in definitions.items():
                        _check(def_schema, f"{path}.{defs_key}.{def_name}")

        _check(schema, "root")

    def _max_retry_attempts(self) -> int:
        try:
            return max(1, int(getattr(config, "MAX_RETRIES", 3)))
        except Exception:
            return 3

    def _retry_delay_seconds(self) -> float:
        try:
            delay = float(getattr(config, "RETRY_DELAY", 10.0))
            return max(0.0, delay)
        except Exception:
            return 10.0

    @staticmethod
    def _extract_request_id(exc: Exception) -> Optional[str]:
        for attr in ("request_id", "response_id", "id"):
            value = getattr(exc, attr, None)
            if value:
                return str(value)
        response = getattr(exc, "response", None)
        if response:
            for attr in ("request_id", "id"):
                value = getattr(response, attr, None)
                if value:
                    return str(value)
        return None

    @staticmethod
    def _should_retry_exception(exc: Exception) -> bool:
        # Network and timeout errors are always retriable
        if isinstance(exc, (asyncio.TimeoutError, aiohttp.ClientError, ConnectionError, OSError)):
            return True

        # HTTP status codes that indicate transient issues
        status = getattr(exc, "status", None) or getattr(exc, "status_code", None)
        if status in {408, 425, 429, 500, 502, 503, 504}:
            return True

        message = str(exc).lower()

        # Catch SDK bugs that manifest as AttributeError during connection issues
        # (e.g., Google GenAI SDK trying to use non-existent aiohttp attributes)
        if isinstance(exc, AttributeError):
            sdk_bug_markers = ["aiohttp", "connector", "client", "http", "socket"]
            if any(marker in message for marker in sdk_bug_markers):
                return True

        # Transient error patterns in exception messages
        transient_markers = [
            "timeout",
            "temporarily unavailable",
            "internal server error",
            "gateway",
            "rate limit",
            "overloaded",
            "unavailable",
            "service unavailable",
            "connection reset",
            "connection refused",
            "dns",
            "network",
        ]
        return any(marker in message for marker in transient_markers)

    async def _execute_with_retries(
        self,
        operation: Callable[[], Awaitable[T]],
        *,
        provider: str,
        model_id: str,
        action: str,
    ) -> T:
        max_attempts = self._max_retry_attempts()
        delay_seconds = self._retry_delay_seconds()
        last_exception: Optional[Exception] = None

        for attempt in range(1, max_attempts + 1):
            try:
                return await operation()
            except AIRequestError:
                raise
            except Exception as exc:
                last_exception = exc
                should_retry = attempt < max_attempts and self._should_retry_exception(exc)

                if not should_retry:
                    raise

                request_id = self._extract_request_id(exc)
                suffix = f" (request_id={request_id})" if request_id else ""
                logger.warning(
                    "AI %s failed for %s via %s on attempt %d/%d%s: %s",
                    action,
                    model_id,
                    provider,
                    attempt,
                    max_attempts,
                    suffix,
                    exc,
                )

                await asyncio.sleep(delay_seconds)

        assert last_exception is not None
        raise AIRequestError(provider, model_id, max_attempts, max_attempts, last_exception) from last_exception

    @staticmethod
    def _normalize_usage(usage_obj: Any) -> Optional[Dict[str, Any]]:
        """Extract token metrics from provider-specific usage objects."""

        if usage_obj is None:
            return None

        def _pluck(obj: Any, *names: str) -> Optional[int]:
            for name in names:
                if isinstance(obj, dict) and name in obj:
                    return obj[name]
                if hasattr(obj, name):
                    value = getattr(obj, name)
                    if value is not None:
                        return value
            return None

        input_tokens = _pluck(usage_obj, "prompt_tokens", "input_tokens", "input_token_count") or 0
        output_tokens = _pluck(usage_obj, "completion_tokens", "output_tokens", "output_token_count") or 0
        total_tokens = _pluck(usage_obj, "total_tokens", "total_token_count")
        reasoning_tokens = _pluck(usage_obj, "reasoning_tokens", "thinking_tokens")

        # Some providers return nested metadata (e.g., Anthropic streaming final response)
        if hasattr(usage_obj, "model_dump"):
            try:
                dumped = usage_obj.model_dump()
                if isinstance(dumped, dict):
                    input_tokens = dumped.get("input_tokens", input_tokens) or input_tokens
                    output_tokens = dumped.get("output_tokens", output_tokens) or output_tokens
                    total_tokens = dumped.get("total_tokens", total_tokens)
            except Exception:
                pass

        return {
            "input_tokens": int(input_tokens or 0),
            "output_tokens": int(output_tokens or 0),
            "total_tokens": int(total_tokens) if total_tokens is not None else None,
            "reasoning_tokens": int(reasoning_tokens) if reasoning_tokens is not None else None,
        }

    def _emit_usage(
        self,
        usage_callback: Optional[Callable[[Dict[str, Any]], None]],
        model_id: str,
        provider: str,
        usage_obj: Any,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Safely emit usage payload to the provided callback."""

        if not usage_callback:
            return

        usage_payload = self._normalize_usage(usage_obj)
        if not usage_payload:
            usage_payload = {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": None,
                "reasoning_tokens": None,
            }

        payload: Dict[str, Any] = {
            "model": model_id,
            "provider": provider,
        }
        payload.update({k: v for k, v in usage_payload.items() if v is not None})

        if extra:
            payload.update(extra)

        try:
            usage_callback(payload)
        except Exception:
            logger.exception("Usage callback failed for model %s", model_id)

    async def generate_content(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 8000,
        system_prompt: Optional[str] = None,
        extra_verbose: bool = False,
        reasoning_effort: Optional[str] = None,
        thinking_budget_tokens: Optional[int] = None,
        content_type: str = "biography",
        json_output: bool = False,
        json_schema: Optional[Dict[str, Any]] = None,
        usage_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        usage_extra: Optional[Dict[str, Any]] = None,
        phase_logger: Optional["PhaseLogger"] = None,
    ) -> str:
        """
        Generate content using specified AI model

        Args:
            prompt: The generation prompt
            model: Model name
            temperature: Generation temperature
            max_tokens: Maximum tokens
            system_prompt: Optional system prompt
            reasoning_effort: Reasoning effort level for GPT-5 models
            thinking_budget_tokens: Budget tokens for thinking/reasoning
            content_type: Type of content being generated (affects system prompt selection)
            json_output: Whether to force JSON output format when supported

        Returns:
            Generated content as string
        """
        # Intelligent validation and adjustment of all token parameters
        token_validation = config.validate_token_limits(
            model, max_tokens, reasoning_effort, thinking_budget_tokens
        )
        adjusted_max_tokens = token_validation["adjusted_tokens"]
        adjusted_reasoning_effort = token_validation["adjusted_reasoning_effort"]
        adjusted_thinking_budget = token_validation["adjusted_thinking_budget_tokens"]
        thinking_validation = token_validation["thinking_validation"]

        # Log adjustments if any were made
        if token_validation["was_adjusted"]:
            logger.warning(f"Token limit adjusted for {model}: {max_tokens} -> {adjusted_max_tokens} (model limit: {token_validation['model_limit']})")

        # Detailed logging for reasoning/thinking parameter handling
        received_tokens = thinking_budget_tokens is not None
        received_effort = reasoning_effort is not None

        if received_tokens and received_effort:
            # Both parameters received
            if adjusted_thinking_budget is None and adjusted_reasoning_effort is not None:
                logger.info(f"Model {model} received both thinking_budget_tokens ({thinking_budget_tokens}) and reasoning_effort ({reasoning_effort}). Using reasoning_effort={adjusted_reasoning_effort}, discarding thinking_budget_tokens.")
            elif adjusted_thinking_budget is not None and adjusted_reasoning_effort != reasoning_effort:
                logger.info(f"Model {model} received both thinking_budget_tokens ({thinking_budget_tokens}) and reasoning_effort ({reasoning_effort}). Using thinking_budget_tokens={adjusted_thinking_budget}, discarding reasoning_effort.")
            elif adjusted_thinking_budget != thinking_budget_tokens or adjusted_reasoning_effort != reasoning_effort:
                logger.info(f"Model {model} received both parameters - thinking_budget_tokens: {thinking_budget_tokens} -> {adjusted_thinking_budget}, reasoning_effort: {reasoning_effort} -> {adjusted_reasoning_effort}")
        elif received_tokens and not received_effort:
            # Only tokens received
            if adjusted_thinking_budget != thinking_budget_tokens:
                reason = thinking_validation.get("reason", "Parameter optimization")
                logger.info(f"Model {model} received thinking_budget_tokens ({thinking_budget_tokens}), adjusted to {adjusted_thinking_budget} ({reason})")
            elif adjusted_reasoning_effort is not None:
                logger.info(f"Model {model} received thinking_budget_tokens ({thinking_budget_tokens}), converted to reasoning_effort={adjusted_reasoning_effort}")
            else:
                logger.info(f"Model {model} using thinking_budget_tokens={adjusted_thinking_budget}")
        elif received_effort and not received_tokens:
            # Only effort received
            if adjusted_reasoning_effort != reasoning_effort:
                logger.info(f"Model {model} received reasoning_effort ({reasoning_effort}), adjusted to {adjusted_reasoning_effort}")
            elif adjusted_thinking_budget is not None:
                logger.info(f"Model {model} received reasoning_effort ({reasoning_effort}), converted to thinking_budget_tokens={adjusted_thinking_budget}")
            else:
                logger.info(f"Model {model} using reasoning_effort={adjusted_reasoning_effort}")
        elif not received_tokens and not received_effort and (adjusted_thinking_budget is not None or adjusted_reasoning_effort is not None):
            # Neither received but defaults applied
            if adjusted_thinking_budget is not None:
                logger.info(f"Model {model} using default thinking_budget_tokens={adjusted_thinking_budget}")
            if adjusted_reasoning_effort is not None:
                logger.info(f"Model {model} using default reasoning_effort={adjusted_reasoning_effort}")

        model_info = token_validation["model_info"]
        provider = model_info["provider"]
        model_id = model_info["model_id"]
        extra_payload = {"requested_model": model}
        if usage_extra:
            extra_payload.update(usage_extra)

        # Enforce structured-output schema compatibility upfront
        # For Gemini, transform schema to match their requirements:
        # 1. Strip additionalProperties (not supported)
        # 2. Convert nullable types from ["type", "null"] to {"type": "type", "nullable": true}
        effective_json_schema = json_schema
        if json_output and json_schema:
            if provider in ("gemini", "google"):
                effective_json_schema = self._strip_additional_properties(json_schema)
                effective_json_schema = self._convert_nullable_to_gemini_format(effective_json_schema)
            self._validate_schema_for_structured_outputs(effective_json_schema, provider, model_id)
        elif json_output and not json_schema and provider == "openai" and any(
            marker in model_id.lower() for marker in ["o3-pro", "gpt-5-pro"]
        ):
            fallback_schema = {"type": "object", "additionalProperties": False, "properties": {}}
            self._validate_schema_for_structured_outputs(fallback_schema, provider, model_id)

        request_timeout = token_validation.get("reasoning_timeout_seconds")
        if request_timeout and request_timeout > 0:
            logger.info(
                f"Applying reasoning timeout of {request_timeout} seconds for model {model_id}"
            )

        # Use the intelligently adjusted parameters for generation
        reasoning_effort = adjusted_reasoning_effort
        thinking_budget_tokens = adjusted_thinking_budget

        temperature, thinking_budget_tokens, forced_temperature = self._apply_temperature_policies(
            model_info, temperature, thinking_budget_tokens
        )

        # Log full prompt if extra_verbose is enabled
        if phase_logger:
            # Use PhaseLogger for enhanced visual logging
            params = {
                "temperature": temperature,
                "max_tokens": adjusted_max_tokens,
            }
            if reasoning_effort:
                params["reasoning_effort"] = reasoning_effort
            if thinking_budget_tokens:
                params["thinking_budget_tokens"] = thinking_budget_tokens
            if forced_temperature:
                params["temperature_override"] = f"Overridden due to reasoning policy"

            phase_logger.log_prompt(
                model=model,
                system_prompt=system_prompt,
                user_prompt=prompt,
                **params
            )
        elif extra_verbose:
            # Fallback to standard logging if phase_logger not available
            logger.info(f"[EXTRA_VERBOSE] AI GENERATION PROMPT for {model}:")
            logger.info(f"[EXTRA_VERBOSE] System: {system_prompt if system_prompt else 'None'}")
            logger.info(f"[EXTRA_VERBOSE] User: {prompt}")
            logger.info(f"[EXTRA_VERBOSE] Temperature: {temperature}, Max Tokens: {adjusted_max_tokens}")
            logger.info(f"[EXTRA_VERBOSE] --- END PROMPT ---")

            if forced_temperature:
                logger.info(
                    f"[EXTRA_VERBOSE] Temperature overridden to {temperature} due to reasoning policy for {model}"
                )

        # Select appropriate system prompt based on content type if not explicitly provided
        if system_prompt is None:
            if content_type in {"other", "json"}:
                system_prompt = config.GENERATOR_SYSTEM_PROMPT_RAW
            else:
                system_prompt = config.GENERATOR_SYSTEM_PROMPT

        # Add language instruction and current date to all prompts
        current_date = datetime.now().strftime("%Y/%m/%d")
        language_instruction = "\n\nIMPORTANT: Always respond in the same language as the user's request, regardless of the language used in these instructions."
        date_instruction = f"\n\nCurrent date: {current_date}"
        prompt = prompt + language_instruction + date_instruction

        async def _single_attempt() -> str:
            if provider == "openai":
                content, usage_meta = await self._generate_openai(
                    prompt,
                    model_id,
                    temperature,
                    adjusted_max_tokens,
                    system_prompt or "",
                    reasoning_effort,
                    extra_verbose=extra_verbose,
                    request_timeout=request_timeout,
                    json_output=json_output,
                    json_schema=json_schema,
                )

                # Log full response if extra_verbose is enabled
                if phase_logger:
                    phase_logger.log_response(
                        model=model_id,
                        response=content,
                        metadata={
                            "input_tokens": usage_meta.get("input_tokens"),
                            "output_tokens": usage_meta.get("output_tokens"),
                            "reasoning_tokens": usage_meta.get("reasoning_tokens"),
                            "provider": provider,
                        }
                    )
                elif extra_verbose:
                    separator = "=" * 80
                    logger.info(f"\n{separator}")
                    logger.info(f"[EXTRA_VERBOSE] AI GENERATION RESPONSE from {model_id}")
                    logger.info(f"{separator}")
                    logger.info(content)
                    logger.info(f"{separator}\n")

                self._emit_usage(usage_callback, model_id, provider, usage_meta, extra_payload)
                return content
            elif provider == "claude":
                content, usage_meta = await self._generate_claude(
                    prompt,
                    model_id,
                    temperature,
                    adjusted_max_tokens,
                    system_prompt or "",
                    thinking_budget_tokens,
                    request_timeout=request_timeout,
                    json_output=json_output,
                    json_schema=json_schema,
                )

                # Log full response if extra_verbose is enabled
                if phase_logger:
                    phase_logger.log_response(
                        model=model_id,
                        response=content,
                        metadata={
                            "input_tokens": usage_meta.get("input_tokens"),
                            "output_tokens": usage_meta.get("output_tokens"),
                            "provider": provider,
                        }
                    )
                elif extra_verbose:
                    separator = "=" * 80
                    logger.info(f"\n{separator}")
                    logger.info(f"[EXTRA_VERBOSE] AI GENERATION RESPONSE from {model_id}")
                    logger.info(f"{separator}")
                    logger.info(content)
                    logger.info(f"{separator}\n")

                self._emit_usage(usage_callback, model_id, provider, usage_meta, extra_payload)
                return content
            elif provider == "gemini":
                content, usage_meta = await self._generate_gemini(
                    prompt,
                    model_id,
                    temperature,
                    adjusted_max_tokens,
                    system_prompt or "",
                    json_output=json_output,
                    json_schema=effective_json_schema,
                )

                # Log full response if extra_verbose is enabled
                if phase_logger:
                    phase_logger.log_response(
                        model=model_id,
                        response=content,
                        metadata={
                            "input_tokens": usage_meta.get("input_tokens"),
                            "output_tokens": usage_meta.get("output_tokens"),
                            "provider": provider,
                        }
                    )
                elif extra_verbose:
                    separator = "=" * 80
                    logger.info(f"\n{separator}")
                    logger.info(f"[EXTRA_VERBOSE] AI GENERATION RESPONSE from {model_id}")
                    logger.info(f"{separator}")
                    logger.info(content)
                    logger.info(f"{separator}\n")

                self._emit_usage(usage_callback, model_id, provider, usage_meta, extra_payload)
                return content
            elif provider == "xai":
                content, usage_meta = await self._generate_xai(
                    prompt,
                    model_id,
                    temperature,
                    adjusted_max_tokens,
                    system_prompt or "",
                    json_output=json_output,
                    json_schema=json_schema,
                )

                # Log full response if extra_verbose is enabled
                if phase_logger:
                    phase_logger.log_response(
                        model=model_id,
                        response=content,
                        metadata={
                            "input_tokens": usage_meta.get("input_tokens"),
                            "output_tokens": usage_meta.get("output_tokens"),
                            "provider": provider,
                        }
                    )
                elif extra_verbose:
                    separator = "=" * 80
                    logger.info(f"\n{separator}")
                    logger.info(f"[EXTRA_VERBOSE] AI GENERATION RESPONSE from {model_id}")
                    logger.info(f"{separator}")
                    logger.info(content)
                    logger.info(f"{separator}\n")

                self._emit_usage(usage_callback, model_id, provider, usage_meta, extra_payload)
                return content
            elif provider == "openrouter":
                content, usage_meta = await self._generate_openrouter(
                    prompt,
                    model_id,
                    temperature,
                    adjusted_max_tokens,
                    system_prompt or "",
                    json_output=json_output,
                    json_schema=json_schema,
                )

                # Log full response if extra_verbose is enabled
                if phase_logger:
                    phase_logger.log_response(
                        model=model_id,
                        response=content,
                        metadata={
                            "input_tokens": usage_meta.get("input_tokens"),
                            "output_tokens": usage_meta.get("output_tokens"),
                            "provider": provider,
                        }
                    )
                elif extra_verbose:
                    separator = "=" * 80
                    logger.info(f"\n{separator}")
                    logger.info(f"[EXTRA_VERBOSE] AI GENERATION RESPONSE from {model_id}")
                    logger.info(f"{separator}")
                    logger.info(content)
                    logger.info(f"{separator}\n")

                self._emit_usage(usage_callback, model_id, provider, usage_meta, extra_payload)
                return content
            elif provider == "ollama":
                content, usage_meta = await self._generate_ollama(
                    prompt,
                    model_id,
                    temperature,
                    adjusted_max_tokens,
                    system_prompt or "",
                    json_output=json_output,
                    json_schema=json_schema,
                )

                # Log full response if extra_verbose is enabled
                if phase_logger:
                    phase_logger.log_response(
                        model=model_id,
                        response=content,
                        metadata={
                            "input_tokens": usage_meta.get("input_tokens"),
                            "output_tokens": usage_meta.get("output_tokens"),
                            "provider": provider,
                        }
                    )
                elif extra_verbose:
                    separator = "=" * 80
                    logger.info(f"\n{separator}")
                    logger.info(f"[EXTRA_VERBOSE] AI GENERATION RESPONSE from {model_id}")
                    logger.info(f"{separator}")
                    logger.info(content)
                    logger.info(f"{separator}\n")

                self._emit_usage(usage_callback, model_id, provider, usage_meta, extra_payload)
                return content
            else:
                raise ValueError(f"Unsupported model provider: {provider}")

        try:
            return await self._execute_with_retries(
                _single_attempt,
                provider=provider,
                model_id=model_id,
                action="generation",
            )
        except Exception as e:
            logger.error(f"Content generation failed for {model}: {str(e)}")
            raise
    
    async def _generate_openai(
        self,
        prompt: str,
        model_id: str,
        temperature: float,
        max_tokens: int,
        system_prompt: str,
        reasoning_effort: Optional[str] = None,
        extra_verbose: bool = False,
        request_timeout: Optional[float] = None,
        json_output: bool = False,
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Any]:
        """Generate content using OpenAI API with optional JSON Schema support"""
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized")

        effective_system_prompt = system_prompt
        request_kwargs: Dict[str, Any] = {}
        if request_timeout and request_timeout > 0:
            request_kwargs["timeout"] = request_timeout

        # Add JSON instruction to prompt if needed
        effective_prompt = prompt
        if json_output:
            effective_prompt = f"{prompt}\n\nIMPORTANT: Output valid JSON only. When including dialogue or quotes in string values, use single quotes (') instead of double quotes (\") to avoid JSON parsing errors. Example: He said 'hello' instead of He said \"hello\"."

        # Default messages (for non-o3/o1/Responses routes)
        messages = [
            {"role": "system", "content": effective_system_prompt},
            {"role": "user", "content": effective_prompt}
        ]

        # --- RAZONAMIENTO o3-pro: Responses API ---
        if "o3-pro" in model_id.lower():
            if not hasattr(self, 'openai_sync_client') or not self.openai_sync_client:
                raise ValueError("OpenAI sync client not initialized for O3-pro")

            # IMPORTANT! In Responses API, use 'instructions' for the "system" prompt.
            create_params = {
                "model": model_id,
                "input": effective_prompt,                     # better as plain string
                "instructions": effective_system_prompt,       # system/developer here
                "max_output_tokens": max_tokens
            }
            if reasoning_effort:
                create_params["reasoning"] = {"effort": reasoning_effort}

            # Configure JSON output format (Responses API uses "text" parameter)
            if json_output:
                # Use provided schema or fallback to flexible schema
                schema_to_use = json_schema if json_schema else {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {}
                }

                # Responses API uses text.format instead of response_format
                create_params["text"] = {
                    "format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "structured_output" if json_schema else "flexible_json_output",
                            "strict": True,
                            "schema": schema_to_use
                        }
                    }
                }

                if json_schema:
                    logger.info(f"Using O3-Pro JSON Schema structured outputs (custom schema)")
                else:
                    logger.info(f"Using O3-Pro JSON Schema structured outputs (flexible schema)")

            if extra_verbose:
                logger.info(f"[EXTRA_VERBOSE] O3-pro responses parameters: {create_params}")

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.openai_sync_client.responses.create(
                    **create_params,
                    **request_kwargs,
                ),
            )

            # Robust text extraction: 1) output_text; 2) assemble from 'output'
            content = getattr(response, "output_text", None) or ""
            if not content.strip():
                try:
                    data = response.model_dump() if hasattr(response, "model_dump") else None
                except Exception:
                    data = None

                if data and isinstance(data, dict):
                    pieces: List[str] = []
                    for item in data.get("output", []) or []:
                        if item.get("type") == "message":
                            for part in item.get("content", []) or []:
                                txt = part.get("text")
                                if txt:
                                    pieces.append(txt)
                    content = "".join(pieces).strip()

            # If still empty, likely token limit or SDK regression
            if not content:
                logger.warning(
                    "O3-pro returned empty output_text. Possible causes: "
                    "1) insufficient max_output_tokens (includes reasoning), "
                    "2) SDK regression in output_text. Suggestion: increase max_output_tokens."
                )

            return content or "", getattr(response, "usage", None)

        # --- GPT-5 Pro: Responses API ---
        elif "gpt-5-pro" in model_id.lower():
            if not hasattr(self, 'openai_sync_client') or not self.openai_sync_client:
                raise ValueError("OpenAI sync client not initialized for GPT-5 Pro")

            # GPT-5 Pro usa Responses API igual que o3-pro
            create_params = {
                "model": model_id,
                "input": effective_prompt,
                "instructions": effective_system_prompt,
                "max_output_tokens": max_tokens
            }

            # GPT-5 Pro defaults to high reasoning effort
            if reasoning_effort:
                create_params["reasoning"] = {"effort": reasoning_effort}
            else:
                create_params["reasoning"] = {"effort": "high"}

            # Configure JSON output format (Responses API uses "text" parameter)
            if json_output:
                # Use provided schema or fallback to flexible schema
                schema_to_use = json_schema if json_schema else {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {}
                }

                # Responses API uses text.format instead of response_format
                create_params["text"] = {
                    "format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "structured_output" if json_schema else "flexible_json_output",
                            "strict": True,
                            "schema": schema_to_use
                        }
                    }
                }

                if json_schema:
                    logger.info(f"Using GPT-5 Pro JSON Schema structured outputs (custom schema)")
                else:
                    logger.info(f"Using GPT-5 Pro JSON Schema structured outputs (flexible schema)")

            if extra_verbose:
                logger.info(f"[EXTRA_VERBOSE] GPT-5 Pro responses parameters: {create_params}")

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.openai_sync_client.responses.create(
                    **create_params,
                    **request_kwargs,
                ),
            )

            # Robust text extraction (same as o3-pro)
            content = getattr(response, "output_text", None) or ""
            if not content.strip():
                try:
                    data = response.model_dump() if hasattr(response, "model_dump") else None
                except Exception:
                    data = None

                if data and isinstance(data, dict):
                    pieces: List[str] = []
                    for item in data.get("output", []) or []:
                        if item.get("type") == "message":
                            for part in item.get("content", []) or []:
                                txt = part.get("text")
                                if txt:
                                    pieces.append(txt)
                    content = "".join(pieces).strip()

            # If still empty, warn
            if not content:
                logger.warning(
                    "GPT-5 Pro returned empty output_text. Possible causes: "
                    "1) insufficient max_output_tokens (includes reasoning tokens), "
                    "2) SDK regression in output_text. Suggestion: increase max_output_tokens."
                )

            return content or "", getattr(response, "usage", None)

        # --- O1/O3 (no pro) por Chat Completions ---
        elif any(x in model_id.lower() for x in ["o1", "o3"]) and "o3-pro" not in model_id.lower():
            # O1/O3: sin system, sin temperature (se ignora)
            messages = [{"role": "user", "content": f"{effective_system_prompt}\n\n{effective_prompt}"}]
            create_params = _build_openai_params(model_id, messages, temperature, max_tokens, reasoning_effort)

            # Configure JSON output format
            if json_output:
                if json_schema:
                    # Use JSON Schema for structured outputs
                    create_params["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "structured_output",
                            "strict": True,
                            "schema": json_schema
                        }
                    }
                    logger.info(f"Using O1/O3 JSON Schema structured outputs for {model_id}")
                else:
                    # Use basic JSON mode (flexible structure)
                    create_params["response_format"] = {"type": "json_object"}
                    logger.info(f"Using O1/O3 JSON mode (flexible) for {model_id}")

            if extra_verbose:
                logger.info(f"[EXTRA_VERBOSE] O1/O3 non-streaming parameters: {create_params}")

            response = await self.openai_client.chat.completions.create(
                **create_params,
                **request_kwargs,
            )
            return response.choices[0].message.content, getattr(response, "usage", None)

        # --- GPT-5 por Chat Completions ---
        elif "gpt-5" in model_id.lower():
            create_params = _build_openai_params(model_id, messages, temperature, max_tokens, reasoning_effort)

            # Configure JSON output format
            if json_output:
                if json_schema:
                    # Use JSON Schema for structured outputs
                    create_params["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "structured_output",
                            "strict": True,
                            "schema": json_schema
                        }
                    }
                    logger.info(f"Using GPT-5 JSON Schema structured outputs for {model_id}")
                else:
                    # Use basic JSON mode (flexible structure)
                    create_params["response_format"] = {"type": "json_object"}
                    logger.info(f"Using GPT-5 JSON mode (flexible) for {model_id}")

            if extra_verbose:
                logger.info(f"[EXTRA_VERBOSE] GPT-5 non-streaming parameters: {create_params}")

            try:
                response = await self.openai_client.chat.completions.create(
                    **create_params,
                    **request_kwargs,
                )
            except Exception as e:
                error_msg = f"GPT-5 model '{model_id}' failed: {str(e)}"
                logger.error(error_msg)
                raise Exception(error_msg) from e

            content = response.choices[0].message.content or ""
            if not content.strip():
                error_msg = f"GPT-5 model '{model_id}' returned empty content. This may indicate the model requires different API endpoint (e.g., gpt-5-pro requires Responses API) or other configuration issues."
                logger.error(error_msg)
                raise Exception(error_msg)

            return content, getattr(response, "usage", None)

        # --- Standard models via Chat Completions ---
        else:
            if extra_verbose:
                logger.info(f"[EXTRA_VERBOSE] Standard OpenAI non-streaming with temperature: {temperature}")
            standard_params = _build_openai_params(model_id, messages, temperature, max_tokens, reasoning_effort)

            # Configure JSON output format
            if json_output:
                if json_schema:
                    # Use JSON Schema for structured outputs (GPT-4o-2024-08-06+)
                    standard_params["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "structured_output",
                            "strict": True,
                            "schema": json_schema
                        }
                    }
                    logger.info(f"Using OpenAI JSON Schema structured outputs for {model_id}")
                else:
                    # Use basic JSON mode (flexible structure)
                    standard_params["response_format"] = {"type": "json_object"}
                    logger.info(f"Using OpenAI JSON mode (flexible) for {model_id}")

            response = await self.openai_client.chat.completions.create(
                **standard_params,
                **request_kwargs,
            )
            return response.choices[0].message.content, getattr(response, "usage", None)
    
    async def generate_content_stream(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 8000,
        system_prompt: Optional[str] = None,
        extra_verbose: bool = False,
        reasoning_effort: Optional[str] = None,
        thinking_budget_tokens: Optional[int] = None,
        content_type: str = "biography",
        json_output: bool = False,
        json_schema: Optional[Dict[str, Any]] = None,
        usage_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        usage_extra: Optional[Dict[str, Any]] = None,
        phase_logger: Optional["PhaseLogger"] = None,
    ):
        """
        Generate content with streaming support

        Args:
            prompt: The generation prompt
            model: Model name
            temperature: Generation temperature
            max_tokens: Maximum tokens
            system_prompt: Optional system prompt
            reasoning_effort: Reasoning effort level for GPT-5 models
            thinking_budget_tokens: Budget tokens for thinking/reasoning
            content_type: Type of content being generated (affects system prompt selection)
            json_output: Whether to force JSON output format (response_format: json_object)

        Yields:
            Generated content chunks as they are produced
        """
        # Intelligent validation and adjustment of all token parameters
        token_validation = config.validate_token_limits(
            model, max_tokens, reasoning_effort, thinking_budget_tokens
        )
        adjusted_max_tokens = token_validation["adjusted_tokens"]
        adjusted_reasoning_effort = token_validation["adjusted_reasoning_effort"]
        adjusted_thinking_budget = token_validation["adjusted_thinking_budget_tokens"]
        thinking_validation = token_validation["thinking_validation"]

        # Log adjustments if any were made
        if token_validation["was_adjusted"]:
            logger.warning(f"Token limit adjusted for {model}: {max_tokens} -> {adjusted_max_tokens} (model limit: {token_validation['model_limit']})")

        # Detailed logging for reasoning/thinking parameter handling
        received_tokens = thinking_budget_tokens is not None
        received_effort = reasoning_effort is not None

        if received_tokens and received_effort:
            # Both parameters received
            if adjusted_thinking_budget is None and adjusted_reasoning_effort is not None:
                logger.info(f"Model {model} received both thinking_budget_tokens ({thinking_budget_tokens}) and reasoning_effort ({reasoning_effort}). Using reasoning_effort={adjusted_reasoning_effort}, discarding thinking_budget_tokens.")
            elif adjusted_thinking_budget is not None and adjusted_reasoning_effort != reasoning_effort:
                logger.info(f"Model {model} received both thinking_budget_tokens ({thinking_budget_tokens}) and reasoning_effort ({reasoning_effort}). Using thinking_budget_tokens={adjusted_thinking_budget}, discarding reasoning_effort.")
            elif adjusted_thinking_budget != thinking_budget_tokens or adjusted_reasoning_effort != reasoning_effort:
                logger.info(f"Model {model} received both parameters - thinking_budget_tokens: {thinking_budget_tokens} -> {adjusted_thinking_budget}, reasoning_effort: {reasoning_effort} -> {adjusted_reasoning_effort}")
        elif received_tokens and not received_effort:
            # Only tokens received
            if adjusted_thinking_budget != thinking_budget_tokens:
                reason = thinking_validation.get("reason", "Parameter optimization")
                logger.info(f"Model {model} received thinking_budget_tokens ({thinking_budget_tokens}), adjusted to {adjusted_thinking_budget} ({reason})")
            elif adjusted_reasoning_effort is not None:
                logger.info(f"Model {model} received thinking_budget_tokens ({thinking_budget_tokens}), converted to reasoning_effort={adjusted_reasoning_effort}")
            else:
                logger.info(f"Model {model} using thinking_budget_tokens={adjusted_thinking_budget}")
        elif received_effort and not received_tokens:
            # Only effort received
            if adjusted_reasoning_effort != reasoning_effort:
                logger.info(f"Model {model} received reasoning_effort ({reasoning_effort}), adjusted to {adjusted_reasoning_effort}")
            elif adjusted_thinking_budget is not None:
                logger.info(f"Model {model} received reasoning_effort ({reasoning_effort}), converted to thinking_budget_tokens={adjusted_thinking_budget}")
            else:
                logger.info(f"Model {model} using reasoning_effort={adjusted_reasoning_effort}")
        elif not received_tokens and not received_effort and (adjusted_thinking_budget is not None or adjusted_reasoning_effort is not None):
            # Neither received but defaults applied
            if adjusted_thinking_budget is not None:
                logger.info(f"Model {model} using default thinking_budget_tokens={adjusted_thinking_budget}")
            if adjusted_reasoning_effort is not None:
                logger.info(f"Model {model} using default reasoning_effort={adjusted_reasoning_effort}")

        model_info = token_validation["model_info"]
        provider = model_info["provider"]
        model_id = model_info["model_id"]
        extra_payload = {"requested_model": model}
        if usage_extra:
            extra_payload.update(usage_extra)

        # Enforce structured-output schema compatibility upfront
        # For Gemini, transform schema to match their requirements:
        # 1. Strip additionalProperties (not supported)
        # 2. Convert nullable types from ["type", "null"] to {"type": "type", "nullable": true}
        effective_json_schema = json_schema
        if json_output and json_schema:
            if provider in ("gemini", "google"):
                effective_json_schema = self._strip_additional_properties(json_schema)
                effective_json_schema = self._convert_nullable_to_gemini_format(effective_json_schema)
            self._validate_schema_for_structured_outputs(effective_json_schema, provider, model_id)
        elif json_output and not json_schema and provider == "openai" and any(
            marker in model_id.lower() for marker in ["o3-pro", "gpt-5-pro"]
        ):
            fallback_schema = {"type": "object", "additionalProperties": False, "properties": {}}
            self._validate_schema_for_structured_outputs(fallback_schema, provider, model_id)

        request_timeout = token_validation.get("reasoning_timeout_seconds")

        # Use the intelligently adjusted parameters for generation
        reasoning_effort = adjusted_reasoning_effort
        thinking_budget_tokens = adjusted_thinking_budget

        temperature, thinking_budget_tokens, forced_temperature = self._apply_temperature_policies(
            model_info, temperature, thinking_budget_tokens
        )

        # Log full prompt if extra_verbose is enabled
        if phase_logger:
            # Use PhaseLogger for enhanced visual logging
            params = {
                "temperature": temperature,
                "max_tokens": adjusted_max_tokens,
            }
            if reasoning_effort:
                params["reasoning_effort"] = reasoning_effort
            if thinking_budget_tokens:
                params["thinking_budget_tokens"] = thinking_budget_tokens
            if forced_temperature:
                params["temperature_override"] = f"Overridden due to reasoning policy"

            phase_logger.log_prompt(
                model=model,
                system_prompt=system_prompt,
                user_prompt=prompt,
                **params
            )
        elif extra_verbose and forced_temperature:
            logger.info(
                f"[EXTRA_VERBOSE] Temperature overridden to {temperature} due to reasoning policy for {model}"
            )

        # Select appropriate system prompt based on content type if not explicitly provided
        if system_prompt is None:
            if content_type in {"other", "json"}:
                system_prompt = config.GENERATOR_SYSTEM_PROMPT_RAW
            else:
                system_prompt = config.GENERATOR_SYSTEM_PROMPT

        # Add language instruction and current date to all prompts
        current_date = datetime.now().strftime("%Y/%m/%d")
        language_instruction = "\n\nIMPORTANT: Always respond in the same language as the user's request, regardless of the language used in these instructions."
        date_instruction = f"\n\nCurrent date: {current_date}"
        prompt = prompt + language_instruction + date_instruction

        async def _dispatch_stream():
            if provider == "openai":
                async for chunk in self._stream_openai(
                    prompt,
                    model_id,
                    temperature,
                    adjusted_max_tokens,
                    system_prompt,
                    extra_verbose,
                    reasoning_effort,
                    thinking_budget_tokens,
                    request_timeout=request_timeout,
                    json_output=json_output,
                    json_schema=json_schema,
                    usage_callback=usage_callback,
                    provider=provider,
                    resolved_model_id=model_id,
                    usage_extra=extra_payload,
                ):
                    yield chunk
            elif provider in {"anthropic", "claude"}:
                async for chunk in self._stream_claude(
                    prompt,
                    model_id,
                    temperature,
                    adjusted_max_tokens,
                    system_prompt,
                    extra_verbose,
                    thinking_budget_tokens,
                    json_output=json_output,
                    json_schema=json_schema,
                    usage_callback=usage_callback,
                    provider=provider,
                    resolved_model_id=model_id,
                    usage_extra=extra_payload,
                ):
                    yield chunk
            elif provider in {"google", "gemini"}:
                async for chunk in self._stream_gemini(
                    prompt,
                    model_id,
                    temperature,
                    adjusted_max_tokens,
                    system_prompt,
                    json_output=json_output,
                    json_schema=effective_json_schema,
                    usage_callback=usage_callback,
                    provider=provider,
                    usage_extra=extra_payload,
                ):
                    yield chunk
            elif provider in {"xai", "grok"}:
                async for chunk in self._stream_xai(
                    prompt,
                    model_id,
                    temperature,
                    adjusted_max_tokens,
                    system_prompt,
                    json_output=json_output,
                    json_schema=json_schema,
                    usage_callback=usage_callback,
                    usage_extra=extra_payload,
                ):
                    yield chunk
            elif provider == "openrouter":
                async for chunk in self._stream_openrouter(
                    prompt,
                    model_id,
                    temperature,
                    adjusted_max_tokens,
                    system_prompt,
                    json_output=json_output,
                    json_schema=json_schema,
                    usage_callback=usage_callback,
                    usage_extra=extra_payload,
                ):
                    yield chunk
            elif provider == "ollama":
                async for chunk in self._stream_ollama(
                    prompt,
                    model_id,
                    temperature,
                    adjusted_max_tokens,
                    system_prompt,
                    json_output=json_output,
                    json_schema=json_schema,
                    usage_callback=usage_callback,
                    usage_extra=extra_payload,
                ):
                    yield chunk
            else:
                raise ValueError(f"Unsupported provider: {provider}")

        max_attempts = self._max_retry_attempts()
        delay_seconds = self._retry_delay_seconds()
        last_exception: Optional[Exception] = None
        attempt = 1

        while attempt <= max_attempts:
            chunks_emitted = 0
            try:
                async for chunk in _dispatch_stream():
                    chunks_emitted += 1
                    yield chunk
                return
            except AIRequestError:
                raise
            except Exception as exc:
                last_exception = exc
                should_retry = (
                    attempt < max_attempts
                    and chunks_emitted == 0
                    and self._should_retry_exception(exc)
                )

                if not should_retry:
                    raise

                request_id = self._extract_request_id(exc)
                suffix = f" (request_id={request_id})" if request_id else ""
                logger.warning(
                    "Streaming failed for %s via %s on attempt %d/%d%s: %s",
                    model_id,
                    provider,
                    attempt,
                    max_attempts,
                    suffix,
                    exc,
                )
                await asyncio.sleep(delay_seconds)
                attempt += 1

        assert last_exception is not None
        raise AIRequestError(provider, model_id, max_attempts, max_attempts, last_exception) from last_exception
    
    async def _generate_claude(
        self,
        prompt: str,
        model_id: str,
        temperature: float,
        max_tokens: int,
        system_prompt: str,
        thinking_budget_tokens: Optional[int] = None,
        request_timeout: Optional[float] = None,
        json_output: bool = False,
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Any]:
        """Generate content using Claude API with optional Structured Outputs support

        Args:
            prompt: Generation prompt
            model_id: Claude model identifier
            temperature: Generation temperature
            max_tokens: Maximum tokens
            system_prompt: System prompt
            thinking_budget_tokens: Thinking budget tokens
            request_timeout: Request timeout
            json_output: Enable JSON output mode
            json_schema: Optional JSON schema for structured outputs (Claude Sonnet 4.5 / Opus 4.1 beta)
        """
        if not self.anthropic_client:
            raise ValueError("Claude client not initialized")

        thinking_enabled = thinking_budget_tokens is not None and thinking_budget_tokens > 0

        # Check if model supports Structured Outputs (Sonnet 4.5, Opus 4.1+, Haiku 4.5 - beta Nov 2025)
        # Note: Sonnet 4.0 (claude-sonnet-4-20250514) does NOT support structured outputs
        model_lower = model_id.lower()
        supports_structured_outputs = (
            "sonnet-4-5" in model_lower or "sonnet-4.5" in model_lower or  # Sonnet 4.5
            "opus-4-1" in model_lower or "opus-4.1" in model_lower or      # Opus 4.1
            "opus-4-5" in model_lower or "opus-4.5" in model_lower or      # Opus 4.5
            "haiku-4-5" in model_lower or "haiku-4.5" in model_lower       # Haiku 4.5
        )
        use_structured_outputs = json_output and json_schema and supports_structured_outputs

        if use_structured_outputs:
            # Use new Structured Outputs (beta) - no prefill, schema-guaranteed
            messages = [{"role": "user", "content": prompt}]
            logger.info(f"Using Claude Structured Outputs (beta) with JSON Schema for {model_id}")
        elif json_output:
            # Fallback to prompt engineering approach
            if thinking_enabled:
                messages = [
                    {
                        "role": "user",
                        "content": f"{prompt}\n\nCRITICAL REQUIREMENT: You MUST respond with valid JSON only. Start your response with '{{' immediately. When including dialogue or quotes in string values, use single quotes (') instead of double quotes (\") to avoid JSON parsing errors. Example: He said 'hello' instead of He said \"hello\". Do not include any text, explanation, or thinking output before or after the JSON object."
                    }
                ]
            else:
                messages = [
                    {
                        "role": "user",
                        "content": f"{prompt}\n\nIMPORTANT: Output valid JSON only. When including dialogue or quotes in string values, use single quotes (') instead of double quotes (\") to avoid JSON parsing errors. Example: He said 'hello' instead of He said \"hello\". Do not include any text before or after the JSON object."
                    },
                    {"role": "assistant", "content": "{"}
                ]
            logger.info(f"Using Claude JSON mode (prompt engineering) for {model_id}")
        else:
            messages = [{"role": "user", "content": prompt}]

        create_params = {
            "model": model_id,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages
        }

        # Only add system prompt if it exists and is not empty
        if system_prompt and system_prompt.strip():
            if json_output and not use_structured_outputs:
                create_params["system"] = f"{system_prompt}\n\nYou must respond with valid JSON only. When including dialogue or quotes in string values, use single quotes (') instead of double quotes (\"). No additional text or explanation."
            else:
                create_params["system"] = system_prompt

        # Add thinking mode for Claude models that support it (Claude 3.7, Claude 4 Sonnet, Claude 4 Opus)
        # Structured outputs + thinking is supported; Claude ignores thinking tokens for tool calls
        self._inject_claude_thinking_params(create_params, model_id, thinking_budget_tokens)

        # Add Structured Outputs configuration if using new beta feature
        if use_structured_outputs:
            # Beta API accepts dict directly when json_schema is already a dict
            request_kwargs: Dict[str, Any] = {}
            create_params["output_format"] = {
                "type": "json_schema",
                "schema": json_schema
            }
            # Add required betas parameter for structured outputs
            create_params["betas"] = ["structured-outputs-2025-11-13"]
        else:
            request_kwargs: Dict[str, Any] = {}

        if request_timeout and request_timeout > 0:
            request_kwargs["timeout"] = request_timeout

        # Use beta API when using structured outputs
        if use_structured_outputs:
            response = await self.anthropic_client.beta.messages.create(
                **create_params,
                **request_kwargs,
            )
        else:
            response = await self.anthropic_client.messages.create(
                **create_params,
                **request_kwargs,
            )

        # Handle thinking mode responses which may have different content structure
        content = self._extract_text_from_claude_response(response)
        return content, getattr(response, "usage", None)

    def _extract_text_from_claude_response(self, response) -> str:
        """
        Extract text content from Claude response, handling both regular and thinking mode responses.

        In thinking mode, response.content may contain:
        - ThinkingBlock objects (which we skip)
        - TextBlock objects (which contain the actual response text)
        """
        text_content = []

        try:
            for content_block in response.content:
                # Handle different block types properly
                block_type = getattr(content_block, 'type', None)

                if block_type == 'thinking':
                    # Skip thinking blocks - we only want the final answer
                    continue
                elif block_type == 'text' and hasattr(content_block, 'text'):
                    # Standard text block
                    text_content.append(content_block.text)
                elif hasattr(content_block, 'text'):
                    # Fallback for blocks with text attribute but no type
                    text_content.append(content_block.text)
                else:
                    # Try to get text from other possible structures
                    if hasattr(content_block, 'content'):
                        if isinstance(content_block.content, str):
                            text_content.append(content_block.content)

            # Join all text content
            result = "".join(text_content).strip()

            # Fallback: if no text found, try original approach
            if not result and response.content:
                try:
                    # Try to get text from the first content block that isn't thinking
                    for block in response.content:
                        if getattr(block, 'type', None) != 'thinking' and hasattr(block, 'text'):
                            result = block.text
                            break

                    # If still no result, try the old method
                    if not result:
                        result = response.content[0].text if hasattr(response.content[0], 'text') else str(response.content[0])
                except (AttributeError, IndexError):
                    result = ""

            return result or "No text content found in response"

        except Exception as e:
            logger.error(f"Error extracting text from Claude response: {e}")
            # Ultimate fallback
            return f"Error extracting response content: {str(e)}"

    async def _generate_gemini(
        self,
        prompt: str,
        model_id: str,
        temperature: float,
        max_tokens: int,
        system_prompt: str,
        json_output: bool = False,
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Any]:
        """Generate content using Gemini API with optional JSON Schema support"""
        if self.google_new_client:
            return await self._generate_gemini_new_sdk(
                prompt,
                model_id,
                temperature,
                max_tokens,
                system_prompt,
                json_output=json_output,
                json_schema=json_schema,
            )
        elif self.genai_client:
            return await self._generate_gemini_legacy_sdk(
                prompt,
                model_id,
                temperature,
                max_tokens,
                system_prompt,
                json_output=json_output,
                json_schema=json_schema,
            )
        else:
            raise ValueError("No Gemini client initialized")

    async def _generate_gemini_new_sdk(
        self,
        prompt: str,
        model_id: str,
        temperature: float,
        max_tokens: int,
        system_prompt: str,
        json_output: bool = False,
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Any]:
        """Generate content using new Google GenAI SDK with optional JSON Schema support

        Args:
            prompt: Generation prompt
            model_id: Gemini model identifier
            temperature: Generation temperature
            max_tokens: Maximum output tokens
            system_prompt: System prompt
            json_output: Enable JSON output mode
            json_schema: Optional JSON schema for structured outputs (requires json_output=True)
        """
        try:
            from google.genai import types

            # Build contents using correct SDK format
            contents = []

            final_prompt = prompt
            if json_output:
                final_prompt = f"{prompt}\n\nIMPORTANT: Output valid JSON only. When including dialogue or quotes in string values, use single quotes (') instead of double quotes (\") to avoid JSON parsing errors. Example: He said 'hello' instead of He said \"hello\"."

            if system_prompt:
                contents.append({
                    "role": "user",  # System messages handled as user in new SDK
                    "parts": [{"text": f"System: {system_prompt}\n\nUser: {final_prompt}"}]
                })
            else:
                contents.append({
                    "parts": [{"text": final_prompt}]
                })

            # Check if model supports thinking
            thinking_budget = self._get_thinking_budget_for_model(model_id)

            # Configure generation
            config_params = {
                "temperature": temperature,
                "max_output_tokens": max_tokens
            }

            # Configure JSON output format
            if json_output:
                if json_schema:
                    # Use JSON Schema for structured outputs (Gemini 2.5+, Nov 2025)
                    config_params["response_mime_type"] = "application/json"
                    config_params["response_schema"] = json_schema
                    logger.info(f"Using Gemini JSON Schema structured outputs for {model_id}")
                else:
                    # Use basic JSON mode (flexible structure)
                    config_params["response_mime_type"] = "application/json"
                    logger.info(f"Using Gemini JSON mode (flexible) for {model_id}")

            if thinking_budget and thinking_budget > 0:
                config_params["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)

            config = types.GenerateContentConfig(**config_params)

            response = await self.google_new_client.aio.models.generate_content(
                model=model_id,
                contents=contents,
                config=config
            )

            # Extract text from response
            if hasattr(response, 'text') and response.text:
                return response.text, getattr(response, 'usage_metadata', None)
            elif hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content and hasattr(candidate.content, 'parts'):
                    text_parts = []
                    for part in candidate.content.parts:
                        if hasattr(part, 'text') and part.text:
                            text_parts.append(part.text)
                    if text_parts:
                        return "".join(text_parts), getattr(response, 'usage_metadata', None)

            return (
                "Unable to generate content. The response may have been blocked by safety filters.",
                getattr(response, 'usage_metadata', None),
            )

        except Exception as e:
            logger.error(f"New Gemini SDK error: {e}")
            raise

    async def _generate_gemini_legacy_sdk(
        self,
        prompt: str,
        model_id: str,
        temperature: float,
        max_tokens: int,
        system_prompt: str,
        json_output: bool = False,
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Any]:
        """Generate content using legacy Google GenerativeAI SDK with optional JSON Schema support

        Args:
            prompt: Generation prompt
            model_id: Gemini model identifier
            temperature: Generation temperature
            max_tokens: Maximum output tokens
            system_prompt: System prompt
            json_output: Enable JSON output mode
            json_schema: Optional JSON schema for structured outputs (requires json_output=True)
        """
        system_instruction = system_prompt
        if json_output and system_prompt:
            system_instruction = f"{system_prompt}\n\nYou must respond with valid JSON only. When including dialogue or quotes in string values, use single quotes (') instead of double quotes (\"). No additional text or explanation."

        model = self.genai_client.GenerativeModel(
            model_name=model_id,
            system_instruction=system_instruction
        )

        final_prompt = prompt
        if json_output:
            final_prompt = f"{prompt}\n\nIMPORTANT: Output valid JSON only. When including dialogue or quotes in string values, use single quotes (') instead of double quotes (\") to avoid JSON parsing errors. Example: He said 'hello' instead of He said \"hello\"."

        config_kwargs = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }

        # Configure JSON output format
        if json_output:
            config_kwargs["response_mime_type"] = "application/json"
            if json_schema:
                # Use JSON Schema for structured outputs (Gemini 2.5+, Nov 2025)
                config_kwargs["response_schema"] = json_schema
                logger.info(f"Using Gemini legacy SDK JSON Schema structured outputs for {model_id}")
            else:
                logger.info(f"Using Gemini legacy SDK JSON mode (flexible) for {model_id}")

        try:
            generation_config = self.genai_client.GenerationConfig(**config_kwargs)
        except TypeError:
            # Older SDKs might not support response_mime_type or response_schema
            config_kwargs.pop("response_schema", None)
            config_kwargs.pop("response_mime_type", None)
            generation_config = self.genai_client.GenerationConfig(**config_kwargs)

        response = await model.generate_content_async(
            final_prompt,
            generation_config=generation_config
        )

        # Check if response was blocked by safety filters
        if not response.parts or not response.text:
            # Check safety ratings
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
                    safety_issues = []
                    for rating in candidate.safety_ratings:
                        if rating.probability.name in ['HIGH', 'MEDIUM']:
                            safety_issues.append(f"{rating.category.name}: {rating.probability.name}")
                    if safety_issues:
                        return (
                            f"Content blocked by safety filters: {'; '.join(safety_issues)}. Please try a different prompt.",
                            getattr(response, 'usage_metadata', None),
                        )

            return (
                "Unable to generate content. The response may have been blocked by safety filters or the model didn't return valid content. Please try rephrasing your request.",
                getattr(response, 'usage_metadata', None),
            )

        return response.text, getattr(response, 'usage_metadata', None)

    def _apply_temperature_policies(
        self,
        model_info: Dict[str, Any],
        requested_temperature: float,
        thinking_budget_tokens: Optional[int]
    ) -> Tuple[float, Optional[int], bool]:
        """Adjust temperature and thinking tokens based on model capabilities."""
        provider = (model_info.get("provider") or "").lower()
        capabilities = {cap.lower() for cap in model_info.get("capabilities", [])}

        effective_temperature = requested_temperature
        adjusted_thinking = thinking_budget_tokens
        forced_temperature = False

        # Claude models with thinking mode enabled REQUIRE temperature = 1.0
        if provider in {"claude", "anthropic"}:
            if adjusted_thinking and adjusted_thinking > 0:
                effective_temperature = 1.0
                forced_temperature = True

        # Reasoning-capable models should run at temperature 1 for consistency
        if "reasoning" in capabilities:
            if provider in {"claude", "anthropic", "xai"}:
                effective_temperature = 1.0
                forced_temperature = True

                # Anthropic reasoning models require thinking mode with temperature=1
                if provider in {"claude", "anthropic"}:
                    if not adjusted_thinking or adjusted_thinking <= 0:
                        default_budget = self._get_thinking_budget_for_model(model_info.get("model_id", ""))
                        if default_budget:
                            adjusted_thinking = default_budget
            elif provider == "openai":
                # OpenAI constraints handled in _build_openai_params
                pass

        return effective_temperature, adjusted_thinking, forced_temperature

    @staticmethod
    def _supports_claude_thinking(model_id: str) -> bool:
        """Return True if the Claude variant supports thinking mode."""
        model_lower = (model_id or "").lower()
        thinking_markers = (
            "claude-3.7",
            "claude-3-7",
            "claude-4",
            "claude-sonnet-4",
            "claude-opus-4",
            "claude-opus-4-1",  
            "claude-opus-4-1-20250805",
        )
        return any(marker in model_lower for marker in thinking_markers)

    def _get_thinking_budget_details(self, model_id: str) -> Optional[Dict[str, int]]:
        """Fetch full thinking budget configuration for a model if available."""
        try:
            model_specs = config.get_model_specs()
            for provider in model_specs.get("model_specifications", {}).values():
                for spec in provider.values():
                    if spec.get("model_id") == model_id:
                        thinking_budget = spec.get("thinking_budget", {})
                        if thinking_budget.get("supported", False):
                            return thinking_budget
            return None
        except Exception:
            return None

    def _inject_claude_thinking_params(
        self,
        params: Dict[str, Any],
        model_id: str,
        thinking_budget_tokens: Optional[int],
        log_context: str = ""
    ) -> None:
        """Attach thinking payload for Claude if supported and requested."""
        if not thinking_budget_tokens or thinking_budget_tokens <= 0:
            return

        if not self._supports_claude_thinking(model_id):
            return

        budget = int(thinking_budget_tokens)
        # Ensure minimum budget for thinking mode (1024 tokens minimum)
        budget = max(budget, 1024)

        details = self._get_thinking_budget_details(model_id)
        if details:
            min_tokens = details.get("min_tokens")
            max_tokens = details.get("max_tokens")
            if isinstance(min_tokens, int):
                budget = max(budget, min_tokens)
            if isinstance(max_tokens, int):
                budget = min(budget, max_tokens)

        # Ensure thinking budget doesn't exceed max_tokens
        if "max_tokens" in params:
            budget = min(budget, params["max_tokens"] - 100)  # Leave some room for response

        params["thinking"] = {
            "type": "enabled",
            "budget_tokens": budget
        }
        logger.info(f"{log_context}Using thinking mode with budget tokens: {budget} for model {model_id} ")

    def _get_thinking_budget_for_model(self, model_id: str) -> int:
        """Return default thinking budget tokens for a model."""
        details = self._get_thinking_budget_details(model_id)
        if not details:
            return 0
        return int(details.get("default_tokens", 0))
    
    async def _generate_xai(
        self,
        prompt: str,
        model_id: str,
        temperature: float,
        max_tokens: int,
        system_prompt: str,
        json_output: bool = False,
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Any]:
        """Generate content using xAI Grok API with optional structured outputs

        Args:
            prompt: Generation prompt
            model_id: xAI model identifier
            temperature: Generation temperature
            max_tokens: Maximum tokens
            system_prompt: System prompt
            json_output: Enable JSON output mode
            json_schema: Optional JSON schema for structured outputs (requires json_output=True)
        """
        if not self.xai_client:
            raise ValueError("xAI client not initialized")

        effective_prompt = prompt
        if json_output:
            effective_prompt = f"{prompt}\n\nIMPORTANT: Output valid JSON only. When including dialogue or quotes in string values, use single quotes (') instead of double quotes (\") to avoid JSON parsing errors. Example: He said 'hello' instead of He said \"hello\"."

        messages = [
            {"role": "user", "content": effective_prompt}
        ]

        if system_prompt:
            system_content = system_prompt
            if json_output:
                system_content = f"{system_prompt}\n\nYou must respond with valid JSON only. When including dialogue or quotes in string values, use single quotes (') instead of double quotes (\"). No additional text or explanation."
            messages.insert(0, {"role": "system", "content": system_content})

        request_params = {
            "model": model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Configure JSON output format
        if json_output:
            if json_schema:
                # Use structured outputs with JSON schema (Grok 2-1212+)
                request_params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "structured_output",
                        "strict": True,
                        "schema": json_schema
                    }
                }
                logger.info(f"Using Grok structured outputs with JSON schema for {model_id}")
            else:
                # Use basic JSON mode (flexible structure)
                request_params["response_format"] = {"type": "json_object"}
                logger.info(f"Using Grok JSON mode (flexible) for {model_id}")

        response = await self.xai_client.chat.completions.create(
            **request_params
        )

        return response.choices[0].message.content, getattr(response, "usage", None)

    async def _generate_openrouter(
        self,
        prompt: str,
        model_id: str,
        temperature: float,
        max_tokens: int,
        system_prompt: str,
        json_output: bool = False,
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Any]:
        """Generate content using OpenRouter unified API with optional JSON Schema support"""
        if not self.openrouter_client:
            raise ValueError("OpenRouter client not initialized")

        effective_prompt = prompt
        if json_output:
            effective_prompt = f"{prompt}\n\nIMPORTANT: Output valid JSON only. When including dialogue or quotes in string values, use single quotes (') instead of double quotes (\") to avoid JSON parsing errors. Example: He said 'hello' instead of He said \"hello\"."

        messages = [
            {"role": "user", "content": effective_prompt}
        ]

        if system_prompt:
            system_content = system_prompt
            if json_output:
                system_content = f"{system_prompt}\n\nYou must respond with valid JSON only. When including dialogue or quotes in string values, use single quotes (') instead of double quotes (\"). No additional text or explanation."
            messages.insert(0, {"role": "system", "content": system_content})

        request_params = {
            "model": model_id,  # Format: "provider/model-name" (e.g., "meta-llama/llama-3.3-70b-instruct")
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Configure JSON output format
        if json_output:
            if json_schema:
                # Use JSON Schema for structured outputs (Mistral, OpenAI models via OpenRouter)
                request_params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "structured_output",
                        "strict": True,
                        "schema": json_schema
                    }
                }
                logger.info(f"Using OpenRouter JSON Schema structured outputs for {model_id}")
            else:
                # Use basic JSON mode (flexible structure)
                request_params["response_format"] = {"type": "json_object"}
                logger.info(f"Using OpenRouter JSON mode (flexible) for {model_id}")

        response = await self.openrouter_client.chat.completions.create(
            **request_params
        )

        return response.choices[0].message.content, getattr(response, "usage", None)

    async def health_check(self) -> Dict[str, bool]:
        """Check health of all AI service providers"""
        health_status = {}
        
        # Test OpenAI
        try:
            if self.openai_client:
                await self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=5
                )
                health_status["openai"] = True
            else:
                health_status["openai"] = False
        except Exception as e:
            logger.error(f"OpenAI health check failed: {str(e)}")
            health_status["openai"] = False
        
        # Test Claude
        try:
            if self.anthropic_client:
                await self.anthropic_client.messages.create(
                    model="claude-3-5-haiku-20241022",
                    max_tokens=5,
                    messages=[{"role": "user", "content": "Hello"}]
                )
                health_status["claude"] = True
            else:
                health_status["claude"] = False
        except Exception as e:
            logger.error(f"Claude health check failed: {str(e)}")
            health_status["claude"] = False
        
        # Test Gemini
        try:
            if self.genai_client:
                model = self.genai_client.GenerativeModel('gemini-1.5-flash')
                await model.generate_content_async("Hello")
                health_status["gemini"] = True
            else:
                health_status["gemini"] = False
        except Exception as e:
            logger.error(f"Gemini health check failed: {str(e)}")
            health_status["gemini"] = False
        
        # Test xAI
        try:
            if self.xai_client:
                await self.xai_client.chat.completions.create(
                    model="grok-4-0709",
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=5
                )
                health_status["xai"] = True
            else:
                health_status["xai"] = False
        except Exception as e:
            logger.error(f"xAI health check failed: {str(e)}")
            health_status["xai"] = False
        
        return health_status
    
    async def _stream_openai(
        self,
        prompt: str,
        model_id: str,
        temperature: float,
        max_tokens: int,
        system_prompt: Optional[str],
        extra_verbose: bool,
        reasoning_effort: Optional[str] = None,
        thinking_budget_tokens: Optional[int] = None,
        request_timeout: Optional[float] = None,
        json_output: bool = False,
        json_schema: Optional[Dict[str, Any]] = None,
        usage_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        provider: str = "openai",
        resolved_model_id: Optional[str] = None,
        usage_extra: Optional[Dict[str, Any]] = None,
    ):
        """Stream OpenAI content generation with optional JSON Schema support"""
        if not self.openai_client:
            self._initialize_clients()

        extra_payload = usage_extra or {}

        effective_system_prompt = system_prompt if system_prompt is not None else config.GENERATOR_SYSTEM_PROMPT
        request_kwargs: Dict[str, Any] = {}
        if request_timeout and request_timeout > 0:
            request_kwargs["timeout"] = request_timeout

        # Add JSON instruction to prompt if needed
        effective_prompt = prompt
        if json_output:
            effective_prompt = f"{prompt}\n\nIMPORTANT: Output valid JSON only. When including dialogue or quotes in string values, use single quotes (') instead of double quotes (\") to avoid JSON parsing errors. Example: He said 'hello' instead of He said \"hello\"."

        messages = [
            {"role": "system", "content": effective_system_prompt},
            {"role": "user", "content": effective_prompt}
        ]
        
        # Log streaming start if extra_verbose is enabled
        if extra_verbose:
            logger.info(f"[EXTRA_VERBOSE] Starting streaming generation for model {model_id}")
            logger.info(f"[EXTRA_VERBOSE] Reasoning effort: {reasoning_effort}")
            logger.info(f"[EXTRA_VERBOSE] Max tokens: {max_tokens}")
        
        try:
            # Handle O3-pro models (no streaming support, fallback to regular generation)
            if "o3-pro" in model_id.lower():
                logger.info(f"O3-pro doesn't support streaming, using regular generation")
                content, usage_meta = await self._generate_openai(
                    prompt,
                    model_id,
                    temperature,
                    max_tokens,
                    effective_system_prompt,
                    reasoning_effort,
                    extra_verbose,
                    request_timeout=request_timeout,
                    json_output=json_output,
                    json_schema=json_schema,
                )
                yield content
                self._emit_usage(
                    usage_callback,
                    resolved_model_id or model_id,
                    provider,
                    usage_meta,
                    extra_payload,
                )
                return

            # Handle GPT-5 Pro models (no streaming support, use regular generation)
            elif "gpt-5-pro" in model_id.lower():
                logger.info(f"GPT-5 Pro doesn't support streaming, using regular generation")
                content, usage_meta = await self._generate_openai(
                    prompt,
                    model_id,
                    temperature,
                    max_tokens,
                    effective_system_prompt,
                    reasoning_effort,
                    extra_verbose,
                    request_timeout=request_timeout,
                    json_output=json_output,
                    json_schema=json_schema,
                )
                yield content
                self._emit_usage(
                    usage_callback,
                    resolved_model_id or model_id,
                    provider,
                    usage_meta,
                    extra_payload,
                )
                return

            # Handle O1/O3 models (excluding O3-pro)
            elif any(x in model_id.lower() for x in ["o1", "o3"]) and "o3-pro" not in model_id.lower():
                # O1/O3 models: no system message, no temperature
                messages = [{"role": "user", "content": f"{effective_system_prompt}\n\n{effective_prompt}"}]

                create_params = _build_openai_params(model_id, messages, temperature, max_tokens, reasoning_effort)
                create_params["stream"] = True  # Add streaming
                create_params.setdefault("stream_options", {})["include_usage"] = True

                # Configure JSON output format
                if json_output:
                    if json_schema:
                        # Use JSON Schema for structured outputs
                        create_params["response_format"] = {
                            "type": "json_schema",
                            "json_schema": {
                                "name": "structured_output",
                                "strict": True,
                                "schema": json_schema
                            }
                        }
                        logger.info(f"Using O1/O3 JSON Schema structured outputs (streaming) for {model_id}")
                    else:
                        # Use basic JSON mode (flexible structure)
                        create_params["response_format"] = {"type": "json_object"}
                        logger.info(f"Using O1/O3 JSON mode (streaming, flexible) for {model_id}")

                if extra_verbose:
                    logger.info(f"[EXTRA_VERBOSE] O1/O3 streaming parameters: {create_params}")

                stream = await self.openai_client.chat.completions.create(
                    **create_params,
                    **request_kwargs,
                )

            elif "gpt-5" in model_id.lower():
                # GPT-5 models: use max_completion_tokens, NO temperature parameter
                create_params = {
                    "model": model_id,
                    "messages": messages,
                    "max_completion_tokens": max_tokens,
                    "stream": True
                }

                # Add reasoning_effort if provided
                if reasoning_effort:
                    create_params["reasoning_effort"] = reasoning_effort
                    logger.info(f"Using reasoning effort: {reasoning_effort} for streaming model {model_id}")

                # Configure JSON output format
                if json_output:
                    if json_schema:
                        # Use JSON Schema for structured outputs
                        create_params["response_format"] = {
                            "type": "json_schema",
                            "json_schema": {
                                "name": "structured_output",
                                "strict": True,
                                "schema": json_schema
                            }
                        }
                        logger.info(f"Using GPT-5 JSON Schema structured outputs (streaming) for {model_id}")
                    else:
                        # Use basic JSON mode (flexible structure)
                        create_params["response_format"] = {"type": "json_object"}
                        logger.info(f"Using GPT-5 JSON mode (streaming, flexible) for {model_id}")

                create_params.setdefault("stream_options", {})["include_usage"] = True

                if extra_verbose:
                    logger.info(f"[EXTRA_VERBOSE] GPT-5 streaming parameters: {create_params}")

                try:
                    stream = await self.openai_client.chat.completions.create(
                        **create_params,
                        **request_kwargs,
                    )
                except Exception as stream_error:
                    logger.error(f"GPT-5 streaming failed to initialize: {str(stream_error)}")
                    logger.error(f"GPT-5 parameters that failed: {create_params}")
                    raise Exception(f"GPT-5 streaming initialization failed: {str(stream_error)}")
            else:
                # Standard GPT models
                if extra_verbose:
                    logger.info(f"[EXTRA_VERBOSE] Standard OpenAI streaming with temperature: {temperature}")

                create_params = {
                    "model": model_id,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": True,
                }

                create_params.setdefault("stream_options", {})["include_usage"] = True

                # Configure JSON output format
                if json_output:
                    if json_schema:
                        # Use JSON Schema for structured outputs (GPT-4o-2024-08-06+)
                        create_params["response_format"] = {
                            "type": "json_schema",
                            "json_schema": {
                                "name": "structured_output",
                                "strict": True,
                                "schema": json_schema
                            }
                        }
                        logger.info(f"Using OpenAI JSON Schema structured outputs (streaming) for {model_id}")
                    else:
                        # Use basic JSON mode (flexible structure)
                        create_params["response_format"] = {"type": "json_object"}
                        logger.info(f"Using OpenAI JSON mode (streaming, flexible) for {model_id}")

                stream = await self.openai_client.chat.completions.create(
                    **create_params,
                    **request_kwargs,
                )

            chunk_count = 0
            total_content = ""
            usage_obj = None
            async for chunk in stream:
                # Verify that choices exists and has elements before accessing
                if chunk.choices and len(chunk.choices) > 0:
                    if chunk.choices[0].delta.content is not None:
                        chunk_count += 1
                        content_piece = chunk.choices[0].delta.content
                        total_content += content_piece
                        yield content_piece
                # Capture usage even if choices is empty
                if hasattr(chunk, "usage") and chunk.usage:
                    usage_obj = chunk.usage

            # CRITICAL: Validate that we received content
            if chunk_count == 0 or not total_content.strip():
                error_msg = f"Streaming completed but no content received from {model_id} (chunks: {chunk_count}, content length: {len(total_content)})"
                logger.error(error_msg)
                logger.error(f"This might indicate a model configuration issue or API problem")
                raise Exception(error_msg)

            if extra_verbose:
                logger.info(f"[EXTRA_VERBOSE] Streaming completed for {model_id} - received {chunk_count} chunks, {len(total_content)} characters")

            if hasattr(stream, "response"):
                usage_obj = getattr(stream.response, "usage", usage_obj)
            elif hasattr(stream, "usage") and stream.usage:
                usage_obj = stream.usage

            self._emit_usage(
                usage_callback,
                resolved_model_id or model_id,
                provider,
                usage_obj,
                extra_payload,
            )

        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            raise
    

    async def _stream_claude(
        self,
        prompt: str,
        model_id: str,
        temperature: float,
        max_tokens: int,
        system_prompt: Optional[str],
        extra_verbose: bool,
        thinking_budget_tokens: Optional[int] = None,
        json_output: bool = False,
        json_schema: Optional[Dict[str, Any]] = None,
        usage_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        provider: str = "claude",
        resolved_model_id: Optional[str] = None,
        usage_extra: Optional[Dict[str, Any]] = None,
    ):
        """Stream Claude content generation with optional Structured Outputs support"""
        if not self.anthropic_client:
            self._initialize_clients()

        extra_payload = usage_extra or {}

        # Check if thinking mode is enabled
        thinking_enabled = thinking_budget_tokens is not None and thinking_budget_tokens > 0

        # Check if model supports Structured Outputs (Sonnet 4.5, Opus 4.1+, Haiku 4.5 - beta Nov 2025)
        # Note: Sonnet 4.0 (claude-sonnet-4-20250514) does NOT support structured outputs
        model_lower = model_id.lower()
        supports_structured_outputs = (
            "sonnet-4-5" in model_lower or "sonnet-4.5" in model_lower or  # Sonnet 4.5
            "opus-4-1" in model_lower or "opus-4.1" in model_lower or      # Opus 4.1
            "opus-4-5" in model_lower or "opus-4.5" in model_lower or      # Opus 4.5
            "haiku-4-5" in model_lower or "haiku-4.5" in model_lower       # Haiku 4.5
        )
        use_structured_outputs = json_output and json_schema and supports_structured_outputs

        if use_structured_outputs:
            # Use new Structured Outputs (beta) - no prefill, schema-guaranteed
            messages = [{"role": "user", "content": prompt}]
            logger.info(f"Using Claude Structured Outputs (beta, streaming) with JSON Schema for {model_id}")
        elif json_output:
            # Fallback to prompt engineering approach
            if thinking_enabled:
                # Cannot prefill when thinking is enabled - use strong prompt instructions instead
                messages = [
                    {"role": "user", "content": f"{prompt}\n\nCRITICAL REQUIREMENT: You MUST respond with valid JSON only. Start your response with '{{' immediately. When including dialogue or quotes in string values, use single quotes (') instead of double quotes (\") to avoid JSON parsing errors. Example: He said 'hello' instead of He said \"hello\". Do not include any text, explanation, or thinking output before or after the JSON object."}
                ]
            else:
                # Standard prefill approach when thinking is not enabled
                messages = [
                    {"role": "user", "content": f"{prompt}\n\nIMPORTANT: Output valid JSON only. When including dialogue or quotes in string values, use single quotes (') instead of double quotes (\") to avoid JSON parsing errors. Example: He said 'hello' instead of He said \"hello\". Do not include any text before or after the JSON object."},
                    {"role": "assistant", "content": "{"}  # Prefill to force JSON start
                ]
            logger.info(f"Using Claude JSON mode (streaming, prompt engineering) for {model_id}")
        else:
            messages = [{"role": "user", "content": prompt}]

        try:
            stream_params = {
                "max_tokens": max_tokens,
                "model": model_id,
                "temperature": temperature,
                "messages": messages
            }

            # Only add system prompt if it exists and is not empty
            if system_prompt and system_prompt.strip():
                if json_output and not use_structured_outputs:
                    # Add JSON instruction to system prompt (only for prompt engineering approach)
                    stream_params["system"] = f"{system_prompt}\n\nYou must respond with valid JSON only. When including dialogue or quotes in string values, use single quotes (') instead of double quotes (\") to avoid JSON parsing errors. No additional text or explanation."
                else:
                    stream_params["system"] = system_prompt

            # Add thinking mode for Claude models that support it (structured outputs compatible)
            self._inject_claude_thinking_params(stream_params, model_id, thinking_budget_tokens, log_context="Streaming: ")

            # Add Structured Outputs configuration if using new beta feature
            if use_structured_outputs:
                # Claude streaming requires Pydantic model, not dict
                # Convert JSON schema to Pydantic model dynamically
                pydantic_model = json_schema_to_pydantic(json_schema)
                stream_params["output_format"] = pydantic_model
                # Add required betas parameter for structured outputs
                stream_params["betas"] = ["structured-outputs-2025-11-13"]

            # If using JSON prefill (only when thinking is disabled AND not using structured outputs), yield the opening brace first
            first_chunk = True
            use_json_prefill = json_output and not thinking_enabled and not use_structured_outputs

            # Use beta API when using structured outputs, regular API otherwise
            stream_context = (
                self.anthropic_client.beta.messages.stream(**stream_params)
                if use_structured_outputs
                else self.anthropic_client.messages.stream(**stream_params)
            )

            async with stream_context as stream:
                final_usage = None
                input_tokens = 0
                output_tokens = 0

                async for event in stream:
                    # Capture usage from message_start event (contains input_tokens)
                    if event.type == "message_start":
                        if hasattr(event, 'message') and hasattr(event.message, 'usage'):
                            usage_obj = event.message.usage
                            input_tokens = getattr(usage_obj, 'input_tokens', 0)

                    # Capture usage from message_delta event (contains cumulative output_tokens)
                    elif event.type == "message_delta":
                        if hasattr(event, 'usage'):
                            usage_obj = event.usage
                            output_tokens = getattr(usage_obj, 'output_tokens', 0)

                    # Handle content streaming
                    elif event.type == "content_block_delta":
                        # Handle both regular text deltas and thinking deltas
                        delta_type = getattr(event.delta, 'type', None)

                        if delta_type == 'thinking_delta':
                            # ThinkingDelta - stream for live monitoring, marked as thinking
                            # so consumers can filter it from final accumulated content.
                            # TODO: In future, consider adding visual tags like [THINKING]...[/THINKING]
                            # for frontend differentiation and styling.
                            thinking_text = getattr(event.delta, 'thinking', '')
                            if thinking_text:
                                yield StreamChunk(thinking_text, is_thinking=True)
                        elif hasattr(event.delta, 'text') and event.delta.text:
                            # For JSON mode with prefill (thinking disabled), include the opening brace
                            if use_json_prefill and first_chunk:
                                yield StreamChunk("{", is_thinking=False)
                                first_chunk = False
                            # Regular text delta
                            yield StreamChunk(event.delta.text, is_thinking=False)
                        elif delta_type == 'text_delta' and hasattr(event.delta, 'text'):
                            # For JSON mode with prefill (thinking disabled), include the opening brace
                            if use_json_prefill and first_chunk:
                                yield StreamChunk("{", is_thinking=False)
                                first_chunk = False
                            # Explicit text delta type
                            yield StreamChunk(event.delta.text, is_thinking=False)

                # Try to get final response, but fallback to captured usage if it fails
                try:
                    final_response = await stream.get_final_response()
                    final_usage = getattr(final_response, "usage", None)
                except Exception:
                    final_usage = None

                # If get_final_response failed, construct usage from captured events
                if not final_usage and (input_tokens > 0 or output_tokens > 0):
                    # Create a simple object with the captured usage
                    class UsageData:
                        def __init__(self, input_tokens, output_tokens):
                            self.input_tokens = input_tokens
                            self.output_tokens = output_tokens
                            self.total_tokens = input_tokens + output_tokens

                    final_usage = UsageData(input_tokens, output_tokens)

                self._emit_usage(
                    usage_callback,
                    resolved_model_id or model_id,
                    provider,
                    final_usage,
                    extra_payload,
                )
        except Exception as e:
            logger.error(f"Claude streaming error: {e}")
            raise

    
    async def _stream_gemini(
        self,
        prompt: str,
        model_id: str,
        temperature: float,
        max_tokens: int,
        system_prompt: Optional[str],
        json_output: bool = False,
        json_schema: Optional[Dict[str, Any]] = None,
        usage_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        provider: str = "gemini",
        usage_extra: Optional[Dict[str, Any]] = None,
    ):
        """Stream Gemini content generation with optional JSON Schema support"""
        extra_payload = usage_extra or {}
        if self.google_new_client:
            async for chunk in self._stream_gemini_new_sdk(
                prompt,
                model_id,
                temperature,
                max_tokens,
                system_prompt,
                json_output,
                json_schema,
                usage_callback,
                provider,
                extra_payload,
            ):
                yield chunk
        elif self.genai_client:
            async for chunk in self._stream_gemini_legacy_sdk(
                prompt,
                model_id,
                temperature,
                max_tokens,
                system_prompt,
                json_output,
                json_schema,
                usage_callback,
                provider,
                extra_payload,
            ):
                yield chunk
        else:
            raise ValueError("No Gemini client initialized")

    async def _stream_gemini_new_sdk(
        self,
        prompt: str,
        model_id: str,
        temperature: float,
        max_tokens: int,
        system_prompt: Optional[str],
        json_output: bool = False,
        json_schema: Optional[Dict[str, Any]] = None,
        usage_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        provider: str = "gemini",
        usage_extra: Optional[Dict[str, Any]] = None,
    ):
        """Stream content using new Google GenAI SDK with optional JSON Schema support"""
        extra_payload = usage_extra or {}
        try:
            from google.genai import types

            # Build contents using correct SDK format
            contents = []

            # Add JSON instruction to prompt if needed
            final_prompt = prompt
            if json_output:
                final_prompt = f"{prompt}\n\nIMPORTANT: Output valid JSON only. When including dialogue or quotes in string values, use single quotes (') instead of double quotes (\") to avoid JSON parsing errors. Example: He said 'hello' instead of He said \"hello\"."

            if system_prompt:
                contents.append({
                    "role": "user",  # System messages handled as user in new SDK
                    "parts": [{"text": f"System: {system_prompt}\n\nUser: {final_prompt}"}]
                })
            else:
                contents.append({
                    "parts": [{"text": final_prompt}]
                })

            # Check if model supports thinking
            thinking_budget = self._get_thinking_budget_for_model(model_id)

            # Configure generation
            config_params = {
                "temperature": temperature,
                "max_output_tokens": max_tokens
            }

            # Configure JSON output format
            if json_output:
                if json_schema:
                    # Use JSON Schema for structured outputs (Gemini 2.5+, Nov 2025)
                    config_params["response_mime_type"] = "application/json"
                    config_params["response_schema"] = json_schema
                    logger.info(f"Using Gemini JSON Schema structured outputs (streaming) for {model_id}")
                else:
                    # Use basic JSON mode (flexible structure)
                    config_params["response_mime_type"] = "application/json"
                    logger.info(f"Using Gemini JSON mode (streaming, flexible) for {model_id}")

            if thinking_budget and thinking_budget > 0:
                config_params["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)

            config = types.GenerateContentConfig(**config_params)

            stream_response = await self.google_new_client.aio.models.generate_content_stream(
                model=model_id,
                contents=contents,
                config=config
            )

            usage_metadata = None
            # Check if stream_response is None or not iterable
            if stream_response is None:
                error_msg = f"New Gemini SDK streaming error: stream_response is None for model {model_id}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            yielded_chunks = False
            fallback_needed = False
            fallback_reason: Optional[BaseException] = None

            try:
                async for chunk in stream_response:
                    # Extract text from streaming chunks
                    if hasattr(chunk, 'text') and chunk.text:
                        yielded_chunks = True
                        yield chunk.text
                    elif hasattr(chunk, 'candidates') and chunk.candidates:
                        candidate = chunk.candidates[0]
                        if hasattr(candidate, 'content') and candidate.content and hasattr(candidate.content, 'parts'):
                            try:
                                for part in candidate.content.parts:
                                    if hasattr(part, 'text') and part.text:
                                        yielded_chunks = True
                                        yield part.text
                            except TypeError as type_err:
                                fallback_needed = not yielded_chunks
                                fallback_reason = type_err
                                logger.error(f"New Gemini SDK streaming error while iterating parts: {type_err}")
                                break

                    if hasattr(chunk, 'usage_metadata') and chunk.usage_metadata:
                        usage_metadata = chunk.usage_metadata
            except TypeError as e:
                # Handle 'NoneType' object is not iterable and similar issues
                fallback_needed = not yielded_chunks
                fallback_reason = e
                logger.error(f"New Gemini SDK streaming error: {e}")
            except Exception as e:
                fallback_needed = not yielded_chunks
                fallback_reason = e
                logger.error(f"New Gemini SDK streaming error: {e}")
            finally:
                if stream_response and hasattr(stream_response, "aclose"):
                    try:
                        await stream_response.aclose()
                    except Exception as close_error:
                        logger.warning(f"Error closing Gemini stream response: {close_error}")

            if fallback_needed:
                try:
                    fallback_text, fallback_usage = await self._generate_gemini_new_sdk(
                        prompt,
                        model_id,
                        temperature,
                        max_tokens,
                        system_prompt,
                        json_output=json_output,
                        json_schema=json_schema,
                    )
                    if fallback_text:
                        yield fallback_text
                    usage_metadata = fallback_usage
                    logger.info("Gemini stream fallback succeeded after streaming error")
                except Exception as fallback_error:
                    logger.error(f"Gemini fallback generation failed after streaming error ({fallback_reason}): {fallback_error}")
                    raise

            self._emit_usage(
                usage_callback,
                model_id,
                provider,
                usage_metadata,
                extra_payload,
            )

        except Exception as e:
            logger.error(f"New Gemini SDK streaming error: {e}")
            raise


    async def _stream_gemini_legacy_sdk(
        self,
        prompt: str,
        model_id: str,
        temperature: float,
        max_tokens: int,
        system_prompt: Optional[str],
        json_output: bool = False,
        json_schema: Optional[Dict[str, Any]] = None,
        usage_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        provider: str = "gemini",
        usage_extra: Optional[Dict[str, Any]] = None,
    ):
        """Stream content using legacy Google GenerativeAI SDK with optional JSON Schema support"""
        extra_payload = usage_extra or {}
        try:
            # Add JSON instruction if needed
            final_prompt = prompt
            if json_output:
                final_prompt = f"{prompt}\n\nIMPORTANT: Output valid JSON only. When including dialogue or quotes in string values, use single quotes (') instead of double quotes (\") to avoid JSON parsing errors. Example: He said 'hello' instead of He said \"hello\"."

            full_prompt = f"{system_prompt}\n\n{final_prompt}" if system_prompt else final_prompt

            # Configure model
            config_params = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }

            # Configure JSON output format
            if json_output:
                config_params["response_mime_type"] = "application/json"
                if json_schema:
                    # Use JSON Schema for structured outputs (Gemini 2.5+, Nov 2025)
                    config_params["response_schema"] = json_schema
                    logger.info(f"Using Gemini legacy SDK JSON Schema structured outputs (streaming) for {model_id}")
                else:
                    logger.info(f"Using Gemini legacy SDK JSON mode (streaming, flexible) for {model_id}")

            generation_config = self.genai_client.types.GenerationConfig(**config_params)

            model = self.genai_client.GenerativeModel(model_id, generation_config=generation_config)
            response = await model.generate_content_async(full_prompt, stream=True)

            usage_metadata = None
            async for chunk in response:
                if chunk.text:
                    yield chunk.text
                if hasattr(chunk, 'usage_metadata') and chunk.usage_metadata:
                    usage_metadata = chunk.usage_metadata
        except Exception as e:
            logger.error(f"Legacy Gemini SDK streaming error: {e}")
            raise
            # Fallback to regular generation
            content, usage_meta = await self._generate_gemini_legacy_sdk(
                prompt,
                model_id,
                temperature,
                max_tokens,
                system_prompt,
                json_output=json_output,
                json_schema=json_schema,
            )
            yield content
            usage_metadata = usage_meta

        self._emit_usage(
            usage_callback,
            model_id,
            provider,
            usage_metadata,
            extra_payload,
        )
    
    async def _stream_xai(
        self,
        prompt: str,
        model_id: str,
        temperature: float,
        max_tokens: int,
        system_prompt: Optional[str],
        json_output: bool = False,
        json_schema: Optional[Dict[str, Any]] = None,
        usage_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        usage_extra: Optional[Dict[str, Any]] = None,
    ):
        """Stream xAI content generation with optional structured outputs

        Args:
            prompt: Generation prompt
            model_id: xAI model identifier
            temperature: Generation temperature
            max_tokens: Maximum tokens
            system_prompt: System prompt
            json_output: Enable JSON output mode
            json_schema: Optional JSON schema for structured outputs (requires json_output=True)
            usage_callback: Callback for usage tracking
            usage_extra: Extra usage tracking data
        """
        if not self.xai_client:
            self._initialize_clients()

        extra_payload = usage_extra or {}
        # Add JSON instruction to prompt if needed
        effective_prompt = prompt
        if json_output:
            effective_prompt = f"{prompt}\n\nIMPORTANT: Output valid JSON only. When including dialogue or quotes in string values, use single quotes (') instead of double quotes (\") to avoid JSON parsing errors. Example: He said 'hello' instead of He said \"hello\"."

        messages = [{"role": "user", "content": effective_prompt}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        try:
            create_params = {
                "model": model_id,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True
            }

            # Include usage stats in streaming response (OpenAI-compatible API)
            create_params.setdefault("stream_options", {})["include_usage"] = True

            # Configure JSON output format
            if json_output:
                if json_schema:
                    # Use structured outputs with JSON schema (Grok 2-1212+)
                    create_params["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "structured_output",
                            "strict": True,
                            "schema": json_schema
                        }
                    }
                    logger.info(f"Using Grok structured outputs (streaming) with JSON schema for {model_id}")
                else:
                    # Use basic JSON mode (flexible structure)
                    create_params["response_format"] = {"type": "json_object"}
                    logger.info(f"Using Grok JSON mode (streaming, flexible) for {model_id}")

            stream = await self.xai_client.chat.completions.create(**create_params)

            usage_obj = None
            async for chunk in stream:
                # Verify that choices exists and has elements before accessing
                if chunk.choices and len(chunk.choices) > 0:
                    if chunk.choices[0].delta.content is not None:
                        yield chunk.choices[0].delta.content
                # Capture usage even if choices is empty
                if hasattr(chunk, "usage") and chunk.usage:
                    usage_obj = chunk.usage

            if hasattr(stream, "response"):
                usage_obj = getattr(stream.response, "usage", usage_obj)
            elif hasattr(stream, "usage") and stream.usage:
                usage_obj = stream.usage

            self._emit_usage(
                usage_callback,
                model_id,
                "xai",
                usage_obj,
                extra_payload,
            )
        except Exception as e:
            logger.error(f"xAI streaming error: {e}")
            raise

    async def _stream_openrouter(
        self,
        prompt: str,
        model_id: str,
        temperature: float,
        max_tokens: int,
        system_prompt: Optional[str],
        json_output: bool = False,
        json_schema: Optional[Dict[str, Any]] = None,
        usage_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        usage_extra: Optional[Dict[str, Any]] = None,
    ):
        """Stream OpenRouter content generation with optional JSON Schema support"""
        if not self.openrouter_client:
            self._initialize_clients()

        extra_payload = usage_extra or {}
        # Add JSON instruction to prompt if needed
        effective_prompt = prompt
        if json_output:
            effective_prompt = f"{prompt}\n\nIMPORTANT: Output valid JSON only. When including dialogue or quotes in string values, use single quotes (') instead of double quotes (\") to avoid JSON parsing errors. Example: He said 'hello' instead of He said \"hello\"."

        messages = [{"role": "user", "content": effective_prompt}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        try:
            create_params = {
                "model": model_id,  # Format: "provider/model-name" (e.g., "meta-llama/llama-3.3-70b-instruct")
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True
            }

            # Include usage stats in streaming response (OpenAI-compatible API)
            create_params.setdefault("stream_options", {})["include_usage"] = True

            # Configure JSON output format (OpenRouter is OpenAI-compatible)
            if json_output:
                if json_schema:
                    # Use JSON Schema for structured outputs (Mistral, OpenAI models via OpenRouter)
                    create_params["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "structured_output",
                            "strict": True,
                            "schema": json_schema
                        }
                    }
                    logger.info(f"Using OpenRouter JSON Schema structured outputs (streaming) for {model_id}")
                else:
                    # Use basic JSON mode (flexible structure)
                    create_params["response_format"] = {"type": "json_object"}
                    logger.info(f"Using OpenRouter JSON mode (streaming, flexible) for {model_id}")

            stream = await self.openrouter_client.chat.completions.create(**create_params)

            usage_obj = None
            async for chunk in stream:
                # Verify that choices exists and has elements before accessing
                if chunk.choices and len(chunk.choices) > 0:
                    if chunk.choices[0].delta.content is not None:
                        yield chunk.choices[0].delta.content
                # Capture usage even if choices is empty
                if hasattr(chunk, "usage") and chunk.usage:
                    usage_obj = chunk.usage

            if hasattr(stream, "response"):
                usage_obj = getattr(stream.response, "usage", usage_obj)
            elif hasattr(stream, "usage") and stream.usage:
                usage_obj = stream.usage

            self._emit_usage(
                usage_callback,
                model_id,
                "openrouter",
                usage_obj,
                extra_payload,
            )
        except Exception as e:
            logger.error(f"OpenRouter streaming error: {e}")
            raise

    async def _generate_ollama(
        self,
        prompt: str,
        model_id: str,
        temperature: float,
        max_tokens: int,
        system_prompt: str,
        json_output: bool = False,
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Any]:
        """Generate content using Ollama local models (OpenAI-compatible API)

        Args:
            prompt: Generation prompt
            model_id: Ollama model identifier (e.g., 'qwen2.5:14b', 'deepseek-r1:8b')
            temperature: Generation temperature
            max_tokens: Maximum tokens
            system_prompt: System prompt
            json_output: Enable JSON output mode
            json_schema: Optional JSON schema (Ollama supports basic JSON mode only)
        """
        if not self.ollama_client:
            raise ValueError("Ollama client not initialized. Set OLLAMA_HOST in your environment.")

        effective_prompt = prompt
        if json_output:
            effective_prompt = f"{prompt}\n\nIMPORTANT: Output valid JSON only. When including dialogue or quotes in string values, use single quotes (') instead of double quotes (\") to avoid JSON parsing errors. Example: He said 'hello' instead of He said \"hello\"."

        messages = [
            {"role": "user", "content": effective_prompt}
        ]

        if system_prompt:
            system_content = system_prompt
            if json_output:
                system_content = f"{system_prompt}\n\nYou must respond with valid JSON only. When including dialogue or quotes in string values, use single quotes (') instead of double quotes (\"). No additional text or explanation."
            messages.insert(0, {"role": "system", "content": system_content})

        request_params = {
            "model": model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Configure JSON output format (Ollama supports basic JSON mode)
        if json_output:
            # Ollama supports json_object mode but not JSON Schema structured outputs
            request_params["response_format"] = {"type": "json_object"}
            if json_schema:
                logger.info(f"Ollama does not support JSON Schema structured outputs; using basic JSON mode for {model_id}")
            else:
                logger.info(f"Using Ollama JSON mode for {model_id}")

        response = await self.ollama_client.chat.completions.create(
            **request_params
        )

        return response.choices[0].message.content, getattr(response, "usage", None)

    async def _stream_ollama(
        self,
        prompt: str,
        model_id: str,
        temperature: float,
        max_tokens: int,
        system_prompt: Optional[str],
        json_output: bool = False,
        json_schema: Optional[Dict[str, Any]] = None,
        usage_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        usage_extra: Optional[Dict[str, Any]] = None,
    ):
        """Stream Ollama content generation (OpenAI-compatible API)

        Args:
            prompt: Generation prompt
            model_id: Ollama model identifier
            temperature: Generation temperature
            max_tokens: Maximum tokens
            system_prompt: System prompt
            json_output: Enable JSON output mode
            json_schema: Optional JSON schema (Ollama supports basic JSON mode only)
            usage_callback: Callback for usage tracking
            usage_extra: Extra usage tracking data
        """
        if not self.ollama_client:
            self._initialize_clients()

        extra_payload = usage_extra or {}
        # Add JSON instruction to prompt if needed
        effective_prompt = prompt
        if json_output:
            effective_prompt = f"{prompt}\n\nIMPORTANT: Output valid JSON only. When including dialogue or quotes in string values, use single quotes (') instead of double quotes (\") to avoid JSON parsing errors. Example: He said 'hello' instead of He said \"hello\"."

        messages = [{"role": "user", "content": effective_prompt}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        try:
            create_params = {
                "model": model_id,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True
            }

            # Include usage stats in streaming response (OpenAI-compatible API)
            create_params.setdefault("stream_options", {})["include_usage"] = True

            # Configure JSON output format (Ollama supports basic JSON mode)
            if json_output:
                create_params["response_format"] = {"type": "json_object"}
                if json_schema:
                    logger.info(f"Ollama does not support JSON Schema; using basic JSON mode (streaming) for {model_id}")
                else:
                    logger.info(f"Using Ollama JSON mode (streaming) for {model_id}")

            stream = await self.ollama_client.chat.completions.create(**create_params)

            usage_obj = None
            async for chunk in stream:
                # Verify that choices exists and has elements before accessing
                if chunk.choices and len(chunk.choices) > 0:
                    if chunk.choices[0].delta.content is not None:
                        yield chunk.choices[0].delta.content
                # Capture usage even if choices is empty
                if hasattr(chunk, "usage") and chunk.usage:
                    usage_obj = chunk.usage

            if hasattr(stream, "response"):
                usage_obj = getattr(stream.response, "usage", usage_obj)
            elif hasattr(stream, "usage") and stream.usage:
                usage_obj = stream.usage

            self._emit_usage(
                usage_callback,
                model_id,
                "ollama",
                usage_obj,
                extra_payload,
            )
        except Exception as e:
            logger.error(f"Ollama streaming error: {e}")
            raise

    async def get_embeddings(self, texts: List[str], model: str = None) -> List[List[float]]:
        """
        Get embeddings for a list of texts

        Args:
            texts: List of texts to embed
            model: Optional model to use (defaults to text-embedding-3-large for OpenAI)

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        # Default to OpenAI embeddings if available
        if self.openai_client and config.OPENAI_API_KEY:
            return await self._get_openai_embeddings(texts, model)

        # Fallback to Google if available (Gemini supports embeddings)
        if self.genai_client and config.GOOGLE_API_KEY:
            return await self._get_google_embeddings(texts, model)

        logger.warning("No embedding provider available")
        return [[] for _ in texts]  # Return empty embeddings

    async def _get_openai_embeddings(self, texts: List[str], model: str = None) -> List[List[float]]:
        """Get embeddings using OpenAI API"""
        if not model:
            model = "text-embedding-3-large"  # Best quality embeddings

        try:
            response = await self.openai_client.embeddings.create(
                model=model,
                input=texts
            )

            return [item.embedding for item in response.data]

        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            # Try with smaller model as fallback
            if model == "text-embedding-3-large":
                try:
                    response = await self.openai_client.embeddings.create(
                        model="text-embedding-3-small",
                        input=texts
                    )
                    return [item.embedding for item in response.data]
                except Exception as e2:
                    logger.error(f"OpenAI embedding fallback failed: {e2}")

            return [[] for _ in texts]

    async def _get_google_embeddings(self, texts: List[str], model: str = None) -> List[List[float]]:
        """Get embeddings using Google/Gemini API"""
        if not model:
            model = "models/text-embedding-004"  # Latest Gemini embedding model

        try:
            embeddings = []

            for text in texts:
                # Google embeddings API requires individual requests
                result = self.genai_client.embed_content(
                    model=model,
                    content=text,
                    task_type="retrieval_document"
                )
                embeddings.append(result['embedding'])

            return embeddings

        except Exception as e:
            logger.error(f"Google embedding error: {e}")
            return [[] for _ in texts]

    async def calculate_similarity(self, text1: str, text2: str, model: str = None) -> float:
        """
        Calculate cosine similarity between two texts using embeddings

        Args:
            text1: First text
            text2: Second text
            model: Optional embedding model to use

        Returns:
            Cosine similarity score (0 to 1)
        """
        embeddings = await self.get_embeddings([text1, text2], model)

        if len(embeddings) != 2 or not embeddings[0] or not embeddings[1]:
            return 0.0

        vec1, vec2 = embeddings[0], embeddings[1]

        # Calculate cosine similarity
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(x * x for x in vec1) ** 0.5
        norm2 = sum(x * x for x in vec2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    async def _make_request(self, method: str, url: str, **kwargs) -> Optional[Dict]:
        """
        Make HTTP request to API endpoints (for custom operations)

        Args:
            method: HTTP method (GET, POST, etc.)
            url: API endpoint URL
            **kwargs: Additional request parameters

        Returns:
            JSON response or None on error
        """
        try:
            if not hasattr(self, 'http_session'):
                import aiohttp
                self.http_session = aiohttp.ClientSession(connector=self.http_connector)

            # Add default headers for OpenAI-style APIs
            if 'api.openai.com' in url and config.OPENAI_API_KEY:
                if 'headers' not in kwargs:
                    kwargs['headers'] = {}
                kwargs['headers']['Authorization'] = f"Bearer {config.OPENAI_API_KEY}"

            async with self.http_session.request(method, url, **kwargs) as response:
                if response.status >= 400:
                    error_text = await response.text()
                    logger.error(f"API request failed: {response.status} - {error_text}")
                    return None

                return await response.json()

        except Exception as e:
            logger.error(f"HTTP request failed: {e}")
            return None

    async def close(self):
        """Close HTTP connector and cleanup resources"""
        try:
            # Close HTTP session if exists
            if hasattr(self, 'http_session') and self.http_session:
                await self.http_session.close()
                logger.info("HTTP session closed successfully")

            # Close HTTP connector
            if hasattr(self, 'http_connector') and self.http_connector:
                await self.http_connector.close()
                logger.info("HTTP connector closed successfully")

            # Close Google new SDK client (async then sync) if available
            if hasattr(self, 'google_new_client') and self.google_new_client:
                async_client = getattr(self.google_new_client, "aio", None)
                if async_client and hasattr(async_client, "aclose"):
                    try:
                        await async_client.aclose()
                        logger.info("Google GenAI async client closed successfully")
                    except Exception as async_close_error:
                        logger.warning(f"Error closing Google GenAI async client: {async_close_error}")

                if hasattr(self.google_new_client, 'close'):
                    if asyncio.iscoroutinefunction(self.google_new_client.close):
                        await self.google_new_client.close()
                        logger.info("Google GenAI client closed successfully")
                    else:
                        self.google_new_client.close()
                        logger.info("Google GenAI client closed successfully")

            logger.info("All AI service resources cleaned up successfully")
        except Exception as e:
            logger.error(f"Error during AI service cleanup: {e}")
