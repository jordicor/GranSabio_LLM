"""
Asynchronous Gran Sabio LLM Client
==================================

High-performance async client for the Gran Sabio LLM Engine API.
Ideal for web applications, parallel generation, and async workflows.

Supports:
- Standard polling-based generation
- SSE streaming with real-time progress, content, QA, and preflight callbacks
- Token budget helpers for model-aware allocation
- Provider API key forwarding
- Specialized repetition analysis wrappers

For synchronous usage, use GranSabioClient instead.
"""

from __future__ import annotations

import json
import os
import time
import asyncio
import logging
from typing import Any, Callable, Coroutine, Dict, List, Optional, Sequence, Tuple

import aiohttp

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Import shared exceptions from package
from . import GranSabioClientError, GranSabioGenerationCancelled, GranSabioGenerationRejected

# Import shared utilities from _common
from ._common import (
    ActivityMonitor,
    is_heartbeat,
    normalize_reasoning_effort,
    compute_generation_timeout,
    compute_token_budgets,
    resolve_model_token_fallback,
    validate_result,
    DEFAULT_STREAM_TIMEOUT_SECONDS,
    STREAM_TIMEOUT_GRACE_SECONDS,
    RESULT_POLL_GRACE_SECONDS,
    RESULT_POLL_INTERVAL_SECONDS,
    STREAM_ACTIVITY_CHECK_SECONDS,
    PROVIDER_KEY_ENV_MAP,
)

logger = logging.getLogger(__name__)


class AsyncGranSabioClient:
    """
    Asynchronous client for Gran Sabio LLM Engine.

    Usage:
        async with AsyncGranSabioClient() as client:
            result = await client.generate("Write a haiku")
            print(result["content"])

    Or without context manager:
        client = AsyncGranSabioClient()
        await client.connect()
        try:
            result = await client.generate("Write a haiku")
        finally:
            await client.close()
    """

    DEFAULT_BASE_URL = "http://localhost:8000"
    DEFAULT_TIMEOUT = aiohttp.ClientTimeout(total=600, connect=30)
    DEFAULT_POLL_INTERVAL = 2.0
    DEFAULT_MAX_WAIT = 600.0

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[aiohttp.ClientTimeout] = None,
        provider_keys: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize the async Gran Sabio client.

        Args:
            base_url: API base URL (default: http://localhost:8000 or GRANSABIO_BASE_URL env)
            api_key: Optional API key (default: GRANSABIO_API_KEY env)
            timeout: Request timeout configuration
            provider_keys: Dict of provider API key headers. If None, loaded from
                environment variables using PROVIDER_KEY_ENV_MAP.
        """
        self.base_url = (base_url or os.getenv("GRANSABIO_BASE_URL", self.DEFAULT_BASE_URL)).rstrip("/")
        self.api_key = api_key or os.getenv("GRANSABIO_API_KEY")
        self.timeout = timeout or self.DEFAULT_TIMEOUT
        self._session: Optional[aiohttp.ClientSession] = None
        self._owns_session = True
        self.provider_keys = provider_keys if provider_keys is not None else self._load_provider_keys()
        self._model_token_cache: Dict[str, int] = {}
        self._model_timeout_cache: Dict[str, int] = {}

    @staticmethod
    def _load_provider_keys() -> Dict[str, str]:
        """Load provider API keys from environment variables."""
        keys: Dict[str, str] = {}
        for header, env_names in PROVIDER_KEY_ENV_MAP.items():
            for env_name in env_names:
                value = os.getenv(env_name)
                if value:
                    keys[header] = value
                    break
        return keys

    async def connect(self) -> None:
        """Initialize the HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self.timeout)
            self._owns_session = True

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and self._owns_session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def __aenter__(self) -> "AsyncGranSabioClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    @property
    def session(self) -> aiohttp.ClientSession:
        """Get the active session, raising if not connected."""
        if self._session is None or self._session.closed:
            raise RuntimeError(
                "Client not connected. Use 'async with AsyncGranSabioClient()' or call connect()"
            )
        return self._session

    def _headers(self) -> Dict[str, str]:
        """Build request headers including provider API keys."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self.provider_keys)
        return headers

    async def _request(
        self,
        method: str,
        endpoint: str,
        json_data: Optional[Dict] = None,
        **kwargs
    ) -> aiohttp.ClientResponse:
        """Make an async HTTP request with error handling."""
        url = f"{self.base_url}{endpoint}"
        kwargs.setdefault("headers", self._headers())

        try:
            response = await self.session.request(method, url, json=json_data, **kwargs)
            return response
        except aiohttp.ClientConnectorError as e:
            raise GranSabioClientError(
                f"Cannot connect to Gran Sabio API at {self.base_url}. "
                "Ensure the server is running."
            ) from e
        except asyncio.TimeoutError as e:
            raise GranSabioClientError(f"Request to {endpoint} timed out") from e
        except aiohttp.ClientError as e:
            raise GranSabioClientError(f"Request failed: {e}") from e

    # =========================================================================
    # Health & Info
    # =========================================================================

    async def health_check(self) -> Dict[str, Any]:
        """
        Check API health status.

        Returns:
            Dict with status, active_sessions, timestamp
        """
        async with await self._request("GET", "/health") as response:
            if response.status != 200:
                raise GranSabioClientError(
                    f"Health check failed: {response.status}",
                    status_code=response.status
                )
            return await response.json()

    async def get_models(self) -> Dict[str, Any]:
        """
        Get available AI models organized by provider.

        Returns:
            Dict mapping provider names to lists of model specifications
        """
        async with await self._request("GET", "/models") as response:
            if response.status != 200:
                raise GranSabioClientError(
                    f"Failed to get models: {response.status}",
                    status_code=response.status
                )
            return await response.json()

    async def is_available(self) -> bool:
        """Check if the API is reachable and healthy."""
        try:
            health = await self.health_check()
            return health.get("status") == "healthy"
        except GranSabioClientError:
            return False

    async def get_info(self) -> Dict[str, Any]:
        """
        Get API information.

        Returns a dict with service name, version, and status.
        """
        health = await self.health_check()
        return {
            "service": "Gran Sabio LLM Engine",
            "version": "1.0.0",
            "status": health.get("status", "unknown"),
            "active_sessions": health.get("active_sessions", 0),
            "timestamp": health.get("timestamp"),
        }

    # =========================================================================
    # Project Management
    # =========================================================================

    async def reserve_project(self, project_id: Optional[str] = None) -> str:
        """
        Reserve a project identifier for grouping related sessions.

        Args:
            project_id: Optional custom ID (auto-generated if not provided)

        Returns:
            The reserved project_id
        """
        payload = {"project_id": project_id} if project_id else {}
        async with await self._request("POST", "/project/new", json_data=payload) as response:
            if response.status not in (200, 201):
                text = await response.text()
                raise GranSabioClientError(
                    f"Failed to reserve project: {text}",
                    status_code=response.status
                )
            data = await response.json()
            return data["project_id"]

    async def start_project(self, project_id: str) -> Dict[str, Any]:
        """Activate/reactivate a project."""
        async with await self._request("POST", f"/project/start/{project_id}") as response:
            if response.status not in (200, 201):
                text = await response.text()
                raise GranSabioClientError(
                    f"Failed to start project: {text}",
                    status_code=response.status
                )
            return await response.json()

    async def stop_project(self, project_id: str) -> Dict[str, Any]:
        """Cancel a project and all its active sessions."""
        async with await self._request("POST", f"/project/stop/{project_id}") as response:
            if response.status not in (200, 202):
                text = await response.text()
                raise GranSabioClientError(
                    f"Failed to stop project: {text}",
                    status_code=response.status
                )
            return await response.json()

    # =========================================================================
    # Content Generation
    # =========================================================================

    async def generate(
        self,
        prompt: str,
        content_type: str = "article",
        generator_model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 4000,
        min_words: Optional[int] = None,
        max_words: Optional[int] = None,
        qa_models: Optional[List[str]] = None,
        qa_layers: Optional[List[Dict[str, Any]]] = None,
        min_global_score: float = 8.0,
        max_iterations: int = 3,
        gran_sabio_model: str = "claude-opus-4-5-20251101",
        gran_sabio_fallback: bool = True,
        verbose: bool = True,
        project_id: Optional[str] = None,
        request_name: Optional[str] = None,
        json_output: bool = False,
        json_schema: Optional[Dict[str, Any]] = None,
        word_count_enforcement: Optional[Dict[str, Any]] = None,
        phrase_frequency: Optional[Dict[str, Any]] = None,
        lexical_diversity: Optional[Dict[str, Any]] = None,
        reasoning_effort: Optional[str] = None,
        thinking_budget_tokens: Optional[int] = None,
        wait_for_completion: bool = True,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        max_wait: float = DEFAULT_MAX_WAIT,
        on_status: Optional[Callable[[Dict[str, Any]], None]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate content with optional QA validation.

        Args:
            prompt: The generation prompt
            content_type: Type of content (article, creative, json, etc.)
            generator_model: AI model to use for generation
            temperature: Creativity level (0.0-2.0)
            max_tokens: Maximum tokens to generate
            min_words: Minimum word count (optional)
            max_words: Maximum word count (optional)
            qa_models: Models for QA evaluation (required if qa_layers not empty)
            qa_layers: QA evaluation layers (empty list = bypass QA)
            min_global_score: Minimum score for approval
            max_iterations: Max generation attempts
            gran_sabio_model: Model for escalation and arbitration
            gran_sabio_fallback: Allow regeneration on exhausted iterations
            verbose: Enable detailed logging
            project_id: Group with existing project
            request_name: Human-readable request label
            json_output: Expect JSON output
            json_schema: JSON Schema for structured output
            word_count_enforcement: Word count validation config
            phrase_frequency: Phrase repetition guard config
            lexical_diversity: Vocabulary diversity guard config
            reasoning_effort: For OpenAI reasoning models (low/medium/high)
            thinking_budget_tokens: For Claude thinking models (min 1024)
            wait_for_completion: Wait for result (True) or return immediately (False)
            poll_interval: Seconds between status checks
            max_wait: Maximum seconds to wait for completion
            on_status: Callback for status updates during polling
            **kwargs: Additional parameters passed to API

        Returns:
            Generation result with content, score, and metadata
        """
        # Build payload
        payload = {
            "prompt": prompt,
            "content_type": content_type,
            "generator_model": generator_model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "min_global_score": min_global_score,
            "max_iterations": max_iterations,
            "gran_sabio_model": gran_sabio_model,
            "gran_sabio_fallback": gran_sabio_fallback,
            "verbose": verbose,
        }

        # Optional parameters
        if min_words is not None:
            payload["min_words"] = min_words
        if max_words is not None:
            payload["max_words"] = max_words
        if qa_models is not None:
            payload["qa_models"] = qa_models
        if qa_layers is not None:
            payload["qa_layers"] = qa_layers
        if project_id:
            payload["project_id"] = project_id
        if request_name:
            payload["request_name"] = request_name
        if json_output:
            payload["json_output"] = json_output
        if json_schema:
            payload["json_schema"] = json_schema
        if word_count_enforcement:
            payload["word_count_enforcement"] = word_count_enforcement
        if phrase_frequency:
            payload["phrase_frequency"] = phrase_frequency
        if lexical_diversity:
            payload["lexical_diversity"] = lexical_diversity
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort
        if thinking_budget_tokens:
            payload["thinking_budget_tokens"] = thinking_budget_tokens

        # Additional kwargs
        payload.update(kwargs)

        # Start generation
        async with await self._request("POST", "/generate", json_data=payload) as response:
            if response.status != 200:
                text = await response.text()
                raise GranSabioClientError(
                    f"Generation request failed: {text}",
                    status_code=response.status,
                    details={"payload": payload}
                )

            result = await response.json()

        # Handle preflight rejection
        if result.get("status") == "rejected":
            feedback = result.get("preflight_feedback", {})
            raise GranSabioClientError(
                f"Request rejected by preflight: {feedback.get('user_feedback', 'Unknown reason')}",
                details={"preflight_feedback": feedback}
            )

        session_id = result.get("session_id")
        if not session_id:
            raise GranSabioClientError("No session_id in response", details=result)

        # Return immediately if not waiting
        if not wait_for_completion:
            return result

        # Poll for completion
        return await self.wait_for_result(
            session_id,
            poll_interval=poll_interval,
            max_wait=max_wait,
            on_status=on_status
        )

    async def get_status(self, session_id: str) -> Dict[str, Any]:
        """Get current status of a generation session."""
        async with await self._request("GET", f"/status/{session_id}") as response:
            if response.status != 200:
                text = await response.text()
                raise GranSabioClientError(
                    f"Failed to get status: {text}",
                    status_code=response.status
                )
            return await response.json()

    async def get_result(self, session_id: str) -> Dict[str, Any]:
        """Get the result of a completed generation session."""
        async with await self._request("GET", f"/result/{session_id}") as response:
            if response.status == 202:
                return {"status": "in_progress", "session_id": session_id}

            if response.status != 200:
                text = await response.text()
                raise GranSabioClientError(
                    f"Failed to get result: {text}",
                    status_code=response.status
                )

            return await response.json()

    async def stop_session(self, session_id: str) -> Dict[str, Any]:
        """Cancel an active generation session."""
        async with await self._request("POST", f"/stop/{session_id}") as response:
            if response.status not in (200, 202):
                text = await response.text()
                raise GranSabioClientError(
                    f"Failed to stop session: {text}",
                    status_code=response.status
                )
            return await response.json()

    async def wait_for_result(
        self,
        session_id: str,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        max_wait: float = DEFAULT_MAX_WAIT,
        on_status: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """
        Poll until generation completes and return result.

        Args:
            session_id: Session to wait for
            poll_interval: Seconds between status checks
            max_wait: Maximum seconds to wait
            on_status: Optional callback for each status update

        Returns:
            Final generation result
        """
        start_time = time.monotonic()

        while True:
            elapsed = time.monotonic() - start_time
            if elapsed > max_wait:
                raise GranSabioClientError(
                    f"Timed out waiting for session {session_id} after {max_wait}s"
                )

            status = await self.get_status(session_id)

            if on_status:
                on_status(status)

            current_status = status.get("status", "")

            if current_status == "completed":
                return await self.get_result(session_id)

            if current_status == "failed":
                raise GranSabioClientError(
                    f"Generation failed: {status.get('error', 'Unknown error')}",
                    details=status
                )

            if current_status == "cancelled":
                raise GranSabioClientError(
                    "Generation was cancelled",
                    details=status
                )

            await asyncio.sleep(poll_interval)

    # Alias for backward compatibility with demos
    async def wait_for_completion(
        self,
        session_id: str,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        timeout: float = DEFAULT_MAX_WAIT,
        on_status: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """Alias for wait_for_result (backward compatibility)."""
        return await self.wait_for_result(
            session_id,
            poll_interval=poll_interval,
            max_wait=timeout,
            on_status=on_status
        )

    # =========================================================================
    # Text Analysis
    # =========================================================================

    async def analyze_lexical_diversity(
        self,
        text: str,
        metrics: str = "auto",
        top_words: int = 20,
        analyze_windows: bool = False,
        language: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Analyze text for vocabulary diversity.

        Args:
            text: Text to analyze
            metrics: Which metrics to compute (auto, all, or specific)
            top_words: Number of top frequent words to return
            analyze_windows: Enable window-based analysis
            language: Language code (en, es, etc.)

        Returns:
            Analysis results with metrics, grades, and decision
        """
        payload = {
            "text": text,
            "metrics": metrics,
            "top_words": top_words,
            "analyze_windows": analyze_windows,
        }
        if language:
            payload["language"] = language
        payload.update(kwargs)

        async with await self._request("POST", "/analysis/lexical-diversity", json_data=payload) as response:
            if response.status != 200:
                text_response = await response.text()
                raise GranSabioClientError(
                    f"Lexical diversity analysis failed: {text_response}",
                    status_code=response.status
                )
            return await response.json()

    async def analyze_repetition(
        self,
        text: str,
        min_n: int = 2,
        max_n: int = 5,
        min_count: int = 2,
        diagnostics: str = "basic",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Analyze text for phrase repetition patterns.

        Args:
            text: Text to analyze
            min_n: Minimum n-gram length
            max_n: Maximum n-gram length
            min_count: Minimum occurrences to report
            diagnostics: Level of diagnostics (none, basic, full)

        Returns:
            Repetition analysis with patterns and diagnostics
        """
        payload = {
            "text": text,
            "min_n": min_n,
            "max_n": max_n,
            "min_count": min_count,
            "diagnostics": diagnostics,
        }
        payload.update(kwargs)

        async with await self._request("POST", "/analysis/repetition", json_data=payload) as response:
            if response.status != 200:
                text_response = await response.text()
                raise GranSabioClientError(
                    f"Repetition analysis failed: {text_response}",
                    status_code=response.status
                )
            return await response.json()

    # =========================================================================
    # Specialized Repetition Analysis Wrappers
    # =========================================================================

    async def fetch_unigram_counts(
        self,
        text: str,
        *,
        min_count: int = 1,
        summary_top_k: int = 1000,
        language: Optional[str] = None,
        filter_stop_words: bool = False,
    ) -> Dict[str, int]:
        """
        Invoke the repetition analysis tool and extract unigram word counts.

        Args:
            text: Source text to analyze.
            min_count: Minimum frequency to include in results.
            summary_top_k: Number of top results per n-gram size to request.
            language: ISO language hint (e.g., 'es', 'en') for stopword filtering.
            filter_stop_words: If True, exclude stopwords from results.

        Returns:
            Mapping of lowercased unigram word to its frequency count.
        """
        if not text or not text.strip():
            return {}

        payload: Dict[str, Any] = {
            "text": text,
            "min_n": 1,
            "max_n": 1,
            "min_count": max(1, min_count),
            "summary_mode": "counts",
            "summary_top_k": max(50, summary_top_k),
            "counts_only_limit_per_n": 0,
            "output_mode": "compact",
            "details": "none",
            "lowercase": True,
            "strip_accents": False,
            "enable_clusters": False,
            "enable_position_metrics": False,
            "enable_windows": False,
            "language": language,
            "filter_stop_words": filter_stop_words,
        }

        async with await self._request("POST", "/analysis/repetition", json_data=payload) as response:
            if response.status != 200:
                text_response = await response.text()
                raise GranSabioClientError(
                    f"Unigram repetition analysis failed: {text_response}",
                    status_code=response.status,
                )
            data = await response.json()

        counts: Dict[str, int] = {}
        summary = data.get("summary") if isinstance(data, dict) else None
        top_by_count = summary.get("top_by_count") if isinstance(summary, dict) else None
        if isinstance(top_by_count, dict):
            for n_key, entries in top_by_count.items():
                try:
                    n_value = int(n_key)
                except (TypeError, ValueError):
                    continue
                if n_value != 1 or not isinstance(entries, list):
                    continue
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    word = entry.get("text")
                    count = entry.get("count")
                    if isinstance(word, str) and isinstance(count, int):
                        counts[word] = count

        return counts

    async def fetch_positional_bias(
        self,
        blocks: Sequence[str],
        *,
        min_count: int = 2,
        confidence_z: float = 1.645,
        bias_threshold: float = 0.40,
        max_summary_sequences: int = 50,
    ) -> Dict[str, Any]:
        """
        Invoke the repetition analysis tool with positional metrics enabled.

        Args:
            blocks: Ordered text blocks (e.g., chapters) to analyze.
            min_count: Minimum absolute repetitions needed to surface a sequence.
            confidence_z: Wilson score z value for the bias calculation.
            bias_threshold: Lower bound threshold to flag a positional bias.
            max_summary_sequences: Maximum sequences to request per positional scope.

        Returns:
            Parsed positional bias payload from the response, or {} if unavailable.
        """
        cleaned_blocks = [block.strip() for block in blocks if isinstance(block, str) and block.strip()]
        if not cleaned_blocks:
            return {}

        text = "\n\n\n".join(cleaned_blocks)
        payload: Dict[str, Any] = {
            "text": text,
            "min_n": 1,
            "max_n": 3,
            "min_count": max(1, min_count),
            "summary_mode": "counts",
            "summary_top_k": max(10, max_summary_sequences),
            "counts_only_limit_per_n": 0,
            "output_mode": "compact",
            "details": "all",
            "lowercase": True,
            "strip_accents": False,
            "enable_clusters": False,
            "enable_position_metrics": True,
            "enable_windows": True,
            "position_metrics": {
                "pos_conf_z": float(confidence_z),
                "pos_bias_threshold": float(bias_threshold),
                "pos_min_count": max(1, min_count),
            },
            "block_break_min_blank_lines": 2,
        }

        async with await self._request("POST", "/analysis/repetition", json_data=payload) as response:
            if response.status != 200:
                text_response = await response.text()
                raise GranSabioClientError(
                    f"Positional bias analysis failed: {text_response}",
                    status_code=response.status,
                )
            data = await response.json()

        if not isinstance(data, dict):
            return {}

        return data.get("position_bias") or {}

    # =========================================================================
    # Attachments
    # =========================================================================

    async def upload_attachment(
        self,
        username: str,
        content: bytes,
        filename: str,
        content_type: str = "text/plain"
    ) -> Dict[str, Any]:
        """
        Upload an attachment file.

        Args:
            username: User namespace for the attachment
            content: File content as bytes
            filename: Original filename
            content_type: MIME type

        Returns:
            Upload result with upload_id
        """
        data = aiohttp.FormData()
        data.add_field("username", username)
        data.add_field(
            "file",
            content,
            filename=filename,
            content_type=content_type
        )

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        async with self.session.post(
            f"{self.base_url}/attachments",
            data=data,
            headers=headers
        ) as response:
            if response.status != 200:
                text = await response.text()
                raise GranSabioClientError(
                    f"Attachment upload failed: {text}",
                    status_code=response.status
                )
            return await response.json()

    async def upload_attachment_base64(
        self,
        username: str,
        content_base64: str,
        filename: str,
        content_type: str = "text/plain"
    ) -> Dict[str, Any]:
        """
        Upload an attachment as base64.

        Args:
            username: User namespace
            content_base64: Base64-encoded content
            filename: Original filename
            content_type: MIME type

        Returns:
            Upload result with upload_id
        """
        payload = {
            "username": username,
            "filename": filename,
            "content_type": content_type,
            "content_base64": content_base64
        }

        async with await self._request("POST", "/attachments/base64", json_data=payload) as response:
            if response.status != 200:
                text = await response.text()
                raise GranSabioClientError(
                    f"Attachment upload failed: {text}",
                    status_code=response.status
                )
            return await response.json()

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    async def generate_json(
        self,
        prompt: str,
        schema: Dict[str, Any],
        model: str = "gpt-4o",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate structured JSON output with schema validation.

        Args:
            prompt: Generation prompt
            schema: JSON Schema for output structure
            model: Generator model
            **kwargs: Additional generate() parameters

        Returns:
            Parsed JSON content and metadata
        """
        result = await self.generate(
            prompt=prompt,
            content_type="json",
            generator_model=model,
            json_output=True,
            json_schema=schema,
            qa_layers=[],
            **kwargs
        )

        content = result.get("content", "{}")
        if isinstance(content, str):
            try:
                result["parsed_content"] = json.loads(content)
            except json.JSONDecodeError:
                result["parsed_content"] = None
                result["parse_error"] = True

        return result

    async def generate_fast(
        self,
        prompt: str,
        model: str = "gpt-4o-mini",
        max_tokens: int = 2000,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Quick generation without QA (bypass mode).

        Args:
            prompt: Generation prompt
            model: Generator model (default: fast model)
            max_tokens: Maximum tokens
            **kwargs: Additional generate() parameters

        Returns:
            Generation result
        """
        return await self.generate(
            prompt=prompt,
            generator_model=model,
            max_tokens=max_tokens,
            qa_layers=[],
            max_iterations=1,
            **kwargs
        )

    async def generate_parallel(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple prompts in parallel.

        Args:
            prompts: List of prompts to generate
            **kwargs: Additional generate() parameters (applied to all)

        Returns:
            List of generation results
        """
        tasks = [
            self.generate(prompt=prompt, **kwargs)
            for prompt in prompts
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)

    # =========================================================================
    # Token Budget Helpers
    # =========================================================================

    async def get_optimal_max_tokens(self, model: str) -> int:
        """Get optimal max_tokens for a model from the server catalog, with fallback.

        Queries the /models endpoint and caches the result. Falls back to
        hardcoded limits from ``_common.resolve_model_token_fallback`` when
        the server is unreachable.

        Args:
            model: Model identifier (e.g. "gpt-5", "claude-opus-4-1-20250805").

        Returns:
            The model's advertised output token limit.
        """
        key = model.strip().lower()
        cached = self._model_token_cache.get(key)
        if cached:
            return cached

        try:
            models_data = await self.get_models()
            for provider_models in models_data.values():
                if not isinstance(provider_models, list):
                    continue
                for spec in provider_models:
                    if spec.get("model_id") == model or spec.get("key") == model:
                        output_tokens = spec.get("output_tokens", 0)
                        if output_tokens > 0:
                            self._model_token_cache[key] = output_tokens
                            return output_tokens
        except GranSabioClientError:
            pass

        fallback = resolve_model_token_fallback(model)
        self._model_token_cache[key] = fallback
        return fallback

    async def allocate_token_budgets(
        self,
        model: str,
        desired_max_tokens: int,
        desired_thinking_tokens: Optional[int] = None,
    ) -> Tuple[int, Optional[int]]:
        """Determine safe max/thinking token budgets honoring model limits.

        Args:
            model: Model identifier.
            desired_max_tokens: Requested max_tokens.
            desired_thinking_tokens: Requested thinking budget (for Claude thinking models).

        Returns:
            ``(max_tokens, thinking_tokens)`` where ``thinking_tokens`` may be ``None``.
        """
        model_limit = await self.get_optimal_max_tokens(model)
        return compute_token_budgets(model_limit, desired_max_tokens, desired_thinking_tokens)

    # =========================================================================
    # Streaming Generation
    # =========================================================================

    async def _resolve_catalog_timeout(self, model: Optional[str]) -> Optional[int]:
        """Look up the advertised reasoning timeout for a model via the catalog."""
        if not model:
            return None
        key = model.strip().lower()
        cached = self._model_timeout_cache.get(key)
        if cached:
            return cached
        try:
            models_data = await self.get_models()
        except GranSabioClientError:
            return None
        for provider_models in models_data.values():
            if not isinstance(provider_models, list):
                continue
            for spec in provider_models:
                if model == spec.get("key") or model == spec.get("model_id"):
                    timeout_val = spec.get("reasoning_timeout_seconds")
                    if isinstance(timeout_val, (int, float)) and timeout_val > 0:
                        timeout_int = int(timeout_val)
                        self._model_timeout_cache[key] = timeout_int
                        return timeout_int
        return None

    async def _fetch_result_polling(
        self,
        session_id: str,
        *,
        timeout_seconds: Optional[float] = None,
        poll_interval: float = RESULT_POLL_INTERVAL_SECONDS,
        activity_monitor: Optional[ActivityMonitor] = None,
    ) -> Dict[str, Any]:
        """Retrieve the final generation payload, polling until completion if required.

        This is the async equivalent of the sync polling loop. It respects the
        ``ActivityMonitor`` to extend deadlines when the server is still making
        progress (e.g. multi-iteration QA).

        Args:
            session_id: The generation session to poll.
            timeout_seconds: Maximum seconds to wait for completion.
            poll_interval: Seconds between status checks.
            activity_monitor: Optional monitor used to extend the deadline when
                the server is still active.

        Returns:
            Final generation result dict.
        """
        result_url = f"/result/{session_id}"
        deadline = time.monotonic() + timeout_seconds if timeout_seconds else None

        while True:
            async with await self._request("GET", result_url) as response:
                if response.status == 200:
                    final_result = await response.json()
                    logger.info(
                        "Final result retrieved for session %s: status = %s",
                        session_id,
                        final_result.get("status", "unknown"),
                    )
                    return final_result

                detail_message = ""
                try:
                    payload = await response.json()
                    if isinstance(payload, dict):
                        detail_message = str(payload.get("detail", ""))
                except Exception:
                    payload = None

                should_retry = False
                if response.status in (202, 425, 503, 400):
                    lower_detail = detail_message.lower()
                    if "not finished" in lower_detail or "still in progress" in lower_detail:
                        should_retry = True
                    elif response.status == 202 and not detail_message:
                        should_retry = True

                if not should_retry:
                    text_body = await response.text()
                    raise GranSabioClientError(
                        f"Result retrieval failed: {response.status} - {text_body}",
                        status_code=response.status,
                    )

            # Retryable -- check deadline
            now = time.monotonic()
            if deadline is not None and now >= deadline:
                if activity_monitor and timeout_seconds:
                    latest_activity = activity_monitor.last_activity_timestamp()
                    if now - latest_activity <= timeout_seconds:
                        deadline = time.monotonic() + timeout_seconds
                        snapshot = activity_monitor.describe()
                        logger.debug(
                            "[RESULT_FETCH] Extended deadline due to stream activity "
                            "(iteration=%s, message=%s)",
                            snapshot.get("last_iteration"),
                            snapshot.get("last_message"),
                        )
                    else:
                        raise GranSabioClientError(
                            f"Timed out waiting for final result of session {session_id}"
                        )
                else:
                    raise GranSabioClientError(
                        f"Timed out waiting for final result of session {session_id}"
                    )

            remaining = None if deadline is None else max(0.0, deadline - now)
            sleep_interval = poll_interval if remaining is None else min(poll_interval, max(1.0, remaining))
            await asyncio.sleep(sleep_interval)

    async def _stream_progress(
        self,
        session_id: str,
        progress_callback: Optional[Callable[[str], None]],
        qa_callback: Optional[Callable[[str, str, str], None]],
        activity_monitor: ActivityMonitor,
        result_queue: "asyncio.Queue[str]",
        timeout: float,
        verbose: bool,
    ) -> None:
        """Consume the SSE progress stream and route events to callbacks.

        Puts a terminal status string onto ``result_queue`` when the generation
        reaches a terminal state (completed, failed, cancelled).
        """
        url = f"{self.base_url}/stream/{session_id}"
        try:
            timeout_cfg = aiohttp.ClientTimeout(total=timeout, connect=30)
            async with self.session.get(url, headers=self._headers(), timeout=timeout_cfg) as response:
                if response.status != 200:
                    logger.warning("Progress stream returned status %s", response.status)
                    return
                activity_monitor.mark(message="progress_stream_connected")

                while not response.content.at_eof():
                    line_bytes = await response.content.readline()
                    line = line_bytes.decode("utf-8", errors="replace").rstrip("\n\r")
                    if not line.startswith("data: "):
                        continue
                    try:
                        data = json.loads(line[6:])
                    except json.JSONDecodeError:
                        if verbose:
                            logger.warning("Failed to parse progress SSE: %s", line[:120])
                        continue

                    if verbose:
                        logger.debug("Progress event: %s", data)

                    # Track activity from iteration / message fields
                    current_iteration = data.get("current_iteration")
                    progress_message = data.get("message") or data.get("status")
                    if isinstance(current_iteration, int):
                        activity_monitor.mark(
                            iteration=current_iteration,
                            message=progress_message,
                        )
                    elif progress_message:
                        activity_monitor.mark(message=progress_message)

                    # Mark activity for verbose_log entries
                    verbose_log = data.get("verbose_log", [])
                    if verbose_log:
                        activity_monitor.mark(message=f"verbose_log[{len(verbose_log)}]")

                    # Classify message as QA vs progress
                    message = data.get("message", "")
                    is_qa_message = any(kw in message for kw in [
                        "quality evaluation", "QA evaluation", "Evaluating layer",
                        "Score:", "/10", "Deal-breaker", "consensus", "accuracy",
                        "QA", "qa_evaluation",
                    ])

                    # Route non-QA messages to progress callback
                    if progress_callback and "message" in data and not is_qa_message:
                        progress_callback(data["message"])

                    # Route QA messages to qa_callback
                    if qa_callback and is_qa_message and message:
                        agent = "QA System"
                        status = "evaluating"

                        for model_name in ["gpt-5", "gpt-4", "claude", "gemini", "o3"]:
                            if model_name in message.lower():
                                agent = model_name.upper()
                                break

                        if "score:" in message.lower() or "/10" in message:
                            status = "score"
                        elif "deal-breaker" in message.lower():
                            status = "critical"
                        elif "consensus" in message.lower():
                            agent = "Consensus"
                            status = "calculating"
                        elif "completed" in message.lower() or "passed" in message.lower():
                            status = "success"
                        elif "failed" in message.lower() or "rejected" in message.lower():
                            status = "error"

                        qa_callback(message, agent, status)
                        activity_monitor.mark(message=f"qa:{agent}:{status}")

                    # Route verbose_log QA entries
                    current_status = data.get("status", "")
                    if qa_callback and (current_status == "qa_evaluation" or verbose_log):
                        for log_entry in verbose_log:
                            if not isinstance(log_entry, str):
                                continue
                            clean_entry = log_entry.strip()
                            if any(kw in clean_entry for kw in [
                                "quality evaluation", "QA evaluation", "Evaluating layer",
                                "Score:", "/10", "Deal-breaker", "consensus", "accuracy",
                            ]):
                                entry_agent = "QA System"
                                entry_status = "evaluating"
                                for model_name in ["gpt-5", "gpt-4", "claude", "gemini", "o3"]:
                                    if model_name in clean_entry.lower():
                                        entry_agent = model_name.upper()
                                        break
                                if "score:" in clean_entry.lower() or "/10" in clean_entry:
                                    entry_status = "score"
                                elif "deal-breaker" in clean_entry.lower():
                                    entry_status = "critical"
                                elif "consensus" in clean_entry.lower():
                                    entry_agent = "Consensus"
                                    entry_status = "calculating"
                                qa_callback(clean_entry, entry_agent, entry_status)
                            elif current_status == "qa_evaluation" and any(term in clean_entry for term in [
                                "Querying", "evaluation", "processing", "analyzing",
                            ]):
                                qa_callback(clean_entry, "QA System", "info")

                    # Check for terminal status
                    if data.get("status") in ("completed", "failed", "cancelled"):
                        await result_queue.put(data.get("status"))
                        break

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("Progress stream error for session %s: %s", session_id, e)

    async def _stream_content(
        self,
        url: str,
        callback: Optional[Callable[[str], None]],
        activity_monitor: ActivityMonitor,
        timeout: float,
        verbose: bool,
        stream_name: str,
    ) -> None:
        """Consume a raw content stream (generation, QA, or preflight).

        Each chunk is forwarded to ``callback`` if provided. Heartbeat messages
        are silently skipped.
        """
        try:
            timeout_cfg = aiohttp.ClientTimeout(total=timeout, connect=30)
            async with self.session.get(url, headers=self._headers(), timeout=timeout_cfg) as response:
                if response.status != 200:
                    logger.warning("%s stream returned status %s", stream_name, response.status)
                    return

                async for chunk_bytes in response.content.iter_any():
                    chunk = chunk_bytes.decode("utf-8", errors="replace")
                    if not chunk or is_heartbeat(chunk):
                        continue
                    activity_monitor.mark(message=f"{stream_name}_chunk")
                    if callback:
                        callback(chunk)
                    if verbose:
                        logger.debug("%s chunk: %s...", stream_name, chunk[:60])

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("%s stream error: %s", stream_name, e)

    async def generate_streaming(
        self,
        prompt: str,
        *,
        content_type: str = "article",
        generator_model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 4000,
        min_words: Optional[int] = None,
        max_words: Optional[int] = None,
        qa_models: Optional[List[str]] = None,
        qa_layers: Optional[List[Dict[str, Any]]] = None,
        min_global_score: float = 8.0,
        max_iterations: int = 3,
        gran_sabio_model: str = "claude-opus-4-5-20251101",
        gran_sabio_fallback: bool = True,
        verbose: bool = True,
        project_id: Optional[str] = None,
        request_name: Optional[str] = None,
        json_output: bool = False,
        json_schema: Optional[Dict[str, Any]] = None,
        word_count_enforcement: Optional[Dict[str, Any]] = None,
        phrase_frequency: Optional[Dict[str, Any]] = None,
        lexical_diversity: Optional[Dict[str, Any]] = None,
        reasoning_effort: Optional[str] = None,
        thinking_budget_tokens: Optional[int] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
        content_callback: Optional[Callable[[str], None]] = None,
        qa_callback: Optional[Callable[[str, str, str], None]] = None,
        on_session_start: Optional[Callable[[str], None]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate content with real-time SSE streaming.

        Starts generation via ``generate(wait_for_completion=False)`` and then
        opens four parallel SSE connections (progress, generation content,
        QA content, preflight content) to surface live updates through the
        provided callbacks.

        Args:
            prompt: The generation prompt.
            content_type: Type of content (article, creative, json, etc.).
            generator_model: AI model to use for generation.
            temperature: Creativity level (0.0-2.0).
            max_tokens: Maximum tokens to generate.
            min_words: Minimum word count (optional).
            max_words: Maximum word count (optional).
            qa_models: Models for QA evaluation.
            qa_layers: QA evaluation layers (empty list = bypass QA).
            min_global_score: Minimum score for approval.
            max_iterations: Max generation attempts.
            gran_sabio_model: Model for escalation and arbitration.
            gran_sabio_fallback: Allow regeneration on exhausted iterations.
            verbose: Enable detailed logging.
            project_id: Group with existing project.
            request_name: Human-readable request label.
            json_output: Expect JSON output.
            json_schema: JSON Schema for structured output.
            word_count_enforcement: Word count validation config.
            phrase_frequency: Phrase repetition guard config.
            lexical_diversity: Vocabulary diversity guard config.
            reasoning_effort: For reasoning models (low/medium/high).
            thinking_budget_tokens: For Claude thinking models (min 1024).
            progress_callback: Called with progress message strings.
            content_callback: Called with raw content chunks as they stream.
            qa_callback: Called with ``(message, agent, status)`` for QA events.
            on_session_start: Called with session_id immediately after the
                generation session is created. May be sync or async.
            **kwargs: Additional parameters passed to API.

        Returns:
            Final generation result dict with content, score, and metadata.
        """
        # Build the kwargs dict for generate(), forwarding all explicit params
        gen_kwargs: Dict[str, Any] = {
            "content_type": content_type,
            "generator_model": generator_model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "min_words": min_words,
            "max_words": max_words,
            "qa_models": qa_models,
            "qa_layers": qa_layers,
            "min_global_score": min_global_score,
            "max_iterations": max_iterations,
            "gran_sabio_model": gran_sabio_model,
            "gran_sabio_fallback": gran_sabio_fallback,
            "verbose": verbose,
            "project_id": project_id,
            "request_name": request_name,
            "json_output": json_output,
            "json_schema": json_schema,
            "word_count_enforcement": word_count_enforcement,
            "phrase_frequency": phrase_frequency,
            "lexical_diversity": lexical_diversity,
            "reasoning_effort": reasoning_effort,
            "thinking_budget_tokens": thinking_budget_tokens,
            "wait_for_completion": False,
        }
        gen_kwargs.update(kwargs)

        # Start generation without waiting
        gen_response = await self.generate(prompt, **gen_kwargs)
        session_id = gen_response["session_id"]

        # Notify caller of the session_id immediately
        if on_session_start:
            ret = on_session_start(session_id)
            if asyncio.iscoroutine(ret):
                await ret

        if progress_callback:
            progress_callback("Generation started...")

        # Build the payload dict for timeout computation (same keys as generate())
        payload_for_timeout: Dict[str, Any] = {
            "content_type": content_type,
            "generator_model": generator_model,
            "reasoning_effort": reasoning_effort,
        }

        # Compute timeout
        catalog_timeout = await self._resolve_catalog_timeout(generator_model)
        stream_timeout = compute_generation_timeout(payload_for_timeout, gen_response, catalog_timeout)

        logger.info(
            "[STREAMING] Session %s | timeout=%ds (%.1f min) | model=%s",
            session_id, stream_timeout, stream_timeout / 60.0, generator_model,
        )

        activity_monitor = ActivityMonitor(stream_timeout + RESULT_POLL_GRACE_SECONDS)
        activity_monitor.mark(message="streams_initializing")
        result_queue: asyncio.Queue[str] = asyncio.Queue()

        # Build stream URLs
        gen_url = f"{self.base_url}/stream-generation/{session_id}"
        qa_url = f"{self.base_url}/stream-qa/{session_id}"
        preflight_url = f"{self.base_url}/stream-preflight/{session_id}"

        # QA content wrapper: route raw chunks through qa_callback with special markers
        def _qa_content_cb(chunk: str) -> None:
            if qa_callback:
                qa_callback(chunk, "QA_CONTENT_STREAM", "raw_chunk")

        def _preflight_content_cb(chunk: str) -> None:
            if qa_callback:
                qa_callback(chunk, "PREFLIGHT_CONTENT_STREAM", "raw_chunk")

        # Launch streaming tasks
        tasks = [
            asyncio.create_task(
                self._stream_progress(
                    session_id, progress_callback, qa_callback,
                    activity_monitor, result_queue, stream_timeout, verbose,
                ),
                name=f"progress-{session_id[:8]}",
            ),
            asyncio.create_task(
                self._stream_content(
                    gen_url, content_callback, activity_monitor,
                    stream_timeout, verbose, "generation",
                ),
                name=f"gen-content-{session_id[:8]}",
            ),
            asyncio.create_task(
                self._stream_content(
                    qa_url, _qa_content_cb, activity_monitor,
                    stream_timeout, verbose, "qa",
                ),
                name=f"qa-content-{session_id[:8]}",
            ),
            asyncio.create_task(
                self._stream_content(
                    preflight_url, _preflight_content_cb, activity_monitor,
                    stream_timeout, verbose, "preflight",
                ),
                name=f"preflight-{session_id[:8]}",
            ),
        ]

        # Wait for completion signal with activity-based deadline extension
        queue_timeout = stream_timeout + RESULT_POLL_GRACE_SECONDS
        wait_start = time.monotonic()
        last_activity = activity_monitor.last_activity_timestamp()

        while True:
            try:
                await asyncio.wait_for(result_queue.get(), timeout=STREAM_ACTIVITY_CHECK_SECONDS)
                break
            except asyncio.TimeoutError:
                latest = activity_monitor.last_activity_timestamp()
                if latest > last_activity:
                    # Server is still active -- reset the inactivity window
                    last_activity = latest
                    wait_start = time.monotonic()
                    continue
                if time.monotonic() - wait_start >= queue_timeout:
                    logger.warning(
                        "[STREAMING] Timed out waiting for completion signal "
                        "for session %s after %.1fs of inactivity",
                        session_id, queue_timeout,
                    )
                    break

        # Cancel remaining tasks and allow brief cleanup
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

        # Fetch final result via polling
        final = await self._fetch_result_polling(
            session_id,
            timeout_seconds=queue_timeout,
            poll_interval=RESULT_POLL_INTERVAL_SECONDS,
            activity_monitor=activity_monitor,
        )

        # Validate the result (raises on cancellation, rejection, or error)
        validate_result(final)

        return final


# Backward compatibility alias
AsyncBioAIClient = AsyncGranSabioClient


# Module-level convenience function
async def create_client(base_url: Optional[str] = None, **kwargs) -> AsyncGranSabioClient:
    """Create and connect an async Gran Sabio client instance."""
    client = AsyncGranSabioClient(base_url=base_url, **kwargs)
    await client.connect()
    return client
