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

import asyncio
import json
import logging
import os
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar

import aiohttp

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Import shared exceptions from package
from . import (
    GranSabioClientError,
    TransientGranSabioError,
)

# Import shared utilities from _common
from ._common import (
    POLL_RETRY_BACKOFF_SECONDS,
    RESULT_POLL_GRACE_SECONDS,
    RESULT_POLL_INTERVAL_SECONDS,
    STREAM_ACTIVITY_CHECK_SECONDS,
    ActivityMonitor,
    compute_generation_timeout,
    compute_token_budgets,
    payload_error_metadata,
    resolve_model_token_fallback,
    validate_result,
)
from ._sse import (
    SessionEventRouter,
    SSEParser,
    StreamCallbacks,
    StreamError,
    decode_event_json,
)

logger = logging.getLogger(__name__)


_POLL_RETRY_BACKOFF_SECONDS = POLL_RETRY_BACKOFF_SECONDS

# HTTP statuses on the polling endpoints that mean "try again later" rather
# than a fatal error: 202/425 (result not ready), 429 (rate limited), 503
# (transient overload / proxy flap). get_status and get_result map them to
# TransientGranSabioError so _call_with_retry retries them within the
# max_wait budget, consistent with _fetch_result_polling.
_RETRYABLE_POLL_STATUS_CODES = (202, 425, 429, 503)


# Generic return type for _call_with_retry so it can wrap any idempotent async
# operation (ClientResponse, parsed JSON dict, etc.).
_RetryT = TypeVar("_RetryT")


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
    DEFAULT_TIMEOUT = aiohttp.ClientTimeout(total=12000, connect=30)
    DEFAULT_POLL_INTERVAL = 2.0
    DEFAULT_MAX_WAIT = 12000.0

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
            provider_keys: Optional provider API key headers. Only values passed
                explicitly here are sent; provider credentials are never loaded
                implicitly from the environment.
        """
        self.base_url = (base_url or os.getenv("GRANSABIO_BASE_URL", self.DEFAULT_BASE_URL)).rstrip("/")
        self.api_key = api_key or os.getenv("GRANSABIO_API_KEY")
        self.timeout = timeout or self.DEFAULT_TIMEOUT
        self._session: Optional[aiohttp.ClientSession] = None
        self._owns_session = True
        self.provider_keys = dict(provider_keys or {})
        self._model_token_cache: Dict[str, int] = {}
        self._model_timeout_cache: Dict[str, int] = {}
        # Streaming failures of the last generate_streaming call. A stream is
        # the visibility channel, not the result channel: failures are
        # surfaced here (and through error_callback) instead of aborting a
        # generation the engine is still running.
        self.last_stream_errors: List[Dict[str, str]] = []

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
        # ClientSSLError must be caught BEFORE ClientConnectorError because
        # aiohttp.ClientSSLError is a subclass of ClientConnectorError; without
        # this explicit branch, genuine TLS problems (bad cert, hostname
        # mismatch, unsupported TLS version) would be silently reclassified as
        # transient and retried. They are not transient -- fail fast.
        except aiohttp.ClientSSLError as e:
            raise GranSabioClientError(
                f"TLS/SSL error connecting to {self.base_url}: {e}"
            ) from e
        except aiohttp.ClientConnectorError as e:
            raise TransientGranSabioError(
                f"Cannot connect to Gran Sabio API at {self.base_url}. "
                "Ensure the server is running."
            ) from e
        except asyncio.TimeoutError as e:
            raise TransientGranSabioError(
                f"Request to {endpoint} timed out"
            ) from e
        # ServerDisconnectedError and ClientPayloadError name specific mid-
        # response failure modes (connection closed or compressed/chunked body
        # truncated while aiohttp was still reading). They are genuinely
        # retryable on an idempotent GET.
        except (aiohttp.ServerDisconnectedError,
                aiohttp.ClientPayloadError) as e:
            raise TransientGranSabioError(
                f"Transient protocol error on request to {endpoint}: {e}"
            ) from e
        except aiohttp.ClientError as e:
            raise GranSabioClientError(f"Request failed: {e}") from e

    # Body-read helpers — aiohttp raises ClientPayloadError / ServerDisconnectedError
    # during `response.json()` / `response.text()` when the connection drops or the
    # chunked/compressed body is truncated mid-read. Those failures occur AFTER
    # `_request()` returned successfully, so without these helpers they would
    # escape the retry envelope in `_call_with_retry` as generic exceptions.
    async def _read_json(self, response: aiohttp.ClientResponse, endpoint: str) -> Any:
        """Read JSON body, converting transient mid-read failures to TransientGranSabioError."""
        try:
            return await response.json()
        except (aiohttp.ClientPayloadError,
                aiohttp.ServerDisconnectedError,
                aiohttp.ClientConnectionError,
                asyncio.TimeoutError,
                aiohttp.ContentTypeError,
                json.JSONDecodeError,
                ValueError) as exc:
            raise TransientGranSabioError(
                f"Transient protocol error reading JSON from {endpoint}: {exc}"
            ) from exc

    async def _read_json_no_retry(self, response: aiohttp.ClientResponse, endpoint: str) -> Any:
        """Read JSON body for non-idempotent POSTs without converting to retryable errors."""
        try:
            return await response.json()
        except (
            aiohttp.ClientPayloadError,
            aiohttp.ServerDisconnectedError,
            aiohttp.ClientConnectionError,
            asyncio.TimeoutError,
            aiohttp.ContentTypeError,
            json.JSONDecodeError,
            ValueError,
        ) as exc:
            raise GranSabioClientError(
                f"Failed to read JSON response from {endpoint}; "
                f"the operation may have been accepted by the server: {exc}"
            ) from exc

    async def _read_text(self, response: aiohttp.ClientResponse, endpoint: str) -> str:
        """Read text body, converting transient mid-read failures to TransientGranSabioError."""
        try:
            return await response.text()
        except (aiohttp.ClientPayloadError,
                aiohttp.ServerDisconnectedError,
                aiohttp.ClientConnectionError,
                asyncio.TimeoutError) as exc:
            raise TransientGranSabioError(
                f"Transient protocol error reading text from {endpoint}: {exc}"
            ) from exc

    async def _read_error_text(self, response: aiohttp.ClientResponse, endpoint: str) -> str:
        """Read a non-2xx error body without reclassifying HTTP errors as retryable."""
        try:
            return await response.text()
        except (aiohttp.ClientPayloadError,
                aiohttp.ServerDisconnectedError,
                aiohttp.ClientConnectionError,
                asyncio.TimeoutError) as exc:
            logger.warning("Could not read error body from %s: %s", endpoint, exc)
            return "<body unreadable>"

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
            return await self._read_json(response, "/health")

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
            return await self._read_json(response, "/models")

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
                text = await self._read_error_text(response, "/project/new")
                raise GranSabioClientError(
                    f"Failed to reserve project: {text}",
                    status_code=response.status
                )
            data = await self._read_json_no_retry(response, "/project/new")
            return data["project_id"]

    async def start_project(self, project_id: str) -> Dict[str, Any]:
        """Activate/reactivate a project."""
        async with await self._request("POST", f"/project/start/{project_id}") as response:
            if response.status not in (200, 201):
                text = await self._read_error_text(response, f"/project/start/{project_id}")
                raise GranSabioClientError(
                    f"Failed to start project: {text}",
                    status_code=response.status
                )
            return await self._read_json_no_retry(response, f"/project/start/{project_id}")

    async def stop_project(self, project_id: str) -> Dict[str, Any]:
        """Cancel a project and all its active sessions."""
        async with await self._request("POST", f"/project/stop/{project_id}") as response:
            if response.status not in (200, 202):
                text = await self._read_error_text(response, f"/project/stop/{project_id}")
                raise GranSabioClientError(
                    f"Failed to stop project: {text}",
                    status_code=response.status
                )
            return await self._read_json_no_retry(response, f"/project/stop/{project_id}")

    # =========================================================================
    # Content Generation
    # =========================================================================

    async def generate(
        self,
        prompt: str,
        content_type: str = "article",
        generator_model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        min_words: Optional[int] = None,
        max_words: Optional[int] = None,
        qa_models: Optional[List[str]] = None,
        qa_layers: Optional[List[Dict[str, Any]]] = None,
        min_global_score: float = 8.0,
        max_iterations: int = 3,
        gran_sabio_model: Optional[str] = None,
        gran_sabio_fallback: bool = True,
        verbose: bool = True,
        show_query_costs: int = 0,
        show_query_stats: int = 0,
        project_id: Optional[str] = None,
        request_name: Optional[str] = None,
        json_output: bool = False,
        json_schema: Optional[Dict[str, Any]] = None,
        word_count_enforcement: Optional[Dict[str, Any]] = None,
        phrase_frequency: Optional[Dict[str, Any]] = None,
        lexical_diversity: Optional[Dict[str, Any]] = None,
        reasoning_effort: Optional[str] = None,
        thinking_budget_tokens: Optional[int] = None,
        timeout_seconds: Optional[float] = None,
        timeouts: Optional[Dict[str, Any]] = None,
        qa_timeout_retries: Optional[int] = None,
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
            max_tokens: Maximum tokens to generate. When omitted, the server
                uses the selected model's max output from model_specs.json
                (fallback: 8192).
            min_words: Minimum word count (optional)
            max_words: Maximum word count (optional)
            qa_models: Models for QA evaluation (required if qa_layers not empty)
            qa_layers: QA evaluation layers (empty list = bypass QA)
            min_global_score: Minimum score for approval
            max_iterations: Max generation attempts
            gran_sabio_model: Model for escalation and arbitration
            gran_sabio_fallback: Allow regeneration on exhausted iterations
            verbose: Enable detailed logging
            show_query_costs: Cost reporting level (0=off, 1=summary, 2=detailed)
            show_query_stats: Execution stats level (0=off, 1=summary, 2=detailed, 3=per-call)
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
            "temperature": temperature,
            "min_global_score": min_global_score,
            "max_iterations": max_iterations,
            "gran_sabio_fallback": gran_sabio_fallback,
            "verbose": verbose,
        }

        # Optional parameters
        if min_words is not None:
            payload["min_words"] = min_words
        if max_words is not None:
            payload["max_words"] = max_words
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if generator_model is not None:
            payload["generator_model"] = generator_model
        if qa_models is not None:
            payload["qa_models"] = qa_models
        if qa_layers is not None:
            payload["qa_layers"] = qa_layers
        if gran_sabio_model is not None:
            payload["gran_sabio_model"] = gran_sabio_model
        if show_query_costs:
            payload["show_query_costs"] = show_query_costs
        if show_query_stats:
            payload["show_query_stats"] = show_query_stats
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
        if timeout_seconds is not None:
            payload["timeout_seconds"] = timeout_seconds
        if timeouts is not None:
            payload["timeouts"] = timeouts
        if qa_timeout_retries is not None:
            payload["qa_timeout_retries"] = qa_timeout_retries

        # Additional kwargs
        payload.update(kwargs)

        # Start generation
        async with await self._request("POST", "/generate", json_data=payload) as response:
            if response.status != 200:
                text = await self._read_error_text(response, "/generate")
                error_details = {"payload_metadata": payload_error_metadata(payload)}
                raise GranSabioClientError(
                    f"Generation request failed: {text}",
                    status_code=response.status,
                    details=error_details,
                )

            result = await self._read_json_no_retry(response, "/generate")

        # Handle preflight rejection
        if result.get("status") in {"preflight_rejected", "rejected"} and result.get("preflight_feedback"):
            feedback = result.get("preflight_feedback", {})
            raise GranSabioClientError(
                f"Request rejected by preflight: {feedback.get('user_feedback', 'Unknown reason')}",
                details={"preflight_feedback": feedback}
            )

        if result.get("status") == "auto_qa_rejected" and result.get("auto_qa_feedback"):
            feedback = result.get("auto_qa_feedback", {})
            raise GranSabioClientError(
                f"Request rejected by Auto-QA: {feedback.get('message', 'Unknown reason')}",
                details={"auto_qa_feedback": feedback}
            )

        session_id = result.get("session_id")
        if not session_id:
            raise GranSabioClientError("No session_id in response", details=result)

        # Return immediately if not waiting
        if not wait_for_completion:
            return result

        effective_max_wait = max_wait
        if max_wait == self.DEFAULT_MAX_WAIT:
            requested_wait = timeout_seconds
            if requested_wait is None and isinstance(timeouts, dict):
                requested_wait = timeouts.get("generation_seconds") or timeouts.get("default_seconds")
            if requested_wait is not None:
                effective_max_wait = float(requested_wait)

        # Poll for completion
        return await self.wait_for_result(
            session_id,
            poll_interval=poll_interval,
            max_wait=effective_max_wait,
            on_status=on_status
        )

    async def get_status(self, session_id: str) -> Dict[str, Any]:
        """Get current status of a generation session."""
        endpoint = f"/status/{session_id}"
        async with await self._request("GET", endpoint) as response:
            if response.status in _RETRYABLE_POLL_STATUS_CODES:
                raise TransientGranSabioError(
                    f"Transient HTTP {response.status} getting status of session {session_id}",
                    status_code=response.status,
                )
            if response.status != 200:
                text = await self._read_error_text(response, endpoint)
                raise GranSabioClientError(
                    f"Failed to get status: {text}",
                    status_code=response.status
                )
            return await self._read_json(response, endpoint)

    async def get_result(self, session_id: str) -> Dict[str, Any]:
        """Get the result of a completed generation session."""
        endpoint = f"/result/{session_id}"
        async with await self._request("GET", endpoint) as response:
            if response.status == 202:
                return {"status": "in_progress", "session_id": session_id}

            if response.status in _RETRYABLE_POLL_STATUS_CODES:
                raise TransientGranSabioError(
                    f"Transient HTTP {response.status} getting result of session {session_id}",
                    status_code=response.status,
                )

            if response.status != 200:
                text = await self._read_error_text(response, endpoint)
                raise GranSabioClientError(
                    f"Failed to get result: {text}",
                    status_code=response.status
                )

            return await self._read_json(response, endpoint)

    async def stop_session(self, session_id: str) -> Dict[str, Any]:
        """Cancel an active generation session."""
        endpoint = f"/stop/{session_id}"
        async with await self._request("POST", endpoint) as response:
            if response.status not in (200, 202):
                text = await self._read_error_text(response, endpoint)
                raise GranSabioClientError(
                    f"Failed to stop session: {text}",
                    status_code=response.status
                )
            return await self._read_json_no_retry(response, endpoint)

    async def _call_with_retry(
        self,
        op_name: str,
        session_id: str,
        func: Callable[[], Awaitable[_RetryT]],
        start_time: float,
        max_wait: float,
    ) -> _RetryT:
        """
        Invoke an idempotent polling operation (get_status / get_result / raw
        GET /result) with exponential backoff on transient network errors.
        Non-transient errors propagate immediately. Retry sleeps count against
        the deadline implied by ``start_time + max_wait``; if the budget
        expires mid-sleep, the call is aborted.

        The retry budget is whatever the caller chose by passing ``start_time``.
        Callers that pass a fresh ``start_time`` per invocation (as
        ``_fetch_result_polling`` does) give each call its own full retry
        budget. Callers that pass a shared ``start_time`` (as
        ``wait_for_result`` does) make retries compete for the overall deadline
        -- intentional, so long-stalled sessions don't silently extend past
        ``max_wait`` via retries.
        """
        retries = len(_POLL_RETRY_BACKOFF_SECONDS)
        last_err: Optional[TransientGranSabioError] = None

        # Initial attempt (not counted as a retry in log messages).
        try:
            return await func()
        except TransientGranSabioError as err:
            last_err = err

        # Retry loop: iterate directly over the backoff curve so retry_number
        # naturally indexes into it without an awkward `+ 1`.
        for retry_number, sleep_for in enumerate(_POLL_RETRY_BACKOFF_SECONDS, start=1):
            logger.warning(
                "Transient error polling session %s via %s (retry %d/%d): %s. "
                "Retrying in %ds.",
                session_id, op_name, retry_number, retries, last_err, sleep_for,
            )

            # Respect the overall max_wait budget. If the backoff would push us
            # past the deadline, abort now rather than sleep past it.
            elapsed = time.monotonic() - start_time
            if elapsed + sleep_for > max_wait:
                assert last_err is not None
                raise GranSabioClientError(
                    f"Timed out waiting for session {session_id} after {max_wait}s "
                    f"while backing off from transient error: {last_err}. "
                    f"Session {session_id} may still be running on the server. "
                    f"Check with GET /result/{session_id} or GET /status/{session_id}."
                ) from last_err
            await asyncio.sleep(sleep_for)

            try:
                return await func()
            except TransientGranSabioError as err:
                last_err = err

        assert last_err is not None
        logger.error(
            "Transient error polling session %s via %s exhausted %d retries: %s. "
            "Session may still be running on the server. "
            "Check with GET /result/%s or GET /status/%s.",
            session_id, op_name, retries, last_err, session_id, session_id,
        )
        raise GranSabioClientError(
            f"Transient network errors exhausted retries for session {session_id} "
            f"during {op_name}: {last_err}. Session {session_id} may still be "
            f"running on the server. Check with GET /result/{session_id} or "
            f"GET /status/{session_id}."
        ) from last_err

    async def wait_for_result(
        self,
        session_id: str,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        max_wait: float = DEFAULT_MAX_WAIT,
        on_status: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """
        Poll until generation completes and return result.

        Transient failures during status or result polling are retried with
        exponential backoff (_POLL_RETRY_BACKOFF_SECONDS) before giving up:
        network errors (connection drops, timeouts, mid-response protocol
        errors) and retryable HTTP statuses (202/425/429/503, e.g. a brief
        proxy flap or rate limit). Retry sleeps count against max_wait.
        Other non-transient errors (HTTP 4xx/5xx, etc.) fail fast.

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

            status = await self._call_with_retry(
                "get_status",
                session_id,
                lambda: self.get_status(session_id),
                start_time,
                max_wait,
            )

            if on_status:
                on_status(status)

            current_status = status.get("status", "")

            if current_status in {"completed", "rejected", "failed", "cancelled"}:
                final = await self._call_with_retry(
                    "get_result",
                    session_id,
                    lambda: self.get_result(session_id),
                    start_time,
                    max_wait,
                )
                if final.get("status") == "in_progress":
                    raise GranSabioClientError(
                        f"Generation returned terminal status '{current_status}' but final result is not ready yet",
                        details=status,
                    )
                validate_result(final)
                return final

            # Bound the poll sleep by the remaining max_wait budget. Without
            # this, a full poll_interval sleep can push total wait up to
            # poll_interval past max_wait, since the timeout check only runs
            # at loop top.
            remaining = max_wait - (time.monotonic() - start_time)
            if remaining <= 0:
                # Fall through: next iteration's max_wait check will raise.
                continue
            await asyncio.sleep(min(poll_interval, remaining))

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
                text_response = await self._read_error_text(response, "/analysis/lexical-diversity")
                raise GranSabioClientError(
                    f"Lexical diversity analysis failed: {text_response}",
                    status_code=response.status
                )
            return await self._read_json_no_retry(response, "/analysis/lexical-diversity")

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
                text_response = await self._read_error_text(response, "/analysis/repetition")
                raise GranSabioClientError(
                    f"Repetition analysis failed: {text_response}",
                    status_code=response.status
                )
            return await self._read_json_no_retry(response, "/analysis/repetition")

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
                text_response = await self._read_error_text(response, "/analysis/repetition")
                raise GranSabioClientError(
                    f"Unigram repetition analysis failed: {text_response}",
                    status_code=response.status,
                )
            data = await self._read_json_no_retry(response, "/analysis/repetition")

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
                text_response = await self._read_error_text(response, "/analysis/repetition")
                raise GranSabioClientError(
                    f"Positional bias analysis failed: {text_response}",
                    status_code=response.status,
                )
            data = await self._read_json_no_retry(response, "/analysis/repetition")

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

        # Multipart uploads cannot reuse the default JSON Content-Type header,
        # but they must go through _request so connection/timeout failures map
        # to the documented GranSabioClientError/TransientGranSabioError
        # hierarchy instead of leaking raw aiohttp exceptions.
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self.provider_keys)

        async with await self._request(
            "POST",
            "/attachments",
            headers=headers,
            data=data,
        ) as response:
            if response.status != 200:
                text = await self._read_error_text(response, "/attachments")
                raise GranSabioClientError(
                    f"Attachment upload failed: {text}",
                    status_code=response.status
                )
            return await self._read_json_no_retry(response, "/attachments")

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
                text = await self._read_error_text(response, "/attachments/base64")
                raise GranSabioClientError(
                    f"Attachment upload failed: {text}",
                    status_code=response.status
                )
            return await self._read_json_no_retry(response, "/attachments/base64")

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    async def generate_json(
        self,
        prompt: str,
        schema: Dict[str, Any],
        model: Optional[str] = None,
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
        if isinstance(content, dict):
            result["parsed_content"] = content
        elif isinstance(content, str):
            try:
                result["parsed_content"] = json.loads(content)
            except (json.JSONDecodeError, ValueError):
                result["parsed_content"] = None
                result["parse_error"] = True

        return result

    async def generate_fast(
        self,
        prompt: str,
        model: Optional[str] = None,
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
        # Per-invocation retry budget for transient network errors on the
        # idempotent GET /result call. When no outer timeout is set, the retry
        # layer still caps its own backoff sleeps (they remain bounded by
        # _POLL_RETRY_BACKOFF_SECONDS).
        retry_max_wait = float(timeout_seconds) if timeout_seconds else float("inf")

        async def _fetch_once() -> Tuple[int, Optional[Dict[str, Any]], str, str]:
            """One GET /result attempt. Returns (status, parsed_json, detail, text_body).

            Transient failures propagate as TransientGranSabioError so the retry
            layer can handle them. Body-read transients include ClientPayloadError,
            ServerDisconnectedError, ClientConnectionError, and asyncio.TimeoutError;
            additional transients may originate in _request. text_body is only
            populated when the caller will need it for an error message
            (non-retryable non-200 status).
            """
            transient_body_errors = (
                aiohttp.ClientPayloadError,
                aiohttp.ServerDisconnectedError,
                aiohttp.ClientConnectionError,
                asyncio.TimeoutError,
            )
            async with await self._request("GET", result_url) as response:
                status_code = response.status
                if status_code == 200:
                    try:
                        body = await response.json()
                    except transient_body_errors as exc:
                        # Mid-response truncation on the success branch is
                        # transient for an idempotent GET -- let the retry
                        # layer try again.
                        raise TransientGranSabioError(
                            f"Transient protocol error reading /result/{session_id}: {exc}"
                        ) from exc
                    except (aiohttp.ContentTypeError, json.JSONDecodeError, ValueError) as exc:
                        raise TransientGranSabioError(
                            f"Transient protocol error reading /result/{session_id}: {exc}"
                        ) from exc
                    return status_code, body, "", ""

                parsed: Optional[Dict[str, Any]] = None
                detail = ""
                is_retryable_status = status_code in (202, 425, 503, 400)
                try:
                    parsed = await response.json()
                    if isinstance(parsed, dict):
                        detail = str(parsed.get("detail", ""))
                except transient_body_errors as exc:
                    if is_retryable_status:
                        raise TransientGranSabioError(
                            f"Transient protocol error reading /result/{session_id}: {exc}"
                        ) from exc
                    parsed = None
                except (aiohttp.ContentTypeError, json.JSONDecodeError, ValueError):
                    parsed = None

                text_body = ""
                # Only bother reading the body text when we may need it for an
                # error message (non-retryable status). Retryable statuses skip
                # this to avoid wasted I/O.
                if not is_retryable_status:
                    try:
                        text_body = await response.text()
                    except transient_body_errors as exc:
                        logger.warning("Could not read /result/%s error body: %s", session_id, exc)
                        text_body = "<body unreadable>"

                return status_code, parsed, detail, text_body

        while True:
            call_start = time.monotonic()
            status_code, _payload, detail_message, text_body = await self._call_with_retry(
                "get_result",
                session_id,
                _fetch_once,
                call_start,
                retry_max_wait,
            )

            if status_code == 200:
                final_result = _payload if isinstance(_payload, dict) else {}
                logger.info(
                    "Final result retrieved for session %s: status = %s",
                    session_id,
                    final_result.get("status", "unknown"),
                )
                return final_result

            should_retry = False
            if status_code in (202, 425, 503, 400):
                lower_detail = detail_message.lower()
                if "not finished" in lower_detail or "still in progress" in lower_detail:
                    should_retry = True
                elif status_code == 202 and not detail_message:
                    should_retry = True

            if not should_retry:
                raise GranSabioClientError(
                    f"Result retrieval failed: {status_code} - {text_body}",
                    status_code=status_code,
                )

            # Retryable (server reports in-progress) -- check deadline.
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
            # Clamp the sleep to remaining budget (not to a 1s floor) so we
            # never overshoot the deadline. If remaining is 0 the sleep is a
            # no-op and the next iteration's deadline check fires and either
            # extends or aborts.
            sleep_interval = poll_interval if remaining is None else max(0.0, min(poll_interval, remaining))
            await asyncio.sleep(sleep_interval)

    def _record_stream_error(
        self,
        stream: str,
        url: str,
        exc: Exception,
        error_callback: Optional[Callable[[str, Exception], None]],
    ) -> None:
        """Surface a streaming failure: log, client state and callback.

        Never raises: the generation keeps running on the server and the
        result is still fetched by polling. Silence is what made the dead
        stream endpoint invisible for months, so this is a warning, not a
        debug line.
        """
        logger.warning("Stream '%s' failed (%s): %s", stream, url, exc)
        self.last_stream_errors.append(
            StreamError(
                stream=stream,
                url=url,
                error=str(exc),
                error_type=type(exc).__name__,
            ).as_dict()
        )
        if error_callback is None:
            return
        try:
            error_callback(stream, exc)
        except Exception:
            logger.exception("error_callback raised while reporting a stream failure")

    async def _stream_session_events(
        self,
        session_id: str,
        callbacks: StreamCallbacks,
        activity_monitor: ActivityMonitor,
        result_queue: "asyncio.Queue[str]",
        timeout: float,
        verbose: bool,
        error_callback: Optional[Callable[[str, Exception], None]] = None,
    ) -> None:
        """Consume ``GET /stream/{session_id}`` and route events to callbacks.

        One connection carries everything the session produces (status,
        generation drafts and commits, QA / preflight / Arbiter / Gran Sabio
        text, thinking, control events, ``session_end``), each emission
        labelled with ``content_status``/``iteration``/``attempt_id``. The
        terminal status is put onto ``result_queue`` when the session ends.

        Failures are surfaced through ``_record_stream_error`` (warning log,
        ``last_stream_errors``, ``error_callback``) and never abort the run:
        the result is still fetched by polling.
        """
        url = f"{self.base_url}/stream/{session_id}"
        router = SessionEventRouter(callbacks, activity=activity_monitor)
        parser = SSEParser()
        try:
            timeout_cfg = aiohttp.ClientTimeout(total=timeout, connect=30)
            async with self.session.get(url, headers=self._headers(), timeout=timeout_cfg) as response:
                if response.status != 200:
                    body = await response.text()
                    raise GranSabioClientError(
                        f"Session stream returned status {response.status}: {body[:200]}",
                        status_code=response.status,
                    )
                activity_monitor.mark(message="session_stream_connected")

                terminal: Optional[str] = None
                async for chunk_bytes in response.content.iter_any():
                    for sse_event in parser.feed(chunk_bytes):
                        if sse_event.is_comment:
                            activity_monitor.mark(message="heartbeat")
                            continue
                        payload = decode_event_json(sse_event)
                        if payload is None:
                            continue
                        if verbose:
                            logger.debug("Session event: %s", payload.get("type"))
                        terminal = router.handle(payload) or terminal
                    if terminal:
                        break

                if not terminal:
                    for sse_event in parser.close():
                        payload = decode_event_json(sse_event)
                        if payload is not None:
                            terminal = router.handle(payload) or terminal

                if terminal:
                    await result_queue.put(terminal)

        except asyncio.CancelledError:
            pass
        except Exception as exc:
            self._record_stream_error("session", url, exc, error_callback)

    async def generate_streaming(
        self,
        prompt: str,
        *,
        content_type: str = "article",
        generator_model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        min_words: Optional[int] = None,
        max_words: Optional[int] = None,
        qa_models: Optional[List[str]] = None,
        qa_layers: Optional[List[Dict[str, Any]]] = None,
        min_global_score: float = 8.0,
        max_iterations: int = 3,
        gran_sabio_model: Optional[str] = None,
        gran_sabio_fallback: bool = True,
        verbose: bool = True,
        show_query_costs: int = 0,
        show_query_stats: int = 0,
        project_id: Optional[str] = None,
        request_name: Optional[str] = None,
        json_output: bool = False,
        json_schema: Optional[Dict[str, Any]] = None,
        word_count_enforcement: Optional[Dict[str, Any]] = None,
        phrase_frequency: Optional[Dict[str, Any]] = None,
        lexical_diversity: Optional[Dict[str, Any]] = None,
        reasoning_effort: Optional[str] = None,
        thinking_budget_tokens: Optional[int] = None,
        timeout_seconds: Optional[float] = None,
        timeouts: Optional[Dict[str, Any]] = None,
        qa_timeout_retries: Optional[int] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
        content_callback: Optional[Callable[[str], None]] = None,
        content_reset_callback: Optional[Callable[[], None]] = None,
        qa_callback: Optional[Callable[[str, str, str], None]] = None,
        on_session_start: Optional[Callable[[str], None]] = None,
        error_callback: Optional[Callable[[str, Exception], None]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate content with real-time SSE streaming.

        Starts generation via ``generate(wait_for_completion=False)`` and then
        opens ONE SSE connection, ``GET /stream/{session_id}``, which carries
        everything the session produces already labelled: status snapshots and
        changes, generation drafts and accepted commits, QA / preflight /
        Arbiter / Gran Sabio text, thinking, control events and the terminal
        ``session_end``.

        Streaming errors do NOT abort the run: the stream is the visibility
        channel and the result still arrives through polling. Each failure is
        logged at warning level, appended to ``self.last_stream_errors`` (a
        list of dicts with ``stream``/``url``/``error``/``error_type``) and
        reported to ``error_callback`` when one is given.

        Args:
            prompt: The generation prompt.
            content_type: Type of content (article, creative, json, etc.).
            generator_model: AI model to use for generation.
            temperature: Creativity level (0.0-2.0).
            max_tokens: Maximum tokens to generate. When omitted, the server
                uses the selected model's max output from model_specs.json
                (fallback: 8192).
            min_words: Minimum word count (optional).
            max_words: Maximum word count (optional).
            qa_models: Models for QA evaluation.
            qa_layers: QA evaluation layers (empty list = bypass QA).
            min_global_score: Minimum score for approval.
            max_iterations: Max generation attempts.
            gran_sabio_model: Model for escalation and arbitration.
            gran_sabio_fallback: Allow regeneration on exhausted iterations.
            verbose: Enable detailed logging.
            show_query_costs: Cost reporting level (0=off, 1=summary, 2=detailed).
            show_query_stats: Execution stats level (0=off, 1=summary, 2=detailed, 3=per-call).
            project_id: Group with existing project.
            request_name: Human-readable request label.
            json_output: Expect JSON output.
            json_schema: JSON Schema for structured output.
            word_count_enforcement: Word count validation config.
            phrase_frequency: Phrase repetition guard config.
            lexical_diversity: Vocabulary diversity guard config.
            reasoning_effort: For reasoning models (low/medium/high).
            thinking_budget_tokens: For Claude thinking models (min 1024).
            progress_callback: Called with lifecycle messages (str). Never
                called with an empty string.
            content_callback: Called with generation text as it streams. Drafts
                arrive as deltas to append; the accepted commit arrives as the
                FULL snapshot of its attempt and replaces what came before it.
            content_reset_callback: Called with no arguments when everything
                delivered so far for the phase becomes dead text: right before
                the delivery that REPLACES it (accepted commit after its drafts,
                or a new attempt superseding the previous one), or on its own
                when the attempt closes with no body to replace it (a
                ``filtered`` provider stop). Provide it to accumulate
                correctly - clear the buffer, then append what arrives next.
                Omitting it keeps the append-everything behaviour, which
                concatenates the dead draft with its replacement (and produces
                unparseable text when ``json_output`` is set).
            qa_callback: Called with ``(text, agent, status)`` for every other
                text channel (QA, preflight, Arbiter, Gran Sabio). ``agent`` is
                the producing model when the engine reports one, otherwise the
                phase; ``status`` is the content label (``draft``,
                ``accepted``, ``rejected``, ``filtered``) or ``thinking``.
            on_session_start: Called with session_id immediately after the
                generation session is created. May be sync or async.
            error_callback: Called with ``(stream_name, exception)`` when the
                event stream fails. The run continues on polling.
            **kwargs: Additional parameters passed to API.

        Returns:
            Final generation result dict with content, score, and metadata.
        """
        # Errors of a previous run must not be mistaken for this one's.
        self.last_stream_errors = []

        # Build the kwargs dict for generate(), forwarding all explicit params
        gen_kwargs: Dict[str, Any] = {
            "content_type": content_type,
            "temperature": temperature,
            "min_words": min_words,
            "max_words": max_words,
            "qa_models": qa_models,
            "qa_layers": qa_layers,
            "min_global_score": min_global_score,
            "max_iterations": max_iterations,
            "gran_sabio_fallback": gran_sabio_fallback,
            "verbose": verbose,
            "show_query_costs": show_query_costs,
            "show_query_stats": show_query_stats,
            "project_id": project_id,
            "request_name": request_name,
            "json_output": json_output,
            "json_schema": json_schema,
            "word_count_enforcement": word_count_enforcement,
            "phrase_frequency": phrase_frequency,
            "lexical_diversity": lexical_diversity,
            "reasoning_effort": reasoning_effort,
            "thinking_budget_tokens": thinking_budget_tokens,
            "timeout_seconds": timeout_seconds,
            "timeouts": timeouts,
            "qa_timeout_retries": qa_timeout_retries,
            "wait_for_completion": False,
        }
        if generator_model is not None:
            gen_kwargs["generator_model"] = generator_model
        if gran_sabio_model is not None:
            gen_kwargs["gran_sabio_model"] = gran_sabio_model
        if max_tokens is not None:
            gen_kwargs["max_tokens"] = max_tokens
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
            "timeout_seconds": timeout_seconds,
            "timeouts": timeouts,
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

        # Launch the single session event stream
        tasks = [
            asyncio.create_task(
                self._stream_session_events(
                    session_id,
                    StreamCallbacks(
                        progress=progress_callback,
                        content=content_callback,
                        qa=qa_callback,
                        content_reset=content_reset_callback,
                    ),
                    activity_monitor,
                    result_queue,
                    stream_timeout,
                    verbose,
                    error_callback,
                ),
                name=f"session-stream-{session_id[:8]}",
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
                if all(task.done() for task in tasks) and result_queue.empty():
                    # The event stream ended without a terminal signal (it
                    # failed, or the server closed it): there is nothing left
                    # to wait for, so go straight to polling instead of
                    # burning the whole inactivity budget in silence.
                    logger.debug(
                        "Session stream ended without a terminal signal for %s",
                        session_id,
                    )
                    break
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


# Module-level convenience function
async def create_client(base_url: Optional[str] = None, **kwargs) -> AsyncGranSabioClient:
    """Create and connect an async Gran Sabio client instance."""
    client = AsyncGranSabioClient(base_url=base_url, **kwargs)
    await client.connect()
    return client
