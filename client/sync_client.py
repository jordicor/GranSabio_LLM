"""
Synchronous Gran Sabio LLM Client
=================================

Thread-safe synchronous client for the Gran Sabio LLM Engine API.
Suitable for scripts, CLI tools, and simple integrations.

For async applications, use AsyncGranSabioClient instead.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar
from urllib.parse import quote

import requests
from requests import exceptions as requests_exceptions

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
    SAAS_REQUEST_ID_HEADER,
    STREAM_ACTIVITY_CHECK_SECONDS,
    ActivityMonitor,
    admission_rejection_error,
    compute_generation_timeout,
    compute_token_budgets,
    generation_http_error_details,
    resolve_model_token_fallback,
    session_list_params,
    validate_result,
    validate_saas_request_id,
)
from ._sse import (
    SessionEventRouter,
    SSEEvent,
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

# Generic return type for _call_with_retry so it can wrap any idempotent GET
# (requests.Response, parsed JSON dict, etc.).
_RetryT = TypeVar("_RetryT")


class GranSabioClient:
    """
    Synchronous client for Gran Sabio LLM Engine.

    Usage:
        client = GranSabioClient()

        # Simple generation (no QA)
        result = client.generate(
            prompt="Write a haiku about coding",
            qa_layers=[]
        )

        # Generation with QA
        result = client.generate(
            prompt="Write a professional article",
            qa_layers=[{"name": "Clarity", "min_score": 7.0, ...}],
            min_global_score=7.5
        )

        # Streaming generation with callbacks
        result = client.generate_streaming(
            prompt="Write a detailed article",
            progress_callback=lambda msg: print(f"Progress: {msg}"),
            content_callback=lambda chunk: print(chunk, end=""),
        )
    """

    DEFAULT_BASE_URL = "http://localhost:8000"
    DEFAULT_TIMEOUT = (30, 12000)  # (connect, read)
    DEFAULT_POLL_INTERVAL = 2.0
    DEFAULT_MAX_WAIT = 12000.0

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: tuple = DEFAULT_TIMEOUT,
        provider_keys: Optional[Dict[str, str]] = None,
        request_headers: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize the Gran Sabio client.

        Args:
            base_url: API base URL (default: http://localhost:8000 or GRANSABIO_BASE_URL env)
            api_key: Optional API key (None uses GRANSABIO_API_KEY; '' sends no Bearer)
            timeout: Request timeout tuple (connect_timeout, read_timeout)
            provider_keys: Optional dict of provider API keys as HTTP headers.
                Only keys passed explicitly here are sent to the server; provider
                credentials are never loaded implicitly from the environment.
            request_headers: Extra HTTP headers copied for this client instance.
                Authorization and Content-Type are reserved; provider_keys must not overlap.
        """
        self.base_url = (base_url or os.getenv("GRANSABIO_BASE_URL", self.DEFAULT_BASE_URL)).rstrip("/")
        self.api_key = os.getenv("GRANSABIO_API_KEY") if api_key is None else api_key
        self.timeout = timeout
        self.provider_keys = dict(provider_keys or {})
        self.request_headers = dict(request_headers or {})
        reserved = {"authorization", "content-type", *(name.lower() for name in self.provider_keys)}
        names = [name.lower() for name in self.request_headers]
        if reserved.intersection(names) or len(names) != len(set(names)):
            raise ValueError("request_headers must not repeat Authorization, Content-Type or provider_keys")
        self._model_token_cache: Dict[str, int] = {}
        self._model_timeout_cache: Dict[str, int] = {}
        # Streaming failures of the last generate_streaming call. A stream is
        # the visibility channel, not the result channel: failures are
        # surfaced here (and through error_callback) instead of aborting a
        # generation the engine is still running.
        self.last_stream_errors: List[Dict[str, str]] = []

    def _headers(self, *, json_content: bool = True) -> Dict[str, str]:
        """Build request headers including provider API keys."""
        headers = {"Content-Type": "application/json"} if json_content else {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self.provider_keys)
        headers.update(self.request_headers)
        return headers

    def _request(
        self,
        method: str,
        endpoint: str,
        json_data: Optional[Dict] = None,
        **kwargs
    ) -> requests.Response:
        """Make an HTTP request with error handling."""
        url = f"{self.base_url}{endpoint}"
        kwargs.setdefault("timeout", self.timeout)
        kwargs.setdefault("headers", self._headers())

        try:
            response = requests.request(method, url, json=json_data, **kwargs)
            return response
        # SSLError must be caught BEFORE ConnectionError because
        # requests.exceptions.SSLError is a subclass of ConnectionError; without
        # this explicit branch, genuine TLS problems (bad cert, hostname
        # mismatch, unsupported TLS version) would be silently reclassified as
        # transient and retried. They are not transient -- fail fast.
        except requests_exceptions.SSLError as e:
            raise GranSabioClientError(
                f"TLS/SSL error connecting to {self.base_url}: {e}"
            ) from e
        except requests_exceptions.ConnectionError as e:
            raise TransientGranSabioError(
                f"Cannot connect to Gran Sabio API at {self.base_url}. "
                "Ensure the server is running."
            ) from e
        except requests_exceptions.Timeout as e:
            raise TransientGranSabioError(
                f"Request to {endpoint} timed out after {self.timeout}s"
            ) from e
        # ChunkedEncodingError and ContentDecodingError name specific mid-response
        # failure modes (VPN flap or server reset while the body is still
        # streaming; gzip stream truncated mid-transfer). They are genuinely
        # retryable on an idempotent GET. urllib3's ProtocolError and
        # ReadTimeoutError surface via requests as ConnectionError /
        # ChunkedEncodingError, so covering these two classes is sufficient.
        except (requests_exceptions.ChunkedEncodingError,
                requests_exceptions.ContentDecodingError) as e:
            raise TransientGranSabioError(
                f"Transient protocol error on request to {endpoint}: {e}"
            ) from e
        except requests_exceptions.RequestException as e:
            raise GranSabioClientError(f"Request failed: {e}") from e

    def _read_json(self, response: requests.Response, endpoint: str) -> Any:
        """Read JSON body, converting transient mid-read failures to TransientGranSabioError."""
        try:
            return response.json()
        except (
            requests_exceptions.ChunkedEncodingError,
            requests_exceptions.ContentDecodingError,
            requests_exceptions.ConnectionError,
            requests_exceptions.Timeout,
            json.JSONDecodeError,
            ValueError,
        ) as exc:
            raise TransientGranSabioError(
                f"Transient protocol error reading JSON from {endpoint}: {exc}"
            ) from exc

    def _read_json_no_retry(self, response: requests.Response, endpoint: str) -> Any:
        """Read JSON body for non-idempotent requests without making it retryable."""
        try:
            return response.json()
        except (
            requests_exceptions.ChunkedEncodingError,
            requests_exceptions.ContentDecodingError,
            requests_exceptions.ConnectionError,
            requests_exceptions.Timeout,
            json.JSONDecodeError,
            ValueError,
        ) as exc:
            raise GranSabioClientError(
                f"Failed to read JSON response from {endpoint}; "
                f"the operation may have been accepted by the server: {exc}"
            ) from exc

    def _read_error_text(self, response: requests.Response, endpoint: str) -> str:
        """Read a non-2xx error body without reclassifying HTTP errors as retryable."""
        try:
            return response.text
        except (
            requests_exceptions.ChunkedEncodingError,
            requests_exceptions.ContentDecodingError,
            requests_exceptions.ConnectionError,
            requests_exceptions.Timeout,
        ) as exc:
            logger.warning("Could not read error body from %s: %s", endpoint, exc)
            return "<body unreadable>"

    # =========================================================================
    # Health & Info
    # =========================================================================

    def health_check(self) -> Dict[str, Any]:
        """
        Check API health status.

        Returns:
            Dict with status, active_sessions, timestamp
        """
        response = self._request("GET", "/health")
        if response.status_code != 200:
            raise GranSabioClientError(
                f"Health check failed: {response.status_code}",
                status_code=response.status_code
            )
        return self._read_json(response, "/health")

    def get_models(self) -> Dict[str, Any]:
        """
        Get available AI models organized by provider.

        Returns:
            Dict mapping provider names to lists of model specifications
        """
        response = self._request("GET", "/models")
        if response.status_code != 200:
            raise GranSabioClientError(
                f"Failed to get models: {response.status_code}",
                status_code=response.status_code
            )
        return self._read_json(response, "/models")

    def evaluate_system_one(
        self,
        state: Any,
        questions: Dict[str, Any],
        model: str,
    ) -> Dict[str, Any]:
        """Evaluate typed System One questions. Does not create a generation session."""

        payload: Dict[str, Any] = {"state": state, "questions": questions, "model": model}
        response = self._request("POST", "/systemone", json_data=payload)
        body = self._read_json(response, "/systemone")
        if response.status_code != 200:
            detail = body.get("detail") if isinstance(body, dict) else body
            raise GranSabioClientError(
                f"System One evaluation failed: {detail}",
                status_code=response.status_code,
            )
        if not isinstance(body, dict) or "answers" not in body:
            raise GranSabioClientError("System One evaluation returned an invalid body")
        return body

    def is_available(self) -> bool:
        """Check if the API is reachable and healthy."""
        try:
            health = self.health_check()
            return health.get("status") == "healthy"
        except GranSabioClientError:
            return False

    def get_info(self) -> Dict[str, Any]:
        """
        Get API information.

        Returns a dict with service name, version, and status.
        """
        health = self.health_check()
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

    def reserve_project(self, project_id: Optional[str] = None) -> str:
        """
        Reserve a project identifier for grouping related sessions.

        Args:
            project_id: Optional custom ID (auto-generated if not provided)

        Returns:
            The reserved project_id
        """
        payload = {"project_id": project_id} if project_id else {}
        response = self._request("POST", "/project/new", json_data=payload)

        if response.status_code not in (200, 201):
            text = self._read_error_text(response, "/project/new")
            raise GranSabioClientError(
                f"Failed to reserve project: {text}",
                status_code=response.status_code
            )

        data = self._read_json_no_retry(response, "/project/new")
        return data["project_id"]

    def start_project(self, project_id: str) -> Dict[str, Any]:
        """Activate/reactivate a project."""
        response = self._request("POST", f"/project/start/{project_id}")
        if response.status_code not in (200, 201):
            endpoint = f"/project/start/{project_id}"
            text = self._read_error_text(response, endpoint)
            raise GranSabioClientError(
                f"Failed to start project: {text}",
                status_code=response.status_code
            )
        return self._read_json_no_retry(response, f"/project/start/{project_id}")

    def stop_project(self, project_id: str) -> Dict[str, Any]:
        """Cancel a project and all its active sessions."""
        response = self._request("POST", f"/project/stop/{project_id}")
        if response.status_code not in (200, 202):
            endpoint = f"/project/stop/{project_id}"
            text = self._read_error_text(response, endpoint)
            raise GranSabioClientError(
                f"Failed to stop project: {text}",
                status_code=response.status_code
            )
        return self._read_json_no_retry(response, f"/project/stop/{project_id}")

    # =========================================================================
    # Content Generation
    # =========================================================================

    def generate(
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
        *,
        request_id: Optional[str] = None,
        shared_context: Optional[str] = None,
        prompt_cache: Optional[Dict[str, Any]] = None,
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
            request_id: Optional idempotency key (1-128 ASCII letters, digits
                or ``._:-``), sent as the ``X-GranSabio-SaaS-Request-ID``
                header of this POST only. Re-sending the same payload with the
                same key replays the original admission (same ``session_id``)
                instead of starting another session, while the server keeps the
                entry (process-local, 24 h). A different payload under a used
                key is rejected with HTTP 409. An invalid key raises
                ``ValueError`` before anything is sent.
            shared_context: Stable source data sent at user level before the task prompt.
            prompt_cache: Cache policy (auto/off/explicit/adaptive); breakpoints count Unicode code points in prompt.
            **kwargs: Additional parameters passed to API

        Returns:
            Generation result with content, score, and metadata

        Raises:
            TransientGranSabioError: HTTP 202 when the original admission
                under ``request_id`` is still in progress; ``details`` carries
                ``retry_after_seconds`` (when sent) and ``request_id``. Retry
                with the same payload and ``request_id``.
            GranSabioClientError: Any other non-200 answer. ``details`` carries
                ``json_error_detail`` (whether the engine itself answered with
                a JSON ``detail`` body, as opposed to a proxy or gateway page),
                ``retry_after_seconds`` when sent, and ``request_id`` when given.
                An HTTP 400, 409 or 422 the engine answered from the payload
                under ``request_id`` is stored and replayed for the key's
                lifetime, so a new attempt needs a new key (a preflight or
                Auto-QA rejection is an HTTP 200 result, replayed the same
                way). A capacity HTTP 503 (admission gate, unavailable local
                runtime, idempotency registry full) is not stored: re-send the
                same payload with the same key, after ``retry_after_seconds``
                when sent; transport failures, proxy or gateway answers and
                HTTP 202 are likewise safe to re-send under the same key.
        """
        if request_id is not None:
            validate_saas_request_id(request_id)

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

        if shared_context is not None:
            payload["shared_context"] = shared_context
        if prompt_cache is not None:
            payload["prompt_cache"] = prompt_cache

        # Additional kwargs
        payload.update(kwargs)

        headers = self._headers()
        if request_id is not None:
            headers[SAAS_REQUEST_ID_HEADER] = request_id

        # Start generation
        response = self._request("POST", "/generate", json_data=payload, headers=headers)

        if response.status_code != 200:
            text = self._read_error_text(response, "/generate")
            error_details = generation_http_error_details(
                payload, body_text=text, headers=response.headers, request_id=request_id,
            )
            if response.status_code == 202 and request_id is not None:
                # Only an idempotent replay answers 202: the original
                # admission under this request ID has not finished yet. A
                # POST without a key cannot be a replay and keeps the plain
                # error path below, so a caller retrying transient errors
                # never re-submits without an idempotency key.
                raise TransientGranSabioError(
                    "Generation admission for this request ID is still in progress "
                    "(HTTP 202); retry with the same payload and request_id.",
                    status_code=202,
                    details=error_details,
                )
            raise GranSabioClientError(
                f"Generation request failed: {text}",
                status_code=response.status_code,
                details=error_details,
            )

        result = self._read_json_no_retry(response, "/generate")

        # A preflight or Auto-QA rejection carries no session to wait for.
        rejection = admission_rejection_error(result)
        if rejection is not None:
            raise rejection

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
        return self.wait_for_result(
            session_id,
            poll_interval=poll_interval,
            max_wait=effective_max_wait,
            on_status=on_status
        )

    def get_status(self, session_id: str) -> Dict[str, Any]:
        """Get current status of a generation session."""
        response = self._request("GET", f"/status/{session_id}")
        if response.status_code in _RETRYABLE_POLL_STATUS_CODES:
            raise TransientGranSabioError(
                f"Transient HTTP {response.status_code} getting status of session {session_id}",
                status_code=response.status_code,
            )
        if response.status_code != 200:
            endpoint = f"/status/{session_id}"
            text = self._read_error_text(response, endpoint)
            raise GranSabioClientError(
                f"Failed to get status: {text}",
                status_code=response.status_code
            )
        return self._read_json(response, f"/status/{session_id}")

    def get_result(self, session_id: str) -> Dict[str, Any]:
        """Get the result of a completed generation session."""
        response = self._request("GET", f"/result/{session_id}")

        if response.status_code == 202:
            # Still in progress
            return {"status": "in_progress", "session_id": session_id}

        if response.status_code in _RETRYABLE_POLL_STATUS_CODES:
            raise TransientGranSabioError(
                f"Transient HTTP {response.status_code} getting result of session {session_id}",
                status_code=response.status_code,
            )

        if response.status_code != 200:
            endpoint = f"/result/{session_id}"
            text = self._read_error_text(response, endpoint)
            raise GranSabioClientError(
                f"Failed to get result: {text}",
                status_code=response.status_code
            )

        return self._read_json(response, f"/result/{session_id}")

    def get_session_details(self, session_id: str) -> Dict[str, Any]:
        """Read original request evidence without submitting or retrying generation.

        Pass the raw session ID; the SDK encodes it as one URL segment. This
        authenticated debugger response can contain private prompts and results.
        """
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("A non-empty session ID is required.")
        endpoint = f"/debugger/sessions/{quote(session_id, safe='')}"
        response = self._request("GET", endpoint)
        if response.status_code in _RETRYABLE_POLL_STATUS_CODES:
            raise TransientGranSabioError(
                f"Transient HTTP {response.status_code} getting session details.",
                status_code=response.status_code,
            )
        if response.status_code != 200:
            raise GranSabioClientError(
                f"Failed to get session details (HTTP {response.status_code}).",
                status_code=response.status_code,
            )
        return self._read_json(response, endpoint)

    def get_request_status(self, request_id: str, *, max_wait: float = 60.0) -> Dict[str, Any]:
        """Look up one idempotent generation submission by its request ID.

        Returns the server's reconciliation snapshot: ``{"found": False,
        "request_id": ...}`` when the server holds no entry (never sent, expired
        after 24 h, or lost on a server restart), otherwise ``found``,
        ``state`` (``pending`` / ``completed`` / ``error``), ``status``,
        ``session_id``, ``project_id`` and ``init_response``. An ``error``
        snapshot also carries ``error`` with the stored ``status_code`` and
        ``detail``. This is an idempotent GET: transient transport failures and
        HTTP 425/429/503 are retried with the polling backoff within
        ``max_wait`` seconds; other HTTP errors (400, 404, ...) raise
        ``GranSabioClientError`` immediately.
        """
        validate_saas_request_id(request_id)
        endpoint = f"/api/saas/requests/{quote(request_id, safe='')}"

        def read() -> Dict[str, Any]:
            response = self._request("GET", endpoint)
            if response.status_code in _RETRYABLE_POLL_STATUS_CODES:
                raise TransientGranSabioError(
                    f"Transient HTTP {response.status_code} looking up a generation request.",
                    status_code=response.status_code,
                )
            if response.status_code != 200:
                raise GranSabioClientError(
                    f"Failed to look up the generation request (HTTP {response.status_code}).",
                    status_code=response.status_code,
                )
            return self._read_json(response, endpoint)

        return self._call_with_retry(
            "get_request_status", request_id, read, time.monotonic(), max_wait,
            resource="request",
        )

    def list_sessions(
        self,
        limit: int = 25,
        offset: int = 0,
        project_id: Optional[str] = None,
        *,
        max_wait: float = 60.0,
    ) -> Dict[str, Any]:
        """List debugger sessions, most recently updated first.

        Returns the ``{"sessions": [...], "project_id": ...}`` body; each row
        has ``session_id``, ``created_at``, ``updated_at`` (UTC
        ``YYYY-MM-DD HH:MM:SS``), ``status``, ``phase``, ``project_id`` and
        ``request_name``. The server caps ``limit`` (100 by default) and answers
        404 when its debugger is disabled. Retries follow
        ``get_request_status``. Meant as secondary reconciliation evidence, not
        as a progress feed.
        """
        params = session_list_params(limit, offset, project_id)
        endpoint = "/debugger/sessions"

        def read() -> Dict[str, Any]:
            response = self._request("GET", endpoint, params=params)
            if response.status_code in _RETRYABLE_POLL_STATUS_CODES:
                raise TransientGranSabioError(
                    f"Transient HTTP {response.status_code} listing debugger sessions.",
                    status_code=response.status_code,
                )
            if response.status_code != 200:
                raise GranSabioClientError(
                    f"Failed to list debugger sessions (HTTP {response.status_code}).",
                    status_code=response.status_code,
                )
            return self._read_json(response, endpoint)

        return self._call_with_retry(
            "list_sessions", endpoint, read, time.monotonic(), max_wait,
            resource="listing",
        )

    def stop_session(self, session_id: str) -> Dict[str, Any]:
        """Cancel an active generation session."""
        response = self._request("POST", f"/stop/{session_id}")
        if response.status_code not in (200, 202):
            endpoint = f"/stop/{session_id}"
            text = self._read_error_text(response, endpoint)
            raise GranSabioClientError(
                f"Failed to stop session: {text}",
                status_code=response.status_code
            )
        return self._read_json_no_retry(response, f"/stop/{session_id}")

    def _call_with_retry(
        self,
        op_name: str,
        resource_id: str,
        func: Callable[[], _RetryT],
        start_time: float,
        max_wait: float,
        *,
        resource: str = "session",
    ) -> _RetryT:
        """
        Invoke an idempotent GET (get_status / get_result, or a ``resource``
        lookup such as get_request_status) with exponential backoff on
        transient transport, JSON, or HTTP errors.
        Non-transient errors propagate immediately. Retry sleeps count against the deadline implied
        by ``start_time + max_wait``; if the budget expires mid-sleep, the call
        is aborted.

        ``wait_for_result`` shares its start time across calls for an overall
        deadline. ``_fetch_result_polling`` supplies the remaining time in its
        current activity-aware window, so retries cannot restart that window.
        ``resource`` and ``resource_id`` name the polled resource (a session
        ID, a request ID, or an endpoint) in messages; only a ``"session"``
        resource adds the hint that the session may still run.
        """
        retries = len(_POLL_RETRY_BACKOFF_SECONDS)
        last_err: Optional[TransientGranSabioError] = None
        hint = (
            f" Session {resource_id} may still be running on the server. "
            f"Check with GET /result/{resource_id} or GET /status/{resource_id}."
            if resource == "session" else ""
        )

        # Initial attempt (not counted as a retry in log messages).
        try:
            return func()
        except TransientGranSabioError as err:
            last_err = err

        # Retry loop: iterate directly over the backoff curve so retry_number
        # naturally indexes into it without an awkward `+ 1`.
        for retry_number, sleep_for in enumerate(_POLL_RETRY_BACKOFF_SECONDS, start=1):
            logger.warning(
                "Transient error polling %s %s via %s (retry %d/%d): %s. "
                "Retrying in %ds.",
                resource, resource_id, op_name, retry_number, retries, last_err, sleep_for,
            )

            # Respect the overall max_wait budget. If the backoff would push us
            # past the deadline, abort now rather than sleep past it.
            elapsed = time.monotonic() - start_time
            if elapsed + sleep_for > max_wait:
                assert last_err is not None
                raise GranSabioClientError(
                    f"Timed out waiting for {resource} {resource_id} after {max_wait}s "
                    f"while backing off from transient error: {last_err}.{hint}"
                ) from last_err
            time.sleep(sleep_for)

            try:
                return func()
            except TransientGranSabioError as err:
                last_err = err

        assert last_err is not None
        logger.error(
            "Transient error polling %s %s via %s exhausted %d retries: %s.%s",
            resource, resource_id, op_name, retries, last_err, hint,
        )
        raise GranSabioClientError(
            f"Transient network errors exhausted retries for {resource} {resource_id} "
            f"during {op_name}: {last_err}.{hint}"
        ) from last_err

    def wait_for_result(
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
        transport or JSON-read errors and retryable HTTP statuses
        (202/425/429/503, e.g. a brief proxy flap or rate limit). HTTP 202
        from get_result keeps polling, including after a terminal status.
        Retry sleeps count against max_wait. Other HTTP errors fail fast.

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

            status = self._call_with_retry(
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
                final = self._call_with_retry(
                    "get_result",
                    session_id,
                    lambda: self.get_result(session_id),
                    start_time,
                    max_wait,
                )
                if final.get("status") != "in_progress":
                    validate_result(final)
                    return final

            # Bound the poll sleep by the remaining max_wait budget. Without this,
            # a full poll_interval sleep can push total wait up to poll_interval
            # past max_wait, since the timeout check only runs at loop top.
            remaining = max_wait - (time.monotonic() - start_time)
            if remaining <= 0:
                # Fall through: next iteration's max_wait check will raise.
                continue
            time.sleep(min(poll_interval, remaining))

    # =========================================================================
    # Streaming Generation
    # =========================================================================

    def _dispatch_stream_event(
        self,
        router: SessionEventRouter,
        sse_event: SSEEvent,
        activity_monitor: ActivityMonitor,
    ) -> Optional[str]:
        """Route one parsed SSE frame; return the terminal status if any.

        Comments (the engine's heartbeats) only refresh the activity monitor:
        they prove the connection is alive without pretending progress.
        """
        if sse_event.is_comment:
            activity_monitor.mark(message="heartbeat")
            return None
        payload = decode_event_json(sse_event)
        if payload is None:
            return None
        return router.handle(payload)

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

    def generate_streaming(
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
        shared_context: Optional[str] = None,
        prompt_cache: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate content with real-time streaming of progress, content, and QA.

        This method starts a generation session and consumes ONE SSE endpoint,
        ``GET /stream/{session_id}``, which carries everything the session
        produces already labelled: status snapshots and changes, generation
        drafts and accepted commits, QA / preflight / Arbiter / Gran Sabio
        text, thinking, control events and the terminal ``session_end``.
        Once generation completes, the final result is fetched by polling and
        returned.

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
            content_callback: Called with generation text (str). Drafts arrive
                as deltas to append; the accepted commit arrives as the FULL
                snapshot of its attempt and replaces what came before it.
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
            on_session_start: Called with session_id once the session starts.
            error_callback: Called with ``(stream_name, exception)`` when the
                event stream fails. The run continues on polling.
            **kwargs: Additional parameters passed to API.

        Returns:
            Final generation result dict (same as generate()).

        Raises:
            GranSabioClientError: On network/API errors.
            GranSabioGenerationCancelled: If the session was cancelled.
            GranSabioGenerationRejected: If QA rejected the final output.
        """
        # Errors of a previous run must not be mistaken for this one's.
        self.last_stream_errors = []

        # Build payload (same as generate())
        payload: Dict[str, Any] = {
            "prompt": prompt,
            "content_type": content_type,
            "temperature": temperature,
            "min_global_score": min_global_score,
            "max_iterations": max_iterations,
            "gran_sabio_fallback": gran_sabio_fallback,
            "verbose": verbose,
        }

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

        if shared_context is not None:
            payload["shared_context"] = shared_context
        if prompt_cache is not None:
            payload["prompt_cache"] = prompt_cache
        payload.update(kwargs)

        # -----------------------------------------------------------------
        # 1. Start generation (non-blocking)
        # -----------------------------------------------------------------
        gen_response = self.generate(
            wait_for_completion=False,
            **payload,
        )

        session_id = gen_response.get("session_id")
        if not session_id:
            raise GranSabioClientError(
                "No session_id returned from generation start",
                details=gen_response,
            )

        if on_session_start:
            on_session_start(session_id)

        # -----------------------------------------------------------------
        # 2. Compute timeout
        # -----------------------------------------------------------------
        catalog_timeout = self._resolve_catalog_timeout(generator_model)
        stream_timeout = compute_generation_timeout(
            payload, gen_response, catalog_timeout,
        )

        # -----------------------------------------------------------------
        # 3. Create activity monitor
        # -----------------------------------------------------------------
        activity_monitor = ActivityMonitor(
            inactivity_window=STREAM_ACTIVITY_CHECK_SECONDS,
        )

        # -----------------------------------------------------------------
        # 4. Launch the session event stream
        # -----------------------------------------------------------------
        completion_queue: queue.Queue[Optional[str]] = queue.Queue()
        router = SessionEventRouter(
            StreamCallbacks(
                progress=progress_callback,
                content=content_callback,
                qa=qa_callback,
                content_reset=content_reset_callback,
            ),
            activity=activity_monitor,
        )

        def _run_session_stream() -> None:
            """Consume the unified session event stream."""
            url = f"{self.base_url}/stream/{session_id}"
            try:
                resp = requests.get(
                    url,
                    headers=self._headers(),
                    stream=True,
                    timeout=(30, stream_timeout),
                )
                try:
                    resp.raise_for_status()
                    parser = SSEParser()
                    terminal: Optional[str] = None
                    for chunk in resp.iter_content(chunk_size=None):
                        for sse_event in parser.feed(chunk):
                            terminal = self._dispatch_stream_event(
                                router, sse_event, activity_monitor,
                            ) or terminal
                        if terminal:
                            break
                    if not terminal:
                        for sse_event in parser.close():
                            terminal = self._dispatch_stream_event(
                                router, sse_event, activity_monitor,
                            ) or terminal
                    if terminal:
                        completion_queue.put(terminal)
                finally:
                    # The session terminal closes the loop before EOF: release
                    # the connection instead of leaving it to the collector.
                    resp.close()
            except Exception as exc:
                self._record_stream_error("session", url, exc, error_callback)
            finally:
                # Ensure queue always gets a signal so main thread never blocks forever
                try:
                    completion_queue.put_nowait(None)
                except queue.Full:
                    pass

        stream_thread = threading.Thread(
            target=_run_session_stream,
            name=f"session-stream-{session_id[:8]}",
            daemon=True,
        )
        stream_thread.start()

        # -----------------------------------------------------------------
        # 5. Wait for completion signal with activity-aware deadline
        # -----------------------------------------------------------------
        deadline = time.monotonic() + stream_timeout

        while True:
            remaining = max(0.0, deadline - time.monotonic())
            if remaining <= 0:
                # Check if server is still active before giving up
                if activity_monitor.has_recent_activity():
                    deadline = time.monotonic() + stream_timeout
                    logger.debug(
                        "Extended streaming deadline due to recent activity"
                    )
                    continue
                raise GranSabioClientError(
                    f"Streaming timed out after {stream_timeout}s for session {session_id}"
                )

            try:
                wait_time = min(remaining, STREAM_ACTIVITY_CHECK_SECONDS)
                completion_queue.get(timeout=wait_time)
                # Got a signal from the stream thread
                break
            except queue.Empty:
                # No signal yet - check activity and continue waiting
                continue

        # -----------------------------------------------------------------
        # 6. Let the stream thread finish its last events
        # -----------------------------------------------------------------
        stream_thread.join(timeout=5.0)

        # -----------------------------------------------------------------
        # 7. Fetch final result via polling
        # -----------------------------------------------------------------
        poll_timeout = stream_timeout + RESULT_POLL_GRACE_SECONDS
        result = self._fetch_result_polling(
            session_id,
            timeout_seconds=poll_timeout,
            poll_interval=RESULT_POLL_INTERVAL_SECONDS,
            activity_monitor=activity_monitor,
        )

        # -----------------------------------------------------------------
        # 8. Validate result (raises on cancellation / rejection / error)
        # -----------------------------------------------------------------
        validate_result(result)

        return result

    # =========================================================================
    # Enhanced Polling
    # =========================================================================

    def _fetch_result_polling(
        self,
        session_id: str,
        *,
        timeout_seconds: Optional[float] = None,
        poll_interval: float = RESULT_POLL_INTERVAL_SECONDS,
        activity_monitor: Optional[ActivityMonitor] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve the final generation result, polling until completion.

        Supports activity-aware deadline extension: if an ActivityMonitor
        is provided and shows recent activity, the polling deadline is
        reset instead of raising a timeout error.

        Args:
            session_id: The session to poll.
            timeout_seconds: Maximum seconds to wait (None = no timeout).
            poll_interval: Seconds between poll attempts.
            activity_monitor: Optional monitor for deadline extension.

        Returns:
            The final result dict from the server.

        Raises:
            GranSabioClientError: On timeout or unexpected server errors.
        """
        deadline = time.monotonic() + timeout_seconds if timeout_seconds else None
        while True:
            call_start = time.monotonic()
            if deadline is not None and call_start >= deadline:
                # Preserve the activity-aware window before budgeting retries.
                if activity_monitor and timeout_seconds:
                    latest = activity_monitor.last_activity_timestamp()
                    if call_start - latest <= timeout_seconds:
                        deadline = call_start + timeout_seconds
                        logger.debug("Extended polling deadline due to recent activity")
                    else:
                        raise GranSabioClientError(
                            f"Timed out waiting for result of session {session_id}"
                        )
                else:
                    raise GranSabioClientError(
                        f"Timed out waiting for result of session {session_id}"
                    )

            retry_max_wait = float("inf") if deadline is None else max(0.0, deadline - call_start)
            result = self._call_with_retry(
                "get_result",
                session_id,
                lambda: self.get_result(session_id),
                call_start,
                retry_max_wait,
            )

            if result.get("status") != "in_progress":
                return result

            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            # Clamp the sleep to the remaining budget so polling cannot
            # overshoot the deadline through a full poll_interval sleep.
            sleep_time = poll_interval if remaining is None else max(0.0, min(poll_interval, remaining))
            time.sleep(sleep_time)

    # =========================================================================
    # Token Budget Helpers
    # =========================================================================

    def get_optimal_max_tokens(self, model: str) -> int:
        """
        Get optimal max_tokens for a model from the server catalog, with fallback.

        Queries the /models endpoint for the model's output_tokens value.
        Falls back to a hardcoded estimate if the server is unreachable or the
        model is not found in the catalog.

        Args:
            model: The model identifier (e.g. "gpt-4o", "claude-opus-4-6").

        Returns:
            The maximum output token limit for the model.
        """
        key = model.strip().lower()
        cached = self._model_token_cache.get(key)
        if cached:
            return cached

        try:
            models_data = self.get_models()
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

    def allocate_token_budgets(
        self,
        model: str,
        desired_max_tokens: int,
        desired_thinking_tokens: Optional[int] = None,
    ) -> Tuple[int, Optional[int]]:
        """
        Determine safe max/thinking token budgets honoring model limits.

        Queries the model catalog for the real output_tokens limit and
        then computes a safe (max_tokens, thinking_tokens) pair.

        Args:
            model: The model identifier.
            desired_max_tokens: Requested max_tokens.
            desired_thinking_tokens: Requested thinking budget (for Claude models).

        Returns:
            Tuple of (max_tokens, thinking_tokens) where thinking_tokens may be None.
        """
        model_limit = self.get_optimal_max_tokens(model)
        return compute_token_budgets(model_limit, desired_max_tokens, desired_thinking_tokens)

    def _resolve_catalog_timeout(self, model: Optional[str]) -> Optional[int]:
        """
        Look up the advertised reasoning timeout for a model via the catalog.

        Args:
            model: The model identifier to look up.

        Returns:
            Timeout in seconds, or None if not found/unavailable.
        """
        if not model:
            return None

        key = model.strip().lower()
        cached = self._model_timeout_cache.get(key)
        if cached:
            return cached

        try:
            models_data = self.get_models()
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

    # =========================================================================
    # Text Analysis
    # =========================================================================

    def analyze_lexical_diversity(
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

        response = self._request("POST", "/analysis/lexical-diversity", json_data=payload)
        if response.status_code != 200:
            text_body = self._read_error_text(response, "/analysis/lexical-diversity")
            raise GranSabioClientError(
                f"Lexical diversity analysis failed: {text_body}",
                status_code=response.status_code
            )
        return self._read_json_no_retry(response, "/analysis/lexical-diversity")

    def analyze_repetition(
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

        response = self._request("POST", "/analysis/repetition", json_data=payload)
        if response.status_code != 200:
            text_body = self._read_error_text(response, "/analysis/repetition")
            raise GranSabioClientError(
                f"Repetition analysis failed: {text_body}",
                status_code=response.status_code
            )
        return self._read_json_no_retry(response, "/analysis/repetition")

    # =========================================================================
    # Specialized Repetition Analysis
    # =========================================================================

    def fetch_unigram_counts(
        self,
        text: str,
        *,
        min_count: int = 1,
        top_k: int = 1000,
        language: Optional[str] = None,
        filter_stop_words: bool = False,
    ) -> Dict[str, int]:
        """
        Get unigram frequency counts from the repetition analysis endpoint.

        Returns a dict mapping each word to its occurrence count, limited to
        the top_k most frequent words with at least min_count occurrences.

        Args:
            text: Text to analyze.
            min_count: Minimum occurrences to include a word.
            top_k: Maximum number of entries to return.
            language: Language code for stop-word filtering.
            filter_stop_words: Whether to exclude stop words.

        Returns:
            Dict mapping word strings to their integer counts.
        """
        if not text or not text.strip():
            return {}

        payload: Dict[str, Any] = {
            "text": text,
            "min_n": 1,
            "max_n": 1,
            "min_count": max(1, min_count),
            "summary_mode": "counts",
            "summary_top_k": max(50, top_k),
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

        response = self._request("POST", "/analysis/repetition", json_data=payload)
        if response.status_code != 200:
            text_body = self._read_error_text(response, "/analysis/repetition")
            raise GranSabioClientError(
                f"Unigram analysis failed: {text_body}",
                status_code=response.status_code,
            )

        data = self._read_json_no_retry(response, "/analysis/repetition")
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

    def fetch_positional_bias(
        self,
        blocks: Sequence[str],
        *,
        min_count: int = 2,
        confidence_z: float = 1.645,
        bias_threshold: float = 0.40,
        max_sequences: int = 50,
    ) -> Dict[str, Any]:
        """
        Get positional bias metrics from the repetition analysis endpoint.

        Analyzes whether certain phrases appear disproportionately at specific
        positions within text blocks (e.g., always at the start or end).

        Args:
            blocks: Sequence of text blocks to analyze for positional patterns.
            min_count: Minimum phrase occurrences to consider.
            confidence_z: Z-score for statistical confidence (1.645 = 90%).
            bias_threshold: Minimum bias ratio to flag (0.40 = 40%).
            max_sequences: Maximum biased sequences to return.

        Returns:
            Dict with positional bias analysis results, or empty dict if unavailable.
        """
        cleaned_blocks = [b.strip() for b in blocks if isinstance(b, str) and b.strip()]
        if not cleaned_blocks:
            return {}

        text = "\n\n\n".join(cleaned_blocks)
        payload: Dict[str, Any] = {
            "text": text,
            "min_n": 1,
            "max_n": 3,
            "min_count": max(1, min_count),
            "summary_mode": "counts",
            "summary_top_k": max(10, max_sequences),
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

        response = self._request("POST", "/analysis/repetition", json_data=payload)
        if response.status_code != 200:
            text_body = self._read_error_text(response, "/analysis/repetition")
            raise GranSabioClientError(
                f"Positional bias analysis failed: {text_body}",
                status_code=response.status_code,
            )

        data = self._read_json_no_retry(response, "/analysis/repetition")
        if not isinstance(data, dict):
            return {}
        position_payload = data.get("position_bias")
        return position_payload if isinstance(position_payload, dict) else {}

    # =========================================================================
    # Attachments
    # =========================================================================

    def upload_attachment(
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
        files = {
            "file": (filename, content, content_type)
        }
        data = {"username": username}

        # Multipart uploads cannot reuse the default JSON Content-Type header,
        # but they must go through _request so connection/timeout failures map
        # to the documented GranSabioClientError/TransientGranSabioError
        # hierarchy instead of leaking raw requests exceptions.
        headers = self._headers(json_content=False)

        response = self._request(
            "POST",
            "/attachments",
            headers=headers,
            files=files,
            data=data,
        )

        if response.status_code != 200:
            text = self._read_error_text(response, "/attachments")
            raise GranSabioClientError(
                f"Attachment upload failed: {text}",
                status_code=response.status_code
            )
        return self._read_json_no_retry(response, "/attachments")

    def upload_attachment_base64(
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

        response = self._request("POST", "/attachments/base64", json_data=payload)
        if response.status_code != 200:
            text = self._read_error_text(response, "/attachments/base64")
            raise GranSabioClientError(
                f"Attachment upload failed: {text}",
                status_code=response.status_code
            )
        return self._read_json_no_retry(response, "/attachments/base64")

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    def generate_json(
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
        result = self.generate(
            prompt=prompt,
            content_type="json",
            generator_model=model,
            json_output=True,
            json_schema=schema,
            qa_layers=[],  # Schema validation only
            **kwargs
        )

        # Parse JSON content
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

    def generate_fast(
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
        return self.generate(
            prompt=prompt,
            generator_model=model,
            max_tokens=max_tokens,
            qa_layers=[],  # Bypass QA
            max_iterations=1,
            **kwargs
        )


# Module-level convenience function
def create_client(base_url: Optional[str] = None, **kwargs) -> GranSabioClient:
    """Create a Gran Sabio client instance."""
    return GranSabioClient(base_url=base_url, **kwargs)
