"""
Synchronous Gran Sabio LLM Client
=================================

Thread-safe synchronous client for the Gran Sabio LLM Engine API.
Suitable for scripts, CLI tools, and simple integrations.

For async applications, use AsyncGranSabioClient instead.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import requests
from requests import exceptions as requests_exceptions

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
    DEFAULT_TIMEOUT = (30, 600)  # (connect, read)
    DEFAULT_POLL_INTERVAL = 2.0
    DEFAULT_MAX_WAIT = 600.0

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: tuple = DEFAULT_TIMEOUT,
        provider_keys: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize the Gran Sabio client.

        Args:
            base_url: API base URL (default: http://localhost:8000 or GRANSABIO_BASE_URL env)
            api_key: Optional API key (default: GRANSABIO_API_KEY env)
            timeout: Request timeout tuple (connect_timeout, read_timeout)
            provider_keys: Optional dict of provider API keys as HTTP headers.
                If not provided, keys are loaded from environment variables
                (OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, etc.).
        """
        self.base_url = (base_url or os.getenv("GRANSABIO_BASE_URL", self.DEFAULT_BASE_URL)).rstrip("/")
        self.api_key = api_key or os.getenv("GRANSABIO_API_KEY")
        self.timeout = timeout
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

    def _headers(self) -> Dict[str, str]:
        """Build request headers including provider API keys."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self.provider_keys)
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
        except requests_exceptions.ConnectionError as e:
            raise GranSabioClientError(
                f"Cannot connect to Gran Sabio API at {self.base_url}. "
                "Ensure the server is running."
            ) from e
        except requests_exceptions.Timeout as e:
            raise GranSabioClientError(
                f"Request to {endpoint} timed out after {self.timeout}s"
            ) from e
        except requests_exceptions.RequestException as e:
            raise GranSabioClientError(f"Request failed: {e}") from e

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
        return response.json()

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
        return response.json()

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
            raise GranSabioClientError(
                f"Failed to reserve project: {response.text}",
                status_code=response.status_code
            )

        data = response.json()
        return data["project_id"]

    def start_project(self, project_id: str) -> Dict[str, Any]:
        """Activate/reactivate a project."""
        response = self._request("POST", f"/project/start/{project_id}")
        if response.status_code not in (200, 201):
            raise GranSabioClientError(
                f"Failed to start project: {response.text}",
                status_code=response.status_code
            )
        return response.json()

    def stop_project(self, project_id: str) -> Dict[str, Any]:
        """Cancel a project and all its active sessions."""
        response = self._request("POST", f"/project/stop/{project_id}")
        if response.status_code not in (200, 202):
            raise GranSabioClientError(
                f"Failed to stop project: {response.text}",
                status_code=response.status_code
            )
        return response.json()

    # =========================================================================
    # Content Generation
    # =========================================================================

    def generate(
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
        response = self._request("POST", "/generate", json_data=payload)

        if response.status_code != 200:
            raise GranSabioClientError(
                f"Generation request failed: {response.text}",
                status_code=response.status_code,
                details={"payload": payload}
            )

        result = response.json()

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
        return self.wait_for_result(
            session_id,
            poll_interval=poll_interval,
            max_wait=max_wait,
            on_status=on_status
        )

    def get_status(self, session_id: str) -> Dict[str, Any]:
        """Get current status of a generation session."""
        response = self._request("GET", f"/status/{session_id}")
        if response.status_code != 200:
            raise GranSabioClientError(
                f"Failed to get status: {response.text}",
                status_code=response.status_code
            )
        return response.json()

    def get_result(self, session_id: str) -> Dict[str, Any]:
        """Get the result of a completed generation session."""
        response = self._request("GET", f"/result/{session_id}")

        if response.status_code == 202:
            # Still in progress
            return {"status": "in_progress", "session_id": session_id}

        if response.status_code != 200:
            raise GranSabioClientError(
                f"Failed to get result: {response.text}",
                status_code=response.status_code
            )

        return response.json()

    def stop_session(self, session_id: str) -> Dict[str, Any]:
        """Cancel an active generation session."""
        response = self._request("POST", f"/stop/{session_id}")
        if response.status_code not in (200, 202):
            raise GranSabioClientError(
                f"Failed to stop session: {response.text}",
                status_code=response.status_code
            )
        return response.json()

    def wait_for_result(
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

            status = self.get_status(session_id)

            if on_status:
                on_status(status)

            current_status = status.get("status", "")

            if current_status == "completed":
                return self.get_result(session_id)

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

            time.sleep(poll_interval)

    # =========================================================================
    # Streaming Generation
    # =========================================================================

    def generate_streaming(
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
        """
        Generate content with real-time streaming of progress, content, and QA.

        This method starts a generation session and then consumes four SSE
        endpoints in parallel (progress, generation content, QA content,
        preflight content), forwarding data to the provided callbacks.
        Once generation completes, the final result is fetched and returned.

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
            progress_callback: Called with progress messages (str).
            content_callback: Called with raw content text chunks (str).
            qa_callback: Called with (chunk, event_type, detail).
                event_type is one of: "QA_PROGRESS", "QA_CONTENT_STREAM",
                "PREFLIGHT_CONTENT_STREAM".
            on_session_start: Called with session_id once the session starts.
            **kwargs: Additional parameters passed to API.

        Returns:
            Final generation result dict (same as generate()).

        Raises:
            GranSabioClientError: On network/API errors.
            GranSabioGenerationCancelled: If the session was cancelled.
            GranSabioGenerationRejected: If QA rejected the final output.
        """
        # Build payload (same as generate())
        payload: Dict[str, Any] = {
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
        # 4. Launch SSE consumer threads
        # -----------------------------------------------------------------
        completion_queue: queue.Queue[Optional[str]] = queue.Queue()
        thread_errors: List[Exception] = []

        def _run_progress_stream() -> None:
            """Consume SSE progress events from /stream/{session_id}."""
            url = f"{self.base_url}/stream/{session_id}"
            try:
                resp = requests.get(
                    url,
                    headers=self._headers(),
                    stream=True,
                    timeout=(30, stream_timeout),
                )
                resp.raise_for_status()
                for raw_line in resp.iter_lines(decode_unicode=True):
                    if raw_line is None:
                        continue
                    line = raw_line.strip()
                    if not line or not line.startswith("data: "):
                        continue

                    json_str = line[6:]  # strip "data: " prefix
                    try:
                        event = json.loads(json_str)
                    except (json.JSONDecodeError, ValueError):
                        continue

                    if not isinstance(event, dict):
                        continue

                    activity_monitor.mark(
                        iteration=event.get("iteration"),
                        message=event.get("message"),
                    )

                    msg = event.get("message", "")
                    event_type = event.get("type", "")

                    # Route QA-related progress to qa_callback
                    is_qa_event = (
                        event_type in ("qa_evaluation", "qa_result", "qa_summary")
                        or "QA" in msg
                        or "qa" in event_type
                    )

                    if is_qa_event and qa_callback:
                        qa_callback(msg, "QA_PROGRESS", json_str)
                    elif progress_callback:
                        progress_callback(msg)

                    # Detect completion signals
                    status = event.get("status", "")
                    if status in ("completed", "failed", "cancelled"):
                        completion_queue.put(status)
                        return

            except Exception as exc:
                logger.debug("Progress stream ended: %s", exc)
                thread_errors.append(exc)
            finally:
                # Ensure queue always gets a signal so main thread never blocks forever
                try:
                    completion_queue.put_nowait(None)
                except queue.Full:
                    pass

        def _run_content_stream() -> None:
            """Consume raw text chunks from /stream-generation/{session_id}."""
            url = f"{self.base_url}/stream-generation/{session_id}"
            try:
                resp = requests.get(
                    url,
                    headers=self._headers(),
                    stream=True,
                    timeout=(30, stream_timeout),
                )
                resp.raise_for_status()
                for chunk in resp.iter_content(chunk_size=None, decode_unicode=True):
                    if not chunk:
                        continue
                    if is_heartbeat(chunk):
                        activity_monitor.mark(message="heartbeat")
                        continue
                    activity_monitor.mark()
                    if content_callback:
                        content_callback(chunk)
            except Exception as exc:
                logger.debug("Content stream ended: %s", exc)

        def _run_qa_stream() -> None:
            """Consume raw QA evaluation chunks from /stream-qa/{session_id}."""
            url = f"{self.base_url}/stream-qa/{session_id}"
            try:
                resp = requests.get(
                    url,
                    headers=self._headers(),
                    stream=True,
                    timeout=(30, stream_timeout),
                )
                resp.raise_for_status()
                for chunk in resp.iter_content(chunk_size=None, decode_unicode=True):
                    if not chunk:
                        continue
                    if is_heartbeat(chunk):
                        activity_monitor.mark(message="heartbeat")
                        continue
                    activity_monitor.mark()
                    if qa_callback:
                        qa_callback(chunk, "QA_CONTENT_STREAM", "raw_chunk")
            except Exception as exc:
                logger.debug("QA stream ended: %s", exc)

        def _run_preflight_stream() -> None:
            """Consume preflight validation chunks from /stream-preflight/{session_id}."""
            url = f"{self.base_url}/stream-preflight/{session_id}"
            try:
                resp = requests.get(
                    url,
                    headers=self._headers(),
                    stream=True,
                    timeout=(30, stream_timeout),
                )
                resp.raise_for_status()
                for chunk in resp.iter_content(chunk_size=None, decode_unicode=True):
                    if not chunk:
                        continue
                    if is_heartbeat(chunk):
                        activity_monitor.mark(message="heartbeat")
                        continue
                    activity_monitor.mark()
                    if qa_callback:
                        qa_callback(chunk, "PREFLIGHT_CONTENT_STREAM", "raw_chunk")
            except Exception as exc:
                logger.debug("Preflight stream ended: %s", exc)

        threads: List[threading.Thread] = []
        for target, name in (
            (_run_progress_stream, f"progress-{session_id[:8]}"),
            (_run_content_stream, f"content-{session_id[:8]}"),
            (_run_qa_stream, f"qa-{session_id[:8]}"),
            (_run_preflight_stream, f"preflight-{session_id[:8]}"),
        ):
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()
            threads.append(t)

        progress_thread = threads[0]
        content_threads = threads[1:]

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
                signal = completion_queue.get(timeout=wait_time)
                # Got a signal from the progress thread
                break
            except queue.Empty:
                # No signal yet - check activity and continue waiting
                continue

        # -----------------------------------------------------------------
        # 6. Join content threads (short grace period)
        # -----------------------------------------------------------------
        for t in content_threads:
            t.join(timeout=5.0)

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
            response = self._request("GET", f"/result/{session_id}")

            if response.status_code == 200:
                return response.json()

            # Parse detail message for retry detection
            detail_message = ""
            try:
                response_payload = response.json()
                if isinstance(response_payload, dict):
                    detail_message = str(response_payload.get("detail", ""))
            except ValueError:
                pass

            should_retry = False
            if response.status_code in (202, 425, 503, 400):
                lower_detail = detail_message.lower()
                if "not finished" in lower_detail or "still in progress" in lower_detail:
                    should_retry = True
                elif response.status_code == 202 and not detail_message:
                    should_retry = True

            if should_retry:
                now = time.monotonic()
                if deadline is not None and now >= deadline:
                    # Check activity monitor for deadline extension
                    if activity_monitor and timeout_seconds:
                        latest = activity_monitor.last_activity_timestamp()
                        if now - latest <= timeout_seconds:
                            deadline = time.monotonic() + timeout_seconds
                            logger.debug(
                                "Extended polling deadline due to recent activity"
                            )
                        else:
                            raise GranSabioClientError(
                                f"Timed out waiting for result of session {session_id}"
                            )
                    else:
                        raise GranSabioClientError(
                            f"Timed out waiting for result of session {session_id}"
                        )

                remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
                sleep_time = poll_interval if remaining is None else min(poll_interval, max(1.0, remaining))
                time.sleep(sleep_time)
                continue

            raise GranSabioClientError(
                f"Result retrieval failed: {response.status_code} - {response.text}",
                status_code=response.status_code,
            )

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
            model: The model identifier (e.g. "gpt-4o", "claude-opus-4-5-20251101").

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
            raise GranSabioClientError(
                f"Lexical diversity analysis failed: {response.text}",
                status_code=response.status_code
            )
        return response.json()

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
            raise GranSabioClientError(
                f"Repetition analysis failed: {response.text}",
                status_code=response.status_code
            )
        return response.json()

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
            raise GranSabioClientError(
                f"Unigram analysis failed: {response.text}",
                status_code=response.status_code,
            )

        data = response.json()
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
            raise GranSabioClientError(
                f"Positional bias analysis failed: {response.text}",
                status_code=response.status_code,
            )

        data = response.json()
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

        response = requests.post(
            f"{self.base_url}/attachments",
            files=files,
            data=data,
            headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
            timeout=self.timeout
        )

        if response.status_code != 200:
            raise GranSabioClientError(
                f"Attachment upload failed: {response.text}",
                status_code=response.status_code
            )
        return response.json()

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
            raise GranSabioClientError(
                f"Attachment upload failed: {response.text}",
                status_code=response.status_code
            )
        return response.json()

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    def generate_json(
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
        if isinstance(content, str):
            try:
                result["parsed_content"] = json.loads(content)
            except (json.JSONDecodeError, ValueError):
                result["parsed_content"] = None
                result["parse_error"] = True

        return result

    def generate_fast(
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
        return self.generate(
            prompt=prompt,
            generator_model=model,
            max_tokens=max_tokens,
            qa_layers=[],  # Bypass QA
            max_iterations=1,
            **kwargs
        )


# Backward compatibility alias
BioAIClient = GranSabioClient


# Module-level convenience function
def create_client(base_url: Optional[str] = None, **kwargs) -> GranSabioClient:
    """Create a Gran Sabio client instance."""
    return GranSabioClient(base_url=base_url, **kwargs)
