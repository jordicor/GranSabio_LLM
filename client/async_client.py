"""
Asynchronous Gran Sabio LLM Client
==================================

High-performance async client for the Gran Sabio LLM Engine API.
Ideal for web applications, parallel generation, and async workflows.

For synchronous usage, use GranSabioClient instead.
"""

from __future__ import annotations

import os
import time
import asyncio
import logging
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime

import aiohttp

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Import shared exception from package
from . import GranSabioClientError

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
    ):
        """
        Initialize the async Gran Sabio client.

        Args:
            base_url: API base URL (default: http://localhost:8000 or GRANSABIO_BASE_URL env)
            api_key: Optional API key (default: GRANSABIO_API_KEY env)
            timeout: Request timeout configuration
        """
        self.base_url = (base_url or os.getenv("GRANSABIO_BASE_URL", self.DEFAULT_BASE_URL)).rstrip("/")
        self.api_key = api_key or os.getenv("GRANSABIO_API_KEY")
        self.timeout = timeout or self.DEFAULT_TIMEOUT
        self._session: Optional[aiohttp.ClientSession] = None
        self._owns_session = True

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
        """Build request headers."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
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
        import json as json_module

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
                result["parsed_content"] = json_module.loads(content)
            except json_module.JSONDecodeError:
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


# Backward compatibility alias
AsyncBioAIClient = AsyncGranSabioClient


# Module-level convenience function
async def create_client(base_url: Optional[str] = None, **kwargs) -> AsyncGranSabioClient:
    """Create and connect an async Gran Sabio client instance."""
    client = AsyncGranSabioClient(base_url=base_url, **kwargs)
    await client.connect()
    return client
