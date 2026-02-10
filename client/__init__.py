"""
Gran Sabio LLM Client SDK
=========================

Python client library for the Gran Sabio LLM Engine API.

Quick Start:
    from client import GranSabioClient

    # Sync usage
    client = GranSabioClient()
    result = client.generate("Write an article about AI")
    print(result["content"])

    # Sync usage with streaming
    result = client.generate_streaming(
        prompt="Write an article about AI",
        progress_callback=lambda msg: print(f"Progress: {msg}"),
        content_callback=lambda chunk: print(chunk, end=""),
    )

    # Async usage
    from client import AsyncGranSabioClient

    async with AsyncGranSabioClient() as client:
        result = await client.generate("Write an article about AI")
        print(result["content"])

Install:
    pip install -e /path/to/GranSabio_LLM/client
"""

from typing import Any, Dict, Optional


class GranSabioClientError(Exception):
    """Exception raised for Gran Sabio client errors."""

    def __init__(self, message: str, status_code: Optional[int] = None, details: Optional[Dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.details = details or {}


class GranSabioGenerationCancelled(GranSabioClientError):
    """Raised when a GranSabio generation session was cancelled by user request."""

    def __init__(self, message: str, *, session_id: Optional[str] = None) -> None:
        super().__init__(message)
        self.session_id = session_id


class GranSabioGenerationRejected(GranSabioClientError):
    """Raised when GranSabio returns content that fails post-generation QA."""

    def __init__(self, message: str, *, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.details = details or {}


# Backward compatibility alias
BioAIClientError = GranSabioClientError

from .sync_client import GranSabioClient
from .async_client import AsyncGranSabioClient

# Backward compatibility aliases
BioAIClient = GranSabioClient
AsyncBioAIClient = AsyncGranSabioClient

__all__ = [
    "GranSabioClient",
    "AsyncGranSabioClient",
    "GranSabioClientError",
    "GranSabioGenerationCancelled",
    "GranSabioGenerationRejected",
    # Backward compatibility
    "BioAIClient",
    "AsyncBioAIClient",
    "BioAIClientError",
]

__version__ = "1.1.0"
