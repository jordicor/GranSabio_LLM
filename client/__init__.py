"""
Gran Sabio LLM Client
=====================

Python client library for the Gran Sabio LLM Engine API.

Quick Start:
    from client import GranSabioClient

    # Sync usage
    client = GranSabioClient()
    result = client.generate("Write an article about AI")
    print(result["content"])

    # Async usage
    from client import AsyncGranSabioClient

    async with AsyncGranSabioClient() as client:
        result = await client.generate("Write an article about AI")
        print(result["content"])

For more examples, see the demos/ folder.
"""

from typing import Dict, Optional


class GranSabioClientError(Exception):
    """Exception raised for Gran Sabio client errors."""

    def __init__(self, message: str, status_code: Optional[int] = None, details: Optional[Dict] = None):
        super().__init__(message)
        self.status_code = status_code
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
    # Backward compatibility
    "BioAIClient",
    "AsyncBioAIClient",
    "BioAIClientError",
]

__version__ = "1.0.0"
