"""
Shared fixtures for integration tests.

These fixtures handle:
- IP filter bypass for TestClient
- Mocked AI services
- Preflight validation mocking
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import ipaddress

from fastapi.testclient import TestClient


# ============================================================================
# IP Filter Bypass
# ============================================================================

@pytest.fixture(autouse=True)
def bypass_ip_filter():
    """
    Automatically bypass IP filter for all integration tests.

    This patches the INTERNAL_NETWORKS to include all IPs and
    allows the test client to access all endpoints.
    """
    # Include all private networks and testclient default
    test_networks = [
        ipaddress.ip_network('0.0.0.0/0'),  # Allow all IPv4
        ipaddress.ip_network('::/0'),        # Allow all IPv6
    ]

    with patch('core.security.INTERNAL_NETWORKS', test_networks), \
         patch('core.security.is_ip_allowed', return_value=True):
        yield


# ============================================================================
# Common Fixtures
# ============================================================================

@pytest.fixture
def mock_ai_service():
    """Mock AI service for integration tests."""
    service = MagicMock()

    async def mock_generate(*args, **kwargs):
        return "This is generated test content for integration testing. " * 50

    service.generate_content = AsyncMock(side_effect=mock_generate)

    async def mock_stream(*args, **kwargs):
        chunks = ["This ", "is ", "streaming ", "content."]
        for chunk in chunks:
            yield chunk

    service.generate_content_stream = mock_stream
    service.health_check = AsyncMock(return_value={"openai": True, "anthropic": True})
    return service


@pytest.fixture
def mock_preflight_proceed():
    """Mock preflight that always proceeds."""
    from models import PreflightResult
    return PreflightResult(
        decision="proceed",
        user_feedback="Request validated successfully",
        summary="All checks passed",
        confidence=0.95,
        enable_algorithmic_word_count=False,
        duplicate_word_count_layers_to_remove=[],
    )


@pytest.fixture
def mock_preflight_reject():
    """Mock preflight that rejects requests."""
    from models import PreflightResult
    return PreflightResult(
        decision="reject",
        user_feedback="Request contains contradictory requirements",
        summary="Cannot process request with conflicting constraints",
        confidence=0.92,
        issues=[
            {
                "code": "contradictory_requirements",
                "severity": "critical",
                "message": "Fiction content conflicts with factual accuracy requirement"
            }
        ],
        enable_algorithmic_word_count=False,
        duplicate_word_count_layers_to_remove=[],
    )


@pytest.fixture
def base_test_client(mock_ai_service, mock_preflight_proceed):
    """
    Base test client with mocked services.

    This client bypasses IP filtering and mocks AI services.
    """
    with patch('ai_service.get_ai_service', return_value=mock_ai_service), \
         patch('core.app_state._ensure_services'), \
         patch('core.app_state.ai_service', mock_ai_service), \
         patch('core.generation_routes.ai_service', mock_ai_service), \
         patch('core.generation_processor.ai_service', mock_ai_service), \
         patch('core.streaming_routes.ai_service', mock_ai_service), \
         patch('preflight_validator.run_preflight_validation', AsyncMock(return_value=mock_preflight_proceed)):
        from main import app
        client = TestClient(app, headers={"X-Forwarded-For": "127.0.0.1"})
        yield client


@pytest.fixture
def rejection_test_client(mock_ai_service, mock_preflight_reject):
    """
    Test client with preflight rejection mock.

    Used for testing preflight rejection scenarios.
    """
    with patch('ai_service.get_ai_service', return_value=mock_ai_service), \
         patch('core.app_state._ensure_services'), \
         patch('core.app_state.ai_service', mock_ai_service), \
         patch('core.generation_routes.ai_service', mock_ai_service), \
         patch('core.generation_processor.ai_service', mock_ai_service), \
         patch('core.streaming_routes.ai_service', mock_ai_service), \
         patch('core.generation_routes.run_preflight_validation', AsyncMock(return_value=mock_preflight_reject)):
        from main import app
        client = TestClient(app, headers={"X-Forwarded-For": "127.0.0.1"})
        yield client


@pytest.fixture
def valid_generation_request():
    """Valid minimal generation request."""
    return {
        "prompt": "Write a detailed article about software testing best practices for developers.",
        "generator_model": "gpt-4o",
        "content_type": "article",
        "max_iterations": 1,
        "qa_layers": [],
        "qa_models": [],
    }


@pytest.fixture
def generation_request_with_qa():
    """Generation request with QA layers."""
    return {
        "prompt": "Write a detailed article about software testing best practices.",
        "generator_model": "gpt-4o",
        "content_type": "article",
        "max_iterations": 3,
        "qa_layers": [
            {
                "name": "Quality Check",
                "description": "Check content quality",
                "criteria": "Content should be accurate and well-written",
                "min_score": 7.0,
                "order": 1
            }
        ],
        "qa_models": ["gpt-4o"],
        "gran_sabio_model": "gpt-4o",
    }
