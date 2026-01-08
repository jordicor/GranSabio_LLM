"""Shared pytest fixtures for Gran Sabio LLM Engine tests."""

import pytest
from unittest.mock import AsyncMock, Mock, patch, MagicMock
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# Environment Fixtures
# ============================================================================

@pytest.fixture
def mock_env_vars():
    """Provide test environment variables."""
    env = {
        "OPENAI_API_KEY": "sk-test-openai-key-12345",
        "ANTHROPIC_API_KEY": "sk-ant-test-key-12345",
        "GOOGLE_API_KEY": "test-google-key-12345",
        "XAI_API_KEY": "xai-test-key-12345",
        "OPENROUTER_API_KEY": "sk-or-test-key-12345",
        "DEFAULT_MIN_GLOBAL_SCORE": "8.0",
        "DEFAULT_MAX_ITERATIONS": "5",
        "APP_HOST": "0.0.0.0",
        "APP_PORT": "8000",
    }
    with patch.dict(os.environ, env, clear=False):
        yield env


@pytest.fixture
def clean_env():
    """Provide clean environment without API keys."""
    keys_to_remove = [
        "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY",
        "XAI_API_KEY", "OPENROUTER_API_KEY"
    ]
    with patch.dict(os.environ, {}, clear=False):
        for key in keys_to_remove:
            os.environ.pop(key, None)
        yield


# ============================================================================
# Model Fixtures
# ============================================================================

@pytest.fixture
def sample_model_specs():
    """Minimal model specs for testing."""
    return {
        "model_specifications": {
            "openai": [
                {
                    "model_id": "gpt-4o",
                    "display_name": "GPT-4o",
                    "max_tokens": 16384,
                    "supports_streaming": True,
                    "supports_vision": True,
                    "pricing_tier": "standard"
                },
                {
                    "model_id": "gpt-4o-mini",
                    "display_name": "GPT-4o Mini",
                    "max_tokens": 16384,
                    "supports_streaming": True,
                    "pricing_tier": "economy"
                }
            ],
            "anthropic": [
                {
                    "model_id": "claude-sonnet-4-20250514",
                    "display_name": "Claude Sonnet 4",
                    "max_tokens": 8192,
                    "supports_streaming": True,
                    "supports_vision": True,
                    "pricing_tier": "standard"
                }
            ]
        },
        "model_aliases": {
            "gpt4": "gpt-4o",
            "claude": "claude-sonnet-4-20250514"
        },
        "default_models": {
            "generator": "gpt-4o",
            "qa": ["gpt-4o"],
            "gran_sabio": "claude-sonnet-4-20250514"
        }
    }


@pytest.fixture
def sample_qa_layer():
    """Standard QA layer for testing."""
    from models import QALayer
    return QALayer(
        name="Test Quality",
        description="Test quality criteria",
        criteria="Content should be accurate and clear",
        min_score=7.0,
        order=1
    )


@pytest.fixture
def sample_qa_evaluation():
    """Sample QA evaluation result."""
    from models import QAEvaluation
    return QAEvaluation(
        model="gpt-4o",
        layer="Test Quality",
        score=8.5,
        feedback="Content meets quality standards",
        deal_breaker=False,
        passes_score=True
    )


@pytest.fixture
def sample_content_request():
    """Standard content request for testing."""
    from models import ContentRequest
    return ContentRequest(
        prompt="Write a test article about software testing",
        content_type="article",
        generator_model="gpt-4o",
        temperature=0.7,
        max_tokens=4000,
        min_words=500,
        max_words=1000,
        qa_layers=[],
        qa_models=[]
    )


# ============================================================================
# AI Service Mocks
# ============================================================================

@pytest.fixture
def mock_ai_service():
    """Mocked AIService for unit tests."""
    service = MagicMock()
    service.generate_content = AsyncMock(return_value="Generated test content")
    service.generate_content_stream = AsyncMock()
    service.health_check = AsyncMock(return_value={"openai": True, "anthropic": True})
    return service


@pytest.fixture
def mock_openai_response():
    """Standard OpenAI completion response."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = "Generated text from OpenAI"
    response.choices[0].finish_reason = "stop"
    response.usage = MagicMock()
    response.usage.prompt_tokens = 100
    response.usage.completion_tokens = 50
    response.usage.total_tokens = 150
    return response


@pytest.fixture
def mock_anthropic_response():
    """Standard Anthropic message response."""
    response = MagicMock()
    response.content = [MagicMock()]
    response.content[0].text = "Generated text from Claude"
    response.content[0].type = "text"
    response.stop_reason = "end_turn"
    response.usage = MagicMock()
    response.usage.input_tokens = 100
    response.usage.output_tokens = 50
    return response


# ============================================================================
# Response Templates
# ============================================================================

@pytest.fixture
def gran_sabio_approve_response():
    """Gran Sabio approval response template."""
    return """
[DECISION]APPROVED[/DECISION]
[SCORE]8.5[/SCORE]
[REASON]Content meets all quality criteria[/REASON]
[MODIFICATIONS_MADE]false[/MODIFICATIONS_MADE]
"""


@pytest.fixture
def gran_sabio_reject_response():
    """Gran Sabio rejection response template."""
    return """
[DECISION]REJECTED[/DECISION]
[SCORE]4.0[/SCORE]
[REASON]Critical factual errors detected[/REASON]
[MODIFICATIONS_MADE]false[/MODIFICATIONS_MADE]
"""


@pytest.fixture
def preflight_proceed_response():
    """Preflight validation proceed response."""
    return {
        "decision": "proceed",
        "summary": "Request validated successfully",
        "user_feedback": "Your request has been approved",
        "issues": [],
        "confidence": 0.95
    }


@pytest.fixture
def preflight_reject_response():
    """Preflight validation reject response."""
    return {
        "decision": "reject",
        "summary": "Request contains contradictions",
        "user_feedback": "Cannot validate historical accuracy for fictional content",
        "issues": [
            {
                "code": "contradictory_requirements",
                "severity": "critical",
                "message": "Fiction content type conflicts with historical accuracy QA layer"
            }
        ],
        "confidence": 0.92
    }


# ============================================================================
# API Test Fixtures
# ============================================================================

@pytest.fixture
def test_client():
    """Synchronous FastAPI test client."""
    from fastapi.testclient import TestClient
    from main import app
    return TestClient(app)


@pytest.fixture
async def async_test_client():
    """Async FastAPI test client."""
    from httpx import AsyncClient, ASGITransport
    from main import app
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        yield client
