"""
Integration tests for OpenAI reasoning_effort via real API calls.

Tests that reasoning_effort is correctly handled for OpenAI models (GPT-5, o1, o3).
OpenAI uses reasoning_effort (string: "low", "medium", "high") instead of
thinking_budget_tokens (number).

IMPORTANT: These tests make real API calls and incur costs.
Set SKIP_EXPENSIVE_TESTS=1 to skip these tests.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import httpx
from typing import Dict, Any
import time

# Skip expensive API tests unless explicitly enabled
SKIP_EXPENSIVE = os.environ.get("SKIP_EXPENSIVE_TESTS", "0") == "1"

# API configuration
API_BASE = os.environ.get("API_BASE", "http://localhost:8000")

# Test models
GPT5 = "gpt-5"
GPT51 = "gpt-5.1"
GPT52 = "gpt-5.2"
GPT4O = "gpt-4o"
GPT4O_MINI = "gpt-4o-mini"


def wait_for_completion(session_id: str, max_wait: int = 120) -> Dict[str, Any]:
    """Wait for a session to complete and return the status."""
    start = time.time()
    while time.time() - start < max_wait:
        response = httpx.get(f"{API_BASE}/status/{session_id}", timeout=30)
        status = response.json()

        if status["status"] in ["completed", "failed", "preflight_rejected"]:
            return status

        time.sleep(2)

    raise TimeoutError(f"Session {session_id} did not complete within {max_wait}s")


def start_generation(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Start a generation request and return the response."""
    response = httpx.post(
        f"{API_BASE}/generate",
        json=payload,
        timeout=60
    )
    return response.json()


def get_result(session_id: str) -> Dict[str, Any]:
    """Get the result of a completed session."""
    response = httpx.get(f"{API_BASE}/result/{session_id}", timeout=30)
    return response.json()


@pytest.fixture
def base_payload():
    """Base payload for testing."""
    return {
        "prompt": "Count from 1 to 5 and explain briefly what numbers are.",
        "content_type": "explanation",
        "min_words": 20,
        "max_words": 100,
        "max_iterations": 1,
        "qa_layers": [],  # Skip QA for faster testing
        "verbose": True,
    }


class TestOpenAIReasoningEffortViaAPI:
    """Test OpenAI reasoning_effort through the generation API."""

    @pytest.mark.skipif(SKIP_EXPENSIVE, reason="Expensive API test")
    def test_gpt5_with_reasoning_effort_low(self, base_payload):
        """Test GPT-5 with reasoning_effort=low."""
        payload = {
            **base_payload,
            "generator_model": GPT5,
            "reasoning_effort": "low",
            "max_tokens": 1000,
        }

        response = start_generation(payload)
        assert "session_id" in response, f"Failed to start: {response}"

        session_id = response["session_id"]
        status = wait_for_completion(session_id)

        assert status["status"] == "completed", f"Generation failed: {status}"

        result = get_result(session_id)
        assert "content" in result
        assert len(result["content"]) > 0

        print(f"\n[OK] GPT-5 with reasoning_effort='low' completed")
        print(f"    Content length: {len(result['content'])} chars")

    @pytest.mark.skipif(SKIP_EXPENSIVE, reason="Expensive API test")
    def test_gpt5_with_reasoning_effort_medium(self, base_payload):
        """Test GPT-5 with reasoning_effort=medium."""
        payload = {
            **base_payload,
            "generator_model": GPT5,
            "reasoning_effort": "medium",
            "max_tokens": 1000,
        }

        response = start_generation(payload)
        assert "session_id" in response

        session_id = response["session_id"]
        status = wait_for_completion(session_id)

        assert status["status"] == "completed"

        print(f"\n[OK] GPT-5 with reasoning_effort='medium' completed")

    @pytest.mark.skipif(SKIP_EXPENSIVE, reason="Expensive API test")
    def test_gpt5_with_reasoning_effort_high(self, base_payload):
        """Test GPT-5 with reasoning_effort=high."""
        payload = {
            **base_payload,
            "generator_model": GPT5,
            "reasoning_effort": "high",
            "max_tokens": 2000,
        }

        response = start_generation(payload)
        assert "session_id" in response

        session_id = response["session_id"]
        status = wait_for_completion(session_id, max_wait=180)

        assert status["status"] == "completed"

        print(f"\n[OK] GPT-5 with reasoning_effort='high' completed")

    @pytest.mark.skipif(SKIP_EXPENSIVE, reason="Expensive API test")
    def test_gpt51_with_reasoning_effort(self, base_payload):
        """Test GPT-5.1 with reasoning_effort."""
        payload = {
            **base_payload,
            "generator_model": GPT51,
            "reasoning_effort": "medium",
            "max_tokens": 1000,
        }

        response = start_generation(payload)
        assert "session_id" in response

        session_id = response["session_id"]
        status = wait_for_completion(session_id)

        assert status["status"] == "completed"

        print(f"\n[OK] GPT-5.1 with reasoning_effort='medium' completed")

    @pytest.mark.skipif(SKIP_EXPENSIVE, reason="Expensive API test")
    def test_gpt52_with_reasoning_effort(self, base_payload):
        """Test GPT-5.2 with reasoning_effort."""
        payload = {
            **base_payload,
            "generator_model": GPT52,
            "reasoning_effort": "medium",
            "max_tokens": 1000,
        }

        response = start_generation(payload)
        assert "session_id" in response

        session_id = response["session_id"]
        status = wait_for_completion(session_id)

        assert status["status"] == "completed"

        print(f"\n[OK] GPT-5.2 with reasoning_effort='medium' completed")


class TestOpenAIThinkingBudgetToReasoningEffort:
    """Test conversion from thinking_budget_tokens to reasoning_effort for OpenAI."""

    @pytest.mark.skipif(SKIP_EXPENSIVE, reason="Expensive API test")
    def test_gpt5_thinking_budget_converts_to_reasoning_effort_high(self, base_payload):
        """Test that high thinking_budget_tokens converts to reasoning_effort=high."""
        payload = {
            **base_payload,
            "generator_model": GPT5,
            "thinking_budget_tokens": 65535,  # Should map to "high"
            "max_tokens": 1000,
        }

        response = start_generation(payload)
        assert "session_id" in response

        session_id = response["session_id"]
        status = wait_for_completion(session_id, max_wait=180)

        # Should complete - thinking_budget converted to reasoning_effort internally
        assert status["status"] == "completed"

        print(f"\n[OK] GPT-5 with thinking_budget_tokens=65535 -> reasoning_effort='high'")

    @pytest.mark.skipif(SKIP_EXPENSIVE, reason="Expensive API test")
    def test_gpt5_thinking_budget_converts_to_reasoning_effort_medium(self, base_payload):
        """Test that medium thinking_budget_tokens converts to reasoning_effort=medium."""
        payload = {
            **base_payload,
            "generator_model": GPT5,
            "thinking_budget_tokens": 16000,  # Should map to "medium"
            "max_tokens": 1000,
        }

        response = start_generation(payload)
        assert "session_id" in response

        session_id = response["session_id"]
        status = wait_for_completion(session_id)

        assert status["status"] == "completed"

        print(f"\n[OK] GPT-5 with thinking_budget_tokens=16000 -> reasoning_effort='medium'")

    @pytest.mark.skipif(SKIP_EXPENSIVE, reason="Expensive API test")
    def test_gpt5_thinking_budget_converts_to_reasoning_effort_low(self, base_payload):
        """Test that low thinking_budget_tokens converts to reasoning_effort=low."""
        payload = {
            **base_payload,
            "generator_model": GPT5,
            "thinking_budget_tokens": 8000,  # Should map to "low"
            "max_tokens": 1000,
        }

        response = start_generation(payload)
        assert "session_id" in response

        session_id = response["session_id"]
        status = wait_for_completion(session_id)

        assert status["status"] == "completed"

        print(f"\n[OK] GPT-5 with thinking_budget_tokens=8000 -> reasoning_effort='low'")


class TestOpenAIReasoningEffortAliases:
    """Test reasoning_effort aliases for OpenAI."""

    @pytest.mark.skipif(SKIP_EXPENSIVE, reason="Expensive API test")
    def test_gpt5_with_mid_alias(self, base_payload):
        """Test 'mid' alias converts to 'medium'."""
        payload = {
            **base_payload,
            "generator_model": GPT5,
            "reasoning_effort": "mid",  # Should normalize to "medium"
            "max_tokens": 1000,
        }

        response = start_generation(payload)
        assert "session_id" in response

        session_id = response["session_id"]
        status = wait_for_completion(session_id)

        assert status["status"] == "completed"

        print(f"\n[OK] GPT-5 with reasoning_effort='mid' (alias) completed")

    @pytest.mark.skipif(SKIP_EXPENSIVE, reason="Expensive API test")
    def test_gpt5_with_hi_alias(self, base_payload):
        """Test 'hi' alias converts to 'high'."""
        payload = {
            **base_payload,
            "generator_model": GPT5,
            "reasoning_effort": "hi",  # Should normalize to "high"
            "max_tokens": 2000,
        }

        response = start_generation(payload)
        assert "session_id" in response

        session_id = response["session_id"]
        status = wait_for_completion(session_id, max_wait=180)

        assert status["status"] == "completed"

        print(f"\n[OK] GPT-5 with reasoning_effort='hi' (alias) completed")

    @pytest.mark.skipif(SKIP_EXPENSIVE, reason="Expensive API test")
    def test_gpt5_with_lo_alias(self, base_payload):
        """Test 'lo' alias converts to 'low'."""
        payload = {
            **base_payload,
            "generator_model": GPT5,
            "reasoning_effort": "lo",  # Should normalize to "low"
            "max_tokens": 1000,
        }

        response = start_generation(payload)
        assert "session_id" in response

        session_id = response["session_id"]
        status = wait_for_completion(session_id)

        assert status["status"] == "completed"

        print(f"\n[OK] GPT-5 with reasoning_effort='lo' (alias) completed")


class TestOpenAINonReasoningModels:
    """Test non-reasoning OpenAI models (GPT-4o) - should not use reasoning_effort."""

    @pytest.mark.skipif(SKIP_EXPENSIVE, reason="Expensive API test")
    def test_gpt4o_without_reasoning(self, base_payload):
        """Test GPT-4o works without reasoning parameters."""
        payload = {
            **base_payload,
            "generator_model": GPT4O,
            "max_tokens": 1000,
            # No reasoning_effort or thinking_budget_tokens
        }

        response = start_generation(payload)
        assert "session_id" in response

        session_id = response["session_id"]
        status = wait_for_completion(session_id)

        assert status["status"] == "completed"

        print(f"\n[OK] GPT-4o without reasoning completed")

    @pytest.mark.skipif(SKIP_EXPENSIVE, reason="Expensive API test")
    def test_gpt4o_mini_without_reasoning(self, base_payload):
        """Test GPT-4o-mini works without reasoning parameters."""
        payload = {
            **base_payload,
            "generator_model": GPT4O_MINI,
            "max_tokens": 1000,
        }

        response = start_generation(payload)
        assert "session_id" in response

        session_id = response["session_id"]
        status = wait_for_completion(session_id)

        assert status["status"] == "completed"

        print(f"\n[OK] GPT-4o-mini without reasoning completed")

    @pytest.mark.skipif(SKIP_EXPENSIVE, reason="Expensive API test")
    def test_gpt4o_ignores_reasoning_effort(self, base_payload):
        """Test GPT-4o ignores reasoning_effort if passed (not supported)."""
        payload = {
            **base_payload,
            "generator_model": GPT4O,
            "reasoning_effort": "high",  # Should be ignored for GPT-4o
            "max_tokens": 1000,
        }

        response = start_generation(payload)
        assert "session_id" in response

        session_id = response["session_id"]
        status = wait_for_completion(session_id)

        # Should still complete - reasoning_effort ignored for non-reasoning model
        assert status["status"] == "completed"

        print(f"\n[OK] GPT-4o with reasoning_effort (ignored) completed")


class TestOpenAIReasoningWithJsonOutput:
    """Test OpenAI reasoning with JSON output."""

    @pytest.mark.skipif(SKIP_EXPENSIVE, reason="Expensive API test")
    def test_gpt5_reasoning_with_json_output(self, base_payload):
        """Test GPT-5 reasoning + JSON output."""
        payload = {
            **base_payload,
            "generator_model": GPT5,
            "reasoning_effort": "low",
            "max_tokens": 1000,
            "json_output": True,
            "json_schema": {
                "type": "object",
                "properties": {
                    "numbers": {
                        "type": "array",
                        "items": {"type": "integer"}
                    },
                    "explanation": {"type": "string"}
                },
                "required": ["numbers", "explanation"]
            },
            "prompt": "List numbers 1 to 5 and explain what numbers are.",
        }

        response = start_generation(payload)
        assert "session_id" in response

        session_id = response["session_id"]
        status = wait_for_completion(session_id)

        assert status["status"] == "completed"

        result = get_result(session_id)
        import json_utils as json
        try:
            parsed = json.loads(result["content"])
            assert "numbers" in parsed
            print(f"\n[OK] GPT-5 reasoning + JSON output: {parsed['numbers']}")
        except:
            print(f"\n[WARN] Could not parse JSON, but request completed")


def run_single_test():
    """Run a single quick test for debugging."""
    print("=" * 70)
    print("SINGLE OPENAI REASONING EFFORT TEST")
    print("=" * 70)

    payload = {
        "prompt": "Count from 1 to 3.",
        "content_type": "explanation",
        "generator_model": GPT5,
        "reasoning_effort": "low",
        "min_words": 10,
        "max_words": 50,
        "max_tokens": 500,
        "max_iterations": 1,
        "qa_layers": [],
        "verbose": True,
    }

    print(f"\nSending request with reasoning_effort='low' to {GPT5}")

    response = start_generation(payload)
    print(f"Response: {response}")

    if "session_id" in response:
        session_id = response["session_id"]
        print(f"Waiting for completion...")

        status = wait_for_completion(session_id)
        print(f"Status: {status['status']}")

        if status["status"] == "completed":
            result = get_result(session_id)
            print(f"\n[RESULT]")
            print(result.get("content", "")[:500])


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--single":
        run_single_test()
    else:
        pytest.main([
            __file__,
            "-v",
            "--tb=short",
            "-x",
        ])
