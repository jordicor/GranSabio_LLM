"""
Integration tests for Claude thinking tokens via real API calls.

Tests that thinking_budget_tokens and reasoning_effort are correctly
handled when making actual requests to Claude models through the API.

IMPORTANT: These tests make real API calls and incur costs.
Set SKIP_EXPENSIVE_TESTS=1 to skip these tests.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import asyncio
import httpx
from typing import Dict, Any, Optional
import time

# Skip expensive API tests unless explicitly enabled
SKIP_EXPENSIVE = os.environ.get("SKIP_EXPENSIVE_TESTS", "0") == "1"

# API configuration
API_BASE = os.environ.get("API_BASE", "http://localhost:8000")

# Test models - using cheaper models for testing
CLAUDE_SONNET = "claude-sonnet-4-20250514"
CLAUDE_SONNET_45 = "claude-sonnet-4-5"
CLAUDE_HAIKU = "claude-haiku-4-5"


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


class TestClaudeThinkingBudgetViaAPI:
    """Test Claude thinking budget through the generation API."""

    @pytest.mark.skipif(SKIP_EXPENSIVE, reason="Expensive API test")
    def test_claude_sonnet_with_thinking_budget_direct(self, base_payload):
        """Test Claude Sonnet with direct thinking_budget_tokens value."""
        payload = {
            **base_payload,
            "generator_model": CLAUDE_SONNET,
            "thinking_budget_tokens": 4096,
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

        print(f"\n[OK] Claude Sonnet with thinking_budget_tokens=4096 completed")
        print(f"    Content length: {len(result['content'])} chars")

    @pytest.mark.skipif(SKIP_EXPENSIVE, reason="Expensive API test")
    def test_claude_sonnet_with_reasoning_effort_low(self, base_payload):
        """Test Claude Sonnet with reasoning_effort=low (should convert to tokens)."""
        payload = {
            **base_payload,
            "generator_model": CLAUDE_SONNET,
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

        print(f"\n[OK] Claude Sonnet with reasoning_effort='low' completed")
        print(f"    Content length: {len(result['content'])} chars")

    @pytest.mark.skipif(SKIP_EXPENSIVE, reason="Expensive API test")
    def test_claude_sonnet_with_reasoning_effort_medium(self, base_payload):
        """Test Claude Sonnet with reasoning_effort=medium."""
        payload = {
            **base_payload,
            "generator_model": CLAUDE_SONNET,
            "reasoning_effort": "medium",
            "max_tokens": 1000,
        }

        response = start_generation(payload)
        assert "session_id" in response, f"Failed to start: {response}"

        session_id = response["session_id"]
        status = wait_for_completion(session_id)

        assert status["status"] == "completed", f"Generation failed: {status}"

        result = get_result(session_id)
        assert "content" in result

        print(f"\n[OK] Claude Sonnet with reasoning_effort='medium' completed")

    @pytest.mark.skipif(SKIP_EXPENSIVE, reason="Expensive API test")
    def test_claude_sonnet_with_reasoning_effort_high(self, base_payload):
        """Test Claude Sonnet with reasoning_effort=high."""
        payload = {
            **base_payload,
            "generator_model": CLAUDE_SONNET,
            "reasoning_effort": "high",
            "max_tokens": 2000,
        }

        response = start_generation(payload)
        assert "session_id" in response, f"Failed to start: {response}"

        session_id = response["session_id"]
        status = wait_for_completion(session_id, max_wait=180)

        assert status["status"] == "completed", f"Generation failed: {status}"

        result = get_result(session_id)
        assert "content" in result

        print(f"\n[OK] Claude Sonnet with reasoning_effort='high' completed")

    @pytest.mark.skipif(SKIP_EXPENSIVE, reason="Expensive API test")
    def test_claude_sonnet_45_with_thinking(self, base_payload):
        """Test Claude Sonnet 4.5 with thinking budget."""
        payload = {
            **base_payload,
            "generator_model": CLAUDE_SONNET_45,
            "thinking_budget_tokens": 8192,
            "max_tokens": 1500,
        }

        response = start_generation(payload)
        assert "session_id" in response, f"Failed to start: {response}"

        session_id = response["session_id"]
        status = wait_for_completion(session_id, max_wait=180)

        assert status["status"] == "completed", f"Generation failed: {status}"

        print(f"\n[OK] Claude Sonnet 4.5 with thinking_budget_tokens=8192 completed")

    @pytest.mark.skipif(SKIP_EXPENSIVE, reason="Expensive API test")
    def test_claude_haiku_with_thinking_disabled_by_default(self, base_payload):
        """Test Claude Haiku 4.5 - thinking disabled by default (default=0)."""
        payload = {
            **base_payload,
            "generator_model": CLAUDE_HAIKU,
            "max_tokens": 1000,
            # No thinking_budget_tokens - should use default of 0
        }

        response = start_generation(payload)
        assert "session_id" in response, f"Failed to start: {response}"

        session_id = response["session_id"]
        status = wait_for_completion(session_id)

        # Should complete without thinking (faster)
        assert status["status"] == "completed", f"Generation failed: {status}"

        print(f"\n[OK] Claude Haiku 4.5 without thinking (default=0) completed")

    @pytest.mark.skipif(SKIP_EXPENSIVE, reason="Expensive API test")
    def test_claude_haiku_with_thinking_enabled(self, base_payload):
        """Test Claude Haiku 4.5 with explicit thinking budget."""
        payload = {
            **base_payload,
            "generator_model": CLAUDE_HAIKU,
            "thinking_budget_tokens": 4096,
            "max_tokens": 1500,
        }

        response = start_generation(payload)
        assert "session_id" in response, f"Failed to start: {response}"

        session_id = response["session_id"]
        status = wait_for_completion(session_id, max_wait=120)

        assert status["status"] == "completed", f"Generation failed: {status}"

        print(f"\n[OK] Claude Haiku 4.5 with thinking_budget_tokens=4096 completed")


class TestClaudeThinkingBudgetAliases:
    """Test that reasoning_effort aliases work correctly for Claude."""

    @pytest.mark.skipif(SKIP_EXPENSIVE, reason="Expensive API test")
    def test_claude_with_min_alias(self, base_payload):
        """Test 'min' alias converts to 'minimal'."""
        payload = {
            **base_payload,
            "generator_model": CLAUDE_SONNET,
            "reasoning_effort": "min",
            "max_tokens": 1000,
        }

        response = start_generation(payload)
        assert "session_id" in response
        session_id = response["session_id"]
        status = wait_for_completion(session_id)
        assert status["status"] == "completed"

        print(f"\n[OK] Claude with reasoning_effort='min' (alias) completed")

    @pytest.mark.skipif(SKIP_EXPENSIVE, reason="Expensive API test")
    def test_claude_with_mid_alias(self, base_payload):
        """Test 'mid' alias converts to 'medium'."""
        payload = {
            **base_payload,
            "generator_model": CLAUDE_SONNET,
            "reasoning_effort": "mid",
            "max_tokens": 1000,
        }

        response = start_generation(payload)
        assert "session_id" in response
        session_id = response["session_id"]
        status = wait_for_completion(session_id)
        assert status["status"] == "completed"

        print(f"\n[OK] Claude with reasoning_effort='mid' (alias) completed")

    @pytest.mark.skipif(SKIP_EXPENSIVE, reason="Expensive API test")
    def test_claude_with_hi_alias(self, base_payload):
        """Test 'hi' alias converts to 'high'."""
        payload = {
            **base_payload,
            "generator_model": CLAUDE_SONNET,
            "reasoning_effort": "hi",
            "max_tokens": 2000,
        }

        response = start_generation(payload)
        assert "session_id" in response
        session_id = response["session_id"]
        status = wait_for_completion(session_id, max_wait=180)
        assert status["status"] == "completed"

        print(f"\n[OK] Claude with reasoning_effort='hi' (alias) completed")


class TestClaudeThinkingTokenValidation:
    """Test thinking token validation and adjustment for Claude."""

    @pytest.mark.skipif(SKIP_EXPENSIVE, reason="Expensive API test")
    def test_claude_thinking_budget_capped_by_model_limit(self, base_payload):
        """Test that extremely large thinking budget is capped."""
        payload = {
            **base_payload,
            "generator_model": CLAUDE_SONNET,
            "thinking_budget_tokens": 999999,  # Way above limit
            "max_tokens": 1000,
        }

        response = start_generation(payload)
        assert "session_id" in response

        session_id = response["session_id"]
        status = wait_for_completion(session_id, max_wait=180)

        # Should complete - budget should be capped, not rejected
        assert status["status"] == "completed", f"Should cap, not reject: {status}"

        print(f"\n[OK] Claude with excessive thinking budget was capped and completed")

    @pytest.mark.skipif(SKIP_EXPENSIVE, reason="Expensive API test")
    def test_claude_thinking_budget_below_minimum(self, base_payload):
        """Test that thinking budget below minimum is raised to minimum."""
        payload = {
            **base_payload,
            "generator_model": CLAUDE_SONNET,
            "thinking_budget_tokens": 100,  # Below 1024 minimum
            "max_tokens": 1000,
        }

        response = start_generation(payload)
        assert "session_id" in response

        session_id = response["session_id"]
        status = wait_for_completion(session_id)

        # Should complete - budget raised to minimum
        assert status["status"] == "completed"

        print(f"\n[OK] Claude with sub-minimum thinking budget was raised and completed")


def run_single_test():
    """Run a single quick test for debugging."""
    print("=" * 70)
    print("SINGLE CLAUDE THINKING TOKEN TEST")
    print("=" * 70)

    payload = {
        "prompt": "Count from 1 to 3.",
        "content_type": "explanation",
        "generator_model": CLAUDE_SONNET,
        "reasoning_effort": "low",
        "min_words": 10,
        "max_words": 50,
        "max_tokens": 500,
        "max_iterations": 1,
        "qa_layers": [],
        "verbose": True,
    }

    print(f"\nSending request with reasoning_effort='low' to {CLAUDE_SONNET}")

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
        # Run all tests
        pytest.main([
            __file__,
            "-v",
            "--tb=short",
            "-x",
        ])
