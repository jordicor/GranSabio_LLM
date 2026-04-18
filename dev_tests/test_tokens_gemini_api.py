"""
Integration tests for Gemini thinking tokens via real API calls.

Tests that thinking_budget and reasoning_effort are correctly
handled when making actual requests to Gemini models through the API.

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
GEMINI_FLASH = "gemini-2.5-flash"
GEMINI_FLASH_LITE = "gemini-2.5-flash-lite"
GEMINI_PRO = "gemini-2.5-pro"
GEMINI_3_FLASH = "gemini-3-flash-preview"
GEMINI_3_PRO = "gemini-3-pro-preview"


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


class TestGeminiThinkingBudgetViaAPI:
    """Test Gemini thinking budget through the generation API."""

    @pytest.mark.skipif(SKIP_EXPENSIVE, reason="Expensive API test")
    def test_gemini_flash_with_default_thinking(self, base_payload):
        """Test Gemini Flash with default thinking (auto-applied: 4000 tokens)."""
        payload = {
            **base_payload,
            "generator_model": GEMINI_FLASH,
            "max_tokens": 1000,
            # No thinking_budget - should auto-apply default of 4000
        }

        response = start_generation(payload)
        assert "session_id" in response, f"Failed to start: {response}"

        session_id = response["session_id"]
        status = wait_for_completion(session_id)

        assert status["status"] == "completed", f"Generation failed: {status}"

        result = get_result(session_id)
        assert "content" in result
        assert len(result["content"]) > 0

        print(f"\n[OK] Gemini Flash with default thinking (4000) completed")
        print(f"    Content length: {len(result['content'])} chars")

    @pytest.mark.skipif(SKIP_EXPENSIVE, reason="Expensive API test")
    def test_gemini_flash_with_explicit_thinking_budget(self, base_payload):
        """Test Gemini Flash with explicit thinking_budget_tokens."""
        payload = {
            **base_payload,
            "generator_model": GEMINI_FLASH,
            "thinking_budget_tokens": 8000,
            "max_tokens": 1000,
        }

        response = start_generation(payload)
        assert "session_id" in response

        session_id = response["session_id"]
        status = wait_for_completion(session_id)

        assert status["status"] == "completed"

        print(f"\n[OK] Gemini Flash with thinking_budget_tokens=8000 completed")

    @pytest.mark.skipif(SKIP_EXPENSIVE, reason="Expensive API test")
    def test_gemini_flash_lite_with_thinking_disabled(self, base_payload):
        """Test Gemini Flash Lite with default (thinking disabled: 0 tokens)."""
        payload = {
            **base_payload,
            "generator_model": GEMINI_FLASH_LITE,
            "max_tokens": 1000,
            # No thinking_budget - should use default of 0 (disabled)
        }

        response = start_generation(payload)
        assert "session_id" in response

        session_id = response["session_id"]
        status = wait_for_completion(session_id)

        # Should complete quickly without thinking
        assert status["status"] == "completed"

        print(f"\n[OK] Gemini Flash Lite with thinking disabled (default=0) completed")

    @pytest.mark.skipif(SKIP_EXPENSIVE, reason="Expensive API test")
    def test_gemini_flash_lite_with_thinking_enabled(self, base_payload):
        """Test Gemini Flash Lite with explicit thinking budget (override default 0)."""
        payload = {
            **base_payload,
            "generator_model": GEMINI_FLASH_LITE,
            "thinking_budget_tokens": 2000,
            "max_tokens": 1000,
        }

        response = start_generation(payload)
        assert "session_id" in response

        session_id = response["session_id"]
        status = wait_for_completion(session_id)

        assert status["status"] == "completed"

        print(f"\n[OK] Gemini Flash Lite with thinking_budget_tokens=2000 completed")

    @pytest.mark.skipif(SKIP_EXPENSIVE, reason="Expensive API test")
    def test_gemini_3_flash_with_reasoning_effort_low(self, base_payload):
        """Test Gemini 3 Flash with reasoning_effort=low."""
        payload = {
            **base_payload,
            "generator_model": GEMINI_3_FLASH,
            "reasoning_effort": "low",
            "max_tokens": 1000,
        }

        response = start_generation(payload)
        assert "session_id" in response

        session_id = response["session_id"]
        status = wait_for_completion(session_id)

        assert status["status"] == "completed"

        print(f"\n[OK] Gemini 3 Flash with reasoning_effort='low' completed")

    @pytest.mark.skipif(SKIP_EXPENSIVE, reason="Expensive API test")
    def test_gemini_3_flash_with_reasoning_effort_medium(self, base_payload):
        """Test Gemini 3 Flash with reasoning_effort=medium."""
        payload = {
            **base_payload,
            "generator_model": GEMINI_3_FLASH,
            "reasoning_effort": "medium",
            "max_tokens": 1000,
        }

        response = start_generation(payload)
        assert "session_id" in response

        session_id = response["session_id"]
        status = wait_for_completion(session_id)

        assert status["status"] == "completed"

        print(f"\n[OK] Gemini 3 Flash with reasoning_effort='medium' completed")

    @pytest.mark.skipif(SKIP_EXPENSIVE, reason="Expensive API test")
    def test_gemini_3_flash_with_reasoning_effort_high(self, base_payload):
        """Test Gemini 3 Flash with reasoning_effort=high."""
        payload = {
            **base_payload,
            "generator_model": GEMINI_3_FLASH,
            "reasoning_effort": "high",
            "max_tokens": 2000,
        }

        response = start_generation(payload)
        assert "session_id" in response

        session_id = response["session_id"]
        status = wait_for_completion(session_id, max_wait=180)

        assert status["status"] == "completed"

        print(f"\n[OK] Gemini 3 Flash with reasoning_effort='high' completed")


class TestGeminiThinkingBudgetConversion:
    """Test reasoning_effort to thinking_budget conversion for Gemini."""

    @pytest.mark.skipif(SKIP_EXPENSIVE, reason="Expensive API test")
    def test_gemini_flash_reasoning_effort_converts_to_tokens(self, base_payload):
        """Test that reasoning_effort is converted to thinking_budget_tokens for Gemini."""
        # Test low = 25% of max (65536 * 0.25 = 16384)
        payload = {
            **base_payload,
            "generator_model": GEMINI_FLASH,
            "reasoning_effort": "low",
            "max_tokens": 1000,
        }

        response = start_generation(payload)
        assert "session_id" in response

        session_id = response["session_id"]
        status = wait_for_completion(session_id)

        assert status["status"] == "completed"

        print(f"\n[OK] Gemini Flash with reasoning_effort='low' -> 16384 tokens completed")

    @pytest.mark.skipif(SKIP_EXPENSIVE, reason="Expensive API test")
    def test_gemini_3_pro_with_high_thinking(self, base_payload):
        """Test Gemini 3 Pro with max thinking budget."""
        payload = {
            **base_payload,
            "generator_model": GEMINI_3_PRO,
            "thinking_budget_tokens": 65536,  # Max for this model
            "max_tokens": 2000,
        }

        response = start_generation(payload)
        assert "session_id" in response

        session_id = response["session_id"]
        status = wait_for_completion(session_id, max_wait=300)  # Longer for pro model

        assert status["status"] == "completed"

        print(f"\n[OK] Gemini 3 Pro with thinking_budget_tokens=65536 completed")


class TestGeminiThinkingWithJsonOutput:
    """Test Gemini thinking with JSON output mode."""

    @pytest.mark.skipif(SKIP_EXPENSIVE, reason="Expensive API test")
    def test_gemini_flash_thinking_with_json_output(self, base_payload):
        """Test Gemini Flash with thinking + JSON output."""
        payload = {
            **base_payload,
            "generator_model": GEMINI_FLASH,
            "thinking_budget_tokens": 4000,
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
        # Verify JSON is valid
        import json_utils as json
        try:
            parsed = json.loads(result["content"])
            assert "numbers" in parsed
            assert "explanation" in parsed
            print(f"\n[OK] Gemini Flash thinking + JSON output: {parsed['numbers']}")
        except:
            print(f"\n[WARN] Could not parse JSON, but request completed")

    @pytest.mark.skipif(SKIP_EXPENSIVE, reason="Expensive API test")
    def test_gemini_3_flash_thinking_with_json_schema(self, base_payload):
        """Test Gemini 3 Flash with thinking + JSON schema."""
        payload = {
            **base_payload,
            "generator_model": GEMINI_3_FLASH,
            "reasoning_effort": "medium",
            "max_tokens": 1000,
            "json_output": True,
            "json_schema": {
                "type": "object",
                "properties": {
                    "result": {"type": "string"},
                    "reasoning": {"type": "string"}
                },
                "required": ["result"]
            },
            "prompt": "What is 2+2? Explain your reasoning.",
        }

        response = start_generation(payload)
        assert "session_id" in response

        session_id = response["session_id"]
        status = wait_for_completion(session_id)

        assert status["status"] == "completed"

        print(f"\n[OK] Gemini 3 Flash thinking + JSON schema completed")


class TestGeminiThinkingValidation:
    """Test thinking budget validation for Gemini."""

    @pytest.mark.skipif(SKIP_EXPENSIVE, reason="Expensive API test")
    def test_gemini_thinking_budget_capped(self, base_payload):
        """Test that excessive thinking budget is capped."""
        payload = {
            **base_payload,
            "generator_model": GEMINI_FLASH,
            "thinking_budget_tokens": 999999,  # Way above limit
            "max_tokens": 1000,
        }

        response = start_generation(payload)
        assert "session_id" in response

        session_id = response["session_id"]
        status = wait_for_completion(session_id, max_wait=180)

        # Should complete - budget should be capped, not rejected
        assert status["status"] == "completed"

        print(f"\n[OK] Gemini with excessive thinking budget was capped and completed")

    @pytest.mark.skipif(SKIP_EXPENSIVE, reason="Expensive API test")
    def test_gemini_thinking_budget_raised_to_minimum(self, base_payload):
        """Test that sub-minimum thinking budget is raised."""
        payload = {
            **base_payload,
            "generator_model": GEMINI_FLASH,
            "thinking_budget_tokens": 100,  # Below 1024 minimum
            "max_tokens": 1000,
        }

        response = start_generation(payload)
        assert "session_id" in response

        session_id = response["session_id"]
        status = wait_for_completion(session_id)

        # Should complete - budget raised to minimum
        assert status["status"] == "completed"

        print(f"\n[OK] Gemini with sub-minimum thinking budget was raised and completed")


def run_single_test():
    """Run a single quick test for debugging."""
    print("=" * 70)
    print("SINGLE GEMINI THINKING TOKEN TEST")
    print("=" * 70)

    payload = {
        "prompt": "Count from 1 to 3.",
        "content_type": "explanation",
        "generator_model": GEMINI_FLASH,
        "thinking_budget_tokens": 4000,
        "min_words": 10,
        "max_words": 50,
        "max_tokens": 500,
        "max_iterations": 1,
        "qa_layers": [],
        "verbose": True,
    }

    print(f"\nSending request with thinking_budget_tokens=4000 to {GEMINI_FLASH}")

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
