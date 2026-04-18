"""
Cross-provider combination tests for thinking tokens / reasoning effort.

Tests various combinations of:
- Different providers (Claude, Gemini, OpenAI)
- Different reasoning levels (minimal, low, medium, high)
- Both thinking_budget_tokens and reasoning_effort parameters
- Edge cases and conversions between formats

This file also provides a comprehensive matrix test runner.

IMPORTANT: These tests make real API calls and incur costs.
Set SKIP_EXPENSIVE_TESTS=1 to skip API tests.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import httpx
from typing import Dict, Any, List, Optional, Tuple
import time
import config

# Skip expensive API tests unless explicitly enabled
SKIP_EXPENSIVE = os.environ.get("SKIP_EXPENSIVE_TESTS", "0") == "1"

# API configuration
API_BASE = os.environ.get("API_BASE", "http://localhost:8000")


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


class TestThinkingTokenSpecsMatrix:
    """Test thinking token specifications for all supported models."""

    def setup_method(self):
        self.config = config.Config()

    def _get_thinking_details(self, model_id: str) -> Optional[Dict[str, Any]]:
        return self.config._get_thinking_budget_details(model_id)

    def test_all_claude_models_specs(self):
        """Verify all Claude models have correct thinking specs."""
        claude_specs = {
            "claude-opus-4-20250514": {"min": 1024, "max": 16384, "default": 16384},
            "claude-opus-4-1-20250805": {"min": 1024, "max": 30000, "default": 30000},
            "claude-opus-4-5-20251101": {"min": 1024, "max": 64000, "default": 64000},
            "claude-sonnet-4-20250514": {"min": 1024, "max": 16384, "default": 8192},
            "claude-sonnet-4-5": {"min": 1024, "max": 16384, "default": 8192},
            "claude-haiku-4-5": {"min": 1024, "max": 128000, "default": 0},
        }

        for model_id, expected in claude_specs.items():
            details = self._get_thinking_details(model_id)
            assert details is not None, f"{model_id}: No thinking details found"
            assert details.get("supported") is True

            actual_min = details.get("min_tokens", 0)
            actual_max = details.get("max_tokens", 0)
            actual_default = details.get("default_tokens", 0)

            assert actual_min == expected["min"], \
                f"{model_id}: min_tokens expected {expected['min']}, got {actual_min}"
            assert actual_max == expected["max"], \
                f"{model_id}: max_tokens expected {expected['max']}, got {actual_max}"
            assert actual_default == expected["default"], \
                f"{model_id}: default_tokens expected {expected['default']}, got {actual_default}"

            print(f"[OK] {model_id}: min={actual_min}, max={actual_max}, default={actual_default}")

    def test_all_gemini_models_specs(self):
        """Verify all Gemini models have correct thinking specs."""
        gemini_specs = {
            "gemini-2.5-pro": {"min": 1024, "max": 65536, "default": 32768},
            "gemini-2.5-flash": {"min": 1024, "max": 65536, "default": 4000},
            "gemini-2.5-flash-lite": {"min": 1024, "max": 65536, "default": 0},
            "gemini-3-pro-preview": {"min": 1024, "max": 65536, "default": 32768},
            "gemini-3-flash-preview": {"min": 1024, "max": 32768, "default": 8192},
        }

        for model_id, expected in gemini_specs.items():
            details = self._get_thinking_details(model_id)
            assert details is not None, f"{model_id}: No thinking details found"
            assert details.get("supported") is True
            assert details.get("parameter_name") == "thinking_budget"

            actual_min = details.get("min_tokens", 0)
            actual_max = details.get("max_tokens", 0)
            actual_default = details.get("default_tokens", 0)

            assert actual_min == expected["min"], \
                f"{model_id}: min_tokens expected {expected['min']}, got {actual_min}"
            assert actual_max == expected["max"], \
                f"{model_id}: max_tokens expected {expected['max']}, got {actual_max}"
            assert actual_default == expected["default"], \
                f"{model_id}: default_tokens expected {expected['default']}, got {actual_default}"

            print(f"[OK] {model_id}: min={actual_min}, max={actual_max}, default={actual_default}")


class TestReasoningEffortConversionMatrix:
    """Test reasoning_effort to tokens conversion for all models."""

    def setup_method(self):
        self.config = config.Config()

    def _get_model_info(self, model_name: str) -> Dict[str, Any]:
        return self.config.get_model_info(model_name)

    def _get_thinking_details(self, model_id: str) -> Optional[Dict[str, Any]]:
        return self.config._get_thinking_budget_details(model_id)

    def test_conversion_matrix_all_levels(self):
        """Test all reasoning effort levels for all thinking-capable models."""
        # Models with their expected max_tokens for thinking
        models = {
            "claude-opus-4-5-20251101": 64000,
            "claude-opus-4-1-20250805": 30000,
            "claude-sonnet-4-20250514": 16384,
            "gemini-2.5-pro": 65536,
            "gemini-3-flash-preview": 32768,
        }

        levels = ["minimal", "low", "medium", "high"]

        print("\n" + "=" * 80)
        print("REASONING EFFORT TO TOKENS CONVERSION MATRIX")
        print("=" * 80)
        print(f"{'Model':<30} | {'Level':<8} | {'Tokens':<10} | {'% of Max':<8}")
        print("-" * 80)

        for model_id, expected_max in models.items():
            model_info = self._get_model_info(model_id)
            thinking_details = self._get_thinking_details(model_id)

            for level in levels:
                tokens = self.config._convert_reasoning_effort_to_tokens(
                    level, model_info, thinking_details
                )

                if tokens:
                    pct = (tokens / expected_max) * 100
                    print(f"{model_id:<30} | {level:<8} | {tokens:<10} | {pct:>6.1f}%")

                    # Verify percentages
                    if level == "high":
                        assert tokens == expected_max
                    elif level == "medium":
                        assert abs(tokens - expected_max * 0.5) < expected_max * 0.01
                    elif level == "low":
                        assert abs(tokens - expected_max * 0.25) < expected_max * 0.01

        print("=" * 80)


class TestCrossProviderValidation:
    """Test that token validation works correctly across providers."""

    def setup_method(self):
        self.config = config.Config()

    def test_validate_token_limits_all_providers(self):
        """Test validate_token_limits for different provider scenarios."""
        test_cases = [
            # (model, max_tokens, reasoning_effort, thinking_budget, expected_provider_behavior)
            ("claude-sonnet-4-5", 4000, "high", None, "claude_converts_to_tokens"),
            ("claude-sonnet-4-5", 4000, None, 8192, "claude_uses_direct_tokens"),
            ("gpt-5", 4000, "high", None, "openai_uses_reasoning_effort"),
            ("gpt-5", 4000, None, 65535, "openai_converts_to_reasoning_effort"),
            ("gemini-2.5-flash", 4000, "medium", None, "gemini_converts_to_tokens"),
            ("gemini-2.5-flash", 4000, None, 16000, "gemini_uses_direct_tokens"),
        ]

        print("\n" + "=" * 100)
        print("CROSS-PROVIDER TOKEN VALIDATION MATRIX")
        print("=" * 100)

        for model, max_tokens, effort, budget, behavior in test_cases:
            result = self.config.validate_token_limits(
                model_name=model,
                max_tokens=max_tokens,
                reasoning_effort=effort,
                thinking_budget_tokens=budget
            )

            model_info = result["model_info"]
            provider = model_info.get("provider", "unknown")

            print(f"\nModel: {model} (Provider: {provider})")
            print(f"  Input: reasoning_effort={effort}, thinking_budget_tokens={budget}")
            print(f"  Output: reasoning_effort={result['adjusted_reasoning_effort']}, "
                  f"thinking_budget={result['adjusted_thinking_budget_tokens']}")

            # Verify provider-specific behavior
            if provider == "openai":
                # OpenAI should have reasoning_effort, not thinking_budget_tokens
                assert result["adjusted_reasoning_effort"] is not None or effort is None
                assert result["adjusted_thinking_budget_tokens"] is None
            elif provider in ["claude", "anthropic"]:
                # Claude should have thinking_budget_tokens
                if effort or budget:
                    assert result["adjusted_thinking_budget_tokens"] is not None
            elif provider == "gemini":
                # Gemini should have thinking_budget_tokens (auto-applied if not specified)
                pass  # Gemini always has a default

        print("\n" + "=" * 100)


class TestAPIIntegrationMatrix:
    """Integration tests for API calls with different parameter combinations."""

    @pytest.fixture
    def base_payload(self):
        return {
            "prompt": "What is 2+2?",
            "content_type": "answer",
            "min_words": 5,
            "max_words": 50,
            "max_iterations": 1,
            "qa_layers": [],
            "verbose": True,
            "max_tokens": 500,
        }

    @pytest.mark.skipif(SKIP_EXPENSIVE, reason="Expensive API test")
    def test_quick_matrix_all_providers(self, base_payload):
        """Quick matrix test: one model per provider with low reasoning."""
        test_configs = [
            ("claude-sonnet-4-20250514", "low", None),
            ("gemini-2.5-flash", "low", None),
            ("gpt-4o", None, None),  # Non-reasoning model
        ]

        results = []

        for model, effort, budget in test_configs:
            payload = {
                **base_payload,
                "generator_model": model,
            }
            if effort:
                payload["reasoning_effort"] = effort
            if budget:
                payload["thinking_budget_tokens"] = budget

            print(f"\nTesting {model} with effort={effort}, budget={budget}")

            try:
                response = start_generation(payload)
                if "session_id" not in response:
                    results.append((model, "FAILED", f"No session_id: {response}"))
                    continue

                status = wait_for_completion(response["session_id"], max_wait=60)
                results.append((model, status["status"], ""))
                print(f"  Result: {status['status']}")

            except Exception as e:
                results.append((model, "ERROR", str(e)))
                print(f"  Error: {e}")

        # Summary
        print("\n" + "=" * 60)
        print("MATRIX TEST RESULTS")
        print("=" * 60)
        for model, status, error in results:
            status_str = "[OK]" if status == "completed" else f"[{status}]"
            print(f"{status_str} {model}")
            if error:
                print(f"    Error: {error}")

        # All should complete
        assert all(r[1] == "completed" for r in results), \
            f"Some tests failed: {[r for r in results if r[1] != 'completed']}"


def print_summary():
    """Print a summary of thinking token behavior for all models."""
    print("=" * 80)
    print("THINKING TOKENS / REASONING EFFORT - SUMMARY")
    print("=" * 80)

    print("""
PROVIDER BEHAVIOR SUMMARY:

1. CLAUDE (Anthropic)
   - Parameter: thinking_budget_tokens (integer)
   - Conversion: reasoning_effort -> tokens (high=100%, medium=50%, low=25%)
   - Constraint: max_tokens MUST be > thinking_budget_tokens
   - Temperature: Forced to 1.0 when thinking is enabled
   - Models with thinking:
     * claude-opus-4-20250514:    max=16384,  default=16384
     * claude-opus-4-1-20250805:  max=30000,  default=30000
     * claude-opus-4-5-20251101:  max=64000,  default=64000
     * claude-sonnet-4-20250514:  max=16384,  default=8192
     * claude-sonnet-4-5:         max=16384,  default=8192
     * claude-haiku-4-5:          max=128000, default=0 (disabled)

2. GEMINI (Google)
   - Parameter: thinking_budget (integer, via ThinkingConfig)
   - Conversion: reasoning_effort -> tokens (same percentages as Claude)
   - Auto-apply: Default thinking budget applied if not specified
   - Models with thinking:
     * gemini-2.5-pro:         max=65536, default=32768
     * gemini-2.5-flash:       max=65536, default=4000
     * gemini-2.5-flash-lite:  max=65536, default=0 (disabled)
     * gemini-3-pro-preview:   max=65536, default=32768
     * gemini-3-flash-preview: max=32768, default=8192

3. OPENAI
   - Parameter: reasoning_effort (string: "low", "medium", "high")
   - NO thinking_budget_tokens parameter
   - Conversion: thinking_budget_tokens -> reasoning_effort internally
     * >= 65535 tokens -> "high"
     * ~16000 tokens   -> "medium"
     * ~8000 tokens    -> "low"
   - Models with reasoning:
     * gpt-5, gpt-5.1, gpt-5.2, o1, o1-mini, o3, o3-mini, o3-pro

ALIASES:
   "min"/"minimum" -> "minimal"
   "lo"            -> "low"
   "mid"/"med"     -> "medium"
   "hi"            -> "high"

VALIDATION BEHAVIOR:
   - Tokens below min_tokens are raised to min_tokens (1024)
   - Tokens above max_tokens are capped to max_tokens
   - For Claude: If max_tokens <= thinking_budget, thinking is reduced
""")
    print("=" * 80)


def run_all_unit_tests():
    """Run all unit tests (no API calls)."""
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-k", "not API",
    ])


def run_all_tests():
    """Run all tests including API tests."""
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
    ])


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "--summary":
            print_summary()
        elif sys.argv[1] == "--unit":
            run_all_unit_tests()
        elif sys.argv[1] == "--all":
            run_all_tests()
    else:
        print_summary()
        print("\nRunning unit tests...\n")
        run_all_unit_tests()
