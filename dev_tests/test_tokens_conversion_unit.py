"""
Unit tests for thinking tokens conversion logic.

Tests the internal conversion between:
- reasoning_effort (high/medium/low/minimal) <-> thinking_budget_tokens (number)
- Model-specific thinking budget calculations
- Provider-specific parameter handling

Does NOT make real API calls - tests config logic only.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from typing import Dict, Any, Optional
import config


class TestReasoningEffortNormalization:
    """Test normalization of reasoning effort labels."""

    def setup_method(self):
        self.config = config.Config()

    def test_normalize_standard_labels(self):
        """Test that standard labels are preserved."""
        assert self.config.normalize_reasoning_effort_label("none") == "none"
        assert self.config.normalize_reasoning_effort_label("minimal") == "minimal"
        assert self.config.normalize_reasoning_effort_label("low") == "low"
        assert self.config.normalize_reasoning_effort_label("medium") == "medium"
        assert self.config.normalize_reasoning_effort_label("high") == "high"
        assert self.config.normalize_reasoning_effort_label("xhigh") == "xhigh"

    def test_normalize_aliases(self):
        """Test that aliases are correctly mapped."""
        # mid -> medium
        assert self.config.normalize_reasoning_effort_label("mid") == "medium"
        assert self.config.normalize_reasoning_effort_label("med") == "medium"

        # hi -> high
        assert self.config.normalize_reasoning_effort_label("hi") == "high"

        # lo -> low
        assert self.config.normalize_reasoning_effort_label("lo") == "low"

        # min -> minimal
        assert self.config.normalize_reasoning_effort_label("min") == "minimal"
        assert self.config.normalize_reasoning_effort_label("minimum") == "minimal"
        assert self.config.normalize_reasoning_effort_label("off") == "none"
        assert self.config.normalize_reasoning_effort_label("xh") == "xhigh"

    def test_normalize_case_insensitive(self):
        """Test that normalization is case-insensitive."""
        assert self.config.normalize_reasoning_effort_label("NONE") == "none"
        assert self.config.normalize_reasoning_effort_label("HIGH") == "high"
        assert self.config.normalize_reasoning_effort_label("Medium") == "medium"
        assert self.config.normalize_reasoning_effort_label("LOW") == "low"
        assert self.config.normalize_reasoning_effort_label("MINIMAL") == "minimal"
        assert self.config.normalize_reasoning_effort_label("MID") == "medium"
        assert self.config.normalize_reasoning_effort_label("XHIGH") == "xhigh"

    def test_normalize_none_input(self):
        """Test that None returns None."""
        assert self.config.normalize_reasoning_effort_label(None) is None

    def test_normalize_invalid_label(self):
        """Test that invalid labels return None."""
        assert self.config.normalize_reasoning_effort_label("invalid") is None
        assert self.config.normalize_reasoning_effort_label("extreme") is None
        assert self.config.normalize_reasoning_effort_label("") is None


class TestConvertReasoningEffortToTokens:
    """Test conversion from reasoning effort labels to token counts."""

    def setup_method(self):
        self.config = config.Config()

    def _get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Helper to get model info."""
        return self.config.get_model_info(model_name)

    def _get_thinking_details(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Helper to get thinking budget details."""
        return self.config._get_thinking_budget_details(model_id)

    # Claude Opus 4.5 tests (max: 64000)
    def test_claude_opus_45_high(self):
        """Claude Opus 4.5 with high effort should use max tokens (64000)."""
        model_info = self._get_model_info("claude-opus-4-5-20251101")
        thinking_details = self._get_thinking_details("claude-opus-4-5-20251101")

        tokens = self.config._convert_reasoning_effort_to_tokens(
            "high", model_info, thinking_details
        )

        assert tokens == 64000, f"Expected 64000, got {tokens}"

    def test_claude_opus_45_medium(self):
        """Claude Opus 4.5 with medium effort should use 50% of max (32000)."""
        model_info = self._get_model_info("claude-opus-4-5-20251101")
        thinking_details = self._get_thinking_details("claude-opus-4-5-20251101")

        tokens = self.config._convert_reasoning_effort_to_tokens(
            "medium", model_info, thinking_details
        )

        assert tokens == 32000, f"Expected 32000, got {tokens}"

    def test_claude_opus_45_low(self):
        """Claude Opus 4.5 with low effort should use 25% of max (16000)."""
        model_info = self._get_model_info("claude-opus-4-5-20251101")
        thinking_details = self._get_thinking_details("claude-opus-4-5-20251101")

        tokens = self.config._convert_reasoning_effort_to_tokens(
            "low", model_info, thinking_details
        )

        assert tokens == 16000, f"Expected 16000, got {tokens}"

    def test_claude_opus_45_minimal(self):
        """Claude Opus 4.5 with minimal effort should use default or min tokens."""
        model_info = self._get_model_info("claude-opus-4-5-20251101")
        thinking_details = self._get_thinking_details("claude-opus-4-5-20251101")

        tokens = self.config._convert_reasoning_effort_to_tokens(
            "minimal", model_info, thinking_details
        )

        # minimal uses max(min_tokens, default_tokens, 1024)
        # For Opus 4.5: min_tokens=1024, default_tokens=64000
        # So it should be 64000 (the default)
        assert tokens >= 1024, f"Expected at least 1024, got {tokens}"

    # Claude Opus 4.1 tests (max: 30000)
    def test_claude_opus_41_high(self):
        """Claude Opus 4.1 with high effort should use max tokens (30000)."""
        model_info = self._get_model_info("claude-opus-4-1-20250805")
        thinking_details = self._get_thinking_details("claude-opus-4-1-20250805")

        tokens = self.config._convert_reasoning_effort_to_tokens(
            "high", model_info, thinking_details
        )

        assert tokens == 30000, f"Expected 30000, got {tokens}"

    def test_claude_opus_41_medium(self):
        """Claude Opus 4.1 with medium effort should use 50% of max (15000)."""
        model_info = self._get_model_info("claude-opus-4-1-20250805")
        thinking_details = self._get_thinking_details("claude-opus-4-1-20250805")

        tokens = self.config._convert_reasoning_effort_to_tokens(
            "medium", model_info, thinking_details
        )

        assert tokens == 15000, f"Expected 15000, got {tokens}"

    def test_claude_opus_41_low(self):
        """Claude Opus 4.1 with low effort should use 25% of max (7500)."""
        model_info = self._get_model_info("claude-opus-4-1-20250805")
        thinking_details = self._get_thinking_details("claude-opus-4-1-20250805")

        tokens = self.config._convert_reasoning_effort_to_tokens(
            "low", model_info, thinking_details
        )

        assert tokens == 7500, f"Expected 7500, got {tokens}"

    # Claude Sonnet 4 tests (max: 16384)
    def test_claude_sonnet_4_high(self):
        """Claude Sonnet 4 with high effort should use max tokens (16384)."""
        model_info = self._get_model_info("claude-sonnet-4-20250514")
        thinking_details = self._get_thinking_details("claude-sonnet-4-20250514")

        tokens = self.config._convert_reasoning_effort_to_tokens(
            "high", model_info, thinking_details
        )

        assert tokens == 16384, f"Expected 16384, got {tokens}"

    def test_claude_sonnet_4_medium(self):
        """Claude Sonnet 4 with medium effort should use 50% of max (8192)."""
        model_info = self._get_model_info("claude-sonnet-4-20250514")
        thinking_details = self._get_thinking_details("claude-sonnet-4-20250514")

        tokens = self.config._convert_reasoning_effort_to_tokens(
            "medium", model_info, thinking_details
        )

        assert tokens == 8192, f"Expected 8192, got {tokens}"

    def test_claude_sonnet_4_low(self):
        """Claude Sonnet 4 with low effort should use 25% of max (4096)."""
        model_info = self._get_model_info("claude-sonnet-4-20250514")
        thinking_details = self._get_thinking_details("claude-sonnet-4-20250514")

        tokens = self.config._convert_reasoning_effort_to_tokens(
            "low", model_info, thinking_details
        )

        assert tokens == 4096, f"Expected 4096, got {tokens}"

    # Gemini 2.5 Pro tests (max: 65536)
    def test_gemini_25_pro_high(self):
        """Gemini 2.5 Pro with high effort should use max tokens (65536)."""
        model_info = self._get_model_info("gemini-2.5-pro")
        thinking_details = self._get_thinking_details("gemini-2.5-pro")

        tokens = self.config._convert_reasoning_effort_to_tokens(
            "high", model_info, thinking_details
        )

        assert tokens == 65536, f"Expected 65536, got {tokens}"

    def test_gemini_25_pro_medium(self):
        """Gemini 2.5 Pro with medium effort should use 50% of max (32768)."""
        model_info = self._get_model_info("gemini-2.5-pro")
        thinking_details = self._get_thinking_details("gemini-2.5-pro")

        tokens = self.config._convert_reasoning_effort_to_tokens(
            "medium", model_info, thinking_details
        )

        assert tokens == 32768, f"Expected 32768, got {tokens}"

    def test_gemini_25_pro_low(self):
        """Gemini 2.5 Pro with low effort should use 25% of max (16384)."""
        model_info = self._get_model_info("gemini-2.5-pro")
        thinking_details = self._get_thinking_details("gemini-2.5-pro")

        tokens = self.config._convert_reasoning_effort_to_tokens(
            "low", model_info, thinking_details
        )

        assert tokens == 16384, f"Expected 16384, got {tokens}"

    # Gemini 3 Flash tests (max: 32768)
    def test_gemini_3_flash_high(self):
        """Gemini 3 Flash with high effort should use max tokens (32768)."""
        model_info = self._get_model_info("gemini-3-flash-preview")
        thinking_details = self._get_thinking_details("gemini-3-flash-preview")

        tokens = self.config._convert_reasoning_effort_to_tokens(
            "high", model_info, thinking_details
        )

        assert tokens == 32768, f"Expected 32768, got {tokens}"

    def test_gemini_3_flash_medium(self):
        """Gemini 3 Flash with medium effort should use 50% of max (16384)."""
        model_info = self._get_model_info("gemini-3-flash-preview")
        thinking_details = self._get_thinking_details("gemini-3-flash-preview")

        tokens = self.config._convert_reasoning_effort_to_tokens(
            "medium", model_info, thinking_details
        )

        assert tokens == 16384, f"Expected 16384, got {tokens}"


class TestConvertTokensToReasoningEffort:
    """Test conversion from token counts to reasoning effort labels (for OpenAI)."""

    def setup_method(self):
        self.config = config.Config()

    def test_high_tokens_to_high_effort(self):
        """Tokens >= 65535 should map to high effort."""
        assert self.config._convert_tokens_to_reasoning_effort(65535) == "high"
        assert self.config._convert_tokens_to_reasoning_effort(100000) == "high"
        assert self.config._convert_tokens_to_reasoning_effort(65536) == "high"

    def test_medium_tokens_to_medium_effort(self):
        """Tokens around 16000 should map to medium effort."""
        assert self.config._convert_tokens_to_reasoning_effort(16000) == "medium"
        assert self.config._convert_tokens_to_reasoning_effort(15000) == "medium"
        assert self.config._convert_tokens_to_reasoning_effort(18000) == "medium"

    def test_low_tokens_to_low_effort(self):
        """Tokens around 8000 should map to low effort."""
        assert self.config._convert_tokens_to_reasoning_effort(8000) == "low"
        assert self.config._convert_tokens_to_reasoning_effort(7000) == "low"
        assert self.config._convert_tokens_to_reasoning_effort(9000) == "low"

    def test_very_low_tokens_to_low_effort(self):
        """Very low positive tokens should still map to low effort."""
        assert self.config._convert_tokens_to_reasoning_effort(1000) == "low"

    def test_zero_tokens_to_none_effort(self):
        """Zero thinking tokens should map to disabled reasoning."""
        assert self.config._convert_tokens_to_reasoning_effort(0) == "none"

    def test_none_defaults_to_medium(self):
        """None should default to medium effort."""
        assert self.config._convert_tokens_to_reasoning_effort(None) == "medium"


class TestInferReasoningEffortFromTokens:
    """Test inference of reasoning effort from token ratio."""

    def setup_method(self):
        self.config = config.Config()

    def _get_thinking_details(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Helper to get thinking budget details."""
        return self.config._get_thinking_budget_details(model_id)

    def test_infer_high_for_full_budget(self):
        """Full budget should infer as high effort."""
        # Claude Opus 4.5: max=64000
        details = self._get_thinking_details("claude-opus-4-5-20251101")

        effort = self.config._infer_reasoning_effort_from_tokens(64000, details)
        assert effort == "high", f"Expected 'high', got '{effort}'"

    def test_infer_medium_for_half_budget(self):
        """Half budget should infer as medium effort."""
        # Claude Opus 4.5: max=64000, half=32000
        details = self._get_thinking_details("claude-opus-4-5-20251101")

        effort = self.config._infer_reasoning_effort_from_tokens(32000, details)
        assert effort == "medium", f"Expected 'medium', got '{effort}'"

    def test_infer_low_for_quarter_budget(self):
        """Quarter budget should infer as low effort."""
        # Claude Opus 4.5: max=64000, quarter=16000
        details = self._get_thinking_details("claude-opus-4-5-20251101")

        effort = self.config._infer_reasoning_effort_from_tokens(16000, details)
        assert effort == "low", f"Expected 'low', got '{effort}'"

    def test_infer_returns_none_for_no_details(self):
        """Should return None if no thinking details."""
        effort = self.config._infer_reasoning_effort_from_tokens(10000, None)
        assert effort is None


class TestValidateTokenLimits:
    """Test the full validate_token_limits function."""

    def setup_method(self):
        self.config = config.Config()

    # Claude tests - should produce thinking_budget_tokens
    # NOTE: Claude has constraint that max_tokens > thinking_budget_tokens
    # So when max_tokens is small, thinking_budget is capped by model's output_tokens limit

    def test_claude_with_reasoning_effort_high(self):
        """Claude with reasoning_effort=high converts to tokens but may be capped by output limit."""
        result = self.config.validate_token_limits(
            model_name="claude-opus-4-5-20251101",
            max_tokens=4000,
            reasoning_effort="high"
        )

        # The conversion to tokens happens (64000 initially)
        # But it gets capped due to model output limit (32000 * 0.95 = 30400)
        # and the Claude constraint (max_tokens > thinking_budget)
        # After adjustment, reasoning_effort is re-inferred from actual tokens
        assert result["adjusted_thinking_budget_tokens"] is not None
        assert result["adjusted_thinking_budget_tokens"] > 0
        # NOTE: After capping, the inferred effort may change from "high" to "medium"
        # because the actual tokens are now ~50% of max (29888 vs 64000)
        assert result["adjusted_reasoning_effort"] in ["high", "medium"]

    def test_claude_with_reasoning_effort_high_large_max_tokens(self):
        """Claude with large max_tokens and high effort should get full budget."""
        result = self.config.validate_token_limits(
            model_name="claude-opus-4-5-20251101",
            max_tokens=30000,  # Large enough to accommodate thinking
            reasoning_effort="high"
        )

        # With sufficient max_tokens, thinking budget can be larger
        assert result["adjusted_thinking_budget_tokens"] is not None
        assert result["adjusted_thinking_budget_tokens"] > 0
        # Should be capped by safe output limit, not the small max_tokens
        assert result["adjusted_thinking_budget_tokens"] <= 64000

    def test_claude_with_reasoning_effort_medium(self):
        """Claude with reasoning_effort=medium converts to tokens but may be capped."""
        result = self.config.validate_token_limits(
            model_name="claude-opus-4-5-20251101",
            max_tokens=4000,
            reasoning_effort="medium"
        )

        # Initially 32000, but gets capped by model output limit
        assert result["adjusted_thinking_budget_tokens"] is not None
        assert result["adjusted_thinking_budget_tokens"] > 0
        # Effort is preserved even when tokens are adjusted
        assert result["adjusted_reasoning_effort"] in ["medium", "high"]

    def test_claude_with_reasoning_effort_low(self):
        """Claude with reasoning_effort=low should convert to 25% thinking_budget_tokens."""
        result = self.config.validate_token_limits(
            model_name="claude-opus-4-5-20251101",
            max_tokens=4000,
            reasoning_effort="low"
        )

        # 16000 is within the model limit, should not be capped much
        assert result["adjusted_thinking_budget_tokens"] == 16000, \
            f"Expected 16000, got {result['adjusted_thinking_budget_tokens']}"

    def test_claude_with_direct_thinking_budget(self):
        """Claude with direct thinking_budget_tokens should preserve the value."""
        result = self.config.validate_token_limits(
            model_name="claude-opus-4-5-20251101",
            max_tokens=4000,
            thinking_budget_tokens=50000
        )

        # Should use the provided value (may be adjusted to limits)
        assert result["adjusted_thinking_budget_tokens"] is not None
        assert result["adjusted_thinking_budget_tokens"] <= 64000

    # OpenAI tests - should produce reasoning_effort
    def test_openai_with_thinking_budget_high(self):
        """OpenAI with high thinking_budget_tokens should convert to reasoning_effort=high."""
        result = self.config.validate_token_limits(
            model_name="gpt-5",
            max_tokens=4000,
            thinking_budget_tokens=65535
        )

        # Should convert to reasoning_effort
        assert result["adjusted_reasoning_effort"] == "high", \
            f"Expected 'high', got '{result['adjusted_reasoning_effort']}'"
        # OpenAI doesn't use thinking_budget_tokens
        assert result["adjusted_thinking_budget_tokens"] is None

    def test_openai_with_thinking_budget_medium(self):
        """OpenAI with medium thinking_budget_tokens should convert to reasoning_effort=medium."""
        result = self.config.validate_token_limits(
            model_name="gpt-5",
            max_tokens=4000,
            thinking_budget_tokens=16000
        )

        assert result["adjusted_reasoning_effort"] == "medium", \
            f"Expected 'medium', got '{result['adjusted_reasoning_effort']}'"

    def test_openai_with_thinking_budget_low(self):
        """OpenAI with low thinking_budget_tokens should convert to reasoning_effort=low."""
        result = self.config.validate_token_limits(
            model_name="gpt-5",
            max_tokens=4000,
            thinking_budget_tokens=8000
        )

        assert result["adjusted_reasoning_effort"] == "low", \
            f"Expected 'low', got '{result['adjusted_reasoning_effort']}'"

    def test_openai_with_direct_reasoning_effort(self):
        """OpenAI with direct reasoning_effort should preserve it."""
        result = self.config.validate_token_limits(
            model_name="gpt-5",
            max_tokens=4000,
            reasoning_effort="high"
        )

        assert result["adjusted_reasoning_effort"] == "high"

    # Gemini tests - should produce thinking_budget_tokens
    def test_gemini_with_reasoning_effort_high(self):
        """Gemini with reasoning_effort=high should convert to max thinking_budget_tokens."""
        result = self.config.validate_token_limits(
            model_name="gemini-2.5-pro",
            max_tokens=4000,
            reasoning_effort="high"
        )

        assert result["adjusted_thinking_budget_tokens"] == 65536, \
            f"Expected 65536, got {result['adjusted_thinking_budget_tokens']}"

    def test_gemini_with_default_thinking(self):
        """Gemini without thinking params should get default thinking budget."""
        result = self.config.validate_token_limits(
            model_name="gemini-2.5-pro",
            max_tokens=4000
        )

        # Gemini 2.5 Pro default is 32768
        assert result["adjusted_thinking_budget_tokens"] == 32768, \
            f"Expected 32768, got {result['adjusted_thinking_budget_tokens']}"


class TestDifferentModelsTokenCalculations:
    """Test token calculations for different models to verify model-specific behavior."""

    def setup_method(self):
        self.config = config.Config()

    def _get_thinking_details(self, model_id: str) -> Optional[Dict[str, Any]]:
        return self.config._get_thinking_budget_details(model_id)

    def test_all_claude_models_high(self):
        """Test high effort produces max_tokens for all Claude models."""
        claude_models = {
            "claude-opus-4-20250514": 16384,
            "claude-opus-4-1-20250805": 30000,
            "claude-opus-4-5-20251101": 64000,
            "claude-sonnet-4-20250514": 16384,
            "claude-sonnet-4-5": 16384,
            "claude-haiku-4-5": 128000,
        }

        for model_id, expected_max in claude_models.items():
            model_info = self.config.get_model_info(model_id)
            thinking_details = self._get_thinking_details(model_id)

            if thinking_details and thinking_details.get("supported"):
                tokens = self.config._convert_reasoning_effort_to_tokens(
                    "high", model_info, thinking_details
                )
                assert tokens == expected_max, \
                    f"{model_id}: Expected {expected_max}, got {tokens}"

    def test_all_gemini_models_high(self):
        """Test high effort produces max_tokens for all Gemini models with thinking."""
        gemini_models = {
            "gemini-2.5-pro": 65536,
            "gemini-2.5-flash": 65536,
            "gemini-2.5-flash-lite": 65536,
            "gemini-3-pro-preview": 65536,
            "gemini-3-flash-preview": 32768,
        }

        for model_id, expected_max in gemini_models.items():
            model_info = self.config.get_model_info(model_id)
            thinking_details = self._get_thinking_details(model_id)

            if thinking_details and thinking_details.get("supported"):
                tokens = self.config._convert_reasoning_effort_to_tokens(
                    "high", model_info, thinking_details
                )
                assert tokens == expected_max, \
                    f"{model_id}: Expected {expected_max}, got {tokens}"

    def test_all_claude_models_low(self):
        """Test low effort produces 25% of max_tokens for Claude models."""
        claude_models = {
            "claude-opus-4-20250514": 16384 // 4,  # 4096
            "claude-opus-4-1-20250805": 30000 // 4,  # 7500
            "claude-opus-4-5-20251101": 64000 // 4,  # 16000
            "claude-sonnet-4-20250514": 16384 // 4,  # 4096
        }

        for model_id, expected_low in claude_models.items():
            model_info = self.config.get_model_info(model_id)
            thinking_details = self._get_thinking_details(model_id)

            if thinking_details and thinking_details.get("supported"):
                tokens = self.config._convert_reasoning_effort_to_tokens(
                    "low", model_info, thinking_details
                )
                assert tokens == expected_low, \
                    f"{model_id}: Expected {expected_low}, got {tokens}"


class TestModelSpecsThinkingBudget:
    """Test that model specs have correct thinking budget configuration."""

    def setup_method(self):
        self.config = config.Config()
        self.model_specs = self.config.model_specs

    def _get_model_spec(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model spec by ID."""
        for provider_specs in self.model_specs.get("model_specifications", {}).values():
            for spec in provider_specs.values():
                if spec.get("model_id") == model_id:
                    return spec
        return None

    def test_claude_models_have_thinking_budget(self):
        """Claude models should have thinking_budget configuration."""
        claude_models = [
            "claude-opus-4-20250514",
            "claude-opus-4-1-20250805",
            "claude-opus-4-5-20251101",
            "claude-sonnet-4-20250514",
            "claude-sonnet-4-5",
            "claude-haiku-4-5",
        ]

        for model_id in claude_models:
            spec = self._get_model_spec(model_id)
            assert spec is not None, f"Model {model_id} not found in specs"

            thinking_budget = spec.get("thinking_budget", {})
            assert thinking_budget.get("supported") is True, \
                f"{model_id}: thinking_budget.supported should be True"
            assert "min_tokens" in thinking_budget, \
                f"{model_id}: missing min_tokens"
            assert "max_tokens" in thinking_budget, \
                f"{model_id}: missing max_tokens"
            assert "default_tokens" in thinking_budget, \
                f"{model_id}: missing default_tokens"

    def test_gemini_models_have_thinking_budget(self):
        """Gemini models should have thinking_budget configuration with parameter_name."""
        gemini_models = [
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-3-pro-preview",
            "gemini-3-flash-preview",
        ]

        for model_id in gemini_models:
            spec = self._get_model_spec(model_id)
            assert spec is not None, f"Model {model_id} not found in specs"

            thinking_budget = spec.get("thinking_budget", {})
            assert thinking_budget.get("supported") is True, \
                f"{model_id}: thinking_budget.supported should be True"
            assert thinking_budget.get("parameter_name") == "thinking_budget", \
                f"{model_id}: parameter_name should be 'thinking_budget'"

    def test_openai_models_have_reasoning_effort(self):
        """OpenAI models should have reasoning_effort configuration."""
        openai_models = [
            "gpt-5",
            "gpt-5.1",
            "gpt-5.2",
        ]

        for model_id in openai_models:
            spec = self._get_model_spec(model_id)
            assert spec is not None, f"Model {model_id} not found in specs"

            reasoning_effort = spec.get("reasoning_effort", {})
            assert reasoning_effort.get("supported") is True, \
                f"{model_id}: reasoning_effort.supported should be True"
            assert "levels" in reasoning_effort, \
                f"{model_id}: missing levels"
            assert "default" in reasoning_effort, \
                f"{model_id}: missing default"


def run_tests():
    """Run all unit tests and display results."""
    print("=" * 70)
    print("THINKING TOKENS CONVERSION UNIT TESTS")
    print("=" * 70)

    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-x",  # Stop on first failure for debugging
    ])


if __name__ == "__main__":
    run_tests()
