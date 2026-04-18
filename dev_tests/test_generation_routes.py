"""Unit tests for generation route helpers."""

from models import ContentRequest
from core.generation_routes import _estimate_tokens_for_word_target


class TestEstimateTokensForWordTarget:
    """Tests for word-target token budgeting."""

    def test_returns_none_without_word_limits(self):
        request = ContentRequest(
            prompt="Write something useful.",
            generator_model="gpt-4o",
            qa_layers=[],
            qa_models=[],
        )

        assert _estimate_tokens_for_word_target(request) is None

    def test_long_form_budget_exceeds_legacy_floor(self):
        request = ContentRequest(
            prompt="Write a complete science-fiction novella.",
            generator_model="gpt-4o",
            min_words=4200,
            max_words=4800,
            qa_layers=[],
            qa_models=[],
        )

        assert _estimate_tokens_for_word_target(request) == 11584

    def test_uses_min_words_when_max_words_missing(self):
        request = ContentRequest(
            prompt="Write a long essay.",
            generator_model="gpt-4o",
            min_words=3000,
            qa_layers=[],
            qa_models=[],
        )

        assert _estimate_tokens_for_word_target(request) == 8000
