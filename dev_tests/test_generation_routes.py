"""Unit tests for generation route helpers."""

from types import SimpleNamespace

from models import ContentRequest
from core.generation_routes import _estimate_tokens_for_word_target, _model_default_max_tokens


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


class TestModelDefaultMaxTokens:
    """Tests for model-spec default generation budgets."""

    def test_uses_model_output_tokens_from_specs(self, monkeypatch):
        fake_config = SimpleNamespace(get_model_info=lambda _model: {"output_tokens": 32000})
        monkeypatch.setattr("core.generation_routes.config", fake_config)

        assert _model_default_max_tokens("claude-opus-4-7") == 32000

    def test_falls_back_to_8192_when_specs_have_no_output_limit(self, monkeypatch):
        fake_config = SimpleNamespace(get_model_info=lambda _model: {})
        monkeypatch.setattr("core.generation_routes.config", fake_config)

        assert _model_default_max_tokens("unknown-output-model") == 8192
