"""Provider finish/stop reason normalization tests."""

from ai_runtime import usage as runtime_usage


def test_provider_token_limit_finish_reasons_are_exact():
    assert runtime_usage.is_token_limit_finish_reason("length", provider="openai")
    assert runtime_usage.is_token_limit_finish_reason("max_tokens", provider="claude")
    assert runtime_usage.is_token_limit_finish_reason("MAX_TOKENS", provider="gemini")
    assert runtime_usage.is_token_limit_finish_reason("length", provider="openrouter")
    assert runtime_usage.is_token_limit_finish_reason("max_output_tokens", provider="xai")
    assert runtime_usage.is_token_limit_finish_reason("length", provider="moonshot")
    assert runtime_usage.is_token_limit_finish_reason("length", provider="minimax")

    assert not runtime_usage.is_token_limit_finish_reason("max_tokenizer_warning", provider="openai")
    assert not runtime_usage.is_token_limit_finish_reason("length", provider="claude")
    assert not runtime_usage.is_token_limit_finish_reason("length")


def test_provider_finish_reasons_get_actionable_categories():
    cases = [
        ("openai", "content_filter", "content_filter", False),
        ("openai", "max_output_tokens", "output_token_limit", True),
        ("claude", "model_context_window_exceeded", "context_limit", False),
        ("claude", "pause_turn", "pause_turn", False),
        ("claude", "refusal", "content_filter", False),
        ("gemini", "SAFETY", "content_filter", False),
        ("gemini", "RECITATION", "content_filter", False),
        ("openrouter", "error", "provider_error", False),
        ("moonshot", "tool_calls", "tool_calls", False),
        ("minimax", "length", "output_token_limit", True),
    ]

    for provider, reason, category, output_truncated in cases:
        classification = runtime_usage.classify_finish_reason(
            provider=provider,
            finish_reason=reason,
        )
        assert classification["finish_reason_category"] == category
        assert classification["output_truncated"] is output_truncated
        assert classification["finish_unusable"] is True


def test_success_finish_reasons_are_not_unusable():
    for provider, reason in [
        ("openai", "stop"),
        ("openai", "completed"),
        ("claude", "end_turn"),
        ("claude", "stop_sequence"),
        ("gemini", "STOP"),
        ("openrouter", "stop"),
        ("xai", "stop"),
        ("moonshot", "stop"),
        ("minimax", "stop"),
    ]:
        classification = runtime_usage.classify_finish_reason(
            provider=provider,
            finish_reason=reason,
        )
        assert classification["finish_reason_category"] == "stop"
        assert classification["finish_unusable"] is False
        assert classification["output_truncated"] is False


def test_unknown_finish_reason_is_not_token_limit_wildcard():
    classification = runtime_usage.classify_finish_reason(
        provider="openai",
        finish_reason="max_tokenizer_warning",
    )

    assert classification["finish_reason_category"] == "unknown"
    assert classification["finish_reason_known"] is False
    assert classification["finish_unusable"] is False
    assert classification["output_truncated"] is False
