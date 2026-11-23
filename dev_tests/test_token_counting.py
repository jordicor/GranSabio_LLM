"""Offline counting reuse and payload identity; no provider or app startup."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

import context_budget as cb
from token_counting import TextComponent, TokenCounter


@pytest.mark.asyncio
async def test_reuse_preserves_duplicate_components_and_metadata():
    counter = TokenCounter()
    text = "Datos públicos conservados completos. 中文資料。"
    with patch.object(cb, "count_tokens", wraps=cb.count_tokens) as count:
        first, = await counter.components([TextComponent("source", text)], "chatgpt", "model")
        second, = await counter.components(
            [TextComponent("schema", text, "schema", keep_text=False)], "chatgpt", "model"
        )
    assert count.call_count == 1
    assert first.text == text and second.text is None
    assert first.name == "source" and second.kind == "schema"
    assert cb.count_tokens([first, second], "chatgpt", "model")[0] == 2 * first.tokens_estimate


@pytest.mark.asyncio
async def test_complete_text_and_count_parameters_are_part_of_identity():
    counter = TokenCounter()
    inputs = [
        ("evidence", "chatgpt", "model", "prose"),
        ("evidence!", "chatgpt", "model", "prose"),
        ("evidence", "ollama", "model", "prose"),
        ("evidence", "chatgpt", "other-model", "prose"),
        ("evidence", "chatgpt", "model", "json"),
    ]
    async def measure(item):
        text, provider, model, hint = item
        return await counter.components([TextComponent("body", text, hint=hint)], provider, model)

    with patch.object(cb, "count_tokens", wraps=cb.count_tokens) as count:
        expected = [await measure(item) for item in inputs]
        assert [await measure(item) for item in inputs] == expected
    assert count.call_count == len(inputs)


@pytest.mark.asyncio
async def test_cache_limit_never_changes_results_or_retains_oversized_text():
    counter = TokenCounter(max_bytes=1)
    specs = [TextComponent("body", "資料 complete evidence")]
    with patch.object(cb, "count_tokens", wraps=cb.count_tokens) as count:
        first = await counter.components(specs, "chatgpt", "model")
        assert await counter.components(specs, "chatgpt", "model") == first
    assert count.call_count == 2
    assert not counter._cache


@pytest.mark.asyncio
async def test_counts_are_not_shared_between_requests():
    specs = [TextComponent("body", "retained source")]
    with patch.object(cb, "count_tokens", wraps=cb.count_tokens) as count:
        await TokenCounter().components(specs, "chatgpt", "model")
        await TokenCounter().components(specs, "chatgpt", "model")
    assert count.call_count == 2


@pytest.mark.asyncio
async def test_debugger_keeps_both_records_of_the_same_initial_snapshot(monkeypatch):
    from core import app_state

    request = {"prompt": "original", "nested": ["complete source"]}

    async def record_start(*args, **kwargs):
        request["prompt"] = "changed after persistence started"
        request["nested"].append("later value")

    recorder = SimpleNamespace(
        record_session_start=AsyncMock(side_effect=record_start),
        record_event=AsyncMock(),
    )
    monkeypatch.setattr(app_state, "debug_logger", recorder)
    await app_state._debug_session_start(
        "snapshot-test", request_payload=request, preflight_payload={"accepted": True}
    )
    first = recorder.record_session_start.call_args.kwargs["request_payload"]
    event = recorder.record_event.call_args.kwargs["payload"]["details"]
    assert first == event["request"] == {"prompt": "original", "nested": ["complete source"]}
    assert event["preflight"] == {"accepted": True}
    assert recorder.record_event.call_args.kwargs["event_type"] == "session_initialized"


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["openai", "chatgpt", "ollama", "openrouter"])
@pytest.mark.parametrize("body", ["Complete café evidence. ", "中文資料を保持します。 "])
async def test_large_async_counts_preserve_provider_method_and_all_component_boundaries(provider, body):
    text = body * 12000
    specs = [
        TextComponent("source", text),
        TextComponent("schema", text, "schema", "json", False),
        TextComponent("duplicate", text),
    ]
    counter = TokenCounter()
    actual = await counter.components(specs, provider, "gpt-6-luna")
    expected = [cb.make_payload_component(
        spec.name, spec.text, spec.kind, provider, "gpt-6-luna", spec.hint,
        keep_text=spec.keep_text,
    ) for spec in specs]
    assert actual == expected
    assert await counter.components(specs, provider, "gpt-6-luna") == expected


@pytest.mark.asyncio
async def test_components_are_not_tokenized_as_a_concatenated_string():
    specs = [TextComponent("first", "a"), TextComponent("second", "b")]
    actual = await TokenCounter().components(specs, "openai", "gpt-6-luna")
    assert sum(item.tokens_estimate for item in actual) == 2
    assert cb.count_tokens("ab", "openai", "gpt-6-luna")[0] == 1


@pytest.mark.asyncio
async def test_missing_optional_native_tokenizer_uses_conservative_capacity(monkeypatch):
    import token_counting

    text = "Complete evidence. " * 12000
    monkeypatch.setattr(token_counting, "_native_tokenizer_available", lambda: False)
    monkeypatch.setattr(cb, "_get_encoding", lambda _: None)
    executor = token_counting.get_counting_executor()
    with patch.object(executor, "run", wraps=executor.run) as run:
        result = await TokenCounter().components([TextComponent("source", text)], "openai", "model")
    assert result[0].tokens_estimate == cb.estimate_tokens(text)
    assert run.call_count == 1 and run.call_args.kwargs["python_work"] is True
