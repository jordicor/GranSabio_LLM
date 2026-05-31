"""Focused tests for runtime console capture and monitor wiring."""

import asyncio
from pathlib import Path

import pytest

from core import console_routes
from services.runtime_console import (
    _reset_runtime_console_for_tests,
    bind_console_context,
    get_recent_console_events,
    publish_console_output,
    reset_console_context,
    subscribe_console,
    unsubscribe_console,
)


ROOT = Path(__file__).resolve().parents[1]


def test_runtime_console_records_context_and_filters():
    _reset_runtime_console_for_tests()
    tokens = bind_console_context(session_id="session-1", project_id="project-1", phase="qa")
    try:
        event = publish_console_output("stderr", "provider failed\n", level="ERROR")
    finally:
        reset_console_context(tokens)

    assert event["seq"] == 1
    assert event["context"]["session_id"] == "session-1"
    assert event["context"]["project_id"] == "project-1"
    assert event["context"]["phase"] == "qa"

    matching = get_recent_console_events(project_id="project-1", session_id="session-1")
    assert [item["text"] for item in matching] == ["provider failed\n"]
    assert get_recent_console_events(project_id="other-project") == []


@pytest.mark.asyncio
async def test_runtime_console_subscription_receives_matching_events():
    _reset_runtime_console_for_tests()
    subscription = await subscribe_console(project_id="project-2")
    try:
        publish_console_output("stdout", "ignored\n")
        tokens = bind_console_context(session_id="session-2", project_id="project-2", phase="generation")
        try:
            publish_console_output("stdout", "visible\n")
        finally:
            reset_console_context(tokens)

        event = await subscription.queue.get()
    finally:
        await unsubscribe_console(subscription)

    assert event["text"] == "visible\n"
    assert event["context"]["project_id"] == "project-2"


def test_recent_console_endpoint_returns_tail(test_client):
    _reset_runtime_console_for_tests()
    publish_console_output("stdout", "hello from console\n")

    response = test_client.get("/monitor/console/recent?limit=10")

    assert response.status_code == 200
    payload = response.json()
    assert any(item["text"] == "hello from console\n" for item in payload["events"])
    assert payload["stats"]["events"] >= 1


@pytest.mark.asyncio
async def test_console_stream_subscribes_before_tail_and_deduplicates(monkeypatch):
    _reset_runtime_console_for_tests()
    original_subscribe = console_routes.subscribe_console
    original_get_recent = console_routes.get_recent_console_events
    subscribed = False

    async def wrapped_subscribe(**filters):
        nonlocal subscribed
        subscription = await original_subscribe(**filters)
        subscribed = True
        return subscription

    def wrapped_get_recent(*args, **kwargs):
        assert subscribed
        publish_console_output("stdout", "during-tail\n")
        return original_get_recent(*args, **kwargs)

    monkeypatch.setattr(console_routes, "subscribe_console", wrapped_subscribe)
    monkeypatch.setattr(console_routes, "get_recent_console_events", wrapped_get_recent)

    response = await console_routes.stream_console_output(
        tail=10,
        project_id=None,
        session_id=None,
        phase=None,
        stream=None,
        level=None,
        _client_ip="127.0.0.1",
    )
    iterator = response.body_iterator.__aiter__()

    try:
        connected_chunk = (await iterator.__anext__()).decode("utf-8")
        tail_chunk = (await iterator.__anext__()).decode("utf-8")
        publish_console_output("stdout", "after-tail\n")
        live_chunk = (await asyncio.wait_for(iterator.__anext__(), timeout=1)).decode("utf-8")
    finally:
        await iterator.aclose()

    assert "console_connected" in connected_chunk
    assert "during-tail" in tail_chunk
    assert "during-tail" not in live_chunk
    assert "after-tail" in live_chunk


def test_monitor_ui_has_runtime_console_panel():
    source = (ROOT / "static" / "js" / "stream_monitor.js").read_text(encoding="utf-8")
    template = (ROOT / "templates" / "stream_monitor.html").read_text(encoding="utf-8")

    assert "'console'" in source
    assert "/stream/console" in source
    assert "consoleEventSource" in source
    assert "let consoleScope = 'global'" in source
    assert "initConsoleScopeListeners()" in source
    assert "consoleParams.set('project_id', currentProjectId)" in source
    assert 'data-console-scope="global"' in template
    assert 'data-console-scope="project"' in template
    assert 'data-phase="console"' in template
    assert 'id="content-console"' in template
    assert "stream_monitor.js?v=runtime-console-scope-20260530" in template
