"""Compact regressions for high-value runtime and API boundary guarantees.

These cases preserve the durable contracts from the retired audit suites
without retaining environment-specific fixtures or one-test-per-variant
duplication.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from starlette.requests import Request

# python-magic is optional in production and its native library is not needed
# for these boundary tests.
sys.modules.setdefault("magic", MagicMock())


def _aiohttp_response(status: int, payload: dict[str, Any] | None = None) -> AsyncMock:
    response = AsyncMock()
    response.status = status
    response.json = AsyncMock(return_value=payload or {})
    response.text = AsyncMock(return_value="temporary response")
    response.__aenter__ = AsyncMock(return_value=response)
    response.__aexit__ = AsyncMock(return_value=None)
    return response


def _request_from(client_host: str) -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/analysis/repetition",
            "client": (client_host, 50000),
            "headers": [],
        }
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method_name", "status_code"),
    [
        pytest.param("get_status", status, id=f"status-{status}")
        for status in (202, 425, 429, 503)
    ]
    + [
        pytest.param("get_result", status, id=f"result-{status}")
        for status in (425, 429, 503)
    ],
)
async def test_async_poll_endpoints_classify_transient_statuses(
    method_name: str,
    status_code: int,
) -> None:
    from client import AsyncGranSabioClient, TransientGranSabioError

    client = AsyncGranSabioClient(
        base_url="https://service.example.test",
        api_key="test-key",
    )
    with patch.object(
        client,
        "_request",
        AsyncMock(return_value=_aiohttp_response(status_code)),
    ):
        with pytest.raises(TransientGranSabioError) as exc_info:
            await getattr(client, method_name)("session-example")

    assert exc_info.value.status_code == status_code


@pytest.mark.asyncio
async def test_async_wait_for_result_recovers_after_transient_status() -> None:
    from client import AsyncGranSabioClient

    client = AsyncGranSabioClient(base_url="https://service.example.test")
    responses = [
        _aiohttp_response(503),
        _aiohttp_response(200, {"status": "completed"}),
        _aiohttp_response(200, {"status": "completed", "content": "done"}),
    ]

    with patch.object(client, "_request", AsyncMock(side_effect=responses)):
        with patch("client.async_client._POLL_RETRY_BACKOFF_SECONDS", (0, 0, 0)):
            result = await client.wait_for_result(
                "session-example",
                poll_interval=0.001,
                max_wait=5,
            )

    assert result["content"] == "done"


@pytest.mark.asyncio
async def test_edit_segment_locator_runs_outside_the_event_loop() -> None:
    import core.generation_processor as generation_processor

    observed: dict[str, bool] = {}

    def locator_spy(*_args: Any, **_kwargs: Any) -> tuple[int, int]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            observed["on_event_loop"] = False
        else:
            observed["on_event_loop"] = True
        return (0, 4)

    with patch.object(
        generation_processor,
        "_locate_edit_segment",
        side_effect=locator_spy,
    ):
        span = await generation_processor._locate_edit_segment_async(
            "sample text",
            object(),
        )

    assert span == (0, 4)
    assert observed == {"on_event_loop": False}


def test_deeply_nested_json_falls_back_without_server_error() -> None:
    from word_count_utils import extract_target_field

    deeply_nested = "[" * 3000 + "]" * 3000

    text, extracted = extract_target_field(deeply_nested, "data.content")

    assert text == deeply_nested
    assert extracted is False


@pytest.mark.asyncio
async def test_repetition_endpoint_limits_client_compute_and_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import analysis_router

    captured_configs: list[Any] = []

    async def fake_to_thread(_function: Any, _text: str, config: Any) -> dict[str, Any]:
        captured_configs.append(config)
        return {"version": "test", "meta": {}, "settings": {}}

    monkeypatch.setattr(analysis_router.asyncio, "to_thread", fake_to_thread)
    limiter = analysis_router.SlidingWindowRateLimiter(limit=2, window_seconds=60)
    request = analysis_router.RepetitionAnalyzerRequest(
        text="sample phrase repeated for bounded analysis " * 20,
        algo_mode="multiprocess",
        mp_threshold_tokens=1000,
        workers=16,
    )

    for _ in range(2):
        await analysis_router.analyze_repetition_endpoint(
            request,
            http_request=_request_from("203.0.113.10"),
            rate_limiter=limiter,
        )

    with pytest.raises(HTTPException) as exc_info:
        await analysis_router.analyze_repetition_endpoint(
            request,
            http_request=_request_from("203.0.113.10"),
            rate_limiter=limiter,
        )

    assert exc_info.value.status_code == 429
    assert captured_configs
    assert all(config.algo_mode == "auto" for config in captured_configs)
    assert all(
        config.mp_threshold_tokens >= analysis_router.MIN_MP_THRESHOLD_TOKENS
        for config in captured_configs
    )


@pytest.mark.parametrize(
    ("module_name", "dependency_name", "env_name"),
    [
        pytest.param(
            "core.console_routes",
            "require_console_exposed",
            "CONSOLE_EXPOSE_ENABLED",
            id="console",
        ),
        pytest.param(
            "core.debug_routes",
            "require_debugger_exposed",
            "DEBUGGER_EXPOSE_ENABLED",
            id="debugger",
        ),
    ],
)
def test_sensitive_operator_routes_can_be_disabled(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    dependency_name: str,
    env_name: str,
) -> None:
    import importlib

    module = importlib.import_module(module_name)
    monkeypatch.setenv(env_name, "false")

    with pytest.raises(HTTPException) as exc_info:
        getattr(module, dependency_name)()

    assert exc_info.value.status_code == 404
