"""Real SDK/loopback transport regression; no generation or provider calls."""
from __future__ import annotations

import asyncio
import threading
from contextlib import contextmanager

import aiohttp
import pytest
from aiohttp import web

from client import AsyncGranSabioClient, TransientGranSabioError


@contextmanager
def idle_closing_server():
    ready = threading.Event()
    idle_closed = threading.Event()
    state = {}
    calls = []

    def serve():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        state["loop"] = loop

        async def submitted(request):
            await request.json()  # Transport JSON only; never interpret a prompt.
            calls.append((request.method, request.path))
            if len(calls) == 1:
                transport = request.transport

                def close_idle():
                    transport.close()
                    idle_closed.set()

                loop.call_later(0.1, close_idle)
            return web.json_response({
                "session_id": "transport-receipt",
                "status": "processing",
            })

        async def start():
            app = web.Application()
            app.router.add_post("/generate", submitted)
            runner = web.AppRunner(app, access_log=None)
            await runner.setup()
            site = web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            state["runner"] = runner
            state["url"] = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
            ready.set()

        try:
            loop.run_until_complete(start())
            loop.run_forever()
        finally:
            if "runner" in state:
                loop.run_until_complete(state["runner"].cleanup())
            loop.close()

    thread = threading.Thread(target=serve, name="sdk-idle-http-fixture")
    thread.start()
    try:
        assert ready.wait(3), "HTTP fixture did not start"
        yield state["url"], idle_closed, calls
    finally:
        if "loop" in state:
            state["loop"].call_soon_threadsafe(state["loop"].stop)
        thread.join(3)
        assert not thread.is_alive(), "HTTP fixture did not stop"


def test_idle_pool_reuse_after_blocked_event_loop_does_not_retry_post(monkeypatch):
    monkeypatch.delenv("GRANSABIO_API_KEY", raising=False)

    async def run():
        with idle_closing_server() as (url, idle_closed, calls):
            async with AsyncGranSabioClient(base_url=url) as client:
                await client.generate(prompt="", wait_for_completion=False)
                # A synchronous checkpoint can keep the client loop from seeing
                # the server's idle FIN before the next non-idempotent POST.
                assert idle_closed.wait(3)
                with pytest.raises(TransientGranSabioError) as error:
                    await client.generate(prompt="", wait_for_completion=False)
                # The closed pooled connection surfaces as ServerDisconnectedError
                # on Linux and as a socket ClientOSError on Windows.
                assert isinstance(error.value.__cause__, (aiohttp.ServerDisconnectedError, aiohttp.ClientOSError))
                assert calls == [("POST", "/generate")]

    asyncio.run(run())


def test_fresh_connection_avoids_idle_pool_race_without_retrying_post(monkeypatch):
    monkeypatch.delenv("GRANSABIO_API_KEY", raising=False)

    async def run():
        with idle_closing_server() as (url, idle_closed, calls):
            async with AsyncGranSabioClient(base_url=url, force_close=True) as client:
                await client.generate(prompt="", wait_for_completion=False)
                assert idle_closed.wait(3)
                response = await client.generate(prompt="", wait_for_completion=False)
                assert response["session_id"] == "transport-receipt"
                assert calls == [("POST", "/generate"), ("POST", "/generate")]

    asyncio.run(run())
