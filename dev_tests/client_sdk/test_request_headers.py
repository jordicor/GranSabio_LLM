"""Generic per-client headers reach normal, SSE and multipart HTTP boundaries."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from client import AsyncGranSabioClient, GranSabioClient


@pytest.fixture
def wire():
    received = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            self.respond()

        def do_POST(self):
            self.respond()

        def respond(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            received.append((self.command, self.path, dict(self.headers), body))
            stream = self.path.startswith("/stream/")
            payload = ({"session_id": "s1", "status": "initialized"} if self.path == "/generate" else
                       {"status": "completed", "content": "Done", "approved": True, "session_id": "s1"})
            data = (b'data: {"type":"session_end","status":"completed"}\n\n' if stream else
                    json.dumps(payload).encode())
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream" if stream else "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    task = threading.Thread(target=server.serve_forever, daemon=True)
    task.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", received
    finally:
        server.shutdown()
        server.server_close()
        task.join(timeout=5)


@pytest.mark.parametrize("cls", [AsyncGranSabioClient, GranSabioClient])
def test_headers_are_copied_and_explicit_empty_key_disables_environment(cls, monkeypatch):
    monkeypatch.setenv("GRANSABIO_API_KEY", "ambient-bearer")
    original = {"X-Operation": "one"}
    first = cls(api_key="", request_headers=original)
    original["X-Operation"] = "changed"
    second = cls(api_key="", request_headers={"X-Operation": "two"})
    assert first._headers()["X-Operation"] == "one"
    assert second._headers()["X-Operation"] == "two"
    assert "Authorization" not in first._headers()
    assert cls()._headers()["Authorization"] == "Bearer ambient-bearer"
    for value in ({"AUTHORIZATION": "hidden"}, {"content-TYPE": "other"}, {"x-provider": "duplicate"}):
        with pytest.raises(ValueError):
            cls(provider_keys={"X-Provider": "key"}, request_headers=value)


def check_wire(received):
    paths = {path for _, path, _, _ in received}
    assert {"/generate", "/status/s1", "/stream/s1", "/attachments"} <= paths
    for _, path, headers, body in received:
        assert headers["X-Operation"] == "one"
        assert not any(name.lower() == "authorization" for name in headers)
        if path == "/attachments":
            assert headers["Content-Type"].startswith("multipart/form-data; boundary=")
            assert b"attachment-body" in body


def test_sync_control_generate_stream_and_multipart_keep_instance_headers(wire, monkeypatch):
    monkeypatch.setenv("GRANSABIO_API_KEY", "ambient-bearer")
    url, received = wire
    client = GranSabioClient(url, api_key="", request_headers={"X-Operation": "one"})
    client.get_status("s1")
    client.generate("Hello", max_tokens=50, wait_for_completion=False)
    client.generate_streaming("Hello", max_tokens=50, generator_model="test-model")
    client.upload_attachment("user", b"attachment-body", "document.txt")
    check_wire(received)


@pytest.mark.asyncio
async def test_async_control_generate_stream_and_multipart_keep_instance_headers(wire, monkeypatch):
    monkeypatch.setenv("GRANSABIO_API_KEY", "ambient-bearer")
    url, received = wire
    async with AsyncGranSabioClient(url, api_key="", request_headers={"X-Operation": "one"}) as client:
        await client.get_status("s1")
        await client.generate("Hello", max_tokens=50, wait_for_completion=False)
        await client.generate_streaming("Hello", max_tokens=50, generator_model="test-model")
        await client.upload_attachment("user", b"attachment-body", "document.txt")
    check_wire(received)
