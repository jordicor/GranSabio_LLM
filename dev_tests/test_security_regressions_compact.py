"""Compact regression coverage for the engine's HTTP security boundary.

The scenarios use documentation-only addresses and domains.  They exercise
the security contracts directly without importing the full application or
depending on deployment-specific network names.
"""

from __future__ import annotations

from collections.abc import Iterable

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

import core.security as security


def _request(client_host: str, headers: dict[str, str]) -> Request:
    raw_headers = [
        (name.lower().encode("latin-1"), value.encode("latin-1"))
        for name, value in headers.items()
    ]
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": raw_headers,
            "client": (client_host, 50000),
        }
    )


def _ok_app(*middleware_classes: type) -> FastAPI:
    app = FastAPI()
    for middleware_class in middleware_classes:
        app.add_middleware(middleware_class)

    @app.api_route("/ok", methods=["GET", "POST"])
    async def ok() -> dict[str, str]:
        return {"status": "ok"}

    return app


@pytest.mark.parametrize(
    ("trust_single_headers", "expected_ip"),
    [
        (False, "203.0.113.10"),
        (True, "198.51.100.20"),
    ],
    ids=["xff-wins-by-default", "single-header-explicitly-trusted"],
)
def test_proxy_ip_resolution_requires_opt_in_for_single_ip_headers(
    monkeypatch: pytest.MonkeyPatch,
    trust_single_headers: bool,
    expected_ip: str,
) -> None:
    if trust_single_headers:
        monkeypatch.setenv(security.TRUST_SINGLE_IP_HEADERS_ENV_VAR, "1")
    else:
        monkeypatch.delenv(
            security.TRUST_SINGLE_IP_HEADERS_ENV_VAR,
            raising=False,
        )

    request = _request(
        "127.0.0.1",
        {
            "CF-Connecting-IP": "198.51.100.20",
            "X-Real-IP": "127.0.0.1",
            "X-Forwarded-For": "203.0.113.10",
        },
    )

    assert security.get_client_ip(request) == expected_ip


@pytest.mark.parametrize(
    ("client_host", "authorization", "expected_status"),
    [
        ("198.51.100.10", None, 401),
        ("198.51.100.10", "Bearer incorrect-secret", 403),
        ("198.51.100.10", "Bearer test-shared-secret", 200),
        ("127.0.0.1", None, 200),
    ],
    ids=["lan-missing", "lan-invalid", "lan-valid", "loopback-exempt"],
)
def test_shared_secret_gates_non_loopback_clients(
    monkeypatch: pytest.MonkeyPatch,
    client_host: str,
    authorization: str | None,
    expected_status: int,
) -> None:
    monkeypatch.setenv(
        "INTERNAL_ALLOWED_CIDRS",
        "127.0.0.0/8,198.51.100.0/24",
    )
    monkeypatch.setenv(
        security.ENGINE_SHARED_SECRET_ENV_VAR,
        "test-shared-secret",
    )
    headers = {"Authorization": authorization} if authorization else {}
    client = TestClient(_ok_app(security.IPFilterMiddleware), client=(client_host, 50000))

    response = client.get("/ok", headers=headers)

    assert response.status_code == expected_status


@pytest.mark.parametrize(
    ("method", "headers", "expected_status"),
    [
        ("POST", {"Sec-Fetch-Site": "cross-site"}, 403),
        ("POST", {"Sec-Fetch-Site": "same-origin"}, 200),
        ("POST", {"Origin": "https://attacker.example.test"}, 403),
        ("POST", {"Origin": "https://api.example.test"}, 200),
        ("POST", {}, 200),
        ("GET", {"Sec-Fetch-Site": "cross-site"}, 200),
    ],
    ids=[
        "cross-site-fetch",
        "same-origin-fetch",
        "foreign-origin",
        "matching-origin",
        "non-browser-client",
        "safe-method",
    ],
)
def test_cross_site_guard_distinguishes_browser_mutations(
    method: str,
    headers: dict[str, str],
    expected_status: int,
) -> None:
    client = TestClient(
        _ok_app(security.CrossSiteRequestGuardMiddleware),
        base_url="https://api.example.test",
    )

    response = client.request(method, "/ok", headers=headers)

    assert response.status_code == expected_status


async def _drive_body_limit(
    *,
    content_length: int | None,
    chunks: Iterable[tuple[bytes, bool]],
    max_bytes: int,
) -> tuple[list[dict], int]:
    sent: list[dict] = []
    inner_calls = 0
    messages = [
        {"type": "http.request", "body": body, "more_body": more_body}
        for body, more_body in chunks
    ]

    async def receive() -> dict:
        if messages:
            return messages.pop(0)
        return {"type": "http.disconnect"}

    async def send(message: dict) -> None:
        sent.append(message)

    async def inner_app(scope: dict, receive, send) -> None:
        nonlocal inner_calls
        inner_calls += 1
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"{}"})

    headers = []
    if content_length is not None:
        headers.append((b"content-length", str(content_length).encode("ascii")))
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/generate",
        "headers": headers,
    }
    middleware = security.RequestBodyLimitMiddleware(
        inner_app,
        default_max_bytes=max_bytes,
    )
    await middleware(scope, receive, send)
    return sent, inner_calls


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("content_length", "chunks", "expected_status", "expected_inner_calls"),
    [
        (2048, [], 413, 0),
        (None, [(b"x" * 60, True), (b"x" * 60, False)], 413, 0),
        (10, [(b"0123456789", False)], 200, 1),
    ],
    ids=["declared-oversize", "chunked-oversize", "within-limit"],
)
async def test_request_body_limit_rejects_before_application_parsing(
    content_length: int | None,
    chunks: list[tuple[bytes, bool]],
    expected_status: int,
    expected_inner_calls: int,
) -> None:
    sent, inner_calls = await _drive_body_limit(
        content_length=content_length,
        chunks=chunks,
        max_bytes=100,
    )

    assert sent[0]["status"] == expected_status
    assert inner_calls == expected_inner_calls
    if expected_status == 413:
        assert (b"x-max-body-bytes", b"100") in sent[0]["headers"]


def test_security_headers_are_added_to_successful_responses() -> None:
    client = TestClient(_ok_app(security.SecurityHeadersMiddleware))

    response = client.get("/ok")

    assert response.status_code == 200
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["X-Content-Type-Options"] == "nosniff"


@pytest.mark.parametrize(
    ("client_ip", "expected_allowed"),
    [
        ("127.0.0.1", True),
        ("198.51.100.42", True),
        ("203.0.113.42", False),
        ("203.0.113.30", False),
        ("not-an-ip", False),
    ],
    ids=["loopback", "configured-lan", "other-lan", "external", "invalid"],
)
def test_internal_cidr_allowlist_is_explicit_and_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    client_ip: str,
    expected_allowed: bool,
) -> None:
    monkeypatch.setenv(
        "INTERNAL_ALLOWED_CIDRS",
        "127.0.0.0/8,198.51.100.0/24",
    )

    assert security.is_ip_allowed(client_ip) is expected_allowed
