"""
Focused unit tests for core/security.py internal CIDR handling.

These tests validate the localhost/LAN allowlist behavior without
importing the FastAPI app.
"""

from __future__ import annotations

import importlib
import os
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from fastapi.testclient import TestClient

import core.security as security


def test_default_internal_networks_allow_localhost_and_default_lan():
    with patch.dict(os.environ, {"INTERNAL_ALLOWED_CIDRS": ""}, clear=False):
        importlib.reload(security)

        assert security.is_ip_allowed("127.0.0.1") is True
        assert security.is_ip_allowed("::1") is True
        assert security.is_ip_allowed("192.168.50.23") is True
        assert security.is_ip_allowed("203.0.113.10") is False


def test_internal_allowed_cidrs_override_replaces_default_allowlist():
    with patch.dict(
        os.environ,
        {"INTERNAL_ALLOWED_CIDRS": "10.10.0.0/16,172.16.5.0/24"},
        clear=False,
    ):
        importlib.reload(security)

        assert security.is_ip_allowed("10.10.12.34") is True
        assert security.is_ip_allowed("172.16.5.77") is True
        assert security.is_ip_allowed("192.168.50.23") is False


def test_invalid_internal_allowed_cidrs_falls_back_to_defaults():
    with patch.dict(
        os.environ,
        {"INTERNAL_ALLOWED_CIDRS": "not-a-cidr"},
        clear=False,
    ):
        importlib.reload(security)

        assert security.is_ip_allowed("127.0.0.1") is True
        assert security.is_ip_allowed("192.168.50.23") is True


def _build_filtered_test_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(security.IPFilterMiddleware)

    @app.get("/health")
    async def health():
        return {"status": "healthy"}

    @app.get("/static/app.js")
    async def static_asset():
        return PlainTextResponse("console.log('ok');", media_type="text/javascript")

    return app


def test_public_paths_are_empty_by_default():
    assert security.PUBLIC_PATHS == set()
    assert security.PUBLIC_PATH_PREFIXES == ()


def test_internal_client_can_reach_docs_health_and_static_paths():
    with patch.dict(os.environ, {"INTERNAL_ALLOWED_CIDRS": ""}, clear=False):
        importlib.reload(security)
        app = _build_filtered_test_app()
        client = TestClient(app, client=("192.168.50.23", 50000))

        assert client.get("/health").status_code == 200
        assert client.get("/openapi.json").status_code == 200
        assert client.get("/docs").status_code == 200
        assert client.get("/redoc").status_code == 200
        assert client.get("/static/app.js").status_code == 200


def test_external_client_cannot_reach_docs_health_or_static_paths():
    with patch.dict(os.environ, {"INTERNAL_ALLOWED_CIDRS": ""}, clear=False):
        importlib.reload(security)
        app = _build_filtered_test_app()
        client = TestClient(app, client=("203.0.113.10", 50000))

        for path in ("/health", "/openapi.json", "/docs", "/redoc", "/static/app.js"):
            response = client.get(path)
            assert response.status_code == 403
            assert response.json()["detail"] == "Access denied: internal access only"
