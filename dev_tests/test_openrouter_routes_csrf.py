"""CSRF hardening tests for OpenRouter admin routes."""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock

# The venv's libmagic native DLL crashes on import in this Windows environment
# (pre-existing); attachment_manager treats python-magic as optional. Stub it
# before importing core.* (same pattern as dev_tests/test_audit_genproc.py).
sys.modules.setdefault("magic", MagicMock())

import pytest
from fastapi.testclient import TestClient

import core.openrouter_routes as openrouter_routes
from core.app_state import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(autouse=True)
def _allow_testserver_admin_host(monkeypatch):
    # API-05: admin mutations now validate the Host header against a hostname
    # allowlist. These tests exercise the same-origin
    # logic, so the TestClient hostname is explicitly allowlisted the same way
    # a real deployment would via ADMIN_ALLOWED_HOSTNAMES.
    monkeypatch.setenv("ADMIN_ALLOWED_HOSTNAMES", "testserver")


@pytest.fixture
def mock_model_sync_service(monkeypatch):
    service = MagicMock()
    service.sync_provider.return_value = {
        "success": True,
        "provider": "openrouter",
        "synced_models": [],
    }
    service.delete_model.return_value = {
        "success": True,
        "provider": "openrouter",
        "deleted_model_id": "openrouter/test",
    }
    service.toggle_model.return_value = {
        "success": True,
        "provider": "openrouter",
        "model_id": "openrouter/test",
        "enabled": True,
    }
    service.fetch_remote_models = AsyncMock(return_value={"models": []})
    monkeypatch.setattr(openrouter_routes, "_get_model_sync_service", lambda: service)
    return service


@pytest.mark.parametrize(
    ("host", "expected_status"),
    [
        ("localhost", 200),
        ("127.0.0.1", 200),
        ("[::1]", 200),
        ("198.51.100.10", 200),
        ("[2001:db8:1::10]", 200),
        ("203.0.113.10", 403),
        ("attacker.example.test", 403),
        ("198.51.100.10.attacker.example.test", 403),
    ],
)
def test_admin_sync_default_hosts_follow_internal_networks(
    monkeypatch, mock_model_sync_service, host, expected_status,
):
    monkeypatch.delenv("ADMIN_ALLOWED_HOSTNAMES", raising=False)
    # Documentation-only networks represent the configured LAN and VPN.
    monkeypatch.setenv(
        "INTERNAL_ALLOWED_CIDRS",
        "127.0.0.0/8,::1/128,198.51.100.0/24,2001:db8:1::/64",
    )
    origin = f"http://{host}:8000"
    client = TestClient(app, client=("198.51.100.42", 50000))

    response = client.post(
        "/api/admin/models/providers/openrouter/sync",
        json={"models": []},
        headers={"host": f"{host}:8000", "origin": origin},
    )

    assert response.status_code == expected_status
    if expected_status == 200:
        mock_model_sync_service.sync_provider.assert_called_once_with("openrouter", [])
    else:
        assert response.json()["detail"] == "Admin mutation rejected: untrusted Host header"
        mock_model_sync_service.sync_provider.assert_not_called()


@pytest.mark.parametrize(
    ("host", "expected_status"),
    [("admin.example.test", 200), ("198.51.100.11", 200), ("198.51.100.10", 403)],
)
def test_admin_sync_explicit_hosts_override_network_defaults(
    monkeypatch, mock_model_sync_service, host, expected_status,
):
    monkeypatch.setenv("ADMIN_ALLOWED_HOSTNAMES", " Admin.Example.Test. , 198.51.100.11 ")
    monkeypatch.setenv("INTERNAL_ALLOWED_CIDRS", "127.0.0.0/8,198.51.100.0/24")
    origin = f"http://{host}:8000"
    client = TestClient(app, base_url=origin)

    response = client.post(
        "/api/admin/models/providers/openrouter/sync",
        json={"models": []},
        headers={"origin": origin},
    )

    assert response.status_code == expected_status
    if expected_status == 200:
        mock_model_sync_service.sync_provider.assert_called_once_with("openrouter", [])
    else:
        mock_model_sync_service.sync_provider.assert_not_called()


@pytest.mark.parametrize("origin", ["http://[::1", "http://testserver:invalid"])
def test_admin_sync_rejects_malformed_origin(client, mock_model_sync_service, origin):
    response = client.post(
        "/api/admin/models/providers/openrouter/sync",
        json={"models": []},
        headers={"origin": origin, "sec-fetch-site": "same-origin"},
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "Admin mutation rejected: invalid Origin/Referer"
    mock_model_sync_service.sync_provider.assert_not_called()


@pytest.mark.parametrize(
    ("header_name", "header_value"),
    [
        ("origin", "http://testserver"),
        ("referer", "http://testserver/admin/models"),
    ],
)
def test_admin_openrouter_sync_allows_same_origin(client, mock_model_sync_service, header_name, header_value):
    response = client.post(
        "/api/admin/models/providers/openrouter/sync",
        json={"models": [{"model_id": "openrouter/test"}]},
        headers={header_name: header_value},
    )

    assert response.status_code == 200
    assert response.json()["provider"] == "openrouter"
    mock_model_sync_service.sync_provider.assert_called_once_with(
        "openrouter",
        [{"model_id": "openrouter/test"}],
    )


@pytest.mark.parametrize(
    "header_value",
    [
        "http://evil.testserver",
        "http://testserver:9999",
        "https://testserver",
    ],
)
def test_admin_openrouter_sync_blocks_cross_origin(client, mock_model_sync_service, header_value):
    response = client.post(
        "/api/admin/models/providers/openrouter/sync",
        json={"models": [{"model_id": "openrouter/test"}]},
        headers={"origin": header_value},
    )

    assert response.status_code == 403
    # Either the global cross-site guard middleware (FE-01) or the route-level
    # same-origin check rejects the request; both enforce the same contract.
    assert response.json()["detail"] in (
        "Forbidden: origin mismatch",
        "Admin mutation rejected: cross-origin request",
    )
    mock_model_sync_service.sync_provider.assert_not_called()


def test_admin_openrouter_sync_blocks_missing_origin_and_referer(client, mock_model_sync_service):
    response = client.post(
        "/api/admin/models/providers/openrouter/sync",
        json={"models": [{"model_id": "openrouter/test"}]},
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "Admin mutation rejected: missing Origin/Referer"
    mock_model_sync_service.sync_provider.assert_not_called()


def test_legacy_openrouter_sync_blocks_missing_origin_and_referer(client, mock_model_sync_service):
    response = client.post(
        "/api/openrouter/sync",
        json={"selected_models": [{"model_id": "openrouter/test"}]},
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "Admin mutation rejected: missing Origin/Referer"
    mock_model_sync_service.sync_provider.assert_not_called()


def test_openrouter_models_get_stays_unrestricted_by_same_origin_guard(client, mock_model_sync_service):
    response = client.get("/api/openrouter/models")

    assert response.status_code == 200
    assert response.json() == {"models": []}
    mock_model_sync_service.fetch_remote_models.assert_awaited_once_with("openrouter")


@pytest.mark.parametrize(
    ("method", "url", "service_method", "expected_kwargs"),
    [
        (
            "patch",
            "/api/admin/models/catalog/openrouter?model_id=openrouter/test&enabled=true",
            "toggle_model",
            ("openrouter", "openrouter/test", True),
        ),
        (
            "delete",
            "/api/admin/models/catalog/openrouter?model_id=openrouter/test",
            "delete_model",
            ("openrouter", "openrouter/test"),
        ),
    ],
)
def test_admin_catalog_mutations_allow_same_origin(
    client,
    mock_model_sync_service,
    method,
    url,
    service_method,
    expected_kwargs,
):
    response = getattr(client, method)(
        url,
        headers={"origin": "http://testserver"},
    )

    assert response.status_code == 200
    getattr(mock_model_sync_service, service_method).assert_called_once_with(*expected_kwargs)


@pytest.mark.parametrize(
    ("method", "url", "service_method"),
    [
        ("patch", "/api/admin/models/catalog/openrouter?model_id=openrouter/test&enabled=true", "toggle_model"),
        ("delete", "/api/admin/models/catalog/openrouter?model_id=openrouter/test", "delete_model"),
    ],
)
def test_admin_catalog_mutations_block_missing_origin_and_referer(
    client,
    mock_model_sync_service,
    method,
    url,
    service_method,
):
    response = getattr(client, method)(url)

    assert response.status_code == 403
    assert response.json()["detail"] == "Admin mutation rejected: missing Origin/Referer"
    getattr(mock_model_sync_service, service_method).assert_not_called()
