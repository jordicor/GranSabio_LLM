"""CSRF hardening tests for OpenRouter admin routes."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from core.app_state import app
import core.openrouter_routes as openrouter_routes


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


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
    assert response.json()["detail"] == "Admin mutation rejected: cross-origin request"
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
