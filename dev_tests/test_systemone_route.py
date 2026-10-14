"""Stateless POST /systemone route tests."""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock

sys.modules.setdefault("magic", MagicMock())

import pytest
from fastapi.testclient import TestClient

from context_budget import ContextBudgetExceeded
from services.typesafe_client import (
    TypeSafeConfigurationError,
    TypeSafeEvaluation,
    TypeSafeRequestError,
)


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, MagicMock]:
    service = MagicMock()
    service.evaluate_system_one = AsyncMock(
        return_value=TypeSafeEvaluation(
            model="jev-1.13.0",
            answers={
                "decision": {
                    "type": "choice",
                    "choice": "proceed",
                    "probabilities": {"proceed": 1.0, "reject": 0.0},
                    "confidence": 0.9,
                }
            },
            usage={"input_tokens": 12, "output_tokens": 0},
        )
    )
    monkeypatch.setattr("core.systemone_routes.get_ai_service", lambda: service)
    monkeypatch.setattr("core.systemone_routes._ensure_services", lambda: None)
    monkeypatch.setattr(
        "config.Config.get_model_info",
        lambda _self, _name, **_kwargs: {
            "provider": "typesafe",
            "model_id": "jev-latest",
            "api_key": "test-key",
            "capabilities": ["text", "system_one"],
        },
    )
    import core.systemone_routes  # noqa: F401
    from core.app_state import app

    return TestClient(app), service


def _payload() -> dict:
    return {
        "model": "jev-latest",
        "state": {"prompt": "Can this proceed?"},
        "questions": {
            "decision": {
                "type": "choice",
                "instructions": "Choose.",
                "criteria": {"proceed": "yes", "reject": "no"},
            }
        },
    }


def test_systemone_happy_path_returns_answers_and_usage(client) -> None:
    http, service = client
    response = http.post("/systemone", json=_payload())
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "jev-1.13.0"
    assert body["answers"]["decision"]["choice"] == "proceed"
    assert body["usage"]["input_tokens"] == 12
    service.evaluate_system_one.assert_awaited_once()
    assert service.evaluate_system_one.await_args.args[2] == "jev-latest"


def test_systemone_missing_questions_is_400(client) -> None:
    http, service = client
    response = http.post("/systemone", json={"state": {}})
    assert response.status_code == 400
    service.evaluate_system_one.assert_not_called()


def test_systemone_invalid_questions_are_400(client) -> None:
    http, service = client
    response = http.post(
        "/systemone",
        json={"state": {}, "questions": {"decision": {"type": "chat", "instructions": "no"}}},
    )
    assert response.status_code == 400
    service.evaluate_system_one.assert_not_called()


def test_systemone_missing_key_is_503(client) -> None:
    http, service = client
    service.evaluate_system_one = AsyncMock(
        side_effect=TypeSafeConfigurationError(
            "Missing API key for provider 'typesafe'. Set 'TYPESAFE_API_KEY' in your environment."
        )
    )
    response = http.post("/systemone", json=_payload())
    assert response.status_code == 503
    assert "TYPESAFE_API_KEY" in response.json()["detail"]


def test_systemone_upstream_http_is_502(client) -> None:
    http, service = client
    service.evaluate_system_one = AsyncMock(
        side_effect=TypeSafeRequestError("TypeSafe HTTP 500", status_code=500)
    )
    response = http.post("/systemone", json=_payload())
    assert response.status_code == 502


@pytest.mark.parametrize("model", [None, "", "   ", 42])
def test_systemone_requires_explicit_model(client, model) -> None:
    http, service = client
    payload = _payload()
    payload["model"] = model
    response = http.post("/systemone", json=payload)
    assert response.status_code == 400
    service.evaluate_system_one.assert_not_called()


def test_systemone_oversized_context_is_400(client) -> None:
    http, service = client
    service.evaluate_system_one.side_effect = ContextBudgetExceeded(call_id="systemone.evaluate")
    response = http.post("/systemone", json=_payload())
    assert response.status_code == 400
    assert response.json()["detail"]["error_code"] == "context_budget_exceeded"
