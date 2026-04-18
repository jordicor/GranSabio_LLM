"""Route-level tests for Long Text Mode request handling."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

@pytest.fixture
def client():
    """Return a TestClient with generation background work disabled."""

    with patch("core.security.is_ip_allowed", return_value=True), \
         patch("core.app_state._ensure_services"), \
         patch("core.generation_routes.ai_service", MagicMock()), \
         patch("core.generation_processor.ai_service", MagicMock()), \
         patch("core.generation_routes.publish_project_phase_chunk", AsyncMock(return_value=None)), \
         patch("core.generation_routes._debug_session_start", AsyncMock(return_value=None)), \
         patch("core.generation_routes.process_content_generation", AsyncMock(return_value=None)), \
         patch("core.generation_routes._debug_record_event", AsyncMock(return_value=None)), \
         patch("core.generation_routes._debug_update_status", AsyncMock(return_value=None)):
        from main import app

        yield TestClient(app, headers={"X-Forwarded-For": "127.0.0.1"})


def _base_payload() -> dict:
    """Return a valid minimal payload for `/generate`."""

    return {
        "prompt": "Write a detailed long-form article about reliable software delivery practices.",
        "generator_model": "gpt-4o",
        "qa_layers": [],
        "qa_models": [],
    }


def test_long_text_on_rejects_json_output(client: TestClient) -> None:
    payload = _base_payload()
    payload.update({"long_text_mode": "on", "json_output": True, "min_words": 3000, "max_words": 3600})
    response = client.post("/generate", json=payload)
    assert response.status_code == 400
    assert "plain-text" in response.json()["detail"]


def test_long_text_on_rejects_json_content_type(client: TestClient) -> None:
    payload = _base_payload()
    payload.update({"long_text_mode": "on", "content_type": "json", "min_words": 3000, "max_words": 3600})
    response = client.post("/generate", json=payload)
    assert response.status_code == 400
    assert "plain-text" in response.json()["detail"]


def test_long_text_on_rejects_target_field(client: TestClient) -> None:
    payload = _base_payload()
    payload.update(
        {
            "long_text_mode": "on",
            "min_words": 3000,
            "max_words": 3600,
            "target_field": "data.body",
        }
    )
    response = client.post("/generate", json=payload)
    assert response.status_code == 400
    assert "target_field" in response.json()["detail"]


def test_long_text_on_rejects_images(client: TestClient) -> None:
    payload = _base_payload()
    payload.update(
        {
            "long_text_mode": "on",
            "min_words": 3000,
            "max_words": 3600,
            "images": [{"username": "alice", "upload_id": "img-0001"}],
        }
    )
    response = client.post("/generate", json=payload)
    assert response.status_code == 400
    assert "image" in response.json()["detail"].lower()


def test_long_text_on_rejects_explicit_arbiter_model(client: TestClient) -> None:
    payload = _base_payload()
    payload.update(
        {
            "long_text_mode": "on",
            "min_words": 3000,
            "max_words": 3600,
            "arbiter_model": "gpt-4o",
        }
    )
    response = client.post("/generate", json=payload)
    assert response.status_code == 400
    assert "arbiter_model" in response.json()["detail"]


def test_min_words_greater_than_max_words_returns_422(client: TestClient) -> None:
    payload = _base_payload()
    payload.update({"min_words": 4000, "max_words": 2000})
    response = client.post("/generate", json=payload)
    assert response.status_code == 422
    assert "min_words" in response.text


def test_long_text_auto_decline_returns_advisory(client: TestClient) -> None:
    payload = _base_payload()
    payload.update({"long_text_mode": "auto", "max_words": 3200})
    response = client.post("/generate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "initialized"
    assert data["recommended_timeout_seconds"] is not None
    assert data["advisories"][0]["code"] == "long_text_auto_declined_one_sided_bounds"


def test_long_text_accepted_init_response_includes_timeout_and_advisories(client: TestClient) -> None:
    payload = _base_payload()
    payload.update({"long_text_mode": "on", "min_words": 3000, "max_words": 3600})
    response = client.post("/generate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "initialized"
    assert data["recommended_timeout_seconds"] is not None
    assert data["advisories"] is not None
    advisory_codes = {item["code"] for item in data["advisories"]}
    assert "long_text_qa_bypass" in advisory_codes


def test_clip_long_text_bound_invariants() -> None:
    """Verify `_clip_long_text_bound` honors request bounds and the hard cap."""

    from core.generation_routes import _clip_long_text_bound
    from models import ContentRequest

    def _make_request(*, min_words=None, max_words=None) -> ContentRequest:
        # model_construct bypasses field validation so we can exercise edge combinations
        # (e.g. min > max) that `_clip_long_text_bound` must still handle defensively.
        return ContentRequest.model_construct(
            prompt="test prompt",
            generator_model="gpt-4o",
            qa_layers=[],
            qa_models=[],
            min_words=min_words,
            max_words=max_words,
        )

    hard_cap = 8000

    # Case 1: both bounds present, value inside the window — value should pass through.
    request = _make_request(min_words=2000, max_words=4000)
    assert _clip_long_text_bound(3000, request=request, hard_cap_words=hard_cap, lower=True) == 3000
    assert _clip_long_text_bound(3000, request=request, hard_cap_words=hard_cap, lower=False) == 3000

    # Case 2: both bounds present, lower candidate below min_words — lifted up to min_words.
    request = _make_request(min_words=2500, max_words=4000)
    assert _clip_long_text_bound(2000, request=request, hard_cap_words=hard_cap, lower=True) == 2500

    # Case 3: both bounds present, upper candidate above max_words — pulled down to max_words.
    request = _make_request(min_words=2000, max_words=3500)
    assert _clip_long_text_bound(5000, request=request, hard_cap_words=hard_cap, lower=False) == 3500

    # Case 4: only min_words present — lower bound lifts to min_words, upper bound also lifts to min_words.
    request = _make_request(min_words=3000, max_words=None)
    assert _clip_long_text_bound(2500, request=request, hard_cap_words=hard_cap, lower=True) == 3000
    assert _clip_long_text_bound(2500, request=request, hard_cap_words=hard_cap, lower=False) == 3000

    # Case 5: only max_words present — upper bound caps at max_words, lower bound is only hard-capped.
    request = _make_request(min_words=None, max_words=3500)
    assert _clip_long_text_bound(5000, request=request, hard_cap_words=hard_cap, lower=False) == 3500
    assert _clip_long_text_bound(2500, request=request, hard_cap_words=hard_cap, lower=True) == 2500

    # Case 6: no bounds present — only hard cap applies.
    request = _make_request(min_words=None, max_words=None)
    assert _clip_long_text_bound(2500, request=request, hard_cap_words=hard_cap, lower=True) == 2500
    assert _clip_long_text_bound(9000, request=request, hard_cap_words=hard_cap, lower=True) == hard_cap
    assert _clip_long_text_bound(2500, request=request, hard_cap_words=hard_cap, lower=False) == 2500
    assert _clip_long_text_bound(9000, request=request, hard_cap_words=hard_cap, lower=False) == hard_cap

    # Case 7 edge: upper candidate gets clipped below min_words — final value must still be >= min_words.
    # value 5000 > max_words 3200 -> min(5000, 3200) = 3200 -> min(3200, hard_cap) = 3200 -> max(3200, 3500) = 3500.
    request = _make_request(min_words=3500, max_words=3200)
    assert _clip_long_text_bound(5000, request=request, hard_cap_words=hard_cap, lower=False) == 3500

    # Case 8 edge: input below hard cap with max_words absent — value must be preserved and hard-capped only.
    request = _make_request(min_words=None, max_words=None)
    assert _clip_long_text_bound(1500, request=request, hard_cap_words=hard_cap, lower=False) == 1500
    assert _clip_long_text_bound(hard_cap, request=request, hard_cap_words=hard_cap, lower=False) == hard_cap

    # Case 9 edge: returned value always >= 1 even with pathological inputs.
    request = _make_request(min_words=None, max_words=None)
    assert _clip_long_text_bound(0, request=request, hard_cap_words=hard_cap, lower=True) == 1
    assert _clip_long_text_bound(-50, request=request, hard_cap_words=hard_cap, lower=False) == 1
