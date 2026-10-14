"""Regression tests for shared Gran Sabio SDK computations."""

import pytest

from client._common import compute_token_budgets


@pytest.mark.parametrize(
    ("model_limit", "desired_max_tokens", "expected_max_tokens"),
    [
        (128_000, 2_048, 2_048),
        (1_024, 2_048, 1_024),
        (0, 2_048, 2_048),
        (-1, 2_048, 2_048),
        (128_000, 0, 2_048),
        (None, None, 2_048),
    ],
)
def test_compute_token_budgets_respects_requested_and_model_limits(
    model_limit,
    desired_max_tokens,
    expected_max_tokens,
):
    max_tokens, thinking_tokens = compute_token_budgets(
        model_limit,
        desired_max_tokens,
    )

    assert max_tokens == expected_max_tokens
    assert thinking_tokens is None


def test_compute_token_budgets_keeps_thinking_inside_requested_maximum():
    max_tokens, thinking_tokens = compute_token_budgets(
        model_limit=128_000,
        desired_max_tokens=2_048,
        desired_thinking_tokens=5_000,
    )

    assert max_tokens == 2_048
    assert thinking_tokens == 1_920


def test_compute_token_budgets_accepts_valid_thinking_budget_unchanged():
    assert compute_token_budgets(128_000, 4_096, 1_000) == (4_096, 1_000)


def test_compute_token_budgets_tolerates_invalid_values():
    assert compute_token_budgets("invalid", "invalid", "invalid") == (2_048, None)


# --- Idempotent submission helpers -------------------------------------------

import email.utils  # noqa: E402
from datetime import datetime, timedelta, timezone  # noqa: E402

from client import GranSabioClientError, admission_rejection_error  # noqa: E402
from client._common import (  # noqa: E402
    SAAS_REQUEST_ID_MAX_CHARS,
    _body_has_json_detail,
    generation_http_error_details,
    parse_retry_after_seconds,
    session_list_params,
    validate_saas_request_id,
)


@pytest.mark.parametrize("request_id", [
    "a", "axv2:" + "f" * 64 + ":" + "0" * 32, "A.b_c:d-e", "x" * SAAS_REQUEST_ID_MAX_CHARS,
])
def test_validate_saas_request_id_accepts_the_server_grammar(request_id):
    assert validate_saas_request_id(request_id) == request_id


@pytest.mark.parametrize("request_id", [
    "", "x" * (SAAS_REQUEST_ID_MAX_CHARS + 1), "has space", "slash/id", "١٢",
    "café", None, 7, b"bytes",
])
def test_validate_saas_request_id_rejects_everything_else(request_id):
    with pytest.raises(ValueError):
        validate_saas_request_id(request_id)


def _http_date(offset_seconds):
    moment = datetime.now(timezone.utc) + timedelta(seconds=offset_seconds)
    return email.utils.format_datetime(moment, usegmt=True)


def test_parse_retry_after_seconds_reads_delta_seconds_and_http_dates():
    assert parse_retry_after_seconds({"Retry-After": "5"}) == 5.0
    assert parse_retry_after_seconds({"Retry-After": " 0 "}) == 0.0
    future = parse_retry_after_seconds({"Retry-After": _http_date(120)})
    assert future is not None and 100.0 < future <= 120.0
    assert parse_retry_after_seconds({"Retry-After": _http_date(-3600)}) == 0.0


@pytest.mark.parametrize("headers", [
    {},
    {"Retry-After": "soon"},
    {"Retry-After": "-5"},
    {"Retry-After": "1.5"},
    {"Retry-After": "١٢"},
    # RFC 5322 "-0000" means an unknown zone: no absolute moment to wait for.
    {"Retry-After": "Wed, 21 Oct 2015 07:28:00 -0000"},
    {"Retry-After": 5},
    None,
    ["Retry-After", "5"],
])
def test_parse_retry_after_seconds_ignores_absent_or_malformed_values(headers):
    assert parse_retry_after_seconds(headers) is None


@pytest.mark.parametrize(("body", "expected"), [
    ('{"detail": "Invalid request"}', True),
    ('{"detail": {"code": "x"}}', True),
    ('{"message": "no detail key"}', False),
    ('[{"detail": "inside a list"}]', False),
    ('"detail"', False),
    ("<html><body>502 Bad Gateway</body></html>", False),
    ("", False),
])
def test_body_has_json_detail_only_for_engine_error_objects(body, expected):
    assert _body_has_json_detail(body) is expected


def test_generation_http_error_details_never_carry_the_body():
    payload = {"prompt": "private prompt", "generator_model": "m"}
    details = generation_http_error_details(
        payload, body_text='{"detail": "private request echo"}',
        headers={"Retry-After": "7"}, request_id="axv2:abc",
    )
    assert details["json_error_detail"] is True
    assert details["retry_after_seconds"] == 7.0
    assert details["request_id"] == "axv2:abc"
    assert "private" not in repr(details)
    bare = generation_http_error_details(payload, body_text="<html/>", headers={}, request_id=None)
    assert bare["json_error_detail"] is False
    assert "retry_after_seconds" not in bare and "request_id" not in bare


def test_session_list_params_validate_paging_before_any_request():
    assert session_list_params(25, 0, None) == {"limit": 25, "offset": 0}
    assert session_list_params(1, 7, "proj-1") == {"limit": 1, "offset": 7, "project_id": "proj-1"}
    for limit, offset, project_id in [
        (0, 0, None), (-1, 0, None), (True, 0, None), (1.0, 0, None),
        (1, -1, None), (1, False, None), (1, 0, ""), (1, 0, "  "), (1, 0, 3),
    ]:
        with pytest.raises(ValueError):
            session_list_params(limit, offset, project_id)


def test_admission_rejection_error_reads_preflight_and_auto_qa_rejections():
    preflight = {"status": "preflight_rejected", "session_id": None,
                 "preflight_feedback": {"decision": "reject", "user_feedback": "off-topic"}}
    error = admission_rejection_error(preflight)
    assert type(error) is GranSabioClientError and error.provider_outcome == "rejected"
    assert error.details == {"preflight_feedback": preflight["preflight_feedback"], "response": preflight}
    assert "off-topic" in str(error)
    legacy_status = {**preflight, "status": "rejected"}
    assert admission_rejection_error(legacy_status).provider_outcome == "rejected"
    auto_qa = {"status": "auto_qa_rejected", "auto_qa_feedback": {"message": "no QA plan"}}
    error = admission_rejection_error(auto_qa)
    assert error.details["auto_qa_feedback"] == {"message": "no QA plan"}
    assert "no QA plan" in str(error)


@pytest.mark.parametrize("response", [
    {"status": "initialized", "session_id": "sid-1"},
    {"status": "preflight_rejected", "preflight_feedback": {}},
    {"status": "preflight_rejected", "preflight_feedback": "not an object"},
    {"status": "auto_qa_rejected", "preflight_feedback": {"decision": "reject"}},
    {"status": "initialized", "preflight_feedback": {"decision": "proceed"}},
    None,
    ["preflight_rejected"],
])
def test_admission_rejection_error_ignores_other_admission_responses(response):
    assert admission_rejection_error(response) is None
