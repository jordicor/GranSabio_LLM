"""Contract tests for the stateless POST /context/preflight endpoint.

The response must satisfy the strict invariants of the
``gransabio_context_preflight.v2`` contract. ``_assert_client_contract``
mirrors those invariants so this suite proves compliance without importing
any downstream client.

The builder module is loaded directly from its file path so these tests do
not need to import the full ``core`` package (heavy route modules); only the
route-surface test imports the app, lazily.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[1] / "core" / "context_preflight.py"
_SPEC = importlib.util.spec_from_file_location("context_preflight", _MODULE_PATH)
context_preflight = importlib.util.module_from_spec(_SPEC)
sys.modules["context_preflight"] = context_preflight
_SPEC.loader.exec_module(context_preflight)


# ---------------------------------------------------------------------------
# Fixtures: payloads and fake catalog limits
# ---------------------------------------------------------------------------


def _policy(**overrides):
    policy = {
        "schema_version": "gransabio_context_budget_policy.v1",
        "context_fraction": 0.25,
        "minimum_working_context_tokens": 16384,
        "maximum_working_context_tokens": 65536,
        "safety_margin_fraction": 0.05,
    }
    policy.update(overrides)
    return policy


def _contracts(max_tokens: int, reasoning: int = 0):
    return {
        "generator": {
            "max_output_tokens": max_tokens,
            "max_reasoning_tokens": reasoning,
            "receives_generated_candidate": False,
        },
        "prompt_preflight": {
            "max_output_tokens": 800,
            "max_reasoning_tokens": 0,
            "receives_generated_candidate": False,
        },
        "qa_evaluator": {
            "default_max_output_tokens": 2048,
            "default_max_reasoning_tokens": 0,
            "receives_generated_candidate": True,
        },
        "arbiter": {
            "max_output_tokens": 2048,
            "max_reasoning_tokens": 0,
            "receives_generated_candidate": True,
        },
        "gran_sabio_review": {
            "max_output_tokens": 2048,
            "max_reasoning_tokens": 0,
            "receives_generated_candidate": True,
        },
        "gran_sabio_regeneration": {
            "max_output_tokens": max_tokens,
            "max_reasoning_tokens": reasoning,
            "receives_generated_candidate": True,
        },
        "gran_sabio_fallback": {
            "max_output_tokens": max_tokens,
            "max_reasoning_tokens": reasoning,
            "receives_generated_candidate": True,
        },
    }


def _qa_layer(name: str = "Factual Fidelity"):
    return {
        "name": name,
        "description": "Evaluate the declared quality dimension.",
        "criteria": "Check the generated candidate against the request.",
    }


def _payload(**overrides):
    payload = {
        "prompt": "Produce the requested structured content.",
        "content_type": "json",
        "model": "gen-model",
        "temperature": 0.7,
        "request_name": "generic_preflight_test",
        "json_output": True,
        "max_tokens": 4096,
        "min_global_score": 8.0,
        "max_iterations": 3,
        "gran_sabio_fallback": False,
        "context_budget_policy": _policy(),
        "context_participant_contracts": _contracts(4096),
    }
    payload.update(overrides)
    # Keep the generator contract coherent with max_tokens/thinking, like the
    # real client does when it builds context_participant_contracts.
    if (
        "max_tokens" in overrides or "thinking_budget_tokens" in overrides
    ) and "context_participant_contracts" not in overrides:
        payload["context_participant_contracts"] = _contracts(
            payload["max_tokens"],
            payload.get("thinking_budget_tokens") or 0,
        )
    return payload


def _limits(
    *,
    context_window=1_000_000,
    max_input=1_000_000,
    max_output=100_000,
    provider="anthropic",
    output_accounting="within_total",
    reasoning_accounting="within_output",
):
    return {
        "provider": provider,
        "max_input_tokens": max_input,
        "max_output_tokens": max_output,
        "max_total_context_tokens": context_window,
        "output_accounting": output_accounting,
        "reasoning_accounting": reasoning_accounting,
        "provenance": "manual_curated",
        "verified_at": "2026-07-01",
        "source": "test",
        "suspect": False,
    }


FAKE_LIMITS = {
    "jev-latest": _limits(
        context_window=32_000, max_input=32_000, max_output=256,
        provider="typesafe", output_accounting="separate",
    ),
    "gen-model": _limits(),
    "qa-model-a": _limits(),
    "qa-model-b": _limits(context_window=200_000, max_input=200_000, max_output=32_000),
    "gs-model": _limits(),
    "openai-model": _limits(provider="openai"),
    "tiny-model": _limits(context_window=16_384, max_input=16_384, max_output=4_096),
    "separate-model": _limits(
        max_input=500_000,
        output_accounting="separate",
        reasoning_accounting="separate",
    ),
}

FAKE_ROUTING = {
    "version": 1,
    "policy": {
        "unsupported_optional_params": "warn_and_drop",
        "missing_required_call": "fail_fast",
    },
    "global": {"model": None, "params": {}},
    "calls": {
        "generation.main": {"model": "gen-model", "params": {}},
        "qa.evaluate_layer": {
            "models": [
                {"model": "qa-model-a", "params": {"max_tokens": 2_048}},
                {"model": "qa-model-b", "params": {"max_tokens": 3_072}},
            ]
        },
        "gransabio.review": {
            "model": "gs-model",
            "params": {"max_tokens": 2_048},
        },
        "gransabio.regenerate": {"model": "gs-model", "params": {}},
        "arbiter.resolve": {
            "model": "qa-model-a",
            "params": {"max_tokens": 2_048},
        },
        "preflight.validate": {
            "model": "gen-model",
            "params": {"max_tokens": 800},
        },
    },
}

_REAL_GET_MODEL_TOKEN_LIMITS = type(
    context_preflight.config
).get_model_token_limits
_REAL_VALIDATE_TOKEN_LIMITS = type(context_preflight.config).validate_token_limits
_REAL_GET_MODEL_INFO = type(context_preflight.config).get_model_info


def test_system_one_qa_admits_typed_payload_without_chat_output_reservation():
    payload = _payload(
        qa_models=["jev-latest", "qa-model-a"],
        qa_global_config={"max_tokens": 2048},
        qa_layers=[{
            **_qa_layer(),
            "deal_breaker_criteria": "The candidate contradicts the supplied source.",
        }],
        source_text="A complete source retained for the typed factual review.",
    )

    response = context_preflight.build_context_preflight_response(payload)

    assert response["accepted"] is True
    _assert_client_contract(response, payload)
    typed = next(item for item in response["participants"] if item["model"] == "jev-latest")
    assert typed["call_kind"] == "system_one"
    assert typed["output_tokens"] == typed["reasoning_tokens"] == 0
    assert typed["generated_input_tokens"] == payload["max_tokens"]
    assert typed["max_output_tokens"] == 256
    specs = context_preflight._build_participant_specs(payload)
    typed_spec = next(item for item in specs if item.requested_model == "jev-latest")
    encoded = typed_spec.fixed_components[0][1]
    state = context_preflight.json.loads(encoded)
    assert state["state"]["source_text"] == payload["source_text"]
    assert state["state"]["original_prompt"] == payload["prompt"]
    assert set(state["questions"]) == {"quality_score", "deal_breaker"}
    chat = next(item for item in response["participants"] if item["participant"] == "qa:2")
    assert chat["call_kind"] == "chat"
    assert chat["output_tokens"] == 2048


@pytest.mark.parametrize("oversized", ["source", "question"])
def test_system_one_qa_rejects_oversized_evidence_or_question(oversized):
    layer = _qa_layer()
    source = "Short source."
    if oversized == "source":
        source = "Complete evidence must not be truncated. " * 10_000
    else:
        layer["criteria"] = "Evaluate every required criterion. " * 10_000
    response = context_preflight.build_context_preflight_response(_payload(
        qa_models=["jev-latest"], qa_layers=[layer], source_text=source,
    ))

    assert response["accepted"] is False
    assert response["rejection"]["code"] == "prompt_too_large"
    typed = next(item for item in response["participants"] if item["model"] == "jev-latest")
    assert typed["evidence_tokens"] < 0
    assert typed["output_tokens"] == 0


@pytest.fixture(autouse=True)
def fake_catalog(monkeypatch):
    def fake_get_model_token_limits(self, model_name):
        entry = FAKE_LIMITS.get(model_name)
        if entry is None:
            return None
        resolved = dict(entry)
        resolved["model"] = model_name
        return resolved

    def fake_get_model_info(self, model_name, *, require_api_key=True):
        entry = FAKE_LIMITS.get(model_name)
        if entry is None:
            return _REAL_GET_MODEL_INFO(
                self,
                model_name,
                require_api_key=False,
            )
        return {
            "provider": entry["provider"],
            "catalog_provider": entry["provider"],
            "model_id": model_name,
            "capabilities": [],
        }

    def fake_validate_token_limits(
        self,
        model_name,
        max_tokens,
        reasoning_effort=None,
        thinking_budget_tokens=None,
        max_tokens_percentage=None,
        *,
        require_api_key=True,
    ):
        return {
            "adjusted_reasoning_effort": reasoning_effort,
            "adjusted_thinking_budget_tokens": thinking_budget_tokens,
        }

    monkeypatch.setattr(
        type(context_preflight.config),
        "get_model_token_limits",
        fake_get_model_token_limits,
    )
    monkeypatch.setattr(
        type(context_preflight.config),
        "validate_token_limits",
        fake_validate_token_limits,
    )
    monkeypatch.setattr(
        type(context_preflight.config),
        "get_model_info",
        fake_get_model_info,
    )
    monkeypatch.setattr(
        context_preflight.llm_routing,
        "get_default_routing",
        lambda: FAKE_ROUTING,
    )
    yield


# ---------------------------------------------------------------------------
# Client-contract mirror (the strict from_payload invariants)
# ---------------------------------------------------------------------------


def _assert_client_contract(response: dict, payload: dict) -> None:
    policy = payload["context_budget_policy"]
    contracts = payload["context_participant_contracts"]
    max_tokens = payload["max_tokens"]

    assert response["schema_version"] == "gransabio_context_preflight.v2"
    assert response["participants_complete"] is True
    assert response["request_name"] == payload.get("request_name")
    assert response["context_budget_policy"] == policy
    assert response["context_participant_contracts"] == contracts
    assert isinstance(response["accepted"], bool)
    if response["accepted"]:
        assert response["status"] in {"accepted", "fits", "proceed", "ok"}
        assert response["rejection"] is None
    else:
        assert response["status"] in {
            "rejected",
            "overflow",
            "prompt_too_large",
            "atomic_contract_does_not_fit",
        }
        rejection = response["rejection"]
        assert isinstance(rejection, dict)
        assert str(rejection.get("code") or "").strip()
        assert str(rejection.get("message") or "").strip()

    participants = response["participants"]
    assert isinstance(participants, list) and participants
    labels = [item["participant"] for item in participants]
    assert len(labels) == len(set(labels)), "duplicate participant entry"

    for item in participants:
        for field in (
            "participant",
            "participant_kind",
            "requested_model",
            "model",
            "count_method",
        ):
            assert isinstance(item[field], str) and item[field].strip()
        assert item["participant_kind"] in contracts
        role_contract = contracts[item["participant_kind"]]
        assert (
            item["receives_generated_candidate"]
            == role_contract["receives_generated_candidate"]
        )
        for field in (
            "context_window_tokens",
            "max_input_tokens",
            "max_output_tokens",
            "working_context_tokens",
            "fixed_input_tokens",
            "generated_input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "safety_margin_tokens",
            "reserved_context_tokens",
            "usable_input_tokens",
        ):
            assert isinstance(item[field], int) and not isinstance(item[field], bool)
            assert item[field] >= 0
            if response["accepted"] and field in {
                "context_window_tokens",
                "max_input_tokens",
                "max_output_tokens",
                "working_context_tokens",
            }:
                assert item[field] > 0
        assert item["output_accounting"] in {"within_total", "separate", "unknown"}
        assert item["reasoning_accounting"] in {"within_output", "separate", "unknown"}
        assert isinstance(item["evidence_tokens"], int)
        if response["accepted"]:
            assert item["evidence_tokens"] >= 0

        if item["participant_kind"] == "generator":
            assert item["output_tokens"] == role_contract["max_output_tokens"]
            assert item["reasoning_tokens"] <= role_contract["max_reasoning_tokens"]
        elif item["participant_kind"] != "qa_evaluator":
            assert item["output_tokens"] >= role_contract["max_output_tokens"]
            assert item["reasoning_tokens"] >= role_contract["max_reasoning_tokens"]

        # Per-participant arithmetic identities.
        context_window = item["context_window_tokens"]
        if context_window == 0:
            assert not response["accepted"]
            continue
        assert item["max_input_tokens"] <= context_window
        expected_working = min(
            context_window,
            max(
                int(policy["minimum_working_context_tokens"]),
                min(
                    int(policy.get("maximum_working_context_tokens") or context_window),
                    math.floor(float(policy["context_fraction"]) * context_window),
                ),
            ),
        )
        assert item["working_context_tokens"] == expected_working
        if response["accepted"]:
            assert item["output_tokens"] <= item["max_output_tokens"]
        expected_margin = math.ceil(
            float(policy["safety_margin_fraction"]) * expected_working
        )
        assert item["safety_margin_tokens"] == expected_margin
        expected_reserved = 0
        if item["output_accounting"] != "separate":
            expected_reserved += item["output_tokens"]
        if item["reasoning_accounting"] != "within_output":
            expected_reserved += item["reasoning_tokens"]
        assert item["reserved_context_tokens"] == expected_reserved
        expected_usable = min(
            item["max_input_tokens"], max(0, expected_working - expected_reserved)
        )
        assert item["usable_input_tokens"] == expected_usable
        assert item["evidence_tokens"] == (
            expected_usable
            - item["fixed_input_tokens"]
            - item["generated_input_tokens"]
            - expected_margin
        )

    generators = [p for p in participants if p["participant_kind"] == "generator"]
    assert len(generators) == 1
    assert generators[0]["requested_model"] == payload["model"]
    assert generators[0]["output_tokens"] == max_tokens

    preflights = [p for p in participants if p["participant_kind"] == "prompt_preflight"]
    assert len(preflights) == 1

    if payload.get("qa_layers"):
        runtime_routing = context_preflight._resolve_runtime_routing(payload)
        qa_entries = context_preflight._normalized_qa_models(
            payload,
            runtime_routing.qa_models,
        )
    else:
        qa_entries = []
    qa_participants = [p for p in participants if p["participant_kind"] == "qa_evaluator"]
    assert len(qa_participants) == len(qa_entries)
    for entry, item in zip(qa_entries, qa_participants):
        assert item["requested_model"] == entry["model"]
        if item.get("call_kind") == "system_one":
            assert item["output_tokens"] == item["reasoning_tokens"] == 0
            assert item["generated_input_tokens"] >= max_tokens
            continue
        assert item["output_tokens"] == int(entry["max_tokens"])
        assert item["reasoning_tokens"] >= int(
            entry.get("thinking_budget_tokens") or 0
        )
        assert (
            item["requested_reasoning_effort"]
            == entry.get("reasoning_effort")
        )
        assert (
            item["requested_thinking_budget_tokens"]
            == entry.get("thinking_budget_tokens")
        )
        assert item["receives_generated_candidate"] is True
        assert item["generated_input_tokens"] >= max_tokens

    for item in participants:
        if item["receives_generated_candidate"]:
            assert item["generated_input_tokens"] >= max_tokens
        else:
            assert item["generated_input_tokens"] == 0

    if qa_entries:
        assert any(p["participant_kind"] == "arbiter" for p in participants)
    expected_gs_kinds = set()
    if qa_entries:
        expected_gs_kinds.add("gran_sabio_review")
    if payload.get("gran_sabio_fallback"):
        expected_gs_kinds.update(
            {"gran_sabio_review", "gran_sabio_regeneration", "gran_sabio_fallback"}
        )
    for kind in expected_gs_kinds:
        assert any(p["participant_kind"] == kind for p in participants), kind

    assert response["working_context_tokens"] == min(
        p["working_context_tokens"] for p in participants
    )
    assert response["evidence_budget_tokens"] == min(
        p["evidence_tokens"] for p in participants
    )


# ---------------------------------------------------------------------------
# Acceptance cases
# ---------------------------------------------------------------------------


def test_readiness_like_payload_accepted():
    payload = _payload(
        prompt="Validate this small stateless readiness request.",
        max_tokens=256,
        content_type="other",
        json_output=False,
    )
    response = context_preflight.build_context_preflight_response(payload)
    assert response["accepted"] is True
    assert response["status"] == "accepted"
    assert {p["participant_kind"] for p in response["participants"]} == {
        "generator",
        "prompt_preflight",
    }
    assert response["evidence_budget_tokens"] > 0
    assert response["working_context_tokens"] == 65_536  # 1M window hits the cap
    _assert_client_contract(response, payload)


@pytest.mark.parametrize("request_name", [None, "", "   ", "  generic request  "])
def test_optional_request_name_matches_generate_contract(request_name):
    payload = _payload(request_name=request_name)
    response = context_preflight.build_context_preflight_response(payload)

    assert response["accepted"] is True
    assert response["request_name"] == request_name
    _assert_client_contract(response, payload)


def test_missing_request_name_is_accepted_and_echoed_as_null():
    payload = _payload()
    del payload["request_name"]

    response = context_preflight.build_context_preflight_response(payload)

    assert response["accepted"] is True
    assert response["request_name"] is None
    _assert_client_contract(response, payload)


@pytest.mark.parametrize("request_name", ["x" * 129, 42])
def test_invalid_optional_request_name_is_rejected(request_name):
    with pytest.raises(context_preflight.MalformedRequest):
        context_preflight.build_context_preflight_response(
            _payload(request_name=request_name)
        )


def test_request_name_accepts_exactly_128_characters():
    payload = _payload(request_name="x" * 128)
    response = context_preflight.build_context_preflight_response(payload)

    assert response["accepted"] is True
    assert response["request_name"] == payload["request_name"]
    _assert_client_contract(response, payload)


def test_legacy_policy_schema_alias_is_accepted_and_echoed_exactly():
    legacy_policy = _policy(schema_version="bmc_context_budget_policy.v1")
    payload = _payload(context_budget_policy=legacy_policy)

    response = context_preflight.build_context_preflight_response(payload)

    assert response["accepted"] is True
    assert response["context_budget_policy"] == legacy_policy
    assert (
        "bmc_context_budget_policy.v1"
        in context_preflight.LEGACY_POLICY_SCHEMA_VERSIONS
    )
    _assert_client_contract(response, payload)


def test_effective_reasoning_default_is_reported_within_output(monkeypatch):
    def fake_validate_token_limits(self, *args, **kwargs):
        return {
            "adjusted_reasoning_effort": "high",
            "adjusted_thinking_budget_tokens": None,
        }

    monkeypatch.setattr(
        type(context_preflight.config),
        "validate_token_limits",
        fake_validate_token_limits,
    )
    payload = _payload(model="gen-model", max_tokens=16_384)
    response = context_preflight.build_context_preflight_response(payload)
    generator = next(
        item
        for item in response["participants"]
        if item["participant_kind"] == "generator"
    )

    assert generator["requested_reasoning_effort"] is None
    assert generator["effective_reasoning_effort"] == "high"
    assert generator["effective_thinking_budget_tokens"] is None
    assert generator["reasoning_accounting"] == "within_output"
    assert generator["reasoning_tokens"] == 0
    assert generator["reserved_context_tokens"] == generator["output_tokens"]
    _assert_client_contract(response, payload)


def test_real_reasoning_resolution_is_provider_free_and_unknown_is_structured(
    monkeypatch,
):
    """Exercise the real catalog validator, not the suite's permissive fake."""

    monkeypatch.setattr(
        type(context_preflight.config),
        "get_model_token_limits",
        _REAL_GET_MODEL_TOKEN_LIMITS,
    )
    monkeypatch.setattr(
        type(context_preflight.config),
        "validate_token_limits",
        _REAL_VALIDATE_TOKEN_LIMITS,
    )
    monkeypatch.setattr(context_preflight.config, "ANTHROPIC_API_KEY", "")

    payload = _payload(
        model="claude-opus-4-8",
        max_tokens=16_384,
        reasoning_effort="low",
        llm_routing={
            "calls": {
                "preflight.validate": {"model": "claude-opus-4-8"},
            }
        },
    )
    response = context_preflight.build_context_preflight_response(payload)
    generator = next(
        item
        for item in response["participants"]
        if item["participant_kind"] == "generator"
    )

    assert response["accepted"] is True
    assert generator["effective_reasoning_effort"] == "low"
    assert generator["effective_thinking_budget_tokens"] is None

    with pytest.raises(
        context_preflight.MalformedRequest,
        match="could not resolve runtime LLM routing",
    ):
        context_preflight.build_context_preflight_response(
            _payload(model="not-in-catalog")
        )


def test_inactive_internal_routes_do_not_require_catalog_models(monkeypatch):
    calls = FAKE_ROUTING["calls"]
    monkeypatch.setitem(calls, "qa.evaluate_layer", {"models": [{"model": "missing-qa"}]})
    for call_id in ("gransabio.review", "gransabio.regenerate", "gransabio.escalation", "arbiter.resolve"):
        monkeypatch.setitem(calls, call_id, {"model": "missing-inactive-model"})

    payload = _payload(qa_layers=[], gran_sabio_fallback=False)
    response = context_preflight.build_context_preflight_response(payload)

    assert response["accepted"] is True
    assert response["participants_complete"] is True
    assert [p["participant_kind"] for p in response["participants"]] == ["generator", "prompt_preflight"]
    _assert_client_contract(response, payload)


def test_implicit_internal_models_match_generate_default_routing():
    payload = _payload(qa_layers=[_qa_layer()])

    response = context_preflight.build_context_preflight_response(payload)

    assert response["accepted"] is True
    assert response["participants_complete"] is True
    assert [
        (item["participant_kind"], item["requested_model"])
        for item in response["participants"]
    ] == [
        ("generator", "gen-model"),
        ("qa_evaluator", "qa-model-a"),
        ("qa_evaluator", "qa-model-b"),
        ("arbiter", "qa-model-a"),
        ("arbiter", "gs-model"),
        ("gran_sabio_review", "gs-model"),
        ("prompt_preflight", "gen-model"),
    ]
    _assert_client_contract(response, payload)


def test_request_llm_routing_overlay_controls_every_internal_participant(
    monkeypatch,
):
    monkeypatch.setitem(
        FAKE_LIMITS["qa-model-b"],
        "reasoning_accounting",
        "separate",
    )
    payload = _payload(
        qa_layers=[_qa_layer()],
        gran_sabio_fallback=True,
        llm_routing={
            "calls": {
                "qa.evaluate_layer": {
                    "models": [
                        {
                            "model": "qa-model-b",
                            "params": {"max_tokens": 5_000},
                        }
                    ]
                },
                "gransabio.review": {
                    "model": "qa-model-b",
                    "params": {
                        "max_tokens": 9_000,
                        "thinking_budget_tokens": 7_000,
                    },
                },
                "gransabio.regenerate": {
                    "model": "qa-model-a",
                    "params": {
                        "max_tokens": 20_000,
                        "thinking_budget_tokens": 3_000,
                    },
                },
                "gransabio.escalation": {
                    "model": "qa-model-a",
                    "params": {
                        "max_tokens": 13_000,
                        "thinking_budget_tokens": 4_000,
                    },
                },
                "arbiter.resolve": {
                    "model": "gs-model",
                    "params": {"max_tokens": 11_000},
                },
                "preflight.validate": {
                    "model": "qa-model-b",
                    "params": {"max_tokens": 12_000},
                },
            }
        },
    )

    response = context_preflight.build_context_preflight_response(payload)

    assert response["accepted"] is True
    assert response["participants_complete"] is True
    assert [
        (item["participant_kind"], item["requested_model"])
        for item in response["participants"]
    ] == [
        ("generator", "gen-model"),
        ("qa_evaluator", "qa-model-b"),
        ("arbiter", "gs-model"),
        ("arbiter", "qa-model-a"),
        ("gran_sabio_review", "qa-model-b"),
        ("gran_sabio_regeneration", "qa-model-a"),
        ("gran_sabio_fallback", "qa-model-a"),
        ("prompt_preflight", "qa-model-b"),
    ]
    participants = {
        item["participant_kind"]: item
        for item in response["participants"]
    }
    participants_by_name = {
        item["participant"]: item
        for item in response["participants"]
    }
    assert participants["gran_sabio_review"]["output_tokens"] == 9_000
    assert participants["gran_sabio_review"]["reasoning_tokens"] == 7_000
    assert (
        participants["gran_sabio_review"]["requested_thinking_budget_tokens"]
        == 7_000
    )
    assert (
        participants["gran_sabio_review"]["effective_thinking_budget_tokens"]
        == 7_000
    )
    assert (
        participants["gran_sabio_review"]["reserved_context_tokens"]
        == 9_000 + 7_000
    )
    assert participants_by_name["arbiter:conditional"]["output_tokens"] == 11_000
    escalation = participants_by_name["arbiter:escalation:conditional"]
    assert escalation["output_tokens"] == 13_000
    assert escalation["reasoning_tokens"] == 4_000
    assert escalation["requested_thinking_budget_tokens"] == 4_000
    assert participants["prompt_preflight"]["output_tokens"] == 12_000
    for kind in ("gran_sabio_regeneration", "gran_sabio_fallback"):
        # The live regenerate call gives request.max_tokens precedence over
        # the larger routed max_tokens, while routed thinking still applies.
        assert participants[kind]["output_tokens"] == payload["max_tokens"]
        assert participants[kind]["reasoning_tokens"] == 3_000
        assert participants[kind]["requested_thinking_budget_tokens"] == 3_000
        assert participants[kind]["effective_thinking_budget_tokens"] == 3_000
    _assert_client_contract(response, payload)


def test_unresolvable_internal_routing_fails_instead_of_claiming_complete():
    payload = _payload(
        qa_layers=[_qa_layer()],
        llm_routing={
            "calls": {
                "arbiter.resolve": {"model": "not-in-catalog"},
            }
        },
    )

    with pytest.raises(
        context_preflight.MalformedRequest,
        match="could not resolve runtime LLM routing",
    ):
        context_preflight.build_context_preflight_response(payload)


def test_full_generation_payload_participants_complete():
    payload = _payload(
        source_text="x" * 50_000,
        qa_models=[
            {"model": "qa-model-a", "max_tokens": 2048},
            {"model": "qa-model-b", "max_tokens": 2048, "thinking_budget_tokens": 4096},
        ],
        qa_layers=[_qa_layer()],
        gran_sabio_model="gs-model",
        gran_sabio_fallback=True,
        thinking_budget_tokens=8192,
        json_schema={"type": "object"},
    )
    response = context_preflight.build_context_preflight_response(payload)
    assert response["accepted"] is True
    kinds = [p["participant_kind"] for p in response["participants"]]
    assert kinds == [
        "generator",
        "qa_evaluator",
        "qa_evaluator",
        "arbiter",
        "arbiter",
        "gran_sabio_review",
        "gran_sabio_regeneration",
        "gran_sabio_fallback",
        "prompt_preflight",
    ]
    arbiters = [
        p for p in response["participants"] if p["participant_kind"] == "arbiter"
    ]
    assert [
        (item["participant"], item["requested_model"])
        for item in arbiters
    ] == [
        ("arbiter:conditional", "qa-model-a"),
        ("arbiter:escalation:conditional", "gs-model"),
    ]
    # Candidate consumers reserve the full generated candidate.
    for item in response["participants"]:
        if item["receives_generated_candidate"]:
            assert item["generated_input_tokens"] == payload["max_tokens"]
    # The least-capable participant (200K-window QA evaluator) drives globals.
    assert response["working_context_tokens"] == 50_000
    _assert_client_contract(response, payload)


def test_qa_default_output_uses_runtime_resolver_not_fixed_floor(monkeypatch):
    def fake_resolve_qa_max_tokens(
        self,
        model_name,
        requested_max_tokens=None,
    ):
        assert model_name == "qa-model-a"
        return requested_max_tokens or 12_000

    monkeypatch.setattr(
        type(context_preflight.config),
        "resolve_qa_max_tokens",
        fake_resolve_qa_max_tokens,
    )
    payload = _payload(
        qa_models=["qa-model-a"],
        qa_layers=[_qa_layer()],
    )

    response = context_preflight.build_context_preflight_response(payload)
    evaluator = next(
        item
        for item in response["participants"]
        if item["participant_kind"] == "qa_evaluator"
    )

    assert response["accepted"] is True
    assert evaluator["output_tokens"] == 12_000
    _assert_client_contract(response, payload)


def test_qa_overlays_match_runtime_normalization_precedence():
    payload = _payload(
        qa_models=[
            {"model": "qa-model-a"},
            {
                "model": "qa-model-b",
                "max_tokens": 9_000,
                "reasoning_effort": "low",
                "thinking_budget_tokens": 4_096,
            },
        ],
        qa_global_config={
            "max_tokens": 6_000,
            "reasoning_effort": "medium",
            "thinking_budget_tokens": 2_048,
        },
        qa_models_config={
            "qa-model-a": {
                "max_tokens": 7_000,
                "reasoning_effort": "high",
                "thinking_budget_tokens": 3_072,
            },
            "qa-model-b": {
                "max_tokens": 12_000,
                "reasoning_effort": "high",
                "thinking_budget_tokens": 8_192,
            },
        },
        qa_layers=[_qa_layer()],
    )

    response = context_preflight.build_context_preflight_response(payload)
    evaluators = [
        item
        for item in response["participants"]
        if item["participant_kind"] == "qa_evaluator"
    ]

    assert response["accepted"] is True
    assert [
        (
            item["requested_model"],
            item["output_tokens"],
            item["requested_reasoning_effort"],
            item["requested_thinking_budget_tokens"],
            item["reasoning_tokens"],
        )
        for item in evaluators
    ] == [
        ("qa-model-a", 7_000, "high", 3_072, 3_072),
        ("qa-model-b", 9_000, "low", 4_096, 4_096),
    ]
    _assert_client_contract(response, payload)


def test_all_role_reasoning_requests_are_preserved_without_underreservation(
    monkeypatch,
):
    def fake_validate_token_limits(
        self,
        model_name,
        max_tokens,
        reasoning_effort=None,
        thinking_budget_tokens=None,
        max_tokens_percentage=None,
        *,
        require_api_key=True,
    ):
        del self, model_name, max_tokens, max_tokens_percentage, require_api_key
        return {
            "adjusted_reasoning_effort": reasoning_effort,
            "adjusted_thinking_budget_tokens": (
                1_024 if thinking_budget_tokens is not None else None
            ),
        }

    monkeypatch.setattr(
        type(context_preflight.config),
        "validate_token_limits",
        fake_validate_token_limits,
    )
    contracts = _contracts(4_096, reasoning=8_192)
    secondary_kinds = {
        "prompt_preflight",
        "arbiter",
        "gran_sabio_review",
        "gran_sabio_regeneration",
        "gran_sabio_fallback",
    }
    for kind in secondary_kinds:
        contracts[kind]["max_reasoning_tokens"] = 4_096
        contracts[kind]["reasoning_effort"] = "medium"
        contracts[kind]["thinking_budget_tokens"] = 4_096

    payload = _payload(
        max_tokens=4_096,
        reasoning_effort="high",
        thinking_budget_tokens=8_192,
        qa_models=[{"model": "qa-model-a", "max_tokens": 2_048}],
        qa_layers=[_qa_layer()],
        gran_sabio_model="gs-model",
        gran_sabio_fallback=True,
        context_participant_contracts=contracts,
    )

    response = context_preflight.build_context_preflight_response(payload)
    participants = {
        item["participant_kind"]: item
        for item in response["participants"]
    }

    assert response["accepted"] is True
    generator = participants["generator"]
    assert generator["requested_reasoning_effort"] == "high"
    assert generator["requested_thinking_budget_tokens"] == 8_192
    assert generator["reasoning_tokens"] == 8_192
    for kind in secondary_kinds:
        participant = participants[kind]
        assert participant["requested_reasoning_effort"] == "medium"
        assert participant["requested_thinking_budget_tokens"] == 4_096
        # Runtime normalization may lower an effective request for a model,
        # but it must never replace the explicit contract maximum.
        assert participant["effective_thinking_budget_tokens"] == 1_024
        assert participant["reasoning_tokens"] == 4_096
    _assert_client_contract(response, payload)


def test_openai_model_count_method_exact_or_degraded():
    payload = _payload(model="openai-model")
    response = context_preflight.build_context_preflight_response(payload)
    generator = response["participants"][0]
    assert generator["count_method"] in {"exact_local", "conservative_estimate"}
    _assert_client_contract(response, payload)


def test_separate_output_accounting_has_no_output_reservation():
    payload = _payload(model="separate-model")
    response = context_preflight.build_context_preflight_response(payload)
    generator = response["participants"][0]
    assert generator["output_accounting"] == "separate"
    assert generator["reserved_context_tokens"] == 0
    assert generator["usable_input_tokens"] == min(
        generator["max_input_tokens"], generator["working_context_tokens"]
    )
    _assert_client_contract(response, payload)


def test_separate_reasoning_is_reserved_on_top_of_output():
    FAKE_LIMITS["reasoning-model"] = _limits(
        output_accounting="within_total", reasoning_accounting="separate"
    )
    try:
        payload = _payload(model="reasoning-model", thinking_budget_tokens=4096)
        response = context_preflight.build_context_preflight_response(payload)
        generator = response["participants"][0]
        assert generator["reserved_context_tokens"] == (
            generator["output_tokens"] + generator["reasoning_tokens"]
        )
        _assert_client_contract(response, payload)
    finally:
        del FAKE_LIMITS["reasoning-model"]


def test_effective_reasoning_budget_is_reported_and_reserved(monkeypatch):
    FAKE_LIMITS["effective-reasoning-model"] = _limits(
        output_accounting="separate",
        reasoning_accounting="separate",
    )

    def fake_validate_token_limits(self, *args, **kwargs):
        model_name = args[0] if args else kwargs.get("model_name")
        return {
            "adjusted_reasoning_effort": None,
            "adjusted_thinking_budget_tokens": (
                6144 if model_name == "effective-reasoning-model" else None
            ),
        }

    monkeypatch.setattr(
        type(context_preflight.config),
        "validate_token_limits",
        fake_validate_token_limits,
    )
    contracts = _contracts(4096, reasoning=8192)
    try:
        payload = _payload(
            model="effective-reasoning-model",
            preflight_model="gen-model",
            context_participant_contracts=contracts,
        )
        response = context_preflight.build_context_preflight_response(payload)
        generator = response["participants"][0]

        assert generator["effective_thinking_budget_tokens"] == 6144
        assert generator["reasoning_tokens"] == 6144
        assert generator["reserved_context_tokens"] == 6144
        _assert_client_contract(response, payload)
    finally:
        del FAKE_LIMITS["effective-reasoning-model"]


def test_evidence_budget_drives_packing_decision():
    payload = _payload(source_text="y" * 30_000)
    response = context_preflight.build_context_preflight_response(payload)
    generator = response["participants"][0]
    # usable - fixed - generated - margin, with the 1M-window cap at 64K working.
    assert generator["usable_input_tokens"] == 65_536 - 4_096
    assert generator["fixed_input_tokens"] > 0
    assert generator["evidence_tokens"] == response["evidence_budget_tokens"]


def test_null_input_split_uses_window_as_ceiling():
    """Curated entries may leave max_input_tokens null (no explicit split)."""
    nullsplit = _limits(context_window=200_000, max_input=200_000, max_output=64_000)
    nullsplit["max_input_tokens"] = None
    FAKE_LIMITS["nullsplit-model"] = nullsplit
    try:
        payload = _payload(model="nullsplit-model")
        response = context_preflight.build_context_preflight_response(payload)
        assert response["accepted"] is True
        generator = response["participants"][0]
        assert generator["context_window_tokens"] == 200_000
        assert generator["max_input_tokens"] == 200_000
        _assert_client_contract(response, payload)
    finally:
        del FAKE_LIMITS["nullsplit-model"]


# ---------------------------------------------------------------------------
# Structured rejections
# ---------------------------------------------------------------------------


def test_unknown_routed_model_is_malformed_like_generate():
    payload = _payload(model="not-in-catalog")
    with pytest.raises(
        context_preflight.MalformedRequest,
        match="could not resolve runtime LLM routing",
    ):
        context_preflight.build_context_preflight_response(payload)


@pytest.mark.parametrize("maximum", [65536, None])
def test_output_overflow_rejected(maximum):
    payload = _payload(
        model="tiny-model", max_tokens=8192,  # tiny allows only 4096 out
        context_budget_policy=_policy(maximum_working_context_tokens=maximum),
    )
    response = context_preflight.build_context_preflight_response(payload)
    assert response["accepted"] is False
    assert response["rejection"]["code"] == "output_budget_exceeds_model_limit"
    assert response["rejection"]["participant"] == "generator"
    _assert_client_contract(response, payload)


@pytest.mark.parametrize("maximum", [65536, None])
def test_tiny_window_prompt_too_large(maximum):
    payload = _payload(
        model="tiny-model",
        max_tokens=4096,
        source_text="z" * 200_000,
        context_budget_policy=_policy(maximum_working_context_tokens=maximum),
    )
    response = context_preflight.build_context_preflight_response(payload)
    assert response["accepted"] is False
    assert response["status"] == "prompt_too_large"
    assert response["rejection"]["code"] == "prompt_too_large"
    _assert_client_contract(response, payload)


@pytest.mark.parametrize("configured,requested", [(0.05, 0.04), (0.08, 0.05)])
def test_minimum_safety_margin_enforced_without_service_ceiling(monkeypatch, configured, requested):
    monkeypatch.setenv("GRANSABIO_CONTEXT_POLICY_SAFETY_MARGIN", str(configured))
    payload = _payload(context_budget_policy=_policy(
        maximum_working_context_tokens=None, safety_margin_fraction=requested,
    ))
    response = context_preflight.build_context_preflight_response(payload)
    assert response["accepted"] is False
    assert response["rejection"]["code"] == "context_budget_policy_mismatch"
    _assert_client_contract(response, payload)


@pytest.mark.parametrize("context_window", [262144, 1050000, 2000000])
def test_model_capacity_replaces_retired_service_ceiling(monkeypatch, context_window):
    monkeypatch.setenv("GRANSABIO_CONTEXT_POLICY_MAX_TOKENS", "262144")
    monkeypatch.setitem(FAKE_LIMITS, "gen-model", _limits(
        context_window=context_window, max_input=context_window - 128000,
        max_output=128000,
    ))
    payload = _payload(max_tokens=32768, context_budget_policy=_policy(
        context_fraction=1, minimum_working_context_tokens=1,
        maximum_working_context_tokens=None,
    ))
    response = context_preflight.build_context_preflight_response(payload)
    assert response["accepted"] is True
    assert response["working_context_tokens"] == context_window
    generator = response["participants"][0]
    assert generator["reserved_context_tokens"] == 32768
    assert generator["safety_margin_tokens"] == math.ceil(context_window * 0.05)
    assert generator["usable_input_tokens"] == context_window - 128000
    _assert_client_contract(response, payload)


def test_omitted_maximum_uses_each_participants_native_window():
    policy = _policy(context_fraction=1, minimum_working_context_tokens=1)
    del policy["maximum_working_context_tokens"]
    payload = _payload(
        context_budget_policy=policy,
        qa_models=["qa-model-a", "qa-model-b"], qa_layers=[_qa_layer()],
    )
    response = context_preflight.build_context_preflight_response(payload)
    assert response["accepted"] is True
    assert response["working_context_tokens"] == 200000
    assert {p["working_context_tokens"] for p in response["participants"]} == {200000, 1000000}
    _assert_client_contract(response, payload)


def test_explicit_ceiling_above_model_capacity_is_clamped_to_model(monkeypatch):
    monkeypatch.setenv("GRANSABIO_CONTEXT_POLICY_MAX_TOKENS", "262144")
    payload = _payload(context_budget_policy=_policy(
        context_fraction=1, maximum_working_context_tokens=2000000,
    ))
    response = context_preflight.build_context_preflight_response(payload)
    assert response["accepted"] is True
    assert response["working_context_tokens"] == 1000000
    _assert_client_contract(response, payload)


def test_run_ceiling_caps_each_participant_independently_of_model_window():
    payload = _payload(context_budget_policy=_policy(
        context_fraction=1, minimum_working_context_tokens=1,
        maximum_working_context_tokens=32000,
    ))
    response = context_preflight.build_context_preflight_response(payload)
    assert response["accepted"] is True
    assert all(item["working_context_tokens"] == 32000 for item in response["participants"])
    _assert_client_contract(response, payload)


def test_native_system_one_preflight_counts_typed_payload_and_enforces_run_ceiling():
    payload = {
        "operation": "system_one", "model": "jev-latest", "state": "State " * 100,
        "questions": {"q": {
            "type": "choice", "instructions": "Select an option.",
            "criteria": {"a": "Option A", "b": "Option B"},
        }},
        "context_budget_policy": _policy(
            context_fraction=1, minimum_working_context_tokens=1,
            maximum_working_context_tokens=32000,
        ),
        "context_participant_contracts": {"system_one": {
            "max_output_tokens": 0, "max_reasoning_tokens": 0,
            "receives_generated_candidate": False,
        }},
    }
    response = context_preflight.build_context_preflight_response(payload)
    assert response["accepted"] is True
    participant, = response["participants"]
    assert participant["call_kind"] == "system_one"
    assert participant["fixed_input_tokens"] > 0
    assert participant["output_tokens"] == participant["generated_input_tokens"] == 0
    payload["context_budget_policy"]["maximum_working_context_tokens"] = 32
    rejected = context_preflight.build_context_preflight_response(payload)
    assert rejected["accepted"] is False
    assert rejected["rejection"]["code"] == "prompt_too_large"


def test_auto_qa_rejected_as_non_deterministic():
    payload = _payload(auto_qa={"enabled": True})
    response = context_preflight.build_context_preflight_response(payload)
    assert response["accepted"] is False
    assert response["rejection"]["code"] == "non_deterministic_request"
    _assert_client_contract(response, payload)


def test_attachments_rejected_as_unsupported():
    payload = _payload(attachments=[{"name": "file.pdf"}])
    response = context_preflight.build_context_preflight_response(payload)
    assert response["accepted"] is False
    assert response["rejection"]["code"] == "unsupported_payload_component"
    _assert_client_contract(response, payload)


# ---------------------------------------------------------------------------
# Malformed payloads (HTTP 400 surface)
# ---------------------------------------------------------------------------


def test_missing_policy_is_malformed():
    payload = _payload()
    del payload["context_budget_policy"]
    with pytest.raises(context_preflight.MalformedRequest):
        context_preflight.build_context_preflight_response(payload)


def test_unknown_policy_schema_is_malformed():
    payload = _payload(
        context_budget_policy=_policy(schema_version="consumer_policy.v99")
    )
    with pytest.raises(context_preflight.MalformedRequest):
        context_preflight.build_context_preflight_response(payload)


@pytest.mark.parametrize("maximum", [0, -1, "not-a-number", float("inf")])
def test_invalid_explicit_maximum_is_malformed(maximum):
    payload = _payload(context_budget_policy=_policy(maximum_working_context_tokens=maximum))
    with pytest.raises(context_preflight.MalformedRequest):
        context_preflight.build_context_preflight_response(payload)


def test_missing_contracts_is_malformed():
    payload = _payload()
    del payload["context_participant_contracts"]
    with pytest.raises(context_preflight.MalformedRequest):
        context_preflight.build_context_preflight_response(payload)


def test_generator_contract_mismatch_is_malformed():
    contracts = _contracts(4096)
    contracts["generator"]["max_output_tokens"] = 9999
    payload = _payload(context_participant_contracts=contracts)
    with pytest.raises(context_preflight.MalformedRequest):
        context_preflight.build_context_preflight_response(payload)


def test_missing_qa_contract_kind_is_malformed():
    contracts = _contracts(4096)
    del contracts["qa_evaluator"]
    payload = _payload(
        context_participant_contracts=contracts,
        qa_models=[{"model": "qa-model-a", "max_tokens": 2048}],
        qa_layers=[_qa_layer()],
    )
    with pytest.raises(context_preflight.MalformedRequest):
        context_preflight.build_context_preflight_response(payload)


def test_invalid_qa_overlay_is_malformed():
    payload = _payload(
        qa_models=["qa-model-a"],
        qa_global_config={"max_tokens": "not-an-integer"},
        qa_layers=[_qa_layer()],
    )
    with pytest.raises(context_preflight.MalformedRequest):
        context_preflight.build_context_preflight_response(payload)


def test_secondary_thinking_request_cannot_exceed_declared_maximum():
    contracts = _contracts(4_096)
    contracts["prompt_preflight"]["thinking_budget_tokens"] = 2_048
    with pytest.raises(context_preflight.MalformedRequest):
        context_preflight.build_context_preflight_response(
            _payload(context_participant_contracts=contracts)
        )


def test_builder_is_pure_and_deterministic():
    payload = _payload(
        source_text="w" * 10_000,
        qa_models=[{"model": "qa-model-a"}],
        qa_layers=[_qa_layer()],
    )
    first = context_preflight.build_context_preflight_response(payload)
    second = context_preflight.build_context_preflight_response(payload)
    assert first == second


# ---------------------------------------------------------------------------
# Route surface
# ---------------------------------------------------------------------------


def test_route_accepts_and_rejects_gracefully():
    import subprocess
    import sys as _sys

    # The full app import pulls in optional native dependencies (python-magic
    # via services/attachment_manager). Where that native import hard-crashes
    # the interpreter, skip instead of killing the whole pytest process.
    try:
        probe = subprocess.run(
            [_sys.executable, "-c", "from main import app"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            timeout=240,
        )
    except subprocess.TimeoutExpired:
        pytest.skip("full app import timed out in this environment")
    if probe.returncode != 0:
        pytest.skip(
            "full app import failed in this environment (see e.g. python-magic "
            "native crash); route surface unverifiable here"
        )

    from fastapi.testclient import TestClient

    from main import app

    client = TestClient(app)
    ok = client.post("/context/preflight", json=_payload())
    assert ok.status_code == 200
    assert ok.json()["schema_version"] == "gransabio_context_preflight.v2"
    assert ok.json()["accepted"] is True

    rejected = client.post("/context/preflight", json=_payload(model="not-in-catalog"))
    assert rejected.status_code == 400

    malformed = client.post("/context/preflight", json={"prompt": "x"})
    assert malformed.status_code == 400

    not_a_dict = client.post("/context/preflight", json=["not", "a", "dict"])
    assert not_a_dict.status_code == 400



@pytest.mark.parametrize("system_prompt", ["Fixed temporal instructions.", "", "   "])
@pytest.mark.parametrize("include_context", [True, False])
def test_context_preflight_preserves_exact_generator_system_prompt(
    system_prompt: str, include_context: bool
) -> None:
    payload = _payload(
        system_prompt=system_prompt,
        include_default_system_context=include_context,
    )
    specs = context_preflight._build_participant_specs(payload)
    generator = next(spec for spec in specs if spec.participant == "generator")
    counted_system = next(value for name, value, _hint in generator.fixed_components if name == "generator_system_prompt")
    if include_context:
        assert counted_system.startswith(system_prompt + "\n\nIMPORTANT: Always respond in the same language")
        assert "\n\nCurrent date:" in counted_system
    else:
        assert counted_system == system_prompt


@pytest.mark.parametrize("qa_active,fallback", [
    (False, False), (False, True), (True, False), (True, True),
])
def test_shared_context_is_counted_once_for_each_chat_participant(
    qa_active: bool, fallback: bool,
) -> None:
    source = 'Complete shared evidence.\nQuoted observation: "verified".\n' * 20
    payload = _payload(
        gran_sabio_fallback=fallback,
        source_text="A separate legacy source remains present.",
        qa_layers=[_qa_layer()] if qa_active else [],
    )
    baseline = context_preflight.build_context_preflight_response(payload)
    sourced = {**payload, "shared_context": source}
    response = context_preflight.build_context_preflight_response(sourced)
    specs = context_preflight._build_participant_specs(sourced)
    expected_kinds = {"generator", "prompt_preflight"}
    if qa_active:
        expected_kinds.update({"qa_evaluator", "arbiter", "gran_sabio_review"})
    if fallback:
        expected_kinds.update({
            "gran_sabio_review", "gran_sabio_regeneration", "gran_sabio_fallback",
        })
    assert {spec.participant_kind for spec in specs} == expected_kinds
    assert baseline["accepted"] is response["accepted"] is True
    baseline_by_name = {item["participant"]: item for item in baseline["participants"]}
    response_by_name = {item["participant"]: item for item in response["participants"]}
    for spec in specs:
        assert spec.call_kind == "chat"
        evidence = [item for item in spec.fixed_components if item[0] == "shared_context"]
        assert evidence == [("shared_context", source, "prose")]
        assert all(source not in text for name, text, _hint in spec.fixed_components
                   if name != "shared_context")
        before = baseline_by_name[spec.participant]
        after = response_by_name[spec.participant]
        capacity = context_preflight._resolve_capacity(spec.requested_model)
        source_tokens = context_preflight.context_budget.make_payload_component(
            "shared_context", source, "message", capacity.provider,
            capacity.effective_model, "prose", keep_text=False,
        ).tokens_estimate
        assert source_tokens > 0
        assert after["fixed_input_tokens"] - before["fixed_input_tokens"] == source_tokens
        assert before["evidence_tokens"] - after["evidence_tokens"] == source_tokens
    _assert_client_contract(response, sourced)


def test_shared_context_stays_inside_the_single_system_one_qa_payload() -> None:
    source = 'Full shared evidence.\nQuoted observation: "verified".\n' * 20
    payload = _payload(
        shared_context=source,
        source_text="Independent legacy reference data.",
        qa_models=["jev-latest", "qa-model-a"],
        qa_global_config={"max_tokens": 2048},
        qa_layers=[_qa_layer(), {
            **_qa_layer("Detailed fidelity"),
            "criteria": "Check the supplied reference in full. " * 20,
        }],
    )
    specs = context_preflight._build_participant_specs(payload)
    typed = next(spec for spec in specs if spec.participant == "qa:1")
    assert typed.call_kind == "system_one"
    assert len(typed.fixed_components) == 1
    name, encoded, hint = typed.fixed_components[0]
    assert name == "system_one_request"
    assert hint == "json"
    state = context_preflight.json.loads(encoded)["state"]
    assert state["shared_context"] == source
    assert state["source_text"] == payload["source_text"]
    assert state["original_prompt"] == payload["prompt"]
    response = context_preflight.build_context_preflight_response(payload)
    assert response["accepted"] is True
    reported = next(item for item in response["participants"]
                    if item["participant"] == typed.participant)
    expected_tokens = context_preflight.context_budget.make_payload_component(
        name, encoded, "message", "typesafe", "jev-latest", hint,
        keep_text=False,
    ).tokens_estimate
    assert reported["fixed_input_tokens"] == expected_tokens
    assert reported["output_tokens"] == 0
    _assert_client_contract(response, payload)


@pytest.mark.parametrize("shared_context", [None, ""])
def test_empty_shared_context_preserves_legacy_component_accounting(shared_context) -> None:
    baseline = _payload()
    payload = {**baseline, "shared_context": shared_context}
    specs = context_preflight._build_participant_specs(payload)
    baseline_specs = context_preflight._build_participant_specs(baseline)
    for actual, previous in zip(specs, baseline_specs, strict=True):
        assert all(name != "shared_context" for name, _text, _hint in actual.fixed_components)
        if actual.participant != "preflight:validator":
            assert actual == previous
    # Null/empty values are not transported as source data by existing runtime
    # calls; retain their prior metadata serialization rather than changing it.
    fields = context_preflight.json.loads(context_preflight._serialized_request_skeleton(payload))
    assert fields["shared_context"] == shared_context
    assert "shared_context" not in context_preflight.json.loads(
        context_preflight._serialized_request_skeleton(baseline)
    )


def test_oversized_shared_context_rejects_the_generator_budget() -> None:
    payload = _payload(model="tiny-model", max_tokens=1024)
    assert context_preflight.build_context_preflight_response(payload)["accepted"] is True
    payload["shared_context"] = "Complete reference data retained without truncation.\n" * 2000
    response = context_preflight.build_context_preflight_response(payload)
    assert response["accepted"] is False
    assert response["rejection"]["code"] == "prompt_too_large"
    assert response["rejection"]["participant"] == "generator"
    generator = next(item for item in response["participants"] if item["participant"] == "generator")
    assert generator["evidence_tokens"] < 0
