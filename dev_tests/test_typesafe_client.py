"""TypeSafe System One client, AIService dispatch, and QA scorer path."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from ai_service import AIService, TypeSafeConfigurationError
from context_budget import ContextBudgetExceeded
from models import QAEvaluation
from qa_evaluation_service import QAEvaluationService
from services.typesafe_client import (
    TypeSafeConfigurationError as TypeSafeClientConfigurationError,
)
from services.typesafe_client import (
    TypeSafeEvaluation,
    TypeSafeRequestError,
    TypeSafeResponseError,
    _validate_usage,
    evaluate,
    system_one_wire_model,
)


@pytest.mark.parametrize("value", [True, 1.5, "120", None, -1])
def test_usage_requires_native_non_negative_integer_counts(value):
    with pytest.raises(TypeSafeResponseError):
        _validate_usage({"input_tokens": value, "output_tokens": 0})


def _choice_questions() -> dict[str, Any]:
    return {
        "decision": {
            "type": "choice",
            "instructions": "Choose one.",
            "criteria": {"proceed": "ok", "reject": "not ok"},
        }
    }


def _choice_payload(choice: str = "proceed") -> dict[str, Any]:
    options = ["proceed", "reject"]
    return {
        "model": "jev-1.13.0",
        "answers": {
            "decision": {
                "type": "choice",
                "choice": choice,
                "probabilities": {option: float(option == choice) for option in options},
                "confidence": 0.91,
            }
        },
        "usage": {"input_tokens": 120, "output_tokens": 0},
    }


@pytest.fixture
def ai_service_instance() -> AIService:
    with patch("ai_service.aiohttp.TCPConnector"):
        with patch("ai_service.aiohttp.ClientTimeout"):
            with patch("ai_service.openai"):
                with patch("ai_service.anthropic"):
                    with patch.object(AIService, "_initialize_clients"):
                        service = AIService()
                        service.openai_client = None
                        service.anthropic_client = None
                        service.google_new_client = None
                        service.xai_client = None
                        service.openrouter_client = None
                        service.ollama_client = None
                        return service


def test_wire_model_strips_provider_prefix() -> None:
    assert system_one_wire_model("typesafe/jev-latest") == "jev-latest"
    assert system_one_wire_model("typesafe::jev-latest") == "jev-latest"
    assert system_one_wire_model("jev-latest") == "jev-latest"


async def test_noul_with_invalid_criteria_fails_before_http() -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json={}))
    ) as http:
        with pytest.raises(TypeSafeClientConfigurationError, match="true and false"):
            await evaluate(
                {"prompt": "x"},
                {"flag": {"type": "noul", "instructions": "Is it true?", "criteria": {"yes": "yes"}}},
                model="jev-latest",
                api_key="test-key",
                client=http,
            )


async def test_happy_choice_uses_systemone_and_never_sends_chat_params() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://api.typesafe.ai/v1/systemone"
        assert request.headers["authorization"] == "Bearer test-key"
        payload = json.loads(request.content)
        seen.append(payload)
        return httpx.Response(200, json=_choice_payload())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        result = await evaluate(
            {"prompt": "Write a biography"},
            _choice_questions(),
            model="typesafe/jev-latest",
            api_key="test-key",
            client=http,
        )
    assert result.answers["decision"]["choice"] == "proceed"
    assert result.model == "jev-1.13.0"
    assert result.usage["input_tokens"] == 120
    assert seen[0]["model"] == "jev-latest"
    assert "temperature" not in seen[0]
    assert "max_tokens" not in seen[0]
    assert "tools" not in seen[0]


@pytest.mark.parametrize(
    "failure",
    ["missing_answer", "extra_answer", "unknown_choice", "nan_probability", "non_max_choice"],
)
async def test_bad_choice_distribution_fails_without_repair(failure: str) -> None:
    payload = _choice_payload("proceed")
    if failure == "missing_answer":
        payload["answers"] = {}
    elif failure == "extra_answer":
        payload["answers"]["unexpected"] = payload["answers"]["decision"]
    elif failure == "unknown_choice":
        payload["answers"]["decision"]["choice"] = "maybe"
    elif failure == "nan_probability":
        payload["answers"]["decision"]["probabilities"]["proceed"] = "NaN"
    else:
        payload["answers"]["decision"]["probabilities"] = {
            "proceed": 0.36,
            "reject": 0.64,
        }
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(TypeSafeResponseError):
            await evaluate(
                {"prompt": "x"},
                _choice_questions(),
                model="jev-latest",
                api_key="test-key",
                client=http,
            )
    assert calls == 1


async def test_http_500_fails_without_body_echo() -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(500, text="sensitive-provider-body")
        )
    ) as http:
        with pytest.raises(TypeSafeRequestError, match="TypeSafe HTTP 500") as caught:
            await evaluate(
                {"prompt": "x"},
                _choice_questions(),
                model="jev-latest",
                api_key="test-key",
                client=http,
            )
    assert "sensitive-provider-body" not in str(caught.value)


async def test_noul_and_score_answers_are_validated() -> None:
    questions = {
        "is_length": {
            "type": "noul",
            "instructions": "Is this a length layer?",
            "criteria": {
                "true": "It judges length or size",
                "false": "It judges something else",
            },
        },
        "quality": {
            "type": "score",
            "instructions": "Rate quality from the lowest to the highest level",
            "criteria": [str(i) for i in range(6)],
        },
    }
    payload = {
        "model": "jev-1.13.0",
        "answers": {
            "is_length": {"type": "noul", "noul": 0.12},
            "quality": {
                "type": "score",
                "score": 4.2,
                "legend": {str(i): str(i) for i in range(6)},
                "probabilities": {"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.0, "4": 0.8, "5": 0.2},
                "confidence": 0.7,
            },
        },
        "usage": {"input_tokens": 40, "output_tokens": 0},
    }

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        result = await evaluate(
            {"content": "draft"},
            questions,
            model="jev-latest",
            api_key="test-key",
            client=http,
        )
    assert result.answers["is_length"]["noul"] == 0.12
    assert result.answers["quality"]["score"] == 4.2


async def test_missing_score_legend_fails() -> None:
    questions = {
        "quality": {
            "type": "score",
            "instructions": "Rate quality from the lowest to the highest level",
            "criteria": [str(i) for i in range(6)],
        }
    }
    payload = {
        "model": "jev-1.13.0",
        "answers": {
            "quality": {
                "type": "score",
                "score": 4.0,
                "probabilities": {str(i): float(i == 4) for i in range(6)},
                "confidence": 0.5,
            }
        },
        "usage": {"input_tokens": 1, "output_tokens": 0},
    }

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload))
    ) as http:
        with pytest.raises(TypeSafeResponseError, match="legend"):
            await evaluate(
                {"content": "draft"},
                questions,
                model="jev-latest",
                api_key="test-key",
                client=http,
            )


async def test_generate_content_rejects_typesafe_model(ai_service_instance: AIService) -> None:
    with patch(
        "ai_service.resolve_model_catalog_entry",
        return_value={
            "matched": True,
            "enabled": True,
            "catalog_provider": "typesafe",
            "provider": "typesafe",
            "model_id": "jev-latest",
        },
    ):
        with pytest.raises(TypeSafeConfigurationError, match="does not generate text"):
            await ai_service_instance.generate_content("hello", "jev-latest")


async def test_generate_content_stream_rejects_typesafe_model(
    ai_service_instance: AIService,
) -> None:
    with patch(
        "ai_service.resolve_model_catalog_entry",
        return_value={
            "matched": True,
            "enabled": True,
            "catalog_provider": "typesafe",
            "provider": "typesafe",
            "model_id": "jev-latest",
        },
    ):
        with pytest.raises(TypeSafeConfigurationError, match="does not generate text"):
            async for _ in ai_service_instance.generate_content_stream("hello", "jev-latest"):
                pass


async def test_evaluate_system_one_emits_usage(
    ai_service_instance: AIService, monkeypatch: pytest.MonkeyPatch
) -> None:
    import config as config_module

    monkeypatch.setattr(
        config_module.Config,
        "get_model_info",
        lambda _self, _name, **_kwargs: {
            "provider": "typesafe",
            "model_id": "jev-latest",
            "api_key": "",
        },
    )
    seen_usage: list[dict[str, Any]] = []

    def handler(_request: httpx.Request) -> httpx.Response:
        assert _request.headers["authorization"] == "Bearer scoped-key"
        assert _request.extensions["timeout"]["read"] == 17.0
        return httpx.Response(200, json=_choice_payload())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        result = await ai_service_instance.evaluate_system_one(
            {"prompt": "x"},
            _choice_questions(),
            "jev-latest",
            usage_callback=seen_usage.append,
            client=http,
            api_key="scoped-key",
            request_timeout=17.0,
        )
    assert result.answers["decision"]["choice"] == "proceed"
    assert seen_usage[0]["provider"] == "typesafe"
    assert seen_usage[0]["input_tokens"] == 120


@pytest.mark.parametrize("raw_score, expected_score", [(0.0, 0.0), (4.1, 8.2), (5.0, 10.0)])
async def test_qa_system_one_scores_without_edit_groups(
    monkeypatch: pytest.MonkeyPatch,
    raw_score: float,
    expected_score: float,
) -> None:
    import config as config_module

    service = QAEvaluationService(ai_service=AsyncMock())
    service.ai_service.evaluate_system_one = AsyncMock(
        return_value=TypeSafeEvaluation(
            model="jev-1.13.0",
            answers={
                "quality_score": {
                    "type": "score",
                    "score": raw_score,
                    "legend": {str(i): str(i) for i in range(6)},
                    "confidence": 0.6,
                },
                "deal_breaker": {"type": "noul", "noul": 0.1},
            },
            usage={"input_tokens": 20, "output_tokens": 0},
        )
    )
    monkeypatch.setattr(
        config_module.Config,
        "get_model_info",
        lambda _self, _name, **_kwargs: {"provider": "typesafe", "model_id": "jev-latest"},
    )
    result = await service.evaluate_content(
        content="draft text",
        criteria="Be accurate",
        model="jev-latest",
        layer_name="Accuracy",
        min_score=7.0,
        deal_breaker_criteria="invents facts",
        original_request=SimpleNamespace(
            prompt="Use only the supplied source.",
            source_text="The researcher was born in Madrid in 1984.",
            content_type="biography",
        ),
    )
    assert isinstance(result, QAEvaluation)
    assert result.score == pytest.approx(expected_score)
    assert result.feedback == "scored_by_jev"
    assert result.deal_breaker is False
    assert result.passes_score is (expected_score >= 7.0)
    assert result.identified_issues is None
    service.ai_service.generate_content.assert_not_called()
    state, questions, _ = service.ai_service.evaluate_system_one.await_args.args
    assert state["original_prompt"] == "Use only the supplied source."
    assert state["source_text"] == "The researcher was born in Madrid in 1984."
    assert 2 <= len(questions["quality_score"]["criteria"]) <= 10


async def test_score_with_eleven_levels_fails_before_http() -> None:
    http = AsyncMock(spec=httpx.AsyncClient)
    with pytest.raises(TypeSafeClientConfigurationError, match="2 to 10"):
        await evaluate(
            "draft",
            {"quality": {"type": "score", "instructions": "Rate quality", "criteria": list(map(str, range(11)))}},
            model="jev-latest", api_key="test-key", client=http,
        )
    http.post.assert_not_called()


@pytest.mark.parametrize("oversized_part", ["state", "questions"])
async def test_system_one_admits_complete_payload_before_http(
    ai_service_instance: AIService, monkeypatch: pytest.MonkeyPatch, oversized_part: str,
) -> None:
    from config import Config

    monkeypatch.setenv("CONTEXT_ADMISSION_MODE", "enforce")
    monkeypatch.setattr(Config, "get_model_info", lambda _self, _model, **_kwargs: {
        "provider": "typesafe", "model_id": "jev-latest", "api_key": "test-key",
    })
    state = "token " * 60000 if oversized_part == "state" else "short state"
    questions = _choice_questions()
    if oversized_part == "questions":
        questions["decision"]["instructions"] = "token " * 60000
    http = AsyncMock(spec=httpx.AsyncClient)
    with pytest.raises(ContextBudgetExceeded):
        await ai_service_instance.evaluate_system_one(state, questions, "jev-latest", client=http)
    http.post.assert_not_called()


async def test_system_one_missing_key_is_a_configuration_error(
    ai_service_instance: AIService, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from config import Config

    monkeypatch.setattr(Config, "get_model_info", lambda _self, _model, **_kwargs: {
        "provider": "typesafe", "model_id": "jev-latest", "api_key": "",
    })
    http = AsyncMock(spec=httpx.AsyncClient)
    with pytest.raises(TypeSafeConfigurationError, match="TYPESAFE_API_KEY"):
        await ai_service_instance.evaluate_system_one("state", _choice_questions(), "jev-latest", client=http)
    http.post.assert_not_called()


async def test_system_one_cancelled_request_never_reaches_http(ai_service_instance: AIService) -> None:
    token = SimpleNamespace(any_cancelled=AsyncMock(return_value=True))
    from config import Config

    http = AsyncMock(spec=httpx.AsyncClient)
    with patch.object(Config, "get_model_info", return_value={
        "provider": "typesafe", "model_id": "jev-latest", "api_key": "test-key",
    }):
        import asyncio

        with pytest.raises(asyncio.CancelledError):
            await ai_service_instance.evaluate_system_one(
                "state", _choice_questions(), "jev-latest", client=http, cancellation_token=token,
            )
    http.post.assert_not_called()


async def test_noul_without_optional_criteria_is_supported() -> None:
    payload = {
        "model": "jev-1.13.0",
        "answers": {"flag": {"type": "noul", "noul": 0.9}},
        "usage": {"input_tokens": 10, "output_tokens": 0},
    }
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload)),
    ) as http:
        result = await evaluate(
            "state", {"flag": {"type": "noul", "instructions": "Is it true?"}},
            model="jev-latest", api_key="test-key", client=http,
        )
    assert result.answers["flag"]["noul"] == 0.9


async def test_qa_fanout_admits_the_system_one_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    from config import Config
    from models import QALayer
    from qa_engine import QAEngine
    from qa_scheduler import QASchedulerSlot

    monkeypatch.setattr(Config, "get_model_info", lambda _self, _model, **_kwargs: {
        "provider": "typesafe", "model_id": "jev-latest", "api_key": "test-key",
    })
    admission = AsyncMock(return_value=[])
    monkeypatch.setattr("context_budget.evaluate_fanout_admission", admission)
    engine = QAEngine(ai_service=AsyncMock())
    original = SimpleNamespace(prompt="Use only this source.", source_text="Born in 1984.")
    await engine._admit_qa_fanout(
        content="Born in 1984.",
        layer=QALayer(name="Accuracy", description="Source fidelity", criteria="Match the source."),
        slots=[QASchedulerSlot(0, "jev-latest", "jev-latest", "jev-latest", "Jev", 30)],
        original_request=original, request_edit_info=False, marker_mode="phrase",
        marker_length=None, word_map_formatted=None, draft_map_formatted=None,
        input_images=None, edit_history=None, model_alias_registry=None,
    )
    budget = admission.await_args.args[0][0]
    assert budget.reserved_output_tokens == 0
    sent = json.loads(budget.components[0].text)
    assert sent["state"]["source_text"] == original.source_text
    assert sent["state"]["original_prompt"] == original.prompt
    assert len(sent["questions"]["quality_score"]["criteria"]) <= 10
