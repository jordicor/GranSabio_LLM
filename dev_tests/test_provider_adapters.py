from __future__ import annotations

from provider_adapters import ProviderCallContext, get_provider_adapter
from provider_errors import ProviderErrorKind
from tool_loop_models import OutputContract


class ProviderStatusError(Exception):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


def test_adapter_classification_preserves_attempt_counts():
    adapter = get_provider_adapter("gemini")
    failure = adapter.classify_exception(
        ProviderStatusError(503, "service unavailable"),
        ProviderCallContext(
            provider="gemini",
            model_id="gemini-test",
            operation="generation",
            attempt=2,
            max_attempts=4,
        ),
    )

    assert failure.kind == ProviderErrorKind.PROVIDER_DOWN
    assert failure.attempt == 2
    assert failure.max_attempts == 4


def test_ai_service_output_plan_uses_patchable_capability_wrappers(monkeypatch):
    from ai_service import AIService

    monkeypatch.setattr(
        AIService,
        "_supports_structured_outputs",
        staticmethod(lambda provider, model_id: False),
    )
    monkeypatch.setattr(
        AIService,
        "_supports_json_object",
        staticmethod(lambda provider, model_id: True),
    )
    monkeypatch.setattr(
        AIService,
        "_is_openai_responses_api_model",
        staticmethod(lambda model_id: False),
    )

    plan = AIService._plan_output_contract(
        "openai",
        "patched-model",
        json_output=True,
        json_schema={"type": "object", "properties": {}},
    )

    assert plan.desired_contract == OutputContract.JSON_STRUCTURED
    assert plan.effective_contract == OutputContract.JSON_LOOSE
    assert plan.schema is None
    assert plan.inject_json_instruction is True
    assert plan.local_validation_required is True
    assert plan.attempted_feature == "response_format.json_object"


def test_ai_service_output_plan_preserves_openai_responses_feature(monkeypatch):
    from ai_service import AIService

    monkeypatch.setattr(
        AIService,
        "_supports_structured_outputs",
        staticmethod(lambda provider, model_id: True),
    )
    monkeypatch.setattr(
        AIService,
        "_supports_json_object",
        staticmethod(lambda provider, model_id: False),
    )
    monkeypatch.setattr(
        AIService,
        "_is_openai_responses_api_model",
        staticmethod(lambda model_id: True),
    )
    monkeypatch.setattr(
        AIService,
        "_prepare_structured_output_schema",
        staticmethod(lambda provider, model_id, schema: {"prepared": True}),
    )

    plan = AIService._plan_output_contract(
        "openai",
        "gpt-5-pro",
        json_output=True,
        json_schema={"type": "object", "properties": {}},
    )

    assert plan.effective_contract == OutputContract.JSON_STRUCTURED
    assert plan.schema == {"prepared": True}
    assert plan.inject_json_instruction is False
    assert plan.attempted_feature == "text.format.json_schema"
