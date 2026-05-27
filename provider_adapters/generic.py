"""Generic provider adapter used until provider-specific adapters need custom logic."""

from __future__ import annotations

from provider_adapters.base import EffectiveOutputPlan, OutputPlanningContext, ProviderCallContext
from provider_errors import ProviderFailure, classify_provider_exception
from tool_loop_models import OutputContract


class GenericProviderAdapter:
    def __init__(self, provider: str) -> None:
        self.provider = provider

    def classify_exception(self, exc: BaseException, context: ProviderCallContext) -> ProviderFailure:
        return classify_provider_exception(
            exc,
            provider=context.provider or self.provider,
            model_id=context.model_id,
            operation=context.operation,
            attempt=context.attempt,
            max_attempts=context.max_attempts,
            attempted_feature=context.attempted_feature,
        )

    def plan_output_contract(self, context: OutputPlanningContext) -> EffectiveOutputPlan:
        """Plan the provider-native output mode without resolving capabilities itself."""

        desired = context.desired
        if desired.contract == OutputContract.FREE_TEXT:
            return EffectiveOutputPlan(
                desired_contract=desired.contract,
                effective_contract=OutputContract.FREE_TEXT,
                native_mode="none",
                schema=None,
                inject_json_instruction=False,
                local_validation_required=False,
            )

        if desired.contract == OutputContract.JSON_STRUCTURED and desired.schema:
            if context.supports_structured_outputs:
                return EffectiveOutputPlan(
                    desired_contract=desired.contract,
                    effective_contract=OutputContract.JSON_STRUCTURED,
                    native_mode="json_schema",
                    schema=context.prepared_schema or desired.schema,
                    inject_json_instruction=False,
                    local_validation_required=desired.local_validation_required,
                    attempted_feature=self._structured_output_feature(context),
                    capability_source="ai_service",
                )
            return EffectiveOutputPlan(
                desired_contract=desired.contract,
                effective_contract=OutputContract.JSON_LOOSE,
                native_mode="json_object" if context.supports_json_object else "prompt_json",
                schema=None,
                inject_json_instruction=True,
                local_validation_required=True,
                attempted_feature=self._json_object_feature(context),
                downgrade_reason="structured_outputs_not_supported",
                capability_source="ai_service",
            )

        return EffectiveOutputPlan(
            desired_contract=desired.contract,
            effective_contract=OutputContract.JSON_LOOSE,
            native_mode="json_object" if self._uses_native_json_object(context) else "prompt_json",
            schema=None,
            inject_json_instruction=True,
            local_validation_required=desired.local_validation_required,
            attempted_feature=self._json_object_feature(context),
            capability_source="ai_service",
        )

    def _structured_output_feature(self, context: OutputPlanningContext) -> str:
        provider = (context.provider or self.provider).lower()
        if provider == "openai":
            return (
                "text.format.json_schema"
                if context.uses_openai_responses_api
                else "response_format.json_schema"
            )
        if provider == "claude":
            return "output_format"
        if provider == "gemini":
            return "response_json_schema"
        return "response_format.json_schema"

    def _uses_native_json_object(self, context: OutputPlanningContext) -> bool:
        provider = (context.provider or self.provider).lower()
        if provider == "gemini":
            return True
        if provider == "openai" and context.uses_openai_responses_api:
            return False
        return context.supports_json_object or provider == "openai"

    def _json_object_feature(self, context: OutputPlanningContext) -> str | None:
        provider = (context.provider or self.provider).lower()
        if provider == "openai":
            if context.uses_openai_responses_api:
                return None
            return "response_format.json_object"
        if provider == "gemini":
            return "response_mime_type"
        if provider in {"openrouter", "xai", "ollama"} and context.supports_json_object:
            return "response_format.json_object"
        return None
