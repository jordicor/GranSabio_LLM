"""Generic provider adapter used until provider-specific adapters need custom logic."""

from __future__ import annotations

from provider_adapters.base import ProviderCallContext
from provider_errors import ProviderFailure, classify_provider_exception


class GenericProviderAdapter:
    def __init__(self, provider: str) -> None:
        self.provider = provider

    def classify_exception(self, exc: BaseException, context: ProviderCallContext) -> ProviderFailure:
        return classify_provider_exception(
            exc,
            provider=context.provider or self.provider,
            model_id=context.model_id,
            operation=context.operation,
            attempted_feature=context.attempted_feature,
        )
