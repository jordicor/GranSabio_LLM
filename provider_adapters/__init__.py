"""Provider adapter package for capability/error hardening."""

from __future__ import annotations

from provider_adapters.anthropic import AnthropicProviderAdapter
from provider_adapters.base import (
    DesiredOutputContract,
    EffectiveOutputPlan,
    OutputPlanningContext,
    ProviderAdapter,
    ProviderCallContext,
)
from provider_adapters.gemini import GeminiProviderAdapter
from provider_adapters.generic import GenericProviderAdapter
from provider_adapters.ollama import OllamaProviderAdapter
from provider_adapters.openai import OpenAIProviderAdapter
from provider_adapters.openrouter import OpenRouterProviderAdapter
from provider_adapters.xai import XAIProviderAdapter

_ADAPTERS: dict[str, ProviderAdapter] = {
    "openai": OpenAIProviderAdapter(),
    "claude": AnthropicProviderAdapter(),
    "anthropic": AnthropicProviderAdapter(),
    "gemini": GeminiProviderAdapter(),
    "google": GeminiProviderAdapter(),
    "openrouter": OpenRouterProviderAdapter(),
    "xai": XAIProviderAdapter(),
    "ollama": OllamaProviderAdapter(),
}


def get_provider_adapter(provider: str) -> ProviderAdapter:
    """Return a provider adapter, falling back to generic behavior."""

    key = (provider or "").strip().lower()
    return _ADAPTERS.get(key) or GenericProviderAdapter(key or "unknown")


__all__ = [
    "DesiredOutputContract",
    "EffectiveOutputPlan",
    "GenericProviderAdapter",
    "OutputPlanningContext",
    "ProviderAdapter",
    "ProviderCallContext",
    "get_provider_adapter",
]
