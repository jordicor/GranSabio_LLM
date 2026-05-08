from provider_adapters.generic import GenericProviderAdapter


class OpenRouterProviderAdapter(GenericProviderAdapter):
    def __init__(self) -> None:
        super().__init__("openrouter")
