from provider_adapters.generic import GenericProviderAdapter


class GeminiProviderAdapter(GenericProviderAdapter):
    def __init__(self) -> None:
        super().__init__("gemini")
