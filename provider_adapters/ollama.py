from provider_adapters.generic import GenericProviderAdapter


class OllamaProviderAdapter(GenericProviderAdapter):
    def __init__(self) -> None:
        super().__init__("ollama")
