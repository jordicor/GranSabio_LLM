from provider_adapters.generic import GenericProviderAdapter


class OpenAIProviderAdapter(GenericProviderAdapter):
    def __init__(self) -> None:
        super().__init__("openai")
