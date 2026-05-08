from provider_adapters.generic import GenericProviderAdapter


class AnthropicProviderAdapter(GenericProviderAdapter):
    def __init__(self) -> None:
        super().__init__("claude")
