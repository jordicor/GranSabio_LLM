from provider_adapters.generic import GenericProviderAdapter


class XAIProviderAdapter(GenericProviderAdapter):
    def __init__(self) -> None:
        super().__init__("xai")
