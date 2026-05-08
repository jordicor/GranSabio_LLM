from qa_evaluation_service import QAEvaluationService


class _ProviderFailure:
    def __init__(self, kind: str) -> None:
        self.kind = kind


class _ProviderFailureError(Exception):
    def __init__(self, kind: str) -> None:
        super().__init__("provider rejected request")
        self.provider_failure = _ProviderFailure(kind)


class _DirectProviderFailure(Exception):
    def __init__(self, kind: str) -> None:
        super().__init__("provider failure")
        self.kind = kind


def test_structured_output_fallback_detects_provider_failure_kind_directly():
    assert QAEvaluationService._is_structured_output_schema_error(
        _ProviderFailureError("unsupported_parameter")
    )


def test_structured_output_fallback_detects_provider_failure_kind_through_cause():
    wrapper = RuntimeError("qa provider failure")
    wrapper.__cause__ = _ProviderFailureError("schema_invalid")

    assert QAEvaluationService._is_structured_output_schema_error(wrapper)


def test_structured_output_fallback_detects_direct_provider_failure_cause():
    wrapper = RuntimeError("qa provider failure")
    wrapper.__cause__ = _DirectProviderFailure("schema_invalid")

    assert QAEvaluationService._is_structured_output_schema_error(wrapper)


def test_structured_output_fallback_keeps_legacy_marker_compatibility():
    assert QAEvaluationService._is_structured_output_schema_error(
        RuntimeError("response_format rejected")
    )
