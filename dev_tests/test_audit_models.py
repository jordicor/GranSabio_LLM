"""Security regression tests for the models/schema package.

Each test reproduces the scenario of its finding and would have failed
before the fix:

- PER-01: combinatorial/depth DoS via user-supplied ``json_schema``.
- API-02: free-text fields and payload lists without size caps.
- API-09: client-controlled phase timeouts without an upper ceiling.
- GEN-10: ``max_iterations`` accepted up to 999 (session memory retention).
- PER-03: ``PhraseFrequencyRule`` accepted without any repetition limit when
  fields fell back to defaults.
- PER-04: ``normalize_qa_models_config`` injected unvalidated values via
  ``setattr`` and allowed renaming the routing model.
- PER-07: Gran Sabio escalation limits defaulted to unlimited (-1).
"""

import pytest
from pydantic import ValidationError

from ai_runtime import schemas as runtime_schemas
from config import config
from models import (
    MAX_CONTEXT_DOCUMENT_REFS,
    MAX_CUMULATIVE_TEXT_CHARS,
    MAX_IMAGE_REFS,
    MAX_ITERATIONS_EFFECTIVE_CAP,
    MAX_PROMPT_CHARS,
    MAX_QA_LAYERS,
    MAX_SOURCE_TEXT_CHARS,
    MAX_SYSTEM_PROMPT_CHARS,
    ContentRequest,
    ContextDocumentRef,
    ImageRef,
    PhraseFrequencyConfig,
    PhraseFrequencyRule,
    QALayer,
    QAModelConfig,
    normalize_qa_models_config,
)
from request_timeouts import MAX_CLIENT_TIMEOUT_SECONDS, resolve_request_timeout
from schema_utils import (
    JSON_SCHEMA_MAX_DEPTH,
    json_schema_to_pydantic,
    validate_json_schema_complexity,
)

VALID_PROMPT = "Write a detailed report about renewable energy"


def _nested_combinator_schema(options: int, levels: int) -> dict:
    """Build a schema with ``options`` oneOf variants nested ``levels`` deep."""

    schema = {"type": "string"}
    for _ in range(levels):
        schema = {"oneOf": [dict(schema) for _ in range(options)]}
    return schema


def _deep_object_schema(depth: int) -> dict:
    """Build a singly-nested object schema ``depth`` levels deep."""

    schema = {"type": "string"}
    for level in range(depth):
        schema = {
            "type": "object",
            "properties": {f"level_{level}": schema},
        }
    return schema


# ---------------------------------------------------------------------------
# PER-01: DoS via combinatorial expansion in a user-controlled json_schema
# ---------------------------------------------------------------------------


def test_schema_complexity_accepts_legitimate_schema():
    """Legitimate structured-output schemas keep passing the pre-pass."""

    schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "score": {"type": ["number", "null"]},
            "tags": {"type": "array", "items": {"type": "string"}},
            "author": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
            },
            "status": {"oneOf": [{"const": "draft"}, {"const": "final"}]},
        },
        "required": ["title"],
    }

    validate_json_schema_complexity(schema)
    model = json_schema_to_pydantic(schema)
    assert model is not None


def test_schema_complexity_rejects_combinatorial_explosion():
    """20 oneOf options nested 8 levels deep is rejected (audit scenario)."""

    with pytest.raises(ValueError, match="combinatorial"):
        validate_json_schema_complexity(_nested_combinator_schema(20, 8))


def test_schema_complexity_rejects_excessive_depth():
    """Nesting beyond the depth ceiling raises ValueError, not RecursionError."""

    with pytest.raises(ValueError, match="depth"):
        validate_json_schema_complexity(_deep_object_schema(JSON_SCHEMA_MAX_DEPTH + 5))


def test_schema_complexity_rejects_excessive_node_count():
    """Schemas with too many nodes are rejected before conversion."""

    schema = {
        "type": "object",
        "properties": {f"prop_{i}": {"type": "string"} for i in range(2000)},
    }
    with pytest.raises(ValueError, match="nodes"):
        validate_json_schema_complexity(schema)


def test_schema_complexity_pre_pass_is_iterative():
    """A schema thousands of levels deep fails with ValueError, never RecursionError."""

    with pytest.raises(ValueError, match="depth"):
        validate_json_schema_complexity(_deep_object_schema(5000))


def test_content_request_rejects_dangerous_json_schema():
    """The admission validator rejects the schema before any pipeline work.

    Note: no ``match=`` here on purpose — str() of a ValidationError reprs the
    input value, and the shared-reference attack structure would expand
    exponentially (an artifact impossible over real JSON bodies).
    """

    with pytest.raises(ValidationError):
        ContentRequest(
            prompt=VALID_PROMPT,
            json_output=True,
            json_schema=_nested_combinator_schema(20, 8),
        )


def test_content_request_accepts_legitimate_json_schema():
    request = ContentRequest(
        prompt=VALID_PROMPT,
        json_output=True,
        json_schema={
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        },
    )
    assert request.json_schema is not None


def test_json_schema_to_pydantic_rejects_combinatorial_schema():
    """The converter itself refuses the dangerous schema instead of hanging."""

    with pytest.raises(ValueError, match="combinatorial"):
        json_schema_to_pydantic(_nested_combinator_schema(20, 8))


@pytest.mark.parametrize("provider", ["gemini", "openai", "claude"])
def test_prepare_structured_output_schema_rejects_dangerous_schema(provider):
    """All provider branches fail fast with ValueError (400-class), not a 500."""

    with pytest.raises(ValueError):
        runtime_schemas.prepare_structured_output_schema(
            provider,
            "fake-model",
            _nested_combinator_schema(20, 8),
            claude_structured_outputs_supported=True,
            json_schema_to_pydantic_fn=json_schema_to_pydantic,
        )


def test_prepare_structured_output_schema_recursion_error_residual(monkeypatch):
    """A residual RecursionError surfaces as ValueError, not as a 500."""

    def _raise_recursion_error(schema):
        raise RecursionError("maximum recursion depth exceeded")

    monkeypatch.setattr(
        runtime_schemas, "normalize_openai_strict_schema", _raise_recursion_error
    )
    with pytest.raises(ValueError, match="recursion"):
        runtime_schemas.prepare_structured_output_schema(
            "openai",
            "fake-model",
            {"type": "object", "properties": {"a": {"type": "string"}}},
            claude_structured_outputs_supported=False,
            json_schema_to_pydantic_fn=json_schema_to_pydantic,
        )


# ---------------------------------------------------------------------------
# API-02: max_length on free-text fields and max items on request lists
# ---------------------------------------------------------------------------


def test_prompt_accepts_boundary_length():
    request = ContentRequest(prompt="x" * MAX_PROMPT_CHARS)
    assert len(request.prompt) == MAX_PROMPT_CHARS


def test_prompt_rejects_excessive_length():
    with pytest.raises(ValidationError):
        ContentRequest(prompt="x" * (MAX_PROMPT_CHARS + 1))


def test_system_prompt_rejects_excessive_length():
    with pytest.raises(ValidationError):
        ContentRequest(
            prompt=VALID_PROMPT,
            system_prompt="x" * (MAX_SYSTEM_PROMPT_CHARS + 1),
        )


def test_source_text_rejects_excessive_length():
    with pytest.raises(ValidationError):
        ContentRequest(
            prompt=VALID_PROMPT,
            source_text="x" * (MAX_SOURCE_TEXT_CHARS + 1),
        )


def test_cumulative_text_rejects_excessive_length():
    with pytest.raises(ValidationError):
        ContentRequest(
            prompt=VALID_PROMPT,
            cumulative_text="x" * (MAX_CUMULATIVE_TEXT_CHARS + 1),
        )


def test_context_documents_rejects_excessive_items():
    refs = [
        ContextDocumentRef(upload_id="upload1234", username="user1")
        for _ in range(MAX_CONTEXT_DOCUMENT_REFS + 1)
    ]
    with pytest.raises(ValidationError):
        ContentRequest(prompt=VALID_PROMPT, context_documents=refs)


def test_images_rejects_excessive_items():
    refs = [
        ImageRef(upload_id="upload1234", username="user1")
        for _ in range(MAX_IMAGE_REFS + 1)
    ]
    with pytest.raises(ValidationError):
        ContentRequest(prompt=VALID_PROMPT, images=refs)


def test_qa_layers_rejects_excessive_items():
    layers = [
        QALayer(name=f"Layer {i}", description="d", criteria="c")
        for i in range(MAX_QA_LAYERS + 1)
    ]
    with pytest.raises(ValidationError):
        ContentRequest(prompt=VALID_PROMPT, qa_layers=layers)


# ---------------------------------------------------------------------------
# API-09: upper ceiling on client-controlled timeouts
# ---------------------------------------------------------------------------


def test_resolve_request_timeout_clamps_phase_field():
    request = ContentRequest(
        prompt=VALID_PROMPT,
        timeouts={"generation_seconds": 99_999_999},
    )
    resolved = resolve_request_timeout(request, "generation_seconds", fallback=1)
    assert resolved == MAX_CLIENT_TIMEOUT_SECONDS


def test_resolve_request_timeout_clamps_generic_timeout_seconds():
    request = ContentRequest(prompt=VALID_PROMPT, timeout_seconds=99_999_999)
    resolved = resolve_request_timeout(request, "qa_model_seconds", fallback=1)
    assert resolved == MAX_CLIENT_TIMEOUT_SECONDS


def test_resolve_request_timeout_preserves_values_below_ceiling():
    request = ContentRequest(
        prompt=VALID_PROMPT,
        timeouts={"generation_seconds": 18000},
    )
    resolved = resolve_request_timeout(request, "generation_seconds", fallback=1)
    assert resolved == 18000


def test_resolve_request_timeout_does_not_clamp_operator_config():
    """Operator-owned config values are not clamped (only client input is)."""

    settings = {"process_timeouts": {"generation_seconds": 100_000}}
    request = ContentRequest(prompt=VALID_PROMPT)
    resolved = resolve_request_timeout(
        request,
        "generation_seconds",
        settings=settings,
        config_path=("process_timeouts", "generation_seconds"),
        fallback=1,
    )
    assert resolved == 100_000


def test_qa_model_config_timeout_is_clamped():
    """The QA engine reads QAModelConfig.timeout_seconds directly; clamp at admission."""

    model = QAModelConfig(model="qa-model", timeout_seconds=99_999_999)
    assert model.timeout_seconds == MAX_CLIENT_TIMEOUT_SECONDS


# ---------------------------------------------------------------------------
# GEN-10: effective upper bound on max_iterations
# ---------------------------------------------------------------------------


def test_max_iterations_clamped_to_productive_ceiling():
    request = ContentRequest(prompt=VALID_PROMPT, max_iterations=999)
    assert request.max_iterations == MAX_ITERATIONS_EFFECTIVE_CAP


def test_max_iterations_below_ceiling_unchanged():
    request = ContentRequest(prompt=VALID_PROMPT, max_iterations=10)
    assert request.max_iterations == 10


# ---------------------------------------------------------------------------
# PER-03: the PhraseFrequencyRule validator also runs when fields fall back to defaults
# ---------------------------------------------------------------------------


def test_phrase_frequency_rule_requires_a_limit_even_with_defaults():
    """Omitting both limits now fails validation (previously accepted silently)."""

    with pytest.raises(
        ValidationError, match="Either max_ratio_tokens or max_repetitions"
    ):
        PhraseFrequencyRule(name="generic", min_length=3)


def test_phrase_frequency_rule_accepts_explicit_limits():
    assert PhraseFrequencyRule(name="a", max_repetitions=3).max_repetitions == 3
    assert PhraseFrequencyRule(name="b", max_ratio_tokens=0.1).max_ratio_tokens == 0.1


def test_phrase_frequency_rule_range_checked_against_default_min_length():
    """max_length=1 is invalid against the default min_length=2 (same default blind spot)."""

    with pytest.raises(ValidationError, match="max_length must be greater"):
        PhraseFrequencyRule(name="bad-range", max_length=1, max_repetitions=2)


def test_phrase_frequency_config_rejects_rule_without_limit():
    """The audit scenario: enabled guard with a limit-less rule fails fast."""

    with pytest.raises(ValidationError):
        PhraseFrequencyConfig(
            enabled=True,
            rules=[{"name": "generic", "min_length": 3}],
        )


# ---------------------------------------------------------------------------
# PER-04: normalize_qa_models_config re-validates the merge and restricts keys
# ---------------------------------------------------------------------------


def test_normalize_qa_models_config_validates_value_types():
    """Garbage values now fail at normalization instead of deep in the pipeline."""

    with pytest.raises(ValidationError):
        normalize_qa_models_config(
            qa_models=["qa-model"],
            qa_global_config={"max_tokens": "abc"},
        )


def test_normalize_qa_models_config_validates_ranges():
    with pytest.raises(ValidationError):
        normalize_qa_models_config(
            qa_models=["qa-model"],
            qa_models_config={"qa-model": {"temperature": 3.5}},
        )


def test_normalize_qa_models_config_cannot_rename_model():
    """The routing identity is no longer configurable from global config."""

    normalized = normalize_qa_models_config(
        qa_models=["qa-model"],
        qa_global_config={"model": "other-model"},
    )
    assert normalized[0].model == "qa-model"


def test_normalize_qa_models_config_ignores_unknown_keys():
    normalized = normalize_qa_models_config(
        qa_models=["qa-model"],
        qa_global_config={"unknown_key": 123},
    )
    assert not hasattr(normalized[0], "unknown_key")


def test_normalize_qa_models_config_preserves_precedence():
    """Explicit fields win over per-model config, which wins over global config."""

    normalized = normalize_qa_models_config(
        qa_models=["model-a", QAModelConfig(model="model-b", timeout_seconds=3333)],
        qa_global_config={"timeout_seconds": 1111, "temperature": 0.9},
        qa_models_config={"model-a": {"timeout_seconds": 2222}},
    )
    assert normalized[0].timeout_seconds == 2222
    assert normalized[0].temperature == 0.9
    assert normalized[1].timeout_seconds == 3333


# ---------------------------------------------------------------------------
# PER-07: Gran Sabio escalation limits are finite by default
# ---------------------------------------------------------------------------


def test_gran_sabio_limits_default_to_finite_config_values():
    """Defaults are finite (config knobs) instead of unlimited -1."""

    request = ContentRequest(prompt=VALID_PROMPT)
    assert request.gran_sabio_call_limit_per_iteration == config.DEFAULT_GRAN_SABIO_LIMIT_PER_ITERATION
    assert request.gran_sabio_call_limit_per_session == config.DEFAULT_GRAN_SABIO_LIMIT_PER_SESSION
    assert request.gran_sabio_call_limit_per_iteration > 0
    assert request.gran_sabio_call_limit_per_session > 0


def test_gran_sabio_limits_unlimited_remains_explicit_opt_in():
    request = ContentRequest(
        prompt=VALID_PROMPT,
        gran_sabio_call_limit_per_iteration=-1,
        gran_sabio_call_limit_per_session=-1,
    )
    assert request.gran_sabio_call_limit_per_iteration == -1
    assert request.gran_sabio_call_limit_per_session == -1
