import logging
from types import SimpleNamespace

from deterministic_validation import evaluate_phrase_frequency_check, has_active_generation_validators
from models import PhraseFrequencyConfig, PhraseFrequencyRule
from validation_context_factory import build_measurement_request_for_layer


def test_phrase_frequency_check_normalizes_enabled_config_with_empty_rules_via_model_construct(caplog):
    config = PhraseFrequencyConfig.model_construct(enabled=True, rules=[])
    request = SimpleNamespace(phrase_frequency=config)

    with caplog.at_level(logging.WARNING):
        result = evaluate_phrase_frequency_check("dummy text", request)

    assert result is None
    assert config.enabled is False
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "setting enabled=False" in messages
    assert "Stack:" not in messages


def test_phrase_frequency_check_normalizes_rules_cleared_after_construction(caplog):
    config = PhraseFrequencyConfig(
        enabled=True,
        rules=[
            PhraseFrequencyRule(
                name="repeat",
                min_length=2,
                max_repetitions=2,
            )
        ],
    )
    config.rules = []
    request = SimpleNamespace(phrase_frequency=config)

    with caplog.at_level(logging.WARNING):
        result = evaluate_phrase_frequency_check("dummy text", request)

    assert result is None
    assert config.enabled is False
    assert "setting enabled=False" in caplog.text


def test_phrase_frequency_config_direct_construction_with_empty_rules_is_disabled(caplog):
    with caplog.at_level(logging.WARNING):
        config = PhraseFrequencyConfig(enabled=True, rules=[])

    assert config.enabled is False
    assert "setting enabled=False" in caplog.text


def test_empty_phrase_frequency_rules_do_not_activate_generation_validators():
    config = PhraseFrequencyConfig.model_construct(enabled=True, rules=[])
    request = SimpleNamespace(phrase_frequency=config)

    assert has_active_generation_validators(request) is False
    assert config.enabled is False


def test_empty_phrase_frequency_rules_do_not_build_measurement_request():
    config = PhraseFrequencyConfig.model_construct(enabled=True, rules=[])
    request = SimpleNamespace(
        min_words=None,
        max_words=None,
        word_count_enforcement=None,
        phrase_frequency=config,
        lexical_diversity=None,
        json_output=False,
        json_schema=None,
        target_field=None,
    )

    result = build_measurement_request_for_layer(request, SimpleNamespace(name="Originality"))

    assert result is None
    assert config.enabled is False
