from types import SimpleNamespace

import pytest

from deterministic_validation import evaluate_phrase_frequency_check
from models import PhraseFrequencyConfig, PhraseFrequencyRule


def test_phrase_frequency_check_skips_enabled_config_with_empty_rules_via_model_construct():
    config = PhraseFrequencyConfig.model_construct(enabled=True, rules=[])
    request = SimpleNamespace(phrase_frequency=config)

    result = evaluate_phrase_frequency_check("dummy text", request)

    assert result is None


def test_phrase_frequency_check_skips_rules_cleared_after_construction():
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

    result = evaluate_phrase_frequency_check("dummy text", request)

    assert result is None


def test_phrase_frequency_config_direct_construction_with_empty_rules_is_rejected():
    with pytest.raises(ValueError, match="At least one phrase frequency rule"):
        PhraseFrequencyConfig(enabled=True, rules=[])
