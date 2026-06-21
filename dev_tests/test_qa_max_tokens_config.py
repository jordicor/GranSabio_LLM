from config import config
from models import normalize_qa_models_config


def _set_model_specs(monkeypatch, *, output_tokens: int = 20000, safety_margin: float = 0.5) -> None:
    monkeypatch.setattr(
        config,
        "spec_catalog",
        {
            "model_specifications": {
                "fake": {
                    "qa-model": {
                        "model_id": "qa-model",
                        "output_tokens": output_tokens,
                    }
                }
            },
            "token_validation": {
                "default_max_output": 8192,
                "fallback_limits": {"output": 4096},
                "safety_margin": safety_margin,
            },
        },
    )


def test_qa_model_string_uses_configured_default(monkeypatch):
    _set_model_specs(monkeypatch, output_tokens=64000, safety_margin=0.95)
    monkeypatch.setattr(config, "QA_DEFAULT_MAX_TOKENS", 12000)

    normalized = normalize_qa_models_config(
        qa_models=["qa-model"],
        qa_global_config=None,
        qa_models_config=None,
    )

    assert normalized[0].max_tokens == 12000


def test_qa_model_string_without_config_uses_model_safe_limit(monkeypatch):
    _set_model_specs(monkeypatch, output_tokens=20000, safety_margin=0.5)
    monkeypatch.setattr(config, "QA_DEFAULT_MAX_TOKENS", None)

    normalized = normalize_qa_models_config(
        qa_models=["qa-model"],
        qa_global_config=None,
        qa_models_config=None,
    )

    assert normalized[0].max_tokens == 10000


def test_qa_request_overrides_take_precedence_over_config_default(monkeypatch):
    _set_model_specs(monkeypatch, output_tokens=64000, safety_margin=0.95)
    monkeypatch.setattr(config, "QA_DEFAULT_MAX_TOKENS", 12000)

    normalized = normalize_qa_models_config(
        qa_models=["qa-model"],
        qa_global_config={"max_tokens": 9000},
        qa_models_config={"qa-model": {"max_tokens": 7000}},
    )

    assert normalized[0].max_tokens == 7000


def test_qa_truncation_retry_uses_larger_safe_model_limit(monkeypatch):
    _set_model_specs(monkeypatch, output_tokens=20000, safety_margin=0.5)
    monkeypatch.setattr(config, "QA_DEFAULT_MAX_TOKENS", None)

    assert config.resolve_qa_truncation_retry_max_tokens("qa-model", 4000) == 10000
    assert config.resolve_qa_truncation_retry_max_tokens("qa-model", 10000) is None
