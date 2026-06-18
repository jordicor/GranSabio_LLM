"""Tests for provider/model request parameter policies."""

from ai_runtime import parameters as runtime_parameters


def test_kimi_fixed_sampling_policy_overrides_stale_catalog() -> None:
    specs = {
        "model_specifications": {
            "moonshot": {
                "models": {
                    "kimi-k2.7-code": {
                        "parameter_constraints": {
                            "temperature": {"mode": "allow"},
                        },
                    },
                },
            },
        },
    }

    assert (
        runtime_parameters.accepts_parameter(
            "moonshot",
            "kimi-k2.7-code",
            "temperature",
            specs=specs,
        )
        is False
    )
