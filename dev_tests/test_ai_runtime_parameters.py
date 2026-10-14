"""Tests for provider/model request parameter policies."""

import pytest

from ai_runtime import parameters as runtime_parameters


@pytest.mark.parametrize("parameter", ["temperature", "top_p", "top_k", "candidate_count"])
def test_gemini_38_omits_sampling_controls(parameter: str) -> None:
    assert not runtime_parameters.accepts_parameter("google", "gemini-3.8-flash", parameter)
    assert runtime_parameters.accepts_parameter("google", "gemini-2.5-flash", parameter)


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
