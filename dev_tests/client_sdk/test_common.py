"""Regression tests for shared Gran Sabio SDK computations."""

import pytest

from client._common import compute_token_budgets


@pytest.mark.parametrize(
    ("model_limit", "desired_max_tokens", "expected_max_tokens"),
    [
        (128_000, 2_048, 2_048),
        (1_024, 2_048, 1_024),
        (0, 2_048, 2_048),
        (-1, 2_048, 2_048),
        (128_000, 0, 2_048),
        (None, None, 2_048),
    ],
)
def test_compute_token_budgets_respects_requested_and_model_limits(
    model_limit,
    desired_max_tokens,
    expected_max_tokens,
):
    max_tokens, thinking_tokens = compute_token_budgets(
        model_limit,
        desired_max_tokens,
    )

    assert max_tokens == expected_max_tokens
    assert thinking_tokens is None


def test_compute_token_budgets_keeps_thinking_inside_requested_maximum():
    max_tokens, thinking_tokens = compute_token_budgets(
        model_limit=128_000,
        desired_max_tokens=2_048,
        desired_thinking_tokens=5_000,
    )

    assert max_tokens == 2_048
    assert thinking_tokens == 1_920


def test_compute_token_budgets_accepts_valid_thinking_budget_unchanged():
    assert compute_token_budgets(128_000, 4_096, 1_000) == (4_096, 1_000)


def test_compute_token_budgets_tolerates_invalid_values():
    assert compute_token_budgets("invalid", "invalid", "invalid") == (2_048, None)
