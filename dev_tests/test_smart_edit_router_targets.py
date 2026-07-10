"""Regression tests for Smart Edit API target semantics."""

import pytest
from pydantic import ValidationError

from smart_edit.operations import SmartTextEditor
from smart_edit.router import (
    ActionRequest,
    ExecuteRequest,
    TextTargetRequest,
    execute_single_action,
)


def test_empty_exact_target_is_rejected() -> None:
    """An empty exact target must never reach string replacement loops."""
    with pytest.raises(ValidationError, match="non-empty string"):
        TextTargetRequest(mode="exact", value="")

    result = SmartTextEditor().replace("abc", "", "x")
    assert result.success is False
    assert result.content_after == "abc"


@pytest.mark.asyncio
async def test_execute_respects_requested_occurrence() -> None:
    """A positive occurrence selects only that exact match."""
    request = ExecuteRequest(
        text="first target, second target",
        action=ActionRequest(
            type="delete",
            target=TextTargetRequest(
                mode="exact",
                value="target",
                occurrence=2,
            ),
        ),
    )

    response = await execute_single_action(request)

    assert response.success is True
    assert response.text_after == "first target, second "


@pytest.mark.asyncio
async def test_execute_respects_regex_and_case_sensitivity() -> None:
    """Regex mode and case-insensitive matching survive the API conversion."""
    regex_response = await execute_single_action(
        ExecuteRequest(
            text="IDs: A-12 and B-34",
            action=ActionRequest(
                type="replace",
                target=TextTargetRequest(
                    mode="regex",
                    value=r"[A-Z]-\d+",
                    occurrence=2,
                ),
                content="REDACTED",
            ),
        )
    )
    case_response = await execute_single_action(
        ExecuteRequest(
            text="Alpha alpha",
            action=ActionRequest(
                type="delete",
                target=TextTargetRequest(
                    mode="exact",
                    value="ALPHA",
                    occurrence=-1,
                    case_sensitive=False,
                ),
            ),
        )
    )

    assert regex_response.success is True
    assert regex_response.text_after == "IDs: A-12 and REDACTED"
    assert case_response.success is True
    assert case_response.text_after == " "
