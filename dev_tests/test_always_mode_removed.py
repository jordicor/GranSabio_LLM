"""Pydantic must reject the legacy ``"always"`` value for ``generation_tools_mode``.

The proposal eliminates the ``"always"`` literal entirely (§2.6, §4.7). This
test guards against accidental regressions that re-introduce the value.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from models import ContentRequest


def _base_request_kwargs() -> dict:
    """Minimal kwargs required to construct a ``ContentRequest``."""
    return {
        "content_type": "article",
        "prompt": "Sample prompt for validation.",
    }


def test_generation_tools_mode_rejects_always_literal():
    """``ContentRequest(generation_tools_mode="always")`` must raise ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        ContentRequest(generation_tools_mode="always", **_base_request_kwargs())

    error_detail = str(exc_info.value)
    assert "generation_tools_mode" in error_detail


def test_generation_tools_mode_accepts_auto_and_never():
    """The remaining valid literals must still work."""
    auto_request = ContentRequest(generation_tools_mode="auto", **_base_request_kwargs())
    assert auto_request.generation_tools_mode == "auto"

    never_request = ContentRequest(generation_tools_mode="never", **_base_request_kwargs())
    assert never_request.generation_tools_mode == "never"


@pytest.mark.parametrize(
    "mode_field",
    ["qa_tools_mode", "arbiter_tools_mode", "gransabio_tools_mode"],
)
def test_new_tools_mode_fields_reject_always(mode_field):
    """The new per-layer tool-mode fields are ``Literal["auto", "never"]`` as well."""
    kwargs = _base_request_kwargs()
    kwargs[mode_field] = "always"
    with pytest.raises(ValidationError) as exc_info:
        ContentRequest(**kwargs)
    assert mode_field in str(exc_info.value)


@pytest.mark.parametrize(
    "mode_field",
    ["qa_tools_mode", "arbiter_tools_mode", "gransabio_tools_mode"],
)
def test_new_tools_mode_fields_default_to_auto(mode_field):
    """Default value for every new per-layer tool-mode field is ``"auto"``."""
    request = ContentRequest(**_base_request_kwargs())
    assert getattr(request, mode_field) == "auto"
