"""Compatibility helpers for Starlette template rendering."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from fastapi import Request
from fastapi.templating import Jinja2Templates


def enable_template_response_compat(templates: Jinja2Templates) -> Jinja2Templates:
    """Allow both old and new ``TemplateResponse`` call styles.

    Starlette 0.50 expects ``TemplateResponse(request, name, context)``. The
    existing app still has many calls using the older
    ``TemplateResponse(name, context)`` order. This adapter keeps the current
    routes working while remaining compatible with the new signature.
    """
    original_template_response = templates.TemplateResponse

    def template_response(*args: Any, **kwargs: Any) -> Any:
        if args and isinstance(args[0], str):
            name = args[0]
            context = args[1] if len(args) > 1 else kwargs.pop("context", None)
            remaining_args = args[2:]
            if context is None:
                context = {}
            if not isinstance(context, Mapping):
                raise TypeError("Legacy TemplateResponse context must be a mapping")
            request = context.get("request")
            if not isinstance(request, Request):
                raise ValueError("Legacy TemplateResponse calls require context['request']")
            return original_template_response(
                request,
                name,
                dict(context),
                *remaining_args,
                **kwargs,
            )
        return original_template_response(*args, **kwargs)

    templates.TemplateResponse = template_response  # type: ignore[method-assign]
    return templates
