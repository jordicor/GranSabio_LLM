"""Loopback-only account management for the local owner's ChatGPT plan."""

from __future__ import annotations

import ipaddress
from pathlib import Path

import httpx
from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from config import config
from personal_subscription import (
    SubscriptionError,
    get_personal_subscription,
    subscriptions_allowed,
)

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parents[1] / "templates"))
CALLBACK_PATH = "/auth/chatgpt/callback"
COOKIE = "gransabio_chatgpt_state"
PAGE = "/admin/subscription"
PRIVATE_HEADERS = {"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"}


def _loopback(request: Request) -> None:
    """Use a literal loopback origin, independently of forwarded headers."""
    try:
        local = bool(request.client and ipaddress.ip_address(request.client.host).is_loopback)
    except ValueError:
        local = False
    if not local or request.url.hostname != "127.0.0.1" or request.url.scheme != "http":
        raise HTTPException(403, "Manage personal accounts on this computer at http://127.0.0.1:<engine-port>/admin/subscription.")


def _mutation(request: Request) -> None:
    _loopback(request)
    # Import here to avoid the app_state/router registration cycle.
    from .openrouter_routes import require_admin_same_origin

    require_admin_same_origin(request)


def _page(request: Request, *, error: str | None = None, status: int = 200):
    enabled = subscriptions_allowed(config)
    summary = {"active": None, "accounts": []}
    models = {}
    if enabled:
        try:
            service = get_personal_subscription(config)
            summary, models = service.summary(), service.catalog()
        except (SubscriptionError, OSError):
            error = error or "Personal account storage could not be opened. Check its permissions and use one engine process."
    # no-referrer makes plain form POSTs send Origin: null. An origin-only
    # referrer preserves CSRF checks without exposing OAuth callback path/query.
    return templates.TemplateResponse(request=request, name="personal_subscription.html", context={
        "enabled": enabled, "summary": summary, "models": models, "error": error,
    }, status_code=status, headers={**PRIVATE_HEADERS, "Referrer-Policy": "origin"})


@router.get(PAGE, response_class=HTMLResponse)
async def personal_subscription_page(request: Request):
    """Show account status without exposing credentials or authorization URLs."""
    if (request.url.hostname in {"localhost", "::1"} and request.url.scheme == "http"
            and request.client and request.client.host in {"127.0.0.1", "::1"}):
        return RedirectResponse(f"http://127.0.0.1:{request.url.port or 80}{PAGE}",
                                status_code=303, headers=PRIVATE_HEADERS)
    _loopback(request)
    return _page(request)


@router.post(PAGE + "/login")
async def personal_subscription_login(request: Request, account_id: str | None = Form(None)):
    """Bind one PKCE transaction to this browser before redirecting to OpenAI."""
    _mutation(request)
    try:
        service = get_personal_subscription(config)
        port = request.url.port or 80
        state, url = service.begin_login(f"http://127.0.0.1:{port}{CALLBACK_PATH}", account_id)
    except (SubscriptionError, OSError) as exc:
        detail = str(exc) if isinstance(exc, SubscriptionError) else "Personal account storage is unavailable."
        return _page(request, error=detail, status=400)
    response = RedirectResponse(url, status_code=303, headers=PRIVATE_HEADERS)
    response.set_cookie(COOKIE, state, max_age=600, httponly=True, samesite="lax", path=CALLBACK_PATH)
    return response


@router.get(CALLBACK_PATH)
async def personal_subscription_callback(request: Request):
    """Consume the callback once and remove its code from HTTP access logs."""
    params = dict(request.query_params)
    request.scope["query_string"] = b""
    _loopback(request)
    try:
        await get_personal_subscription(config).finish_login(params, request.cookies.get(COOKIE, ""))
    except (SubscriptionError, httpx.HTTPError, ValueError, KeyError, TypeError, OSError) as exc:
        detail = str(exc) if isinstance(exc, SubscriptionError) else "ChatGPT sign-in could not complete. Please try again."
        response = _page(request, error=detail, status=400)
    else:
        response = RedirectResponse(PAGE, status_code=303, headers=PRIVATE_HEADERS)
    response.delete_cookie(COOKIE, path=CALLBACK_PATH, httponly=True, samesite="lax")
    return response


@router.post(PAGE + "/{action}")
async def personal_subscription_action(request: Request, action: str, account_id: str | None = Form(None)):
    """Select, disconnect or refresh the explicitly chosen local account."""
    _mutation(request)
    if action not in {"select", "logout", "refresh"}:
        raise HTTPException(404, "Unknown account action")
    try:
        service = get_personal_subscription(config)
        if action == "select":
            await service.select(account_id or "")
        elif action == "logout":
            await service.logout()
        else:
            await service.refresh_models()
    except (SubscriptionError, httpx.HTTPError, ValueError, OSError) as exc:
        detail = str(exc) if isinstance(exc, SubscriptionError) else "The ChatGPT account could not be updated. Please reconnect."
        return _page(request, error=detail, status=400)
    return RedirectResponse(PAGE, status_code=303, headers=PRIVATE_HEADERS)
