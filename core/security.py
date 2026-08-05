"""
Security utilities for Gran Sabio LLM Engine.
=============================================

Provides IP-based access control for internal-only endpoints, cross-site
request (CSRF) protection, security headers, and early request-body limits.

Deployment note: the client IP is derived from the X-Forwarded-For chain
only. CF-Connecting-IP / X-Real-IP are honored solely when
GRANSABIO_TRUST_SINGLE_IP_HEADERS is set, and the front proxy of the
deployment MUST rewrite/sanitize client-supplied forwarding headers.

Deployment mode "engine as private SaaS backend" (audit SAAS-03): when
GRANSABIO_ENGINE_SHARED_SECRET is set, non-loopback requests must present
that value as `Authorization: Bearer <secret>`; IPFilterMiddleware answers
401 when the header is missing and 403 when it does not match. The Gran
Sabio SaaS sends exactly this header (its SAAS_ENGINE_SHARED_SECRET
setting), so both variables must hold the same value in that topology.
Loopback clients stay exempt so the local web UI and the in-process
TestClient keep working without a credential. Leave the variable unset for
the classic self-hosted/LAN deployment, where the IP filter alone applies.

Usage:
    # As a FastAPI dependency (for specific endpoints/routers):
    from core.security import require_internal_ip

    @app.get("/admin")
    async def admin_endpoint(_ip: str = Depends(require_internal_ip)):
        ...

    # Or for entire routers:
    router = APIRouter(dependencies=[Depends(require_internal_ip)])

    # As middleware (for global protection):
    from core.security import IPFilterMiddleware
    app.add_middleware(IPFilterMiddleware)
"""

from __future__ import annotations

import hmac
import ipaddress
import logging
import os
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Set
from urllib.parse import urlsplit

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Networks allowed to access the API.
# Override with INTERNAL_ALLOWED_CIDRS="127.0.0.0/8,::1/128,192.168.0.0/16"
DEFAULT_INTERNAL_NETWORK_CIDRS: tuple[str, ...] = (
    "127.0.0.0/8",    # localhost IPv4
    "::1/128",        # localhost IPv6
    "192.168.0.0/16",  # RFC1918 private range — override via INTERNAL_ALLOWED_CIDRS
)

# Trusted proxies file (one CIDR per line).
TRUSTED_PROXIES_PATH = Path(__file__).with_name("trusted_proxies.txt")
DEFAULT_TRUSTED_PROXY_NETWORKS: List[ipaddress.IPv4Network | ipaddress.IPv6Network] = [
    ipaddress.ip_network("127.0.0.1/32"),
    ipaddress.ip_network("::1/128"),
]

# Headers to consult when the request is from a trusted proxy.
#
# Single-IP headers (CF-Connecting-IP, X-Real-IP) are NOT trusted by default:
# any client can send them, and a front proxy that does not rewrite them lets
# the value through intact, which bypasses the IP filter entirely (audit
# API-01). They are honored only when GRANSABIO_TRUST_SINGLE_IP_HEADERS is set
# to a truthy value, which signals that the real edge of the deployment
# guarantees (rewrites/sanitizes) these headers. The X-Forwarded-For chain is
# always evaluated with the rightmost-untrusted algorithm regardless.
FORWARDED_SINGLE_IP_HEADERS: tuple[str, ...] = (
    "CF-Connecting-IP",
    "X-Real-IP",
)
FORWARDED_CHAIN_HEADER = "X-Forwarded-For"

TRUST_SINGLE_IP_HEADERS_ENV_VAR = "GRANSABIO_TRUST_SINGLE_IP_HEADERS"
_TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}


def _trust_single_ip_headers() -> bool:
    """Return True when single-IP proxy headers were explicitly opted in."""
    return os.getenv(TRUST_SINGLE_IP_HEADERS_ENV_VAR, "").strip().lower() in _TRUTHY_ENV_VALUES


# Shared secret gating non-loopback access in the "engine as private SaaS
# backend" deployment (audit SAAS-03). Read at request time, like the other
# env-driven switches in this module, so tests and operators can change it
# without reloading the module.
ENGINE_SHARED_SECRET_ENV_VAR = "GRANSABIO_ENGINE_SHARED_SECRET"


def _engine_shared_secret() -> str | None:
    """Return the configured engine shared secret, or None when gate is off."""
    secret = os.getenv(ENGINE_SHARED_SECRET_ENV_VAR, "").strip()
    return secret or None


def _is_loopback_ip(client_ip: str | None) -> bool:
    """Return True when the client IP is a loopback address."""
    if not client_ip:
        return False
    try:
        return ipaddress.ip_address(client_ip).is_loopback
    except ValueError:
        return False


def _bearer_token(request: Request) -> str | None:
    """Extract the Bearer credential from the Authorization header, if present."""
    scheme, _, token = request.headers.get("Authorization", "").partition(" ")
    if scheme.lower() != "bearer":
        return None
    token = token.strip()
    return token or None


def _load_trusted_proxy_networks() -> List[ipaddress.IPv4Network | ipaddress.IPv6Network]:
    networks: List[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
    if TRUSTED_PROXIES_PATH.exists():
        for line in TRUSTED_PROXIES_PATH.read_text(encoding="utf-8").splitlines():
            entry = line.strip()
            if not entry or entry.startswith("#"):
                continue
            try:
                networks.append(ipaddress.ip_network(entry))
            except ValueError:
                logger.warning("Invalid trusted proxy CIDR ignored: %s", entry)
    if not networks:
        logger.warning("Trusted proxies file missing or empty: %s", TRUSTED_PROXIES_PATH)
        networks = list(DEFAULT_TRUSTED_PROXY_NETWORKS)
    return networks


TRUSTED_PROXY_NETWORKS = _load_trusted_proxy_networks()


def _load_internal_networks() -> List[ipaddress.IPv4Network | ipaddress.IPv6Network]:
    raw_value = os.getenv("INTERNAL_ALLOWED_CIDRS", "")
    cidr_values = [value.strip() for value in raw_value.split(",") if value.strip()]
    if not cidr_values:
        cidr_values = list(DEFAULT_INTERNAL_NETWORK_CIDRS)

    networks: List[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
    for cidr in cidr_values:
        try:
            networks.append(ipaddress.ip_network(cidr))
        except ValueError:
            logger.warning("Invalid internal CIDR ignored: %s", cidr)

    if not networks:
        networks = [ipaddress.ip_network(cidr) for cidr in DEFAULT_INTERNAL_NETWORK_CIDRS]
    return networks

# Paths/prefixes that bypass IP filtering.
#
# Gran Sabio's open-source server is intended for self-hosted/internal use.
# Keep these empty by default so health checks, API docs, static assets, and
# every operational endpoint all require an allowed internal client IP.
PUBLIC_PATHS: Set[str] = set()
PUBLIC_PATH_PREFIXES: tuple[str, ...] = ()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def is_ip_allowed(client_ip: str | None) -> bool:
    """
    Check if an IP address is in the allowed networks.

    Args:
        client_ip: The IP address to check (string format)

    Returns:
        True if the IP is allowed, False otherwise
    """
    if not client_ip:
        return False

    try:
        ip = ipaddress.ip_address(client_ip)
        return any(ip in network for network in _load_internal_networks())
    except ValueError:
        logger.warning("Invalid IP address format: %s", client_ip)
        return False


def is_trusted_proxy(client_ip: str | None) -> bool:
    """Return True when the request source is a trusted proxy."""
    if not client_ip:
        return False
    try:
        ip = ipaddress.ip_address(client_ip)
    except ValueError:
        logger.warning("Invalid proxy IP address format: %s", client_ip)
        return False
    return any(ip in network for network in TRUSTED_PROXY_NETWORKS)


def _is_valid_ip(value: str) -> bool:
    try:
        ipaddress.ip_address(value)
    except ValueError:
        return False
    return True


def _parse_forwarded_for(value: str) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def get_client_ip(request: Request) -> str | None:
    """
    Extract the client IP from a request.

    If the request comes from a trusted proxy, the client IP is derived from
    the X-Forwarded-For chain using the rightmost-untrusted algorithm.
    Single-IP headers (CF-Connecting-IP, X-Real-IP) are honored only when
    GRANSABIO_TRUST_SINGLE_IP_HEADERS is set, because they are spoofable
    unless the real edge of the deployment rewrites them.

    Args:
        request: The FastAPI/Starlette request

    Returns:
        The client IP as a string, or None if unavailable
    """
    client_ip = request.client.host if request.client else None
    if not client_ip:
        return None

    # Starlette/FastAPI TestClient uses the sentinel host "testclient".
    if client_ip == "testclient":
        return "127.0.0.1"

    if not is_trusted_proxy(client_ip):
        return client_ip

    if _trust_single_ip_headers():
        for header in FORWARDED_SINGLE_IP_HEADERS:
            forwarded = request.headers.get(header)
            if forwarded and _is_valid_ip(forwarded):
                return forwarded

    forwarded_for = request.headers.get(FORWARDED_CHAIN_HEADER, "")
    chain = _parse_forwarded_for(forwarded_for)
    if chain:
        chain.append(client_ip)
        while chain and is_trusted_proxy(chain[-1]):
            chain.pop()
        for ip in reversed(chain):
            if _is_valid_ip(ip):
                return ip

    try:
        if ipaddress.ip_address(client_ip).is_loopback:
            return client_ip
    except ValueError:
        logger.warning("Invalid proxy IP address format: %s", client_ip)

    return None


# =============================================================================
# FASTAPI DEPENDENCY
# =============================================================================

def require_internal_ip(request: Request) -> str:
    """
    FastAPI dependency that enforces internal IP access.

    Use this on endpoints or routers that should only be accessible
    from internal networks (localhost, LAN).

    Usage:
        @app.get("/admin")
        async def admin_endpoint(_ip: str = Depends(require_internal_ip)):
            ...

        # Or for entire routers:
        router = APIRouter(dependencies=[Depends(require_internal_ip)])

    Returns:
        The client IP if access is allowed

    Raises:
        HTTPException: 403 if the IP is not in allowed networks
    """
    client_ip = get_client_ip(request)

    if not client_ip:
        logger.warning("IP filter: Unable to determine client IP")
        raise HTTPException(
            status_code=403,
            detail="Access denied: unable to determine client IP"
        )

    if not is_ip_allowed(client_ip):
        logger.warning("IP filter: Access denied for IP %s", client_ip)
        raise HTTPException(
            status_code=403,
            detail="Access denied: internal access only"
        )

    return client_ip


# =============================================================================
# MIDDLEWARE (for global protection)
# =============================================================================

class IPFilterMiddleware(BaseHTTPMiddleware):
    """
    Middleware that filters requests by client IP.

    Use this for global protection of all endpoints.
    Allows certain public paths (health checks, static files).

    Usage:
        app.add_middleware(IPFilterMiddleware)

    Note:
        When user authentication is implemented, this middleware
        can be removed and replaced with auth-based access control.
        The debugger should continue using require_internal_ip directly.
    """

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Allow public paths
        if path in PUBLIC_PATHS:
            return await call_next(request)

        # Allow public path prefixes
        if path.startswith(PUBLIC_PATH_PREFIXES):
            return await call_next(request)

        # Check IP
        client_ip = get_client_ip(request)

        if not is_ip_allowed(client_ip):
            logger.warning(
                "IP filter middleware: Blocked request from %s to %s",
                client_ip,
                path
            )
            return JSONResponse(
                status_code=403,
                content={"detail": "Access denied: internal access only"}
            )

        # Optional shared-secret gate (audit SAAS-03): with
        # GRANSABIO_ENGINE_SHARED_SECRET set, non-loopback clients must
        # authenticate with `Authorization: Bearer <secret>`. Loopback stays
        # exempt so the local UI and TestClient keep working.
        secret = _engine_shared_secret()
        if secret is not None and not _is_loopback_ip(client_ip):
            token = _bearer_token(request)
            if token is None:
                logger.warning(
                    "Engine shared secret: missing bearer token from %s to %s",
                    client_ip,
                    path,
                )
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Unauthorized: bearer token required"},
                    headers={"WWW-Authenticate": "Bearer"},
                )
            if not hmac.compare_digest(token, secret):
                logger.warning(
                    "Engine shared secret: invalid bearer token from %s to %s",
                    client_ip,
                    path,
                )
                return JSONResponse(
                    status_code=403,
                    content={"detail": "Forbidden: invalid bearer token"},
                )

        return await call_next(request)


# =============================================================================
# CSRF / ORIGIN GUARD (audit FE-01)
# =============================================================================

# Methods that never mutate state and are exempt from the cross-site guard.
SAFE_METHODS: frozenset = frozenset({"GET", "HEAD", "OPTIONS", "TRACE"})

# Sec-Fetch-Site values that prove the browser initiated the request from the
# same site (or from a direct user gesture, e.g. typing the URL).
ALLOWED_SEC_FETCH_SITE_VALUES: frozenset = frozenset({"same-origin", "same-site", "none"})


def _origin_matches_host(origin: str, host: str | None) -> bool:
    """Return True when the Origin header targets exactly the served host."""
    if not host:
        return False
    try:
        origin_netloc = urlsplit(origin.strip()).netloc
    except ValueError:
        return False
    if not origin_netloc:
        return False
    return origin_netloc.lower() == host.strip().lower()


class CrossSiteRequestGuardMiddleware(BaseHTTPMiddleware):
    """
    Reject cross-site browser mutations (CSRF defense).

    The engine's only access control is the client IP, which the browser
    supplies automatically — ambient authority a malicious page can ride on.
    For mutating methods (POST/PUT/PATCH/DELETE) this middleware requires
    ``Sec-Fetch-Site`` to be same-origin/same-site/none when present; if it
    is absent, ``Origin`` must match the served host. When both headers are
    missing the request is allowed, so non-browser clients (curl, the SDK,
    TestClient, server-to-server calls) keep working unchanged.
    """

    async def dispatch(self, request: Request, call_next):
        if request.method in SAFE_METHODS:
            return await call_next(request)

        sec_fetch_site = request.headers.get("sec-fetch-site")
        if sec_fetch_site is not None:
            if sec_fetch_site.strip().lower() not in ALLOWED_SEC_FETCH_SITE_VALUES:
                logger.warning(
                    "Cross-site guard: blocked %s %s (Sec-Fetch-Site: %s)",
                    request.method,
                    request.url.path,
                    sec_fetch_site,
                )
                return JSONResponse(
                    status_code=403,
                    content={"detail": "Forbidden: cross-site request"}
                )
            return await call_next(request)

        origin = request.headers.get("origin")
        if origin is not None:
            if not _origin_matches_host(origin, request.headers.get("host")):
                logger.warning(
                    "Cross-site guard: blocked %s %s (Origin: %s, Host: %s)",
                    request.method,
                    request.url.path,
                    origin,
                    request.headers.get("host"),
                )
                return JSONResponse(
                    status_code=403,
                    content={"detail": "Forbidden: origin mismatch"}
                )
            return await call_next(request)

        # No browser context headers: non-browser client, allow.
        return await call_next(request)


# =============================================================================
# SECURITY HEADERS (audit FE-03)
# =============================================================================

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Add anti-clickjacking and MIME-sniffing headers to every response.

    ``X-Frame-Options: DENY`` prevents the UI from being framed (the engine
    has no session cookies; the client IP is the credential, so a framed UI
    runs with the victim's network authority). ``X-Content-Type-Options:
    nosniff`` disables MIME sniffing.
    """

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        return response


# =============================================================================
# REQUEST BODY LIMIT (audit API-02)
# =============================================================================

# Default ceiling for JSON request bodies (generate, analysis, admin...).
# Generous enough for long-text prompts and source documents, far below the
# "hundreds of MB" abuse scenario. Override with GRANSABIO_MAX_BODY_BYTES.
DEFAULT_MAX_BODY_BYTES = 8 * 1024 * 1024
MAX_BODY_BYTES_ENV_VAR = "GRANSABIO_MAX_BODY_BYTES"
# Extra room for multipart framing and base64 inflation on attachment uploads.
MULTIPART_OVERHEAD_BYTES = 512 * 1024

ASGIApp = Callable[[Dict[str, Any], Callable[..., Awaitable[dict]], Callable[..., Awaitable[None]]], Awaitable[None]]


def _env_int(name: str, default: int) -> int:
    """Read a positive int from the environment, falling back to default."""
    raw_value = os.getenv(name, "")
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


class RequestBodyLimitMiddleware:
    """
    Reject oversized request bodies before the app parses them.

    Pure-ASGI middleware following the SaaS ``RequestBodyLimitMiddleware``
    pattern: rejects by ``Content-Length`` when the header is present and
    otherwise counts the streamed body chunks, so neither JSON parsing nor
    multipart disk spooling ever runs past the limit. Attachment uploads get
    a dedicated ceiling derived from the configured attachment size plus
    base64 inflation (~4/3) and multipart framing overhead.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        default_max_bytes: int | None = None,
        attachment_max_bytes: int | None = None,
    ) -> None:
        self.app = app
        self.default_max_bytes = (
            default_max_bytes
            if default_max_bytes is not None
            else _env_int(MAX_BODY_BYTES_ENV_VAR, DEFAULT_MAX_BODY_BYTES)
        )
        base_attachment_bytes = (
            attachment_max_bytes
            if attachment_max_bytes is not None
            else DEFAULT_MAX_BODY_BYTES
        )
        self.attachment_max_bytes = int(base_attachment_bytes * 4 / 3) + MULTIPART_OVERHEAD_BYTES

    async def __call__(self, scope: Dict[str, Any], receive, send) -> None:
        if scope.get("type") != "http" or scope.get("method") in {"GET", "HEAD", "OPTIONS"}:
            await self.app(scope, receive, send)
            return

        max_bytes = self._limit_for_scope(scope)
        content_length = _content_length(scope)
        if content_length is not None and content_length > max_bytes:
            await _send_too_large(send, max_bytes)
            return

        messages = []
        received_bytes = 0
        while True:
            message = await receive()
            messages.append(message)
            if message.get("type") != "http.request":
                break
            received_bytes += len(message.get("body", b""))
            if received_bytes > max_bytes:
                await _send_too_large(send, max_bytes)
                return
            if not message.get("more_body", False):
                break

        message_index = 0

        async def replay_receive() -> dict:
            nonlocal message_index
            if message_index >= len(messages):
                return {"type": "http.disconnect"}
            message = messages[message_index]
            message_index += 1
            return message

        await self.app(scope, replay_receive, send)

    def _limit_for_scope(self, scope: Dict[str, Any]) -> int:
        path = str(scope.get("path") or "").rstrip("/")
        if scope.get("method") == "POST" and (
            path == "/attachments" or path.startswith("/attachments/")
        ):
            return self.attachment_max_bytes
        return self.default_max_bytes


def _content_length(scope: Dict[str, Any]) -> int | None:
    for raw_name, raw_value in scope.get("headers", []):
        if raw_name.lower() != b"content-length":
            continue
        try:
            value = int(raw_value)
        except ValueError:
            return None
        return max(value, 0)
    return None


async def _send_too_large(send, max_bytes: int) -> None:
    body = b'{"detail":"Request body too large"}'
    await send(
        {
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
                (b"x-max-body-bytes", str(max_bytes).encode("ascii")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})
