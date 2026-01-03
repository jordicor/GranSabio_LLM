"""
Security utilities for Gran Sabio LLM Engine.
=============================================

Provides IP-based access control for internal-only endpoints.

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

import ipaddress
import logging
from pathlib import Path
from typing import List, Set

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Networks allowed to access the API
# Override by modifying this list before app startup if needed
INTERNAL_NETWORKS: List[ipaddress.IPv4Network | ipaddress.IPv6Network] = [
    ipaddress.ip_network('127.0.0.0/8'),       # localhost IPv4
    ipaddress.ip_network('::1/128'),           # localhost IPv6
    ipaddress.ip_network('192.168.50.0/24'),   # specific LAN
]

# Trusted proxies file (one CIDR per line).
TRUSTED_PROXIES_PATH = Path(__file__).with_name("trusted_proxies.txt")
DEFAULT_TRUSTED_PROXY_NETWORKS: List[ipaddress.IPv4Network | ipaddress.IPv6Network] = [
    ipaddress.ip_network("127.0.0.1/32"),
    ipaddress.ip_network("::1/128"),
]

# Headers to consult when the request is from a trusted proxy.
FORWARDED_SINGLE_IP_HEADERS: tuple[str, ...] = (
    "CF-Connecting-IP",
    "X-Real-IP",
)
FORWARDED_CHAIN_HEADER = "X-Forwarded-For"


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

# Paths that are always allowed (health checks, static files, etc.)
# These bypass IP filtering even when middleware is active
PUBLIC_PATHS: Set[str] = {
    '/health',
    '/openapi.json',
    '/docs',
    '/redoc',
}

# Path prefixes that are always allowed
PUBLIC_PATH_PREFIXES: tuple[str, ...] = (
    '/static/',
    '/templates/docs/',  # documentation static files
)


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
        return any(ip in network for network in INTERNAL_NETWORKS)
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

    If the request comes from a trusted proxy, honor forwarded
    headers (CF-Connecting-IP, X-Forwarded-For, X-Real-IP).

    Args:
        request: The FastAPI/Starlette request

    Returns:
        The client IP as a string, or None if unavailable
    """
    client_ip = request.client.host if request.client else None
    if not client_ip:
        return None

    if not is_trusted_proxy(client_ip):
        return client_ip

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

        return await call_next(request)
