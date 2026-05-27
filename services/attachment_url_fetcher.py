"""Secure remote attachment fetching with SSRF and rebinding protections."""

from __future__ import annotations

import asyncio
import ipaddress
import mimetypes
import socket
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple
from urllib.parse import unquote, urljoin, urlparse

import httpx

from services.attachment_types import (
    AttachmentTooLargeError,
    AttachmentValidationError,
)


@dataclass
class FetchedAttachmentStream:
    """Validated URL response metadata plus a response byte stream."""

    url: str
    parsed: Any
    response: httpx.Response
    declared_size: Optional[int]
    declared_mime: Optional[str]
    filename: str
    data_stream: AsyncIterator[bytes]


class AttachmentUrlFetcher:
    """Fetch remote attachments while preserving existing SSRF protections."""

    REDIRECT_STATUS_CODES = {301, 302, 303, 307, 308}
    ALLOWED_PORTS: Dict[str, int] = {"http": 80, "https": 443}
    MAX_REDIRECT_LIMIT = 35

    def __init__(self, *, settings: Any, chunk_size: int, logger: Any) -> None:
        self.settings = settings
        self.chunk_size = chunk_size
        self.logger = logger
        self.denied_networks = self.build_denied_networks()

    @asynccontextmanager
    async def fetch(self, url: str) -> AsyncIterator[FetchedAttachmentStream]:
        """Yield a validated streaming response and always close it afterwards."""

        normalized_url = (url or "").strip()
        if not normalized_url:
            raise AttachmentValidationError("URL is required for remote attachment ingestion")

        parsed = self.validate_url_structure(normalized_url)
        try:
            port_value = parsed.port
        except ValueError as exc:
            raise AttachmentValidationError("URL port is invalid") from exc
        port = self.determine_port(parsed.scheme.lower(), port_value)

        headers = {"User-Agent": self.settings.url_user_agent}
        timeout = httpx.Timeout(
            timeout=self.settings.url_timeout_seconds,
            connect=self.settings.url_connect_timeout_seconds,
            read=self.settings.url_read_timeout_seconds,
            write=self.settings.url_read_timeout_seconds,
            pool=self.settings.url_connect_timeout_seconds,
        )
        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=False,
            trust_env=False,
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=0, keepalive_expiry=0),
        ) as client:
            current_url = normalized_url
            current_parsed = parsed
            current_port = port
            redirects = 0
            max_redirects = min(self.settings.url_max_redirects, self.MAX_REDIRECT_LIMIT)

            while True:
                if not current_parsed.hostname:
                    raise AttachmentValidationError("Unable to determine hostname from URL")
                response = await self.send_pinned_request(
                    client,
                    url=current_url,
                    hostname=current_parsed.hostname,
                    port=current_port,
                    headers=headers,
                )
                try:
                    if response.status_code in self.REDIRECT_STATUS_CODES:
                        location = response.headers.get("location")
                        if not location:
                            raise AttachmentValidationError("Redirect response missing Location header")
                        next_url = urljoin(current_url, location)
                        current_parsed = self.validate_url_structure(next_url)
                        try:
                            next_port_value = current_parsed.port
                        except ValueError as exc:
                            raise AttachmentValidationError("URL port is invalid") from exc
                        current_port = self.determine_port(current_parsed.scheme.lower(), next_port_value)
                        redirects += 1
                        if redirects > max_redirects:
                            raise AttachmentValidationError("Too many redirects while fetching remote attachment")
                        current_url = next_url
                        continue

                    if response.status_code >= 400:
                        raise AttachmentValidationError(f"URL responded with HTTP {response.status_code}")

                    declared_size = self.extract_declared_size(response)
                    if declared_size is not None and declared_size > self.settings.max_size_bytes:
                        raise AttachmentTooLargeError(
                            f"Attachment exceeds allowed size of {self.settings.max_size_bytes} bytes"
                        )

                    declared_mime = self.extract_declared_mime(response)
                    filename = self.resolve_filename_from_response(
                        current_parsed,
                        response,
                        declared_mime,
                    )
                    fetched = FetchedAttachmentStream(
                        url=current_url,
                        parsed=current_parsed,
                        response=response,
                        declared_size=declared_size,
                        declared_mime=declared_mime,
                        filename=filename,
                        data_stream=self.iter_http_response(response),
                    )
                    yield fetched
                    return
                finally:
                    await response.aclose()

    def build_pinned_request(
        self,
        client: httpx.AsyncClient,
        method: str,
        url: str,
        *,
        hostname: str,
        ip_address: str,
        headers: Dict[str, str],
    ) -> httpx.Request:
        """Build a request pinned to a resolved IP while preserving host/SNI."""

        url_obj = httpx.URL(url)
        pinned_url = url_obj.copy_with(host=ip_address)
        request = client.build_request(method, pinned_url, headers=headers)
        request.headers["host"] = hostname
        request.extensions["sni_hostname"] = hostname
        return request

    async def send_pinned_request(
        self,
        client: httpx.AsyncClient,
        *,
        url: str,
        hostname: str,
        port: int,
        headers: Dict[str, str],
    ) -> httpx.Response:
        """Send a GET request pinned to DNS-resolved IPs."""

        resolved_ips = await self.resolve_and_validate_host(hostname, port)
        last_exc: Optional[Exception] = None
        for ip_address in resolved_ips:
            request = self.build_pinned_request(
                client,
                "GET",
                url,
                hostname=hostname,
                ip_address=ip_address,
                headers=headers,
            )
            try:
                return await client.send(request, stream=True)
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                last_exc = exc
                continue
        if last_exc:
            raise AttachmentValidationError("Unable to connect to resolved host") from last_exc
        raise AttachmentValidationError("Unable to connect to resolved host")

    def validate_url_structure(self, url: str):
        parsed = urlparse(url)
        scheme = parsed.scheme.lower()
        if scheme not in self.ALLOWED_PORTS:
            raise AttachmentValidationError("URL scheme is not permitted for attachments")
        allowed_schemes = {scheme.lower() for scheme in self.settings.allowed_url_schemes}
        if allowed_schemes and scheme not in allowed_schemes:
            raise AttachmentValidationError("URL scheme is not permitted for attachments")
        if parsed.username or parsed.password:
            raise AttachmentValidationError("URL must not include embedded credentials")
        if not parsed.netloc:
            raise AttachmentValidationError("URL must include a hostname")
        hostname = parsed.hostname
        if not hostname:
            raise AttachmentValidationError("Unable to determine hostname from URL")
        self.validate_hostname_format(hostname)
        if not self.hostname_allowed(hostname):
            raise AttachmentValidationError("URL hostname is not permitted")
        return parsed

    def hostname_allowed(self, hostname: str) -> bool:
        host = hostname.lower()
        blocked = {entry.lower() for entry in self.settings.blocked_url_hostnames}
        if any(host == entry or host.endswith(f".{entry}") for entry in blocked):
            return False
        allowed = {entry.lower() for entry in self.settings.allowed_url_hostnames}
        if allowed and not any(host == entry or host.endswith(f".{entry}") for entry in allowed):
            return False
        return True

    def validate_hostname_format(self, hostname: str) -> None:
        cleaned = hostname.strip()
        if hostname != cleaned:
            raise AttachmentValidationError("Hostname contains disallowed leading or trailing characters")
        if len(cleaned) > 253:
            raise AttachmentValidationError("Hostname exceeds maximum length")
        if any(ord(ch) < 32 for ch in cleaned):
            raise AttachmentValidationError("Hostname contains control characters")
        if cleaned.startswith("[") or cleaned.endswith("]"):
            raise AttachmentValidationError("IPv6 literals are not permitted for attachments")
        if "@" in cleaned:
            raise AttachmentValidationError("Hostname must not include '@'")
        if ".." in cleaned:
            raise AttachmentValidationError("Hostname contains empty labels")
        try:
            cleaned.encode("ascii")
        except UnicodeEncodeError as exc:
            raise AttachmentValidationError("Hostname must be ASCII") from exc
        if set(cleaned) <= set("0123456789."):
            try:
                ip_obj = ipaddress.ip_address(cleaned)
            except ValueError as exc:
                raise AttachmentValidationError("Hostname must be a valid IPv4 address") from exc
            if not isinstance(ip_obj, ipaddress.IPv4Address):
                raise AttachmentValidationError("IPv6 addresses are not permitted for attachments")
            if cleaned != str(ip_obj):
                raise AttachmentValidationError("IPv4 address must use canonical dotted-decimal notation")
            return
        labels = cleaned.split(".")
        for label in labels:
            if not label:
                raise AttachmentValidationError("Hostname contains empty labels")
            if label.startswith("-") or label.endswith("-"):
                raise AttachmentValidationError("Hostname labels must not start or end with '-'")
            lower = label.lower()
            for ch in lower:
                if ch not in "abcdefghijklmnopqrstuvwxyz0123456789-":
                    raise AttachmentValidationError("Hostname contains invalid characters")

    def determine_port(self, scheme: str, port: Optional[int]) -> int:
        expected = self.ALLOWED_PORTS[scheme]
        if port is None:
            return expected
        if port != expected:
            raise AttachmentValidationError("URL port is not permitted for attachments")
        return port

    async def resolve_and_validate_host(self, hostname: str, port: int) -> List[str]:
        self.validate_hostname_format(hostname)
        loop = asyncio.get_running_loop()
        try:
            addr_info = await loop.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
        except socket.gaierror as exc:
            raise AttachmentValidationError("Unable to resolve URL hostname") from exc
        addresses: List[str] = []
        seen: set[str] = set()
        for _family, _, _, _, sockaddr in addr_info:
            ip_literal = sockaddr[0]
            if ip_literal in seen:
                continue
            try:
                ip_obj = ipaddress.ip_address(ip_literal)
            except ValueError as exc:
                raise AttachmentValidationError("Resolved address is invalid") from exc
            self.validate_ip_address(ip_obj)
            if isinstance(ip_obj, ipaddress.IPv6Address):
                raise AttachmentValidationError("IPv6 addresses are not permitted for attachments")
            addresses.append(str(ip_obj))
            seen.add(ip_literal)
        if not addresses:
            raise AttachmentValidationError("Unable to resolve URL hostname to permitted addresses")
        return addresses

    def validate_ip_address(self, ip_obj: ipaddress._BaseAddress) -> None:
        if not getattr(ip_obj, "is_global", False):
            raise AttachmentValidationError("Resolved address is not routable on the public internet")
        for network in self.denied_networks:
            if ip_obj in network:
                raise AttachmentValidationError("Resolved address is not permitted")

    def build_denied_networks(self) -> Tuple[ipaddress._BaseNetwork, ...]:
        networks = [
            ipaddress.ip_network("0.0.0.0/8"),
            ipaddress.ip_network("10.0.0.0/8"),
            ipaddress.ip_network("100.64.0.0/10"),
            ipaddress.ip_network("127.0.0.0/8"),
            ipaddress.ip_network("169.254.0.0/16"),
            ipaddress.ip_network("169.254.169.254/32"),
            ipaddress.ip_network("169.254.169.253/32"),
            ipaddress.ip_network("169.254.170.2/32"),
            ipaddress.ip_network("172.16.0.0/12"),
            ipaddress.ip_network("192.0.0.0/24"),
            ipaddress.ip_network("192.0.2.0/24"),
            ipaddress.ip_network("192.168.0.0/16"),
            ipaddress.ip_network("198.18.0.0/15"),
            ipaddress.ip_network("198.51.100.0/24"),
            ipaddress.ip_network("203.0.113.0/24"),
            ipaddress.ip_network("224.0.0.0/4"),
            ipaddress.ip_network("240.0.0.0/4"),
            ipaddress.ip_network("255.255.255.255/32"),
            ipaddress.ip_network("100.100.100.200/32"),
            ipaddress.ip_network("::/128"),
            ipaddress.ip_network("::1/128"),
            ipaddress.ip_network("fe80::/10"),
            ipaddress.ip_network("fc00::/7"),
            ipaddress.ip_network("ff00::/8"),
            ipaddress.ip_network("2001:db8::/32"),
        ]
        return tuple(networks)

    def extract_declared_size(self, response: httpx.Response) -> Optional[int]:
        header = response.headers.get("Content-Length")
        if header is None:
            return None
        value = header.strip()
        if not value:
            return None
        try:
            size = int(value)
        except ValueError as exc:
            raise AttachmentValidationError("Invalid Content-Length header") from exc
        if size < 0:
            raise AttachmentValidationError("Content-Length must be non-negative")
        return size

    def extract_declared_mime(self, response: httpx.Response) -> Optional[str]:
        header = response.headers.get("Content-Type")
        if not header:
            return None
        mime = header.split(";", 1)[0].strip().lower()
        return mime or None

    def resolve_filename_from_response(
        self,
        parsed: Any,
        response: httpx.Response,
        declared_mime: Optional[str],
    ) -> str:
        disposition = response.headers.get("Content-Disposition")
        filename = self.filename_from_content_disposition(disposition) if disposition else None
        if not filename:
            filename = Path(parsed.path).name or "download"
        filename = filename.strip() or "download"
        filename = Path(filename).name or "download"
        if not Path(filename).suffix and declared_mime:
            guessed = mimetypes.guess_extension(declared_mime)
            if guessed:
                filename = f"{filename}{guessed}"
        return filename

    def filename_from_content_disposition(self, header: str) -> Optional[str]:
        parts = header.split(";")
        for part in parts[1:]:
            name, separator, value = part.strip().partition("=")
            if not separator:
                continue
            name = name.lower()
            cleaned = value.strip().strip("'\"")
            if name == "filename*":
                segments = cleaned.split("'", 2)
                if len(segments) == 3:
                    _, _, encoded = segments
                    decoded = unquote(encoded)
                    if decoded:
                        return decoded
                continue
            if name == "filename" and cleaned:
                return cleaned
        return None

    async def iter_http_response(self, response: httpx.Response) -> AsyncIterator[bytes]:
        min_rate = max(self.settings.url_min_bytes_per_second, 1)
        window_seconds = max(self.settings.url_min_speed_window_seconds, 1)
        window_start = time.monotonic()
        window_bytes = 0

        async for chunk in response.aiter_bytes(self.chunk_size):
            if not chunk:
                continue
            window_bytes += len(chunk)
            now = time.monotonic()
            elapsed = now - window_start
            if elapsed >= window_seconds:
                if window_bytes / max(elapsed, 1e-6) < min_rate:
                    raise AttachmentValidationError("Download speed below safety threshold")
                window_start = now
                window_bytes = 0
            yield chunk

        if window_bytes:
            now = time.monotonic()
            elapsed = max(now - window_start, 1e-6)
            if elapsed >= window_seconds and window_bytes / elapsed < min_rate:
                raise AttachmentValidationError("Download speed below safety threshold")
