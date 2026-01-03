"""Attachment management utilities for Gran Sabio LLM."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import ipaddress
import logging
import mimetypes
import re
import secrets
import socket
import time
import unicodedata
import uuid
import json as std_json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Set
from urllib.parse import unquote, urljoin, urlparse

import httpx
from fastapi import UploadFile
from pydantic import BaseModel, Field, ValidationError

import json_utils as json
from config import AttachmentSettings


# Temporarily disabled to fix segfault
# try:
#     import magic  # type: ignore
# except ImportError:  # pragma: no cover - optional dependency
magic = None  # type: ignore[var-annotated]

logger = logging.getLogger(__name__)


class AttachmentError(Exception):
    """Base exception for attachment errors."""


class AttachmentValidationError(AttachmentError):
    """Raised when incoming attachment data fails validation."""


class AttachmentTooLargeError(AttachmentError):
    """Raised when an attachment exceeds configured limits."""


class AttachmentNotFoundError(AttachmentError):
    """Raised when a requested attachment cannot be located."""


class AttachmentRecord(BaseModel):
    """Normalized metadata describing an ingested attachment."""

    upload_id: str = Field(..., description="Unique identifier for this attachment upload")
    origin: str = Field(default="upload", description="Origin of the attachment (upload|base64|url)")
    intended_usage: str = Field(default="context", description="High level usage hint for orchestration")
    original_filename: str = Field(..., description="Filename provided by the client")
    stored_filename: str = Field(..., description="Sanitized filename stored on disk")
    mime_type: str = Field(..., description="Validated MIME type for the attachment")
    size_bytes: int = Field(..., ge=0, description="Total bytes written to disk")
    declared_size: Optional[int] = Field(default=None, description="Size declared by client when available")
    declared_mime: Optional[str] = Field(default=None, description="Original MIME type claimed by client")
    detected_mime: Optional[str] = Field(default=None, description="MIME type detected via content inspection")
    sha256: str = Field(..., description="SHA-256 checksum of the stored file")
    metadata_signature: str = Field(..., description="Integrity signature for persisted metadata")
    original_url: Optional[str] = Field(default=None, description="Original source URL when origin is 'url'")
    created_at: str = Field(..., description="UTC timestamp when the file was persisted")
    hash_prefix1: str = Field(..., min_length=3, max_length=3, description="First prefix derived from user hash")
    hash_prefix2: str = Field(..., min_length=4, max_length=4, description="Second prefix derived from user hash")
    user_hash: str = Field(..., min_length=40, max_length=40, description="Full SHA1 hash for the user")
    storage_path: str = Field(..., description="Relative path (from user root) to the stored file")
    metadata_path: str = Field(..., description="Relative path (from user root) to the metadata JSON")




@dataclass
class ResolvedAttachment:
    """Resolved attachment pointing to on-disk resources."""

    username: str
    record: AttachmentRecord
    binary_path: Path
    metadata_path: Path

    def summary(self) -> Dict[str, Any]:
        """Provide a lightweight summary for validation and logging."""
        return {
            "username": self.username,
            "upload_id": self.record.upload_id,
            "original_filename": self.record.original_filename,
            "mime_type": self.record.mime_type,
            "size_bytes": self.record.size_bytes,
            "created_at": self.record.created_at,
            "intended_usage": self.record.intended_usage,
        }


@dataclass
class CleanupAction:
    """Single maintenance action recorded during cleanup."""

    category: str
    path: str
    detail: str


@dataclass
class CleanupReport:
    """Aggregated report produced by attachment cleanup."""

    dry_run: bool
    users_scanned: int = 0
    attachments_scanned: int = 0
    attachments_removed: int = 0
    metadata_removed: int = 0
    index_entries_rebuilt: int = 0
    retention_expired: int = 0
    orphaned_metadata: int = 0
    issues: List[str] = field(default_factory=list)
    actions: List[CleanupAction] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dry_run": self.dry_run,
            "users_scanned": self.users_scanned,
            "attachments_scanned": self.attachments_scanned,
            "attachments_removed": self.attachments_removed,
            "metadata_removed": self.metadata_removed,
            "index_entries_rebuilt": self.index_entries_rebuilt,
            "retention_expired": self.retention_expired,
            "orphaned_metadata": self.orphaned_metadata,
            "issues": list(self.issues),
            "actions": [action.__dict__ for action in self.actions],
        }


class AttachmentManager:
    """Persist attachments with SPARK-compatible layout and metadata."""

    CHUNK_SIZE = 1024 * 1024  # 1 MB per chunk when streaming uploads
    _REDIRECT_STATUS_CODES = {301, 302, 303, 307, 308}
    _ALLOWED_PORTS: Dict[str, int] = {"http": 80, "https": 443}
    _MAX_REDIRECT_LIMIT = 35
    _SAFE_MIME_WHEN_MAGIC_MISSING = frozenset(
        {
            "text/plain",
            "text/markdown",
            "text/csv",
            "application/pdf",
            "image/jpeg",
            "image/png",
            "image/gif",
            "image/webp",
        }
    )

    def __init__(self, *, settings: AttachmentSettings, pepper: str) -> None:
        if not pepper:
            raise ValueError("PEPPER must be configured before using AttachmentManager")

        self.settings = settings
        self.base_path = Path(settings.base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.pepper = pepper
        self.index_history_limit = settings.index_history_limit
        self._recent_url_cache: Dict[Tuple[str, str], Tuple[str, str, float]] = {}
        self._url_cache_lock = asyncio.Lock()
        # Per-user index file locks to prevent concurrent write corruption
        self._index_locks: Dict[str, asyncio.Lock] = {}
        self._index_locks_lock = asyncio.Lock()
        self._denied_networks = self._build_denied_networks()
        if magic is None:
            logger.warning("python-magic is not available; attachment MIME hardening is degraded")

    async def store_upload(
        self,
        *,
        username: str,
        upload_file: UploadFile,
        intended_usage: str = "context",
        origin: str = "upload",
    ) -> AttachmentRecord:
        """Persist a multipart/form-data upload and return normalized metadata."""
        if not username:
            raise AttachmentValidationError("username is required for attachment ingestion")

        original_filename = upload_file.filename or "attachment"
        sanitized_filename = self._sanitize_filename(original_filename)
        extension = Path(sanitized_filename).suffix.lower()
        self._validate_extension(extension)

        resolved_mime = upload_file.content_type or mimetypes.guess_type(sanitized_filename)[0] or "application/octet-stream"
        self._validate_mime(resolved_mime, source="declared")

        record = await self._persist_content(
            username=username,
            sanitized_filename=sanitized_filename,
            original_filename=original_filename,
            intended_usage=intended_usage,
            origin=origin,
            resolved_mime=resolved_mime,
            declared_mime=upload_file.content_type,
            declared_size=self._safe_content_length(upload_file),
            data_stream=self._iter_upload_file(upload_file),
            source_url=None,
        )

        await upload_file.close()
        return record

    async def store_bytes(
        self,
        *,
        username: str,
        data: bytes,
        filename: str,
        intended_usage: str = "context",
        origin: str = "base64",
        mime_type: Optional[str] = None,
    ) -> AttachmentRecord:
        """Persist an attachment provided as raw bytes (base64, etc.)."""
        if not username:
            raise AttachmentValidationError("username is required for attachment ingestion")
        if data is None:
            raise AttachmentValidationError("No data provided for attachment")

        sanitized_filename = self._sanitize_filename(filename or "attachment")
        extension = Path(sanitized_filename).suffix.lower()
        self._validate_extension(extension)

        resolved_mime = mime_type or mimetypes.guess_type(sanitized_filename)[0] or "application/octet-stream"
        self._validate_mime(resolved_mime, source="declared")

        return await self._persist_content(
            username=username,
            sanitized_filename=sanitized_filename,
            original_filename=filename or sanitized_filename,
            intended_usage=intended_usage,
            origin=origin,
            resolved_mime=resolved_mime,
            declared_mime=mime_type,
            declared_size=len(data),
            data_stream=self._iter_bytes(data),
            source_url=None,
        )

    def _build_pinned_request(
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

    async def _send_pinned_request(
        self,
        client: httpx.AsyncClient,
        *,
        url: str,
        hostname: str,
        port: int,
        headers: Dict[str, str],
    ) -> httpx.Response:
        """Send a GET request pinned to the DNS-resolved IPs (anti-rebinding)."""
        resolved_ips = await self._resolve_and_validate_host(hostname, port)
        last_exc: Optional[Exception] = None
        for ip_address in resolved_ips:
            request = self._build_pinned_request(
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

    async def store_from_url(
        self,
        *,
        username: str,
        url: str,
        intended_usage: str = "context",
    ) -> AttachmentRecord:
        """Download an attachment from a remote URL and persist it locally."""
        if not username:
            raise AttachmentValidationError("username is required for attachment ingestion")
        normalized_url = (url or "").strip()
        if not normalized_url:
            raise AttachmentValidationError("URL is required for remote attachment ingestion")

        parsed = self._validate_url_structure(normalized_url)
        try:
            port_value = parsed.port
        except ValueError as exc:
            raise AttachmentValidationError("URL port is invalid") from exc
        port = self._determine_port(parsed.scheme.lower(), port_value)

        cached = await self._get_cached_url_record(username=username, url=normalized_url)
        if cached:
            logger.info("Using cached attachment %s for URL %s", cached.upload_id, normalized_url)
            return cached

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
            max_redirects = min(self.settings.url_max_redirects, self._MAX_REDIRECT_LIMIT)

            while True:
                if not current_parsed.hostname:
                    raise AttachmentValidationError("Unable to determine hostname from URL")
                response = await self._send_pinned_request(
                    client,
                    url=current_url,
                    hostname=current_parsed.hostname,
                    port=current_port,
                    headers=headers,
                )

                if response.status_code in self._REDIRECT_STATUS_CODES:
                    location = response.headers.get("location")
                    await response.aclose()
                    if not location:
                        raise AttachmentValidationError("Redirect response missing Location header")
                    next_url = urljoin(current_url, location)
                    current_parsed = self._validate_url_structure(next_url)
                    try:
                        next_port_value = current_parsed.port
                    except ValueError as exc:
                        raise AttachmentValidationError("URL port is invalid") from exc
                    current_port = self._determine_port(current_parsed.scheme.lower(), next_port_value)
                    redirects += 1
                    if redirects > max_redirects:
                        raise AttachmentValidationError("Too many redirects while fetching remote attachment")
                    current_url = next_url
                    continue

                if response.status_code >= 400:
                    status_code = response.status_code
                    await response.aclose()
                    raise AttachmentValidationError(f"URL responded with HTTP {status_code}")

                declared_size = self._extract_declared_size(response)
                if declared_size is not None and declared_size > self.settings.max_size_bytes:
                    await response.aclose()
                    raise AttachmentTooLargeError(
                        f"Attachment exceeds allowed size of {self.settings.max_size_bytes} bytes"
                    )

                declared_mime = self._extract_declared_mime(response)
                filename = self._resolve_filename_from_response(current_parsed, response, declared_mime)
                sanitized_filename = self._sanitize_filename(filename)
                extension = Path(sanitized_filename).suffix.lower()
                self._validate_extension(extension)

                resolved_mime = (
                    declared_mime
                    or mimetypes.guess_type(sanitized_filename)[0]
                    or "application/octet-stream"
                )
                self._validate_mime(resolved_mime, source="declared")
                if magic is None:
                    if not resolved_mime or resolved_mime == "application/octet-stream" or resolved_mime not in self._SAFE_MIME_WHEN_MAGIC_MISSING:
                        await response.aclose()
                        raise AttachmentValidationError("MIME type is not permitted without python-magic")

                try:
                    record = await self._persist_content(
                        username=username,
                        sanitized_filename=sanitized_filename,
                        original_filename=filename,
                        intended_usage=intended_usage,
                        origin="url",
                        resolved_mime=resolved_mime,
                        declared_mime=declared_mime,
                        declared_size=declared_size,
                        data_stream=self._iter_http_response(response),
                        source_url=current_url,
                    )
                finally:
                    await response.aclose()

                break

        await self._update_url_cache(username=username, url=normalized_url, record=record)
        return record

    def get_metadata(self, *, username: str, upload_id: str) -> AttachmentRecord:
        """Retrieve stored metadata for a given user/upload combination."""
        if not username:
            raise AttachmentValidationError("username is required to fetch attachment metadata")
        if not upload_id:
            raise AttachmentValidationError("upload_id is required to fetch attachment metadata")

        prefix1, prefix2, user_hash = self.generate_user_hash(username)
        user_root = self._user_root(prefix1, prefix2, user_hash)
        index_path = user_root / "uploads" / "index.json"

        if not index_path.exists():
            raise AttachmentNotFoundError("No attachments registered for this user")

        try:
            with index_path.open("r", encoding="utf-8") as fh:
                index_data: List[Dict[str, str]] = json.load(fh)
        except json.JSONDecodeError as exc:
            raise AttachmentNotFoundError("Attachment index is corrupted") from exc

        relative_metadata = None
        for entry in index_data:
            if entry.get("upload_id") == upload_id:
                relative_metadata = entry.get("metadata_path")
                break

        if not relative_metadata:
            raise AttachmentNotFoundError("Attachment metadata not found")

        metadata_file = user_root / Path(relative_metadata)
        if not metadata_file.exists():
            raise AttachmentNotFoundError("Attachment metadata file is missing")

        try:
            with metadata_file.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except json.JSONDecodeError as exc:
            raise AttachmentValidationError("Attachment metadata file is corrupted") from exc

        try:
            record = AttachmentRecord(**payload)
        except ValidationError as exc:
            raise AttachmentValidationError("Attachment metadata failed validation") from exc

        expected_signature = self._compute_metadata_signature(payload)
        if not record.metadata_signature:
            raise AttachmentValidationError("Attachment metadata is missing integrity signature")
        if not secrets.compare_digest(record.metadata_signature, expected_signature):
            raise AttachmentValidationError("Attachment metadata signature mismatch")

        return record


    def resolve_attachment(
        self,
        *,
        username: str,
        upload_id: str,
    ) -> ResolvedAttachment:
        """Resolve metadata into absolute paths ensuring resources exist."""
        record = self.get_metadata(username=username, upload_id=upload_id)
        prefix1, prefix2, user_hash = self.generate_user_hash(username)
        user_root = self._user_root(prefix1, prefix2, user_hash)
        binary_path = user_root / Path(record.storage_path)
        metadata_path = user_root / Path(record.metadata_path)

        if not binary_path.exists():
            raise AttachmentNotFoundError("Attachment binary is missing from storage")
        if not metadata_path.exists():
            raise AttachmentNotFoundError("Attachment metadata file is missing from storage")

        return ResolvedAttachment(
            username=username,
            record=record,
            binary_path=binary_path,
            metadata_path=metadata_path,
        )

    def build_preflight_summary(self, resolved: ResolvedAttachment) -> Dict[str, Any]:
        """Return metadata snapshot safe for validation and logging layers."""
        summary = resolved.summary()
        summary.update(
            {
                "storage_path": resolved.record.storage_path,
                "metadata_path": resolved.record.metadata_path,
                "declared_mime": resolved.record.declared_mime,
            }
        )
        return summary

    def load_text_preview(self, resolved: ResolvedAttachment, *, max_bytes: int = 4096) -> Optional[str]:
        """Load a small preview of textual attachments for prompt context."""
        if not self._is_text_like(resolved.record):
            return None
        try:
            with resolved.binary_path.open("rb") as fh:
                data = fh.read(max_bytes)
        except FileNotFoundError as exc:
            raise AttachmentNotFoundError("Attachment binary is missing from storage") from exc
        except OSError as exc:
            raise AttachmentError("Unable to read attachment content") from exc

        return data.decode("utf-8", errors="replace")


    def run_cleanup(
        self,
        *,
        dry_run: bool = True,
        username: Optional[str] = None,
        user_hash: Optional[str] = None,
        retention_days: Optional[int] = None,
    ) -> CleanupReport:
        """Validate attachment storage and optionally prune stale data."""
        if username and user_hash:
            raise AttachmentValidationError("Provide either username or user_hash, not both")
        report = CleanupReport(dry_run=dry_run)
        targets: List[Path] = []
        if username:
            prefix1, prefix2, hashed = self.generate_user_hash(username)
            targets.append(self._user_root(prefix1, prefix2, hashed))
        elif user_hash:
            normalized = user_hash.strip().lower()
            if len(normalized) < 7:
                raise AttachmentValidationError("user_hash must include at least 7 characters")
            targets.append(self.base_path / normalized[:3] / normalized[3:7] / normalized)
        else:
            targets.extend(self._iter_user_roots())

        now = datetime.utcnow()
        effective_retention = retention_days
        if effective_retention is None:
            effective_retention = self.settings.retention_days
        if effective_retention is not None and effective_retention <= 0:
            effective_retention = None

        valid_targets: List[Path] = []
        for root in targets:
            if root.exists():
                valid_targets.append(root)
            else:
                report.issues.append(f"User root missing during cleanup: {root}")

        for root in valid_targets:
            self._cleanup_user_root(
                root,
                report=report,
                dry_run=dry_run,
                retention_days=effective_retention,
                now=now,
            )

        return report

    def _iter_user_roots(self) -> List[Path]:
        """Return all user root directories under the attachment base path."""
        if not self.base_path.exists():
            return []
        roots: List[Path] = []
        for prefix1_dir in self.base_path.iterdir():
            if not prefix1_dir.is_dir():
                continue
            for prefix2_dir in prefix1_dir.iterdir():
                if not prefix2_dir.is_dir():
                    continue
                for user_dir in prefix2_dir.iterdir():
                    if user_dir.is_dir():
                        roots.append(user_dir)
        return roots

    def _cleanup_user_root(
        self,
        user_root: Path,
        *,
        report: CleanupReport,
        dry_run: bool,
        retention_days: Optional[int],
        now: datetime,
    ) -> None:
        """Cleanup a single user root directory."""
        uploads_dir = user_root / "uploads"
        if not uploads_dir.exists():
            return

        report.users_scanned += 1
        metadata_files = sorted(uploads_dir.rglob("metadata.json"))
        valid_records: List[AttachmentRecord] = []
        seen_uploads: Set[str] = set()

        relative_parts = self._relative_user_parts(user_root)
        if relative_parts is None:
            report.issues.append(f"Skipping unexpected user root structure: {user_root}")
            return
        prefix1, prefix2, hashed = relative_parts

        for metadata_file in metadata_files:
            record = self._load_metadata_for_cleanup(metadata_file=metadata_file, report=report, dry_run=dry_run)
            if record is None:
                continue

            if record.upload_id in seen_uploads:
                report.metadata_removed += 1
                report.issues.append(f"Duplicate upload_id {record.upload_id} in {metadata_file}")
                self._handle_attachment_removal(
                    report=report,
                    dry_run=dry_run,
                    binary_path=None,
                    metadata_file=metadata_file,
                    reason="Duplicate upload identifier",
                )
                continue

            seen_uploads.add(record.upload_id)
            report.attachments_scanned += 1

            metadata_dirty = False
            expected_metadata_rel = self._relative_path(user_root, metadata_file)
            if expected_metadata_rel is None:
                report.metadata_removed += 1
                report.issues.append(f"Metadata outside user root: {metadata_file}")
                self._handle_attachment_removal(
                    report=report,
                    dry_run=dry_run,
                    binary_path=None,
                    metadata_file=metadata_file,
                    reason="Metadata outside expected directory",
                )
                continue

            if record.metadata_path != expected_metadata_rel:
                metadata_dirty = True
                record.metadata_path = expected_metadata_rel

            expected_storage_rel = self._relative_path(
                user_root, metadata_file.parent / record.stored_filename
            )
            binary_path = user_root / Path(record.storage_path)
            if expected_storage_rel and not binary_path.exists():
                candidate = metadata_file.parent / record.stored_filename
                if candidate.exists():
                    binary_path = candidate
                    record.storage_path = expected_storage_rel
                    metadata_dirty = True

            if not binary_path.exists():
                report.orphaned_metadata += 1
                report.metadata_removed += 1
                report.issues.append(f"Missing binary for metadata {metadata_file}")
                self._handle_attachment_removal(
                    report=report,
                    dry_run=dry_run,
                    binary_path=None,
                    metadata_file=metadata_file,
                    reason="Binary payload missing",
                )
                continue

            if record.user_hash != hashed or record.hash_prefix1 != prefix1 or record.hash_prefix2 != prefix2:
                metadata_dirty = True
                record.user_hash = hashed
                record.hash_prefix1 = prefix1
                record.hash_prefix2 = prefix2

            try:
                checksum, size_bytes = self._compute_file_digest(binary_path)
            except OSError as exc:
                report.metadata_removed += 1
                report.issues.append(f"Unreadable attachment {binary_path}: {exc}")
                self._handle_attachment_removal(
                    report=report,
                    dry_run=dry_run,
                    binary_path=None,
                    metadata_file=metadata_file,
                    reason="Attachment unreadable",
                )
                continue

            if size_bytes != record.size_bytes:
                report.attachments_removed += 1
                report.metadata_removed += 1
                self._handle_attachment_removal(
                    report=report,
                    dry_run=dry_run,
                    binary_path=binary_path,
                    metadata_file=metadata_file,
                    reason=f"Size mismatch (expected {record.size_bytes}, got {size_bytes})",
                )
                continue

            if not secrets.compare_digest(checksum, record.sha256):
                report.attachments_removed += 1
                report.metadata_removed += 1
                self._handle_attachment_removal(
                    report=report,
                    dry_run=dry_run,
                    binary_path=binary_path,
                    metadata_file=metadata_file,
                    reason="Checksum mismatch",
                )
                continue

            created_at = self._parse_created_at(record.created_at)
            if created_at is None:
                report.attachments_removed += 1
                report.metadata_removed += 1
                self._handle_attachment_removal(
                    report=report,
                    dry_run=dry_run,
                    binary_path=binary_path,
                    metadata_file=metadata_file,
                    reason="Invalid created_at timestamp",
                )
                continue

            if retention_days is not None and created_at < now - timedelta(days=retention_days):
                report.attachments_removed += 1
                report.metadata_removed += 1
                report.retention_expired += 1
                self._handle_attachment_removal(
                    report=report,
                    dry_run=dry_run,
                    binary_path=binary_path,
                    metadata_file=metadata_file,
                    reason=f"Attachment expired retention window ({retention_days} days)",
                )
                continue

            expected_signature = self._compute_metadata_signature(record.model_dump())
            if not record.metadata_signature or not secrets.compare_digest(
                record.metadata_signature, expected_signature
            ):
                metadata_dirty = True
                record.metadata_signature = expected_signature
                report.issues.append(f"Repaired metadata signature for {metadata_file}")

            if metadata_dirty:
                self._record_metadata_update(
                    metadata_file=metadata_file,
                    record=record,
                    dry_run=dry_run,
                    report=report,
                )

            valid_records.append(record)

        index_file = uploads_dir / "index.json"
        self._rebuild_index(index_file, valid_records, dry_run=dry_run, report=report)

    def _load_metadata_for_cleanup(
        self,
        *,
        metadata_file: Path,
        report: CleanupReport,
        dry_run: bool,
    ) -> Optional[AttachmentRecord]:
        """Load metadata from disk, removing the file if invalid."""
        try:
            with metadata_file.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            report.metadata_removed += 1
            report.issues.append(f"Unable to read metadata {metadata_file}: {exc}")
            self._handle_attachment_removal(
                report=report,
                dry_run=dry_run,
                binary_path=None,
                metadata_file=metadata_file,
                reason="Invalid metadata JSON",
            )
            return None

        try:
            return AttachmentRecord(**payload)
        except ValidationError as exc:
            report.metadata_removed += 1
            report.issues.append(f"Metadata validation failed for {metadata_file}: {exc}")
            self._handle_attachment_removal(
                report=report,
                dry_run=dry_run,
                binary_path=None,
                metadata_file=metadata_file,
                reason="Metadata schema check failed",
            )
            return None

    def _record_metadata_update(
        self,
        *,
        metadata_file: Path,
        record: AttachmentRecord,
        dry_run: bool,
        report: CleanupReport,
    ) -> None:
        """Persist metadata updates during cleanup when not in dry-run mode."""
        detail = "Updated metadata to reflect on-disk state"
        if dry_run:
            detail += " (dry-run)"
        report.actions.append(CleanupAction("update_metadata", str(metadata_file), detail))
        if not dry_run:
            self._write_metadata(metadata_file, record)

    def _handle_attachment_removal(
        self,
        *,
        report: CleanupReport,
        dry_run: bool,
        binary_path: Optional[Path],
        metadata_file: Optional[Path],
        reason: str,
    ) -> None:
        """Record and optionally perform attachment or metadata removal."""
        detail = reason if not dry_run else f"{reason} (dry-run)"
        if binary_path is not None:
            report.actions.append(CleanupAction("remove_attachment", str(binary_path), detail))
            if not dry_run:
                self._safe_remove(binary_path, report)
        if metadata_file is not None:
            report.actions.append(CleanupAction("remove_metadata", str(metadata_file), detail))
            if not dry_run:
                self._safe_remove(metadata_file, report)

    def _rebuild_index(
        self,
        index_file: Path,
        records: List[AttachmentRecord],
        *,
        dry_run: bool,
        report: CleanupReport,
    ) -> None:
        """Rewrite the per-user index file after cleanup."""
        sorted_records = sorted(records, key=lambda item: item.created_at, reverse=True)
        limited = sorted_records[: self.index_history_limit]
        entries = [
            {
                "upload_id": record.upload_id,
                "original_filename": record.original_filename,
                "stored_filename": record.stored_filename,
                "mime_type": record.mime_type,
                "size_bytes": record.size_bytes,
                "created_at": record.created_at,
                "storage_path": record.storage_path,
                "metadata_path": record.metadata_path,
            }
            for record in limited
        ]
        detail = f"Rebuilt index with {len(entries)} entries"
        if dry_run:
            detail += " (dry-run)"
        report.index_entries_rebuilt += 1
        report.actions.append(CleanupAction("rebuild_index", str(index_file), detail))
        if dry_run:
            return
        index_file.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: write to temp file, then rename
        temp_file = index_file.with_suffix(".tmp")
        try:
            with temp_file.open("w", encoding="utf-8") as fh:
                json.dump(entries, fh, ensure_ascii=True, indent=2)
            temp_file.replace(index_file)
        except Exception:
            temp_file.unlink(missing_ok=True)
            raise

    def _relative_user_parts(self, user_root: Path) -> Optional[Tuple[str, str, str]]:
        try:
            relative = user_root.relative_to(self.base_path)
        except ValueError:
            return None
        parts = relative.parts
        if len(parts) != 3:
            return None
        return parts[0], parts[1], parts[2]

    def _relative_path(self, root: Path, path: Path) -> Optional[str]:
        try:
            return path.relative_to(root).as_posix()
        except ValueError:
            return None

    def _compute_file_digest(self, binary_path: Path) -> Tuple[str, int]:
        hasher = hashlib.sha256()
        size = 0
        with binary_path.open("rb") as fh:
            while True:
                chunk = fh.read(self.CHUNK_SIZE)
                if not chunk:
                    break
                size += len(chunk)
                hasher.update(chunk)
        return hasher.hexdigest(), size

    def _parse_created_at(self, value: str) -> Optional[datetime]:
        if not value:
            return None
        cleaned = value.strip()
        if cleaned.endswith("Z"):
            cleaned = cleaned[:-1]
        try:
            return datetime.fromisoformat(cleaned)
        except ValueError:
            return None

    def _safe_remove(self, path: Path, report: CleanupReport) -> None:
        try:
            path.unlink()
        except FileNotFoundError:
            return
        except OSError as exc:
            report.issues.append(f"Failed to remove {path}: {exc}")

    async def _persist_content(
        self,
        *,
        username: str,
        sanitized_filename: str,
        original_filename: str,
        intended_usage: str,
        origin: str,
        resolved_mime: str,
        declared_mime: Optional[str],
        declared_size: Optional[int],
        data_stream: AsyncIterator[bytes],
        source_url: Optional[str],
    ) -> AttachmentRecord:
        prefix1, prefix2, user_hash = self.generate_user_hash(username)
        user_root = self._user_root(prefix1, prefix2, user_hash)

        now = datetime.utcnow()
        upload_id = uuid.uuid4().hex
        relative_dir = Path("uploads") / f"{now.year:04d}" / f"{now.month:02d}" / upload_id
        target_dir = user_root / relative_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        binary_path = target_dir / sanitized_filename
        sha256 = hashlib.sha256()
        total_bytes = 0
        sample = bytearray()

        try:
            with binary_path.open("wb") as destination:
                async for chunk in data_stream:
                    if not chunk:
                        continue
                    if len(sample) < self.settings.magic_sample_bytes:
                        needed = self.settings.magic_sample_bytes - len(sample)
                        sample.extend(chunk[:needed])
                    total_bytes += len(chunk)
                    if total_bytes > self.settings.max_size_bytes:
                        raise AttachmentTooLargeError(
                            f"Attachment exceeds allowed size of {self.settings.max_size_bytes} bytes"
                        )
                    destination.write(chunk)
                    sha256.update(chunk)
        except AttachmentTooLargeError:
            binary_path.unlink(missing_ok=True)
            raise
        except Exception as exc:  # pragma: no cover - unexpected I/O failure
            binary_path.unlink(missing_ok=True)
            raise AttachmentError("Unable to persist attachment contents") from exc

        try:
            self._validate_declared_size(actual=total_bytes, declared=declared_size)
        except AttachmentValidationError:
            binary_path.unlink(missing_ok=True)
            raise

        detected_mime = self._detect_mime(bytes(sample), binary_path)
        final_mime = resolved_mime
        try:
            if detected_mime:
                self._validate_mime(detected_mime, source="detected")
                if declared_mime and not self._mime_equivalent(detected_mime, declared_mime):
                    raise AttachmentValidationError(
                        "Detected MIME does not match declared MIME type"
                    )
                if resolved_mime and not self._mime_equivalent(detected_mime, resolved_mime):
                    raise AttachmentValidationError(
                        "Detected MIME does not align with file extension"
                    )
                final_mime = detected_mime
            else:
                self._validate_mime(resolved_mime, source="declared")
        except AttachmentValidationError:
            binary_path.unlink(missing_ok=True)
            raise

        if magic is None:
            safe_mime = (final_mime or "").lower()
            if safe_mime not in self._SAFE_MIME_WHEN_MAGIC_MISSING:
                binary_path.unlink(missing_ok=True)
                raise AttachmentValidationError("MIME type is not permitted without python-magic")

        created_at = now.replace(microsecond=0).isoformat() + "Z"
        storage_path = (relative_dir / sanitized_filename).as_posix()
        metadata_path = (relative_dir / "metadata.json").as_posix()

        record_payload = {
            "upload_id": upload_id,
            "origin": origin,
            "intended_usage": intended_usage,
            "original_filename": original_filename,
            "stored_filename": sanitized_filename,
            "mime_type": final_mime,
            "detected_mime": detected_mime,
            "size_bytes": total_bytes,
            "declared_size": declared_size,
            "declared_mime": declared_mime,
            "sha256": sha256.hexdigest(),
            "original_url": source_url,
            "created_at": created_at,
            "hash_prefix1": prefix1,
            "hash_prefix2": prefix2,
            "user_hash": user_hash,
            "storage_path": storage_path,
            "metadata_path": metadata_path,
        }
        record_payload["metadata_signature"] = self._compute_metadata_signature(record_payload)
        record = AttachmentRecord(**record_payload)

        self._write_metadata(user_root / metadata_path, record)
        await self._update_index_async(user_root / "uploads" / "index.json", record, user_hash)

        logger.info("Stored attachment %s for hash %s", upload_id, user_hash)
        return record



    def _write_metadata(self, metadata_file: Path, record: AttachmentRecord) -> None:
        metadata_file.parent.mkdir(parents=True, exist_ok=True)
        with metadata_file.open("w", encoding="utf-8") as fh:
            json.dump(record.model_dump(), fh, ensure_ascii=True, indent=2)

    async def _get_index_lock(self, user_hash: str) -> asyncio.Lock:
        """Get or create a lock for a specific user's index file."""
        async with self._index_locks_lock:
            if user_hash not in self._index_locks:
                self._index_locks[user_hash] = asyncio.Lock()
            return self._index_locks[user_hash]

    async def _update_index_async(self, index_file: Path, record: AttachmentRecord, user_hash: str) -> None:
        """Update index file with proper locking and atomic write to prevent corruption."""
        lock = await self._get_index_lock(user_hash)
        async with lock:
            index_file.parent.mkdir(parents=True, exist_ok=True)
            entries: List[Dict[str, str]] = []
            if index_file.exists():
                try:
                    with index_file.open("r", encoding="utf-8") as fh:
                        entries = json.load(fh)
                except json.JSONDecodeError:
                    logger.warning("Attachment index %s is corrupted; rebuilding", index_file)
                    entries = []

            index_entry = {
                "upload_id": record.upload_id,
                "original_filename": record.original_filename,
                "stored_filename": record.stored_filename,
                "mime_type": record.mime_type,
                "size_bytes": record.size_bytes,
                "created_at": record.created_at,
                "storage_path": record.storage_path,
                "metadata_path": record.metadata_path,
            }

            entries = [entry for entry in entries if entry.get("upload_id") != record.upload_id]
            entries.insert(0, index_entry)
            entries = entries[: self.index_history_limit]

            # Atomic write: write to temp file, then rename
            temp_file = index_file.with_suffix(".tmp")
            try:
                with temp_file.open("w", encoding="utf-8") as fh:
                    json.dump(entries, fh, ensure_ascii=True, indent=2)
                temp_file.replace(index_file)
            except Exception:
                temp_file.unlink(missing_ok=True)
                raise

    def _update_index(self, index_file: Path, record: AttachmentRecord) -> None:
        """Synchronous index update - use _update_index_async when possible."""
        # Note: This sync version is kept for backward compatibility with cleanup
        # For new code, prefer _update_index_async which has proper locking
        index_file.parent.mkdir(parents=True, exist_ok=True)
        entries: List[Dict[str, str]] = []
        if index_file.exists():
            try:
                with index_file.open("r", encoding="utf-8") as fh:
                    entries = json.load(fh)
            except json.JSONDecodeError:
                logger.warning("Attachment index %s is corrupted; rebuilding", index_file)
                entries = []

        index_entry = {
            "upload_id": record.upload_id,
            "original_filename": record.original_filename,
            "stored_filename": record.stored_filename,
            "mime_type": record.mime_type,
            "size_bytes": record.size_bytes,
            "created_at": record.created_at,
            "storage_path": record.storage_path,
            "metadata_path": record.metadata_path,
        }

        entries = [entry for entry in entries if entry.get("upload_id") != record.upload_id]
        entries.insert(0, index_entry)
        entries = entries[: self.index_history_limit]

        # Atomic write: write to temp file, then rename
        temp_file = index_file.with_suffix(".tmp")
        try:
            with temp_file.open("w", encoding="utf-8") as fh:
                json.dump(entries, fh, ensure_ascii=True, indent=2)
            temp_file.replace(index_file)
        except Exception:
            temp_file.unlink(missing_ok=True)
            raise

    def _user_root(self, prefix1: str, prefix2: str, user_hash: str) -> Path:
        return self.base_path / prefix1 / prefix2 / user_hash

    def _is_text_like(self, record: AttachmentRecord) -> bool:
        """Return True when the attachment content is safe to surface as text."""
        if not record.mime_type:
            return False
        if record.mime_type.startswith("text/"):
            return True
        if record.mime_type in {"application/json"}:
            return True
        extension = Path(record.stored_filename).suffix.lower()
        return extension in {".json", ".txt", ".md"}

    def _validate_declared_size(self, *, actual: int, declared: Optional[int]) -> None:
        if declared is None:
            return
        if declared <= 0:
            raise AttachmentValidationError("Declared size must be a positive integer")
        if actual > declared * self.settings.max_compression_ratio:
            raise AttachmentValidationError("Attachment size exceeds allowed compression ratio")
        if actual != declared:
            logger.warning(
                "Declared size %s does not match actual bytes %s",
                declared,
                actual,
            )
            raise AttachmentValidationError("Attachment size mismatches declared Content-Length")

    def _detect_mime(self, sample: bytes, binary_path: Path) -> Optional[str]:
        heuristic = self._heuristic_mime(sample)
        if magic is None:
            return heuristic
        try:
            if sample:
                detected = magic.from_buffer(sample, mime=True)  # type: ignore[attr-defined]
                if detected:
                    return str(detected)
            detected = magic.from_file(str(binary_path), mime=True)  # type: ignore[attr-defined]
            if detected:
                return str(detected)
        except Exception as exc:  # pragma: no cover - libmagic failure
            logger.warning("python-magic failed to detect MIME for %s: %s", binary_path, exc)
        return heuristic

    def _heuristic_mime(self, sample: bytes) -> Optional[str]:
        if not sample:
            return None
        header = sample[:8]
        if header.startswith(b"%PDF-"):
            return "application/pdf"
        if header.startswith((b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")):
            return "application/zip"
        if header.startswith(b"\x1f\x8b\x08"):
            return "application/gzip"
        if header.startswith(b"\x89PNG"):
            return "image/png"
        if header.startswith(b"\xff\xd8"):
            return "image/jpeg"
        if self._looks_textual(sample):
            stripped = sample.lstrip()
            if stripped.startswith((b"{", b"[")):
                return "application/json"
            return "text/plain"
        return None

    def _looks_textual(self, sample: bytes) -> bool:
        if not sample:
            return False
        if b"\x00" in sample:
            return False
        try:
            sample.decode("utf-8")
        except UnicodeDecodeError:
            return False
        return True

    def _mime_equivalent(self, detected: str, declared: str) -> bool:
        detected = detected.lower()
        declared = declared.lower()
        if detected == declared:
            return True
        text_equivalents = {"text/plain", "text/markdown"}
        if {detected, declared} <= text_equivalents:
            return True
        return False

    def _compute_metadata_signature(self, payload: Dict[str, Any]) -> str:
        data = {k: v for k, v in payload.items() if k != "metadata_signature"}
        canonical = std_json.dumps(data, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _sanitize_filename(self, filename: str) -> str:
        normalized = unicodedata.normalize('NFC', filename or '')
        cleaned = ''.join(ch for ch in normalized if unicodedata.category(ch)[0] != 'C')
        candidate = Path(cleaned).name
        candidate = candidate.strip(' .')
        candidate = re.sub(r"[^A-Za-z0-9._-]", "_", candidate)
        candidate = candidate.lstrip('.')
        if not candidate:
            candidate = 'attachment'
        return candidate[:120]

    def _validate_extension(self, extension: str) -> None:
        normalized = (extension or "").lower()
        blocked = {ext.lower() for ext in self.settings.disallowed_extensions}
        if blocked and normalized in blocked:
            raise AttachmentValidationError(
                f"File extension '{normalized or 'unknown'}' is not permitted"
            )
        allowed = {ext.lower() for ext in self.settings.allowed_extensions}
        if allowed and normalized not in allowed:
            raise AttachmentValidationError(
                f"File extension '{normalized or 'unknown'}' is not permitted"
            )

    def _validate_mime(self, mime_type: str, *, source: str) -> None:
        if not mime_type:
            raise AttachmentValidationError(f"Unable to determine MIME type from {source}")
        normalized = mime_type.lower()
        blocked = {m.lower() for m in self.settings.disallowed_mime_types}
        if blocked and normalized in blocked:
            raise AttachmentValidationError(f"MIME type '{mime_type}' is not permitted ({source})")
        allowed = {m.lower() for m in self.settings.allowed_mime_types}
        if allowed and normalized not in allowed:
            raise AttachmentValidationError(f"MIME type '{mime_type}' is not permitted ({source})")

    def _safe_content_length(self, upload_file: UploadFile) -> Optional[int]:
        try:
            headers = upload_file.headers or {}
            content_length = headers.get("content-length") or headers.get("Content-Length")
            return int(content_length) if content_length is not None else None
        except (ValueError, TypeError):
            return None

    async def _iter_upload_file(self, upload_file: UploadFile) -> AsyncIterator[bytes]:
        while True:
            chunk = await upload_file.read(self.CHUNK_SIZE)
            if not chunk:
                break
            yield chunk

    async def _iter_bytes(self, data: bytes) -> AsyncIterator[bytes]:
        view = memoryview(data)
        for start in range(0, len(view), self.CHUNK_SIZE):
            chunk = view[start : start + self.CHUNK_SIZE]
            if chunk:
                yield bytes(chunk)

    async def _iter_http_response(self, response: httpx.Response) -> AsyncIterator[bytes]:
        min_rate = max(self.settings.url_min_bytes_per_second, 1)
        window_seconds = max(self.settings.url_min_speed_window_seconds, 1)
        window_start = time.monotonic()
        window_bytes = 0

        async for chunk in response.aiter_bytes(self.CHUNK_SIZE):
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

    def generate_user_hash(self, username: str) -> tuple[str, str, str]:
        """Reproduce SPARK's hashing scheme for per-user storage layout."""
        # FIXME: deduplicate with SPARK/common.py when projects converge
        data_to_hash = f"{username}{self.pepper}"
        hash_hex = hashlib.sha1(data_to_hash.encode("utf-8")).hexdigest()
        return hash_hex[:3], hash_hex[3:7], hash_hex

    @staticmethod
    def decode_base64_payload(
        payload: str,
        max_decoded_size: int = 10 * 1024 * 1024  # 10MB default, matches config default
    ) -> bytes:
        """Decode a base64 string raising an AttachmentValidationError if invalid.

        Security: Estimates decoded size BEFORE decoding to prevent memory exhaustion
        attacks via oversized payloads.
        """
        # Security: Estimate decoded size before allocating memory
        # Base64 encoding expands data by ~4/3, so decoded size is ~3/4 of encoded
        estimated_size = len(payload) * 3 // 4
        if estimated_size > max_decoded_size:
            raise AttachmentValidationError(
                f"Payload too large: estimated {estimated_size} bytes exceeds limit of {max_decoded_size} bytes"
            )
        try:
            return base64.b64decode(payload, validate=True)
        except (base64.binascii.Error, ValueError) as exc:
            raise AttachmentValidationError("Invalid base64 payload") from exc

    def _validate_url_structure(self, url: str):
        parsed = urlparse(url)
        scheme = parsed.scheme.lower()
        if scheme not in self._ALLOWED_PORTS:
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
        self._validate_hostname_format(hostname)
        if not self._hostname_allowed(hostname):
            raise AttachmentValidationError("URL hostname is not permitted")
        return parsed

    def _hostname_allowed(self, hostname: str) -> bool:
        host = hostname.lower()
        blocked = {entry.lower() for entry in self.settings.blocked_url_hostnames}
        if any(host == entry or host.endswith(f".{entry}") for entry in blocked):
            return False
        allowed = {entry.lower() for entry in self.settings.allowed_url_hostnames}
        if allowed and not any(host == entry or host.endswith(f".{entry}") for entry in allowed):
            return False
        return True

    def _validate_hostname_format(self, hostname: str) -> None:
        cleaned = hostname.strip()
        if hostname != cleaned:
            raise AttachmentValidationError("Hostname contains disallowed leading or trailing characters")
        if len(cleaned) > 253:
            raise AttachmentValidationError("Hostname exceeds maximum length")
        if any(ord(ch) < 32 for ch in cleaned):
            raise AttachmentValidationError("Hostname contains control characters")
        if cleaned.startswith('[') or cleaned.endswith(']'):
            raise AttachmentValidationError("IPv6 literals are not permitted for attachments")
        if '@' in cleaned:
            raise AttachmentValidationError("Hostname must not include '@'")
        if '..' in cleaned:
            raise AttachmentValidationError("Hostname contains empty labels")
        try:
            cleaned.encode('ascii')
        except UnicodeEncodeError as exc:
            raise AttachmentValidationError("Hostname must be ASCII") from exc
        if set(cleaned) <= set('0123456789.'):
            try:
                ip_obj = ipaddress.ip_address(cleaned)
            except ValueError as exc:
                raise AttachmentValidationError("Hostname must be a valid IPv4 address") from exc
            if not isinstance(ip_obj, ipaddress.IPv4Address):
                raise AttachmentValidationError("IPv6 addresses are not permitted for attachments")
            if cleaned != str(ip_obj):
                raise AttachmentValidationError("IPv4 address must use canonical dotted-decimal notation")
            return
        labels = cleaned.split('.')
        for label in labels:
            if not label:
                raise AttachmentValidationError("Hostname contains empty labels")
            if label.startswith('-') or label.endswith('-'):
                raise AttachmentValidationError("Hostname labels must not start or end with '-'")
            lower = label.lower()
            for ch in lower:
                if ch not in 'abcdefghijklmnopqrstuvwxyz0123456789-':
                    raise AttachmentValidationError("Hostname contains invalid characters")

    def _determine_port(self, scheme: str, port: Optional[int]) -> int:
        expected = self._ALLOWED_PORTS[scheme]
        if port is None:
            return expected
        if port != expected:
            raise AttachmentValidationError("URL port is not permitted for attachments")
        return port

    async def _resolve_and_validate_host(self, hostname: str, port: int) -> List[str]:
        self._validate_hostname_format(hostname)
        loop = asyncio.get_running_loop()
        try:
            addr_info = await loop.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
        except socket.gaierror as exc:
            raise AttachmentValidationError("Unable to resolve URL hostname") from exc
        addresses: List[str] = []
        seen: Set[str] = set()
        for family, _, _, _, sockaddr in addr_info:
            ip_literal = sockaddr[0]
            if ip_literal in seen:
                continue
            try:
                ip_obj = ipaddress.ip_address(ip_literal)
            except ValueError as exc:
                raise AttachmentValidationError("Resolved address is invalid") from exc
            self._validate_ip_address(ip_obj)
            if isinstance(ip_obj, ipaddress.IPv6Address):
                raise AttachmentValidationError("IPv6 addresses are not permitted for attachments")
            addresses.append(str(ip_obj))
            seen.add(ip_literal)
        if not addresses:
            raise AttachmentValidationError("Unable to resolve URL hostname to permitted addresses")
        return addresses

    def _validate_ip_address(self, ip_obj: ipaddress._BaseAddress) -> None:
        if not getattr(ip_obj, 'is_global', False):
            raise AttachmentValidationError("Resolved address is not routable on the public internet")
        for network in self._denied_networks:
            if ip_obj in network:
                raise AttachmentValidationError("Resolved address is not permitted")

    def _build_denied_networks(self) -> Tuple[ipaddress._BaseNetwork, ...]:
        networks = [
            ipaddress.ip_network('0.0.0.0/8'),
            ipaddress.ip_network('10.0.0.0/8'),
            ipaddress.ip_network('100.64.0.0/10'),
            ipaddress.ip_network('127.0.0.0/8'),
            ipaddress.ip_network('169.254.0.0/16'),
            ipaddress.ip_network('169.254.169.254/32'),
            ipaddress.ip_network('169.254.169.253/32'),
            ipaddress.ip_network('169.254.170.2/32'),
            ipaddress.ip_network('172.16.0.0/12'),
            ipaddress.ip_network('192.0.0.0/24'),
            ipaddress.ip_network('192.0.2.0/24'),
            ipaddress.ip_network('192.168.0.0/16'),
            ipaddress.ip_network('198.18.0.0/15'),
            ipaddress.ip_network('198.51.100.0/24'),
            ipaddress.ip_network('203.0.113.0/24'),
            ipaddress.ip_network('224.0.0.0/4'),
            ipaddress.ip_network('240.0.0.0/4'),
            ipaddress.ip_network('255.255.255.255/32'),
            ipaddress.ip_network('100.100.100.200/32'),
            ipaddress.ip_network('::/128'),
            ipaddress.ip_network('::1/128'),
            ipaddress.ip_network('fe80::/10'),
            ipaddress.ip_network('fc00::/7'),
            ipaddress.ip_network('ff00::/8'),
            ipaddress.ip_network('2001:db8::/32'),
        ]
        return tuple(networks)

    def _extract_declared_size(self, response: httpx.Response) -> Optional[int]:
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

    def _extract_declared_mime(self, response: httpx.Response) -> Optional[str]:
        header = response.headers.get("Content-Type")
        if not header:
            return None
        mime = header.split(";", 1)[0].strip().lower()
        return mime or None

    def _resolve_filename_from_response(
        self,
        parsed,
        response: httpx.Response,
        declared_mime: Optional[str],
    ) -> str:
        disposition = response.headers.get("Content-Disposition")
        filename = self._filename_from_content_disposition(disposition) if disposition else None
        if not filename:
            filename = Path(parsed.path).name or "download"
        filename = filename.strip() or "download"
        filename = Path(filename).name or "download"
        if not Path(filename).suffix and declared_mime:
            guessed = mimetypes.guess_extension(declared_mime)
            if guessed:
                filename = f"{filename}{guessed}"
        return filename

    def _filename_from_content_disposition(self, header: str) -> Optional[str]:
        parts = header.split(";")
        for part in parts[1:]:
            name, _, value = part.strip().partition("=")
            if not _:
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

    async def _get_cached_url_record(self, *, username: str, url: str) -> Optional[AttachmentRecord]:
        if self.settings.url_cache_ttl_seconds <= 0:
            return None
        key = (username, url)
        async with self._url_cache_lock:
            entry = self._recent_url_cache.get(key)
            if not entry:
                return None
            upload_id, cached_hash, expires_at = entry
            if expires_at <= time.time():
                self._recent_url_cache.pop(key, None)
                return None

        try:
            record = self.get_metadata(username=username, upload_id=upload_id)
        except AttachmentError:
            await self._prune_cache_entry(key)
            return None

        if not secrets.compare_digest(record.sha256, cached_hash):
            await self._prune_cache_entry(key)
            return None
        return record

    async def _update_url_cache(self, *, username: str, url: str, record: AttachmentRecord) -> None:
        if self.settings.url_cache_ttl_seconds <= 0:
            return
        key = (username, url)
        expires_at = time.time() + self.settings.url_cache_ttl_seconds
        async with self._url_cache_lock:
            self._recent_url_cache[key] = (record.upload_id, record.sha256, expires_at)

    async def _prune_cache_entry(self, key: Tuple[str, str]) -> None:
        async with self._url_cache_lock:
            self._recent_url_cache.pop(key, None)
