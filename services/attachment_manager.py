"""Attachment management utilities for Gran Sabio LLM."""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json as std_json
import logging
import mimetypes
import secrets
import tempfile
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import httpx
from fastapi import UploadFile

from config import AttachmentSettings
from services.attachment_store import (
    AttachmentStore,
    AttachmentStoreConflictError,
    AttachmentStoreError,
    AttachmentStoreIntegrityError,
    AttachmentUploadRow,
)
from services.attachment_types import (
    AttachmentError,
    AttachmentNotFoundError,
    AttachmentRecord,
    AttachmentTooLargeError,
    AttachmentValidationError,
    CleanupAction,
    CleanupReport,
    ResolvedAttachment,
)
from services import attachment_validation
from services.attachment_url_fetcher import AttachmentUrlFetcher

try:
    import magic  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    magic = None  # type: ignore[var-annotated]

logger = logging.getLogger(__name__)


class AttachmentManager:
    """Persist attachments with hashed per-user storage layout and metadata."""

    CHUNK_SIZE = 1024 * 1024  # 1 MB per chunk when streaming uploads
    _REDIRECT_STATUS_CODES = {301, 302, 303, 307, 308}
    _ALLOWED_PORTS: Dict[str, int] = {"http": 80, "https": 443}
    _MAX_REDIRECT_LIMIT = 35

    def __init__(self, *, settings: AttachmentSettings, pepper: str) -> None:
        if not pepper:
            raise ValueError("PEPPER must be configured before using AttachmentManager")

        self.settings = settings
        self.store = AttachmentStore(settings=settings)
        self.store.ensure_schema()
        self.pepper = pepper
        self._recent_url_cache: Dict[Tuple[str, str], Tuple[str, str, float]] = {}
        self._url_cache_lock = asyncio.Lock()
        self.url_fetcher = AttachmentUrlFetcher(
            settings=settings,
            chunk_size=self.CHUNK_SIZE,
            logger=logger,
        )
        self._denied_networks = self.url_fetcher.denied_networks
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
        return self.url_fetcher.build_pinned_request(
            client,
            method,
            url,
            hostname=hostname,
            ip_address=ip_address,
            headers=headers,
        )

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
        return await self.url_fetcher.send_pinned_request(
            client,
            url=url,
            hostname=hostname,
            port=port,
            headers=headers,
        )

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

        self._validate_url_structure(normalized_url)

        cached = await self._get_cached_url_record(username=username, url=normalized_url)
        if cached:
            logger.info("Using cached attachment %s for URL %s", cached.upload_id, normalized_url)
            return cached

        async with self.url_fetcher.fetch(normalized_url) as fetched:
            declared_size = fetched.declared_size
            declared_mime = fetched.declared_mime
            filename = fetched.filename
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
                if (
                    not resolved_mime
                    or resolved_mime == "application/octet-stream"
                    or resolved_mime not in attachment_validation.SAFE_MIME_WHEN_MAGIC_MISSING
                ):
                    raise AttachmentValidationError("MIME type is not permitted without python-magic")

            record = await self._persist_content(
                username=username,
                sanitized_filename=sanitized_filename,
                original_filename=filename,
                intended_usage=intended_usage,
                origin="url",
                resolved_mime=resolved_mime,
                declared_mime=declared_mime,
                declared_size=declared_size,
                data_stream=fetched.data_stream,
                source_url=fetched.url,
            )

        await self._update_url_cache(username=username, url=normalized_url, record=record)
        return record

    def get_metadata(self, *, username: str, upload_id: str) -> AttachmentRecord:
        """Retrieve stored metadata for a given user/upload combination."""
        if not username:
            raise AttachmentValidationError("username is required to fetch attachment metadata")
        if not upload_id:
            raise AttachmentValidationError("upload_id is required to fetch attachment metadata")

        _prefix1, _prefix2, user_hash = self.generate_user_hash(username)
        if self.store.deletion_exists(user_hash=user_hash, upload_id=upload_id):
            raise AttachmentNotFoundError("Attachment metadata not found")

        row = self.store.get_upload(user_hash=user_hash, upload_id=upload_id)
        if row is None:
            raise AttachmentNotFoundError("Attachment metadata not found")
        if row.status != "active" or row.blob.status != "ready":
            raise AttachmentNotFoundError("Attachment is not available")
        return self._record_from_store_row(row)

    def resolve_attachment(
        self,
        *,
        username: str,
        upload_id: str,
    ) -> ResolvedAttachment:
        """Resolve metadata into absolute paths ensuring resources exist."""
        _prefix1, _prefix2, user_hash = self.generate_user_hash(username)
        if self.store.deletion_exists(user_hash=user_hash, upload_id=upload_id):
            raise AttachmentNotFoundError("Attachment metadata not found")

        row = self.store.get_upload(user_hash=user_hash, upload_id=upload_id)
        if row is None:
            raise AttachmentNotFoundError("Attachment metadata not found")
        if row.status != "active" or row.blob.status != "ready":
            raise AttachmentNotFoundError("Attachment is not available")

        record = self._record_from_store_row(row)
        binary_path = self.store.blob_path(row.blob.storage_key)
        if not binary_path.exists():
            raise AttachmentNotFoundError("Attachment binary is missing from storage")
        metadata_path = self.store.db_path
        if not metadata_path.exists():
            raise AttachmentNotFoundError("Attachment metadata database is missing from storage")

        return ResolvedAttachment(
            username=username,
            record=record,
            binary_path=binary_path,
            metadata_path=metadata_path,
        )

    def delete_attachment(self, *, username: str, upload_id: str, reason: str = "manual delete") -> Dict[str, Any]:
        """Delete a logical attachment without directly unlinking shared DB blobs."""
        if not username:
            raise AttachmentValidationError("username is required to delete attachment")
        if not upload_id:
            raise AttachmentValidationError("upload_id is required to delete attachment")

        _prefix1, _prefix2, user_hash = self.generate_user_hash(username)
        if self.store.deletion_exists(user_hash=user_hash, upload_id=upload_id):
            raise AttachmentNotFoundError("Attachment metadata not found")
        row = self.store.get_upload(user_hash=user_hash, upload_id=upload_id)
        if row is None or row.status != "active":
            raise AttachmentNotFoundError("Attachment metadata not found")
        deleted = self.store.delete_upload(user_hash=user_hash, upload_id=upload_id, reason=reason)
        if not deleted:
            raise AttachmentNotFoundError("Attachment metadata not found")
        return {
            "db_backed": True,
            "removed_files": 0,
            "blob_deleted": False,
        }

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

    def _record_from_store_row(self, row: AttachmentUploadRow) -> AttachmentRecord:
        """Convert a DB-backed upload row to the public attachment metadata schema."""
        storage_path = f"dedupe/{row.upload_id}/{row.stored_filename}"
        metadata_path = f"dedupe/{row.upload_id}/metadata.json"
        return AttachmentRecord(
            upload_id=row.upload_id,
            origin=row.origin,
            intended_usage=row.intended_usage,
            original_filename=row.original_filename,
            stored_filename=row.stored_filename,
            mime_type=row.mime_type,
            size_bytes=row.blob.size_bytes,
            declared_size=row.declared_size,
            declared_mime=row.declared_mime,
            detected_mime=row.detected_mime,
            sha256=row.blob.sha256,
            metadata_signature=row.metadata_signature,
            original_url=row.original_url,
            created_at=row.created_at,
            hash_prefix1=row.hash_prefix1,
            hash_prefix2=row.hash_prefix2,
            user_hash=row.user_hash,
            storage_path=storage_path,
            metadata_path=metadata_path,
        )

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

    def get_image_dimensions(self, resolved: ResolvedAttachment) -> Optional[Tuple[int, int]]:
        """
        Get image dimensions (width, height) using PIL/Pillow.

        Returns None if dimensions cannot be determined or attachment is not an image.
        """
        if not self._is_image(resolved.record):
            return None

        try:
            from PIL import Image
            with Image.open(resolved.binary_path) as img:
                return img.size  # Returns (width, height)
        except ImportError:
            logger.warning("Pillow not installed, cannot get image dimensions")
            return None
        except Exception as exc:
            logger.warning("Failed to get dimensions for %s: %s", resolved.binary_path, exc)
            return None

    def resize_image_if_needed(
        self,
        resolved: ResolvedAttachment,
        max_edge: Optional[int] = None,
        max_size_bytes: Optional[int] = None,
    ) -> Tuple[bytes, str]:
        """
        Resize image if it exceeds limits. Returns (image_bytes, mime_type).

        May convert format (e.g., HEIC -> JPEG) for broader API compatibility.
        Uses config.IMAGE settings for defaults.
        Respects config.IMAGE.auto_resize setting.

        Raises:
            AttachmentError: If Pillow is not installed.
        """
        try:
            from PIL import Image
        except ImportError as exc:
            raise AttachmentError(
                "Pillow is required for image processing. Install with: pip install Pillow"
            ) from exc
        import io

        from config import config

        # Check if auto_resize is disabled
        if not config.IMAGE.auto_resize:
            # Auto-resize disabled: return original bytes with detected MIME type
            # Still convert HEIC/HEIF to JPEG for API compatibility
            mime_type = resolved.record.mime_type
            if mime_type in ("image/heic", "image/heif"):
                with Image.open(resolved.binary_path) as img:
                    if img.mode in ("RGBA", "LA", "P"):
                        img = img.convert("RGB")
                    buffer = io.BytesIO()
                    img.save(buffer, format="JPEG", quality=95)
                    return buffer.getvalue(), "image/jpeg"
            # Return original bytes for all other formats
            with open(resolved.binary_path, "rb") as f:
                return f.read(), mime_type

        if max_edge is None:
            max_edge = config.IMAGE.optimal_max_edge
        if max_size_bytes is None:
            max_size_bytes = config.IMAGE.max_image_size_bytes

        with Image.open(resolved.binary_path) as img:
            # Determine output format based on input MIME type
            output_format = "JPEG"
            mime_type = "image/jpeg"

            if resolved.record.mime_type == "image/png":
                output_format = "PNG"
                mime_type = "image/png"
            elif resolved.record.mime_type == "image/webp":
                output_format = "WEBP"
                mime_type = "image/webp"
            elif resolved.record.mime_type == "image/gif":
                # GIF: use first frame, convert to PNG for quality
                output_format = "PNG"
                mime_type = "image/png"
            # HEIC/HEIF -> JPEG for broad API compatibility

            # Check if resize is needed
            width, height = img.size
            needs_resize = max(width, height) > max_edge

            if needs_resize:
                ratio = max_edge / max(width, height)
                new_size = (int(width * ratio), int(height * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)

            # Prepare for save - JPEG doesn't support alpha channel
            if output_format == "JPEG":
                if img.mode in ("RGBA", "LA", "P"):
                    img = img.convert("RGB")

            # Save to buffer with initial quality
            buffer = io.BytesIO()
            save_kwargs: Dict[str, Any] = {"format": output_format}
            if output_format in ("JPEG", "WEBP"):
                save_kwargs["quality"] = 85

            img.save(buffer, **save_kwargs)
            image_bytes = buffer.getvalue()

            # Reduce quality if still exceeds size limit (JPEG/WEBP only)
            if len(image_bytes) > max_size_bytes and output_format in ("JPEG", "WEBP"):
                for quality in [70, 55, 40, 25]:
                    buffer = io.BytesIO()
                    img.save(buffer, format=output_format, quality=quality)
                    image_bytes = buffer.getvalue()
                    if len(image_bytes) <= max_size_bytes:
                        break

            return image_bytes, mime_type

    def run_cleanup(
        self,
        *,
        dry_run: bool = True,
        username: Optional[str] = None,
        user_hash: Optional[str] = None,
        retention_days: Optional[int] = None,
    ) -> CleanupReport:
        """Prune DB-backed attachment rows and optionally collect unused blobs."""
        if username and user_hash:
            raise AttachmentValidationError("Specify either username or user_hash, not both")

        target_hash: Optional[str] = None
        if username:
            _prefix1, _prefix2, target_hash = self.generate_user_hash(username)
        elif user_hash:
            normalized = user_hash.strip().lower()
            if len(normalized) != 40 or any(ch not in "0123456789abcdef" for ch in normalized):
                raise AttachmentValidationError("user_hash must be a full 40-character SHA1 hex digest")
            target_hash = normalized

        report = CleanupReport(dry_run=dry_run)
        effective_retention_days = retention_days if retention_days is not None else self.settings.retention_days
        if effective_retention_days and effective_retention_days > 0:
            cutoff = datetime.utcnow() - timedelta(days=effective_retention_days)
            cutoff_text = cutoff.replace(microsecond=0).isoformat() + "Z"
            expired = self.store.expire_uploads(
                cutoff_created_at=cutoff_text,
                reason=f"retention expired after {effective_retention_days} days",
                dry_run=dry_run,
                user_hash=target_hash,
            )
            report.attachments_scanned += expired
            report.attachments_removed += expired
            report.retention_expired += expired
            if expired:
                action = "Would expire" if dry_run else "Expired"
                report.actions.append(
                    CleanupAction(
                        "expire_uploads",
                        "attachment_uploads",
                        f"{action} {expired} DB upload(s) older than {effective_retention_days} days",
                    )
                )

        gc_dry_run = dry_run or not self.settings.blob_gc_enabled
        gc_result = self.store.gc_unreferenced_blobs(dry_run=gc_dry_run)
        if gc_result["scanned"]:
            if not dry_run and not self.settings.blob_gc_enabled:
                report.issues.append("Blob garbage collection is disabled; unreferenced blobs were left in place")
            action = "Would remove" if gc_dry_run else "Removed"
            detail = (
                f"{action} {gc_result['removed']} unreferenced blob(s); "
                f"scanned {gc_result['scanned']}, quarantined {gc_result['quarantined']}"
            )
            report.actions.append(
                CleanupAction("garbage_collect_blobs", str(self.store.blob_base_path), detail)
            )

        return report

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
        return await self._persist_content_dedupe(
            username=username,
            sanitized_filename=sanitized_filename,
            original_filename=original_filename,
            intended_usage=intended_usage,
            origin=origin,
            resolved_mime=resolved_mime,
            declared_mime=declared_mime,
            declared_size=declared_size,
            data_stream=data_stream,
            source_url=source_url,
        )

    async def _persist_content_dedupe(
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
        now = datetime.utcnow()
        upload_id = uuid.uuid4().hex
        created_at = now.replace(microsecond=0).isoformat() + "Z"

        temp_dir = self.store.blob_base_path / "_tmp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_handle = tempfile.NamedTemporaryFile(
            mode="wb",
            prefix="upload-",
            suffix=".tmp",
            dir=temp_dir,
            delete=False,
        )
        temp_path = Path(temp_handle.name)
        sha256 = hashlib.sha256()
        total_bytes = 0
        sample = bytearray()

        try:
            with temp_handle as destination:
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
            temp_path.unlink(missing_ok=True)
            raise
        except Exception as exc:  # pragma: no cover - unexpected I/O failure
            temp_path.unlink(missing_ok=True)
            raise AttachmentError("Unable to persist attachment contents") from exc

        try:
            self._validate_declared_size(actual=total_bytes, declared=declared_size)
        except AttachmentValidationError:
            temp_path.unlink(missing_ok=True)
            raise

        detected_mime = self._detect_mime(bytes(sample), temp_path)
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
            temp_path.unlink(missing_ok=True)
            raise

        if magic is None:
            safe_mime = (final_mime or "").lower()
            if safe_mime not in attachment_validation.SAFE_MIME_WHEN_MAGIC_MISSING:
                temp_path.unlink(missing_ok=True)
                raise AttachmentValidationError("MIME type is not permitted without python-magic")

        blob_kind = self._attachment_kind(final_mime=final_mime, stored_filename=sanitized_filename)
        public_storage_path = f"dedupe/{upload_id}/{sanitized_filename}"
        public_metadata_path = f"dedupe/{upload_id}/metadata.json"

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
            "storage_path": public_storage_path,
            "metadata_path": public_metadata_path,
        }
        record_payload["metadata_signature"] = self._compute_metadata_signature(record_payload)

        try:
            row = self.store.create_upload_from_temp(
                upload_id=upload_id,
                user_hash=user_hash,
                hash_prefix1=prefix1,
                hash_prefix2=prefix2,
                origin=origin,
                intended_usage=intended_usage,
                original_filename=original_filename,
                stored_filename=sanitized_filename,
                mime_type=final_mime,
                declared_size=declared_size,
                declared_mime=declared_mime,
                detected_mime=detected_mime,
                original_url=source_url,
                sha256=record_payload["sha256"],
                size_bytes=total_bytes,
                kind=blob_kind,
                temp_path=temp_path,
                metadata_signature=record_payload["metadata_signature"],
                created_at=created_at,
            )
        except AttachmentStoreConflictError as exc:
            raise AttachmentValidationError("Attachment upload could not be registered") from exc
        except (AttachmentStoreIntegrityError, AttachmentStoreError) as exc:
            raise AttachmentError("Unable to persist attachment contents") from exc

        record = self._record_from_store_row(row)

        logger.info("Stored deduplicated attachment %s for hash %s", upload_id, user_hash)
        return record



    def _is_text_like(self, record: AttachmentRecord) -> bool:
        """Return True when the attachment content is safe to surface as text."""
        return attachment_validation.is_text_like(record)

    def _is_image(self, record: AttachmentRecord) -> bool:
        """Return True when the attachment is an image."""
        return attachment_validation.is_image(record)

    def is_image(self, record: AttachmentRecord) -> bool:
        """Public wrapper for image checks used by generation input resolution."""
        return self._is_image(record)

    def _attachment_kind(self, *, final_mime: str, stored_filename: str) -> str:
        """Classify an allowed attachment into a physical dedupe namespace."""
        return attachment_validation.attachment_kind(
            final_mime=final_mime,
            stored_filename=stored_filename,
        )

    def _validate_declared_size(self, *, actual: int, declared: Optional[int]) -> None:
        if declared is not None and actual != declared and actual <= declared * self.settings.max_compression_ratio:
            logger.warning(
                "Declared size %s does not match actual bytes %s",
                declared,
                actual,
            )
        attachment_validation.validate_declared_size(
            actual=actual,
            declared=declared,
            settings=self.settings,
        )

    def _detect_mime(self, sample: bytes, binary_path: Path) -> Optional[str]:
        return attachment_validation.detect_mime(
            sample,
            binary_path,
            magic_module=magic,
            logger=logger,
        )

    def _heuristic_mime(self, sample: bytes) -> Optional[str]:
        return attachment_validation.heuristic_mime(sample)

    def _looks_textual(self, sample: bytes) -> bool:
        return attachment_validation.looks_textual(sample)

    def _mime_equivalent(self, detected: str, declared: str) -> bool:
        return attachment_validation.mime_equivalent(detected, declared)

    def _compute_metadata_signature(self, payload: Dict[str, Any]) -> str:
        data = {k: v for k, v in payload.items() if k != "metadata_signature"}
        canonical = std_json.dumps(data, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _sanitize_filename(self, filename: str) -> str:
        return attachment_validation.sanitize_filename(filename)

    def _validate_extension(self, extension: str) -> None:
        attachment_validation.validate_extension(extension, settings=self.settings)

    def _validate_mime(self, mime_type: str, *, source: str) -> None:
        attachment_validation.validate_mime(mime_type, source=source, settings=self.settings)

    def _safe_content_length(self, upload_file: UploadFile) -> Optional[int]:
        return attachment_validation.safe_content_length(upload_file)

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
        async for chunk in self.url_fetcher.iter_http_response(response):
            yield chunk

    def generate_user_hash(self, username: str) -> tuple[str, str, str]:
        """Hash-based per-user storage layout."""
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
        return attachment_validation.decode_base64_payload(
            payload,
            max_decoded_size=max_decoded_size,
        )

    def _validate_url_structure(self, url: str):
        return self.url_fetcher.validate_url_structure(url)

    def _hostname_allowed(self, hostname: str) -> bool:
        return self.url_fetcher.hostname_allowed(hostname)

    def _validate_hostname_format(self, hostname: str) -> None:
        self.url_fetcher.validate_hostname_format(hostname)

    def _determine_port(self, scheme: str, port: Optional[int]) -> int:
        return self.url_fetcher.determine_port(scheme, port)

    async def _resolve_and_validate_host(self, hostname: str, port: int) -> List[str]:
        return await self.url_fetcher.resolve_and_validate_host(hostname, port)

    def _validate_ip_address(self, ip_obj: ipaddress._BaseAddress) -> None:
        self.url_fetcher.validate_ip_address(ip_obj)

    def _build_denied_networks(self) -> Tuple[ipaddress._BaseNetwork, ...]:
        return self.url_fetcher.build_denied_networks()

    def _extract_declared_size(self, response: httpx.Response) -> Optional[int]:
        return self.url_fetcher.extract_declared_size(response)

    def _extract_declared_mime(self, response: httpx.Response) -> Optional[str]:
        return self.url_fetcher.extract_declared_mime(response)

    def _resolve_filename_from_response(
        self,
        parsed,
        response: httpx.Response,
        declared_mime: Optional[str],
    ) -> str:
        return self.url_fetcher.resolve_filename_from_response(parsed, response, declared_mime)

    def _filename_from_content_disposition(self, header: str) -> Optional[str]:
        return self.url_fetcher.filename_from_content_disposition(header)

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
