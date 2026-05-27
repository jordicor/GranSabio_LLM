"""Attachment input, filename, MIME, and kind validation helpers."""

from __future__ import annotations

import base64
import mimetypes
import re
import unicodedata
from pathlib import Path
from typing import Any, Optional

from fastapi import UploadFile

from services.attachment_types import (
    AttachmentRecord,
    AttachmentValidationError,
)

SAFE_MIME_WHEN_MAGIC_MISSING = frozenset(
    {
        "text/plain",
        "text/markdown",
        "text/csv",
        "application/json",
        "application/pdf",
        "image/jpeg",
        "image/png",
        "image/gif",
        "image/webp",
    }
)


def is_text_like(record: AttachmentRecord) -> bool:
    """Return True when the attachment content is safe to surface as text."""

    if not record.mime_type:
        return False
    if record.mime_type.startswith("text/"):
        return True
    if record.mime_type in {"application/json"}:
        return True
    extension = Path(record.stored_filename).suffix.lower()
    return extension in {".json", ".txt", ".md"}


def is_image(record: AttachmentRecord) -> bool:
    """Return True when the attachment is an image."""

    if not record.mime_type:
        return False
    return record.mime_type.startswith("image/")


def attachment_kind(*, final_mime: str, stored_filename: str) -> str:
    """Classify an allowed attachment into a physical dedupe namespace."""

    normalized_mime = (final_mime or "").lower()
    extension = Path(stored_filename).suffix.lower()
    if normalized_mime.startswith("image/"):
        return "image"
    if normalized_mime == "application/pdf":
        return "pdf"
    if normalized_mime == "application/json" or extension == ".json":
        return "json"
    if normalized_mime.startswith("text/") or extension in {".txt", ".md", ".csv"}:
        return "text"
    raise AttachmentValidationError("Attachment MIME type is not supported by dedupe storage")


def validate_declared_size(*, actual: int, declared: Optional[int], settings: Any) -> None:
    if declared is None:
        return
    if declared <= 0:
        raise AttachmentValidationError("Declared size must be a positive integer")
    if actual > declared * settings.max_compression_ratio:
        raise AttachmentValidationError("Attachment size exceeds allowed compression ratio")
    if actual != declared:
        raise AttachmentValidationError("Attachment size mismatches declared Content-Length")


def detect_mime(
    sample: bytes,
    binary_path: Path,
    *,
    magic_module: Any,
    logger: Any,
) -> Optional[str]:
    heuristic = heuristic_mime(sample)
    if magic_module is None:
        return heuristic
    try:
        if sample:
            detected = magic_module.from_buffer(sample, mime=True)
            if detected:
                return str(detected)
        detected = magic_module.from_file(str(binary_path), mime=True)
        if detected:
            return str(detected)
    except Exception as exc:  # pragma: no cover - libmagic failure
        logger.warning("python-magic failed to detect MIME for %s: %s", binary_path, exc)
    return heuristic


def heuristic_mime(sample: bytes) -> Optional[str]:
    if not sample:
        return None
    header = sample[:12]
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
    if header.startswith(b"GIF87a") or header.startswith(b"GIF89a"):
        return "image/gif"
    if header[:4] == b"RIFF" and len(header) >= 12 and header[8:12] == b"WEBP":
        return "image/webp"
    if looks_textual(sample):
        stripped = sample.lstrip()
        if stripped.startswith((b"{", b"[")):
            return "application/json"
        return "text/plain"
    return None


def looks_textual(sample: bytes) -> bool:
    if not sample:
        return False
    if b"\x00" in sample:
        return False
    try:
        sample.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def mime_equivalent(detected: str, declared: str) -> bool:
    detected = detected.lower()
    declared = declared.lower()
    if detected == declared:
        return True
    text_equivalents = {"text/plain", "text/markdown"}
    if {detected, declared} <= text_equivalents:
        return True
    return False


def sanitize_filename(filename: str) -> str:
    normalized = unicodedata.normalize("NFC", filename or "")
    cleaned = "".join(ch for ch in normalized if unicodedata.category(ch)[0] != "C")
    candidate = Path(cleaned).name
    candidate = candidate.strip(" .")
    candidate = re.sub(r"[^A-Za-z0-9._-]", "_", candidate)
    candidate = candidate.lstrip(".")
    if not candidate:
        candidate = "attachment"
    if candidate.lower() == "metadata.json":
        candidate = f"attachment_{candidate}"
    return candidate[:120]


def validate_extension(extension: str, *, settings: Any) -> None:
    normalized = (extension or "").lower()
    blocked = {ext.lower() for ext in settings.disallowed_extensions}
    if blocked and normalized in blocked:
        raise AttachmentValidationError(
            f"File extension '{normalized or 'unknown'}' is not permitted"
        )
    allowed = {ext.lower() for ext in settings.allowed_extensions}
    if allowed and normalized not in allowed:
        raise AttachmentValidationError(
            f"File extension '{normalized or 'unknown'}' is not permitted"
        )


def validate_mime(mime_type: str, *, source: str, settings: Any) -> None:
    if not mime_type:
        raise AttachmentValidationError(f"Unable to determine MIME type from {source}")
    normalized = mime_type.lower()
    blocked = {m.lower() for m in settings.disallowed_mime_types}
    if blocked and normalized in blocked:
        raise AttachmentValidationError(f"MIME type '{mime_type}' is not permitted ({source})")
    allowed = {m.lower() for m in settings.allowed_mime_types}
    if allowed and normalized not in allowed:
        raise AttachmentValidationError(f"MIME type '{mime_type}' is not permitted ({source})")


def safe_content_length(upload_file: UploadFile) -> Optional[int]:
    try:
        headers = upload_file.headers or {}
        content_length = headers.get("content-length") or headers.get("Content-Length")
        return int(content_length) if content_length is not None else None
    except (ValueError, TypeError):
        return None


def decode_base64_payload(
    payload: str,
    max_decoded_size: int = 10 * 1024 * 1024,
) -> bytes:
    """Decode a base64 string after estimating decoded size."""

    estimated_size = len(payload) * 3 // 4
    if estimated_size > max_decoded_size:
        raise AttachmentValidationError(
            f"Payload too large: estimated {estimated_size} bytes exceeds limit of {max_decoded_size} bytes"
        )
    try:
        return base64.b64decode(payload, validate=True)
    except (base64.binascii.Error, ValueError) as exc:
        raise AttachmentValidationError("Invalid base64 payload") from exc


def guess_mime_from_filename(filename: str) -> str:
    return mimetypes.guess_type(filename)[0] or "application/octet-stream"
