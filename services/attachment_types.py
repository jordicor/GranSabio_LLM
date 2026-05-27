"""Public attachment models and exceptions."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


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
    storage_path: str = Field(..., description="Logical relative path to the stored attachment")
    metadata_path: str = Field(..., description="Logical relative path to the metadata record")


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
    attachments_scanned: int = 0
    attachments_removed: int = 0
    retention_expired: int = 0
    issues: List[str] = field(default_factory=list)
    actions: List[CleanupAction] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dry_run": self.dry_run,
            "attachments_scanned": self.attachments_scanned,
            "attachments_removed": self.attachments_removed,
            "retention_expired": self.retention_expired,
            "issues": list(self.issues),
            "actions": [action.__dict__ for action in self.actions],
        }
