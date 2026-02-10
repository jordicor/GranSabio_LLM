"""FastAPI router handling attachment ingestion and metadata access."""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from time import monotonic
from typing import Deque, Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile, status
from pydantic import BaseModel

from config import config
from core.security import get_client_ip
from services.attachment_manager import (
    AttachmentManager,
    AttachmentNotFoundError,
    AttachmentRecord,
    AttachmentTooLargeError,
    AttachmentValidationError,
)

logger = logging.getLogger(__name__)


class SimpleRateLimiter:
    """Naive in-memory sliding window limiter for attachment ingestion.

    Supports dual-key rate limiting: a stricter limit per IP and a secondary
    limit per username. This prevents both IP-based abuse and username spoofing.
    """

    def __init__(
        self,
        *,
        limit: int,
        window_seconds: int,
        ip_limit: Optional[int] = None,
        ip_window_seconds: Optional[int] = None,
    ) -> None:
        # Per-username limit (legacy behavior)
        self.limit = limit
        self.window_seconds = max(window_seconds, 1)
        # Per-IP limit (stricter, prevents abuse from single source)
        # Default: 2x the per-user limit to allow multiple users behind NAT
        self.ip_limit = ip_limit if ip_limit is not None else (limit * 2 if limit > 0 else 0)
        self.ip_window_seconds = ip_window_seconds if ip_window_seconds is not None else self.window_seconds
        self._events: Dict[str, Deque[float]] = {}
        self._lock = asyncio.Lock()

    async def _check_single(self, key: str, limit: int, window: int, now: float) -> None:
        """Check rate limit for a single key. Must be called with lock held."""
        if limit <= 0:
            return
        bucket = self._events.setdefault(key, deque())
        cutoff = now - window
        while bucket and bucket[0] <= cutoff:
            bucket.popleft()
        if len(bucket) >= limit:
            logger.warning("Rate limit exceeded for attachment key %s", key)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Attachment ingestion rate limit exceeded",
            )
        bucket.append(now)

    async def purge_stale_keys(self) -> int:
        """Remove empty buckets from the events dict. Returns count of purged keys."""
        async with self._lock:
            stale = [k for k, v in self._events.items() if not v]
            for k in stale:
                del self._events[k]
            return len(stale)

    async def check(self, key: str) -> None:
        """Legacy single-key check for backwards compatibility."""
        if self.limit <= 0:
            return
        now = monotonic()
        async with self._lock:
            await self._check_single(key, self.limit, self.window_seconds, now)

    async def check_dual(self, ip_key: str, user_key: str) -> None:
        """Check both IP-based and user-based rate limits.

        The IP limit is checked first (stricter) to prevent abuse from a single
        source. The user limit is checked second to prevent username spoofing
        from exhausting resources for legitimate users.
        """
        now = monotonic()
        async with self._lock:
            # Lazy purge: remove stale empty buckets periodically
            if len(self._events) > 500:
                stale = [k for k, v in self._events.items() if not v]
                for k in stale:
                    del self._events[k]
            # Check IP limit first (stricter, non-evadable)
            await self._check_single(f"ip:{ip_key}", self.ip_limit, self.ip_window_seconds, now)
            # Check user limit second (prevents username exhaustion)
            await self._check_single(f"user:{user_key}", self.limit, self.window_seconds, now)


def _get_rate_limit_keys(request: Request, username: Optional[str]) -> Tuple[str, str]:
    """Extract rate limit keys from request.

    Returns:
        Tuple of (ip_key, user_key) for dual rate limiting.
        - ip_key: Client IP (real IP behind proxies via CF-Connecting-IP)
        - user_key: Username or 'anonymous' if not provided
    """
    client_ip = get_client_ip(request) or "unknown"
    user_key = username or "anonymous"
    return (client_ip, user_key)


router = APIRouter(prefix="/attachments", tags=["attachments"])

_attachment_manager: Optional[AttachmentManager] = None
_rate_limiter: Optional[SimpleRateLimiter] = None


def get_attachment_manager() -> AttachmentManager:
    """Resolve or initialize the shared AttachmentManager instance."""
    global _attachment_manager
    if _attachment_manager is None:
        pepper = config.PEPPER
        if not pepper:
            raise RuntimeError("PEPPER is not configured; set the PEPPER environment variable")
        _attachment_manager = AttachmentManager(settings=config.ATTACHMENTS, pepper=pepper)
    return _attachment_manager


def get_rate_limiter() -> SimpleRateLimiter:
    """Provide the process-wide rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        settings = config.ATTACHMENTS
        _rate_limiter = SimpleRateLimiter(
            limit=settings.rate_limit_per_minute,
            window_seconds=settings.rate_limit_window_seconds,
        )
    return _rate_limiter

class AttachmentUploadResponse(BaseModel):
    """Response wrapper for successful attachment ingestion."""

    upload_id: str
    metadata: AttachmentRecord


class Base64AttachmentRequest(BaseModel):
    """Payload for base64 attachment ingestion."""

    username: str
    filename: str
    content_base64: str
    intended_usage: Optional[str] = "context"
    mime_type: Optional[str] = None


class URLAttachmentRequest(BaseModel):
    """Payload for URL-based attachment ingestion."""

    username: str
    url: str
    intended_usage: Optional[str] = "context"


@router.post("", response_model=AttachmentUploadResponse)
async def upload_attachment(
    request: Request,
    username: str = Form(..., description="User identifier used to scope storage"),
    file: UploadFile = File(..., description="Attachment to store"),
    intended_usage: str = Form("context", description="High-level hint about how this attachment will be used"),
    manager: AttachmentManager = Depends(get_attachment_manager),
    rate_limiter: SimpleRateLimiter = Depends(get_rate_limiter),
) -> AttachmentUploadResponse:
    """Ingest a multipart/form-data attachment for later reuse."""
    ip_key, user_key = _get_rate_limit_keys(request, username)
    await rate_limiter.check_dual(ip_key, user_key)
    try:
        record = await manager.store_upload(
            username=username,
            upload_file=file,
            intended_usage=intended_usage or "context",
        )
    except AttachmentTooLargeError as exc:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(exc)) from exc
    except AttachmentValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - unexpected failures
        logger.exception("Unexpected error storing attachment", exc_info=exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unable to store attachment") from exc

    return AttachmentUploadResponse(upload_id=record.upload_id, metadata=record)


@router.post("/base64", response_model=AttachmentUploadResponse)
async def upload_attachment_base64(
    request: Request,
    payload: Base64AttachmentRequest,
    manager: AttachmentManager = Depends(get_attachment_manager),
    rate_limiter: SimpleRateLimiter = Depends(get_rate_limiter),
) -> AttachmentUploadResponse:
    """Ingest an attachment supplied as base64 encoded content."""
    ip_key, user_key = _get_rate_limit_keys(request, payload.username)
    await rate_limiter.check_dual(ip_key, user_key)
    try:
        data = AttachmentManager.decode_base64_payload(
            payload.content_base64,
            max_decoded_size=manager.settings.max_size_bytes
        )
        record = await manager.store_bytes(
            username=payload.username,
            data=data,
            filename=payload.filename,
            intended_usage=payload.intended_usage or "context",
            origin="base64",
            mime_type=payload.mime_type,
        )
    except AttachmentTooLargeError as exc:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(exc)) from exc
    except AttachmentValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - unexpected failures
        logger.exception("Unexpected error storing base64 attachment", exc_info=exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unable to store attachment") from exc

    return AttachmentUploadResponse(upload_id=record.upload_id, metadata=record)


@router.post("/url", response_model=AttachmentUploadResponse)
async def upload_attachment_url(
    request: Request,
    payload: URLAttachmentRequest,
    manager: AttachmentManager = Depends(get_attachment_manager),
    rate_limiter: SimpleRateLimiter = Depends(get_rate_limiter),
) -> AttachmentUploadResponse:
    """Download and store an attachment supplied via URL."""
    ip_key, user_key = _get_rate_limit_keys(request, payload.username)
    await rate_limiter.check_dual(ip_key, user_key)
    try:
        record = await manager.store_from_url(
            username=payload.username,
            url=payload.url,
            intended_usage=payload.intended_usage or "context",
        )
    except AttachmentTooLargeError as exc:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(exc)) from exc
    except AttachmentValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - unexpected failures
        logger.exception("Unexpected error storing URL attachment", exc_info=exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unable to store attachment") from exc

    return AttachmentUploadResponse(upload_id=record.upload_id, metadata=record)


@router.get("/{upload_id}", response_model=AttachmentRecord)
async def get_attachment_metadata(
    upload_id: str,
    username: str = Query(..., description="User identifier used when the attachment was uploaded"),
    manager: AttachmentManager = Depends(get_attachment_manager),
) -> AttachmentRecord:
    """Return previously stored metadata for an attachment."""
    try:
        return manager.get_metadata(username=username, upload_id=upload_id)
    except AttachmentValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except AttachmentNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
