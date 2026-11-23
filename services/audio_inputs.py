"""Resolve audio attachments through the existing owner-scoped attachment store."""

from __future__ import annotations

import asyncio
import base64
import hashlib
from typing import Any

from fastapi import HTTPException

from ai_runtime.audio import AUDIO_MIME_FORMATS, AudioInputError, inspect_audio, validate_audio_model
from models import AudioData, ContentRequest
from services.attachment_types import AttachmentNotFoundError, AttachmentValidationError


async def resolve_audio_inputs(request: ContentRequest, manager: Any, config_obj: Any) -> list[AudioData]:
    """Resolve and validate each reference once, with a bounded aggregate payload."""
    if not request.audios:
        return []
    if not request.username:
        raise HTTPException(400, "username is required when providing audios")
    if len(request.audios) > config_obj.ATTACHMENTS.max_files_per_request:
        raise HTTPException(400, "Too many audio attachments")
    seen: set[str] = set()
    result: list[AudioData] = []
    total_bytes = 0
    for ref in request.audios:
        if ref.username != request.username:
            raise HTTPException(403, "Audio does not belong to the requesting user")
        if ref.upload_id in seen:
            raise HTTPException(400, "Duplicate audio reference")
        seen.add(ref.upload_id)
        try:
            resolved = manager.resolve_attachment(username=request.username, upload_id=ref.upload_id)
            record = resolved.record
            if record.mime_type not in AUDIO_MIME_FORMATS:
                raise AudioInputError("audio_format_unsupported", "Referenced attachment is not supported audio.")
            total_bytes += record.size_bytes
            if total_bytes > config_obj.ATTACHMENTS.max_size_bytes:
                raise HTTPException(413, {"code": "audio_too_large", "message": "Combined audio exceeds the attachment size limit."})
            data = await asyncio.to_thread(resolved.binary_path.read_bytes)
            if len(data) != record.size_bytes or hashlib.sha256(data).hexdigest() != record.sha256:
                raise AudioInputError("audio_integrity_error", "Audio attachment failed integrity verification.")
            mime, fmt, duration = await asyncio.to_thread(inspect_audio, data, record.mime_type)
            result.append(AudioData(
                base64_data=base64.b64encode(data).decode("ascii"), mime_type=mime, format=fmt,
                original_filename=record.original_filename, size_bytes=len(data),
                duration_seconds=duration, sha256=record.sha256,
                upload_id=ref.upload_id, source_offset_seconds=ref.source_offset_seconds,
            ))
        except AttachmentNotFoundError as exc:
            raise HTTPException(404, "Audio attachment not found") from exc
        except AttachmentValidationError as exc:
            raise HTTPException(400, str(exc)) from exc
        except AudioInputError as exc:
            raise HTTPException(400, exc.details) from exc
    validate_request_audio_models(request, result, config_obj)
    return result


def validate_request_audio_models(request: ContentRequest, audios: list[AudioData], config_obj: Any) -> None:
    """Check the generator and every explicitly acoustic QA receptor before work."""
    if not audios:
        if any(layer.include_input_audio for layer in request.qa_layers or []):
            raise HTTPException(400, {"code": "audio_required", "message": "An acoustic QA layer requires audios."})
        return
    models = [request.generator_model]
    if request.gran_sabio_fallback:
        from llm_routing import resolve_call

        models.append(resolve_call("gransabio.regenerate", request=request).model)
    if any(layer.include_input_audio for layer in request.qa_layers or []):
        models.extend(model if isinstance(model, str) else model.model for model in request.qa_models or [])
    for model in dict.fromkeys(models):
        info = config_obj.get_model_info(model)
        try:
            validate_audio_model(audios, specs=config_obj.model_specs, provider=info["provider"], model_id=info["model_id"])
        except AudioInputError as exc:
            raise HTTPException(400, exc.details) from exc
