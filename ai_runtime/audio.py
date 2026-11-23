"""Audio validation, capability admission and provider content parts."""

from __future__ import annotations

import base64
import io
import math
import wave
from typing import Any, Mapping, Sequence

from model_capability_registry import model_supports, model_uses_responses_api, normalize_provider

# Container aliases are syntax, not semantic inference. Values are provider formats.
AUDIO_MIME_FORMATS = {
    "audio/wav": "wav", "audio/x-wav": "wav", "audio/wave": "wav",
    "audio/vnd.wave": "wav", "audio/mpeg": "mp3", "audio/mp3": "mp3",
    "audio/flac": "flac", "audio/x-flac": "flac", "audio/ogg": "ogg",
    "audio/opus": "ogg", "application/ogg": "ogg", "audio/mp4": "m4a",
    "audio/m4a": "m4a", "audio/x-m4a": "m4a", "audio/aac": "aac",
    "audio/x-aac": "aac", "audio/aiff": "aiff", "audio/x-aiff": "aiff",
}
_CANONICAL_MIME = {
    "wav": "audio/wav", "mp3": "audio/mpeg", "flac": "audio/flac",
    "ogg": "audio/ogg", "m4a": "audio/m4a", "aac": "audio/aac", "aiff": "audio/aiff",
}


class AudioInputError(ValueError):
    """A deterministic audio rejection, safe to expose before provider dispatch."""

    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.details = {"code": code, "message": message, **details}


def inspect_audio(data: bytes, mime_type: str) -> tuple[str, str, float]:
    """Validate an audio container and return canonical MIME, format and duration.

    Mutagen reads container metadata locally; it neither transcodes nor calls a
    provider. Duration is measured from the file, never trusted from the caller.
    """
    import mutagen
    from mutagen.aac import AAC
    from mutagen.aiff import AIFF
    from mutagen.flac import FLAC
    from mutagen.mp3 import MP3
    from mutagen.mp4 import MP4
    from mutagen.oggopus import OggOpus
    from mutagen.oggvorbis import OggVorbis
    from mutagen.wave import WAVE

    expected = AUDIO_MIME_FORMATS.get(mime_type.lower())
    if expected is None:
        raise AudioInputError("audio_format_unsupported", "Unsupported audio MIME type.", mime_type=mime_type)
    try:
        media = AAC(io.BytesIO(data)) if expected == "aac" else mutagen.File(io.BytesIO(data))
        formats = {WAVE: "wav", MP3: "mp3", FLAC: "flac", OggOpus: "ogg",
                   OggVorbis: "ogg", MP4: "m4a", AAC: "aac", AIFF: "aiff"}
        detected = formats.get(type(media))
        duration = float(media.info.length) if media is not None else 0.0
        if detected == "wav":
            # Pipe-produced RIFF files may declare 0xffffffff bytes. Count the
            # bounded PCM frames actually present, not that placeholder length.
            with wave.open(io.BytesIO(data), "rb") as wav:
                frame_size = wav.getnchannels() * wav.getsampwidth()
                frames = wav.readframes(len(data) // frame_size + 1)
                if len(frames) % frame_size:
                    raise ValueError("Incomplete PCM frame")
                duration = len(frames) / (frame_size * wav.getframerate())
        if detected != expected or not math.isfinite(duration) or duration <= 0:
            raise ValueError("Invalid container or duration")
    except Exception as exc:
        raise AudioInputError("audio_invalid", "Invalid audio or MIME type does not match its container.") from exc
    return _CANONICAL_MIME[detected], detected, duration


def audio_support_info(*, specs: Mapping[str, Any], provider: str, model_id: str) -> dict[str, Any]:
    """Describe model evidence separately from the implemented engine transport."""
    provider = normalize_provider(provider)
    state = model_supports(specs=specs, provider=provider, model_id=model_id, capability="audio_input")
    transport = provider in {"openrouter", "gemini"} or (
        provider == "openai" and not model_uses_responses_api(provider, model_id, specs)
    )
    if provider == "openrouter" and model_id.endswith(":batch"):
        transport = False  # Inline audio is not supported by ordinary OpenRouter Batch.
    formats = ["wav", "mp3"] if provider == "openai" else list(_CANONICAL_MIME)
    declared = state.details.get("formats")
    if isinstance(declared, list):
        formats = [fmt for fmt in formats if fmt in declared]
    return {
        "support": state.support.value, "source": state.source.value,
        "transport_available": transport, "available": state.supported and transport,
        "formats": formats if transport else [],
    }


def validate_audio_model(audios: Sequence[Any], *, specs: Mapping[str, Any], provider: str, model_id: str) -> None:
    """Reject absent evidence, unsupported transports and formats without fallback."""
    if not audios:
        return
    info = audio_support_info(specs=specs, provider=provider, model_id=model_id)
    if not info["available"]:
        code = "audio_input_unsupported" if info["support"] == "unsupported" else "audio_capability_unknown"
        if not info["transport_available"]:
            code = "audio_transport_unsupported"
        raise AudioInputError(code, f"Audio input is not available for model '{model_id}'.", model=model_id, **info)
    for audio in audios:
        if audio.format not in info["formats"]:
            raise AudioInputError("audio_format_unsupported", f"Audio format '{audio.format}' is not supported by this route.", model=model_id, formats=info["formats"])


def estimate_audio_tokens(audios: Sequence[Any]) -> int:
    """Conservative admission estimate; never a usage measurement or billed cost."""
    from config import AUDIO_ADMISSION_TOKENS_PER_SECOND

    return sum(math.ceil(audio.duration_seconds * AUDIO_ADMISSION_TOKENS_PER_SECOND) for audio in audios)


def audio_usage_metadata(audios: Sequence[Any]) -> dict[str, Any]:
    """Return provenance for receipts without retaining encoded audio content."""
    if not audios:
        return {}
    return {"audio_input": True, "audio_inputs": [audio.model_dump(exclude={"base64_data"}) for audio in audios]}


def build_openai_audio_content(audios: Sequence[Any]) -> list[dict[str, Any]]:
    """Build Chat Completions audio parts for OpenAI and OpenRouter."""
    return [{"type": "input_audio", "input_audio": {"data": audio.base64_data, "format": audio.format}}
            for audio in audios]


def build_gemini_audio_parts(audios: Sequence[Any]) -> list[Any]:
    """Build native Gemini inline audio parts, preserving the supplied bytes."""
    from google.genai import types

    return [types.Part.from_bytes(data=base64.b64decode(audio.base64_data), mime_type=audio.mime_type)
            for audio in audios]
