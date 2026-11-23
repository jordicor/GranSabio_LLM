# Audio input

GranSabio accepts uploaded recordings alongside system/user prompts through its
normal generation API. Results remain text or JSON, including client-defined
JSON schemas where the selected model supports structured output. The engine
does not transcribe, segment or convert a recording before sending it to the
chosen model. Applications own segmentation, timestamps and output schemas.

## Upload and generate

```python
from pathlib import Path
from client import GranSabioClient

client = GranSabioClient(base_url="http://localhost:8000")
username = "audio-demo"
model_key = "YOUR_AUDIO_CAPABLE_MODEL"  # Select using GET /models.
recording = Path("recording.flac")
uploaded = client.upload_attachment(
    username=username,
    content=recording.read_bytes(),
    filename=recording.name,
    content_type="audio/flac",
)
result = client.generate(
    prompt="Describe what is audible. Return JSON with a summary string.",
    system_prompt="Use only evidence from the recording.",
    generator_model=model_key,
    username=username,
    audios=[{
        "username": username,
        "upload_id": uploaded["upload_id"],
        "source_offset_seconds": 120.0,
    }],
    content_type="json",
    json_output=True,
    json_schema={
        "type": "object",
        "properties": {"summary": {"type": "string"}},
        "required": ["summary"],
        "additionalProperties": False,
    },
    qa_layers=[],
    qa_models=[],
    gran_sabio_fallback=False,
    feedback_memory_enabled=False,
    max_iterations=1,
    strict_single_generation_attempt=True,
    max_tokens=512,
    show_query_costs=0,
    show_query_stats=3,
)
```

The same fields work in `POST /generate`, streaming generation and the async
client. Upload via `POST /attachments` (multipart), `/attachments/base64` or the
existing guarded URL-ingestion route. A generation request contains upload
references, never filesystem paths or arbitrary provider URLs. Use `audios`, not
`context_documents`, for recordings. References must belong to the request's
username, and duplicate references are rejected.

`source_offset_seconds` is optional consumer metadata. It neither trims the file
nor adjusts returned timestamps; specify the required timestamp convention in
the prompt/schema. Original bytes are retained across retries and tool turns.
`strict_single_generation_attempt` controls provider attempts independently of
the outer iteration count. The SDK's `show_query_costs=0` omits that request
field, preserving the server's default aggregate cost receipt without appending
a cost block to the transcript/JSON. `show_query_stats=3` supplies per-call usage.
Explicit `show_query_costs=2` also embeds the cost block in returned content.

For plain transcription, use `content_type="analysis"` and a transcription
prompt; `transcription` is not a supported content category. Keep the returned
status check before processing content: a truncated generation is a failed
result with `error_code="output_token_limit"`, not a usable transcript.

A small output budget must also allow for the model's reasoning tokens. For an
OpenRouter model that supports disabling reasoning, set
`reasoning_effort="none"` explicitly for a short transcription probe; a model
that requires reasoning needs sufficient output headroom instead. See the
[provider's reasoning controls](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens).

## Models and formats

`GET /models` and `GET /models/{model_name}` expose `audio_input`:

```json
{
  "support": "supported",
  "source": "provider_api",
  "transport_available": true,
  "available": true,
  "formats": ["wav", "mp3", "flac", "ogg", "m4a", "aac", "aiff"]
}
```

Support can be `supported`, `unsupported`, `unknown` or `conditional`; only
explicitly supported routes are admitted. Catalog refresh reads
explicit input-modality evidence from OpenRouter metadata and official
OpenAI/Gemini documentation; a model name does not establish audio capability.
Refresh the installed catalog when adding a newly available model. Missing fresh
evidence preserves previous declarations and curated format restrictions.

| Engine transport | Accepted containers |
| --- | --- |
| OpenAI Chat Completions audio models | WAV (PCM), MP3 |
| Native Gemini | WAV (PCM), MP3, FLAC, OGG/Opus, M4A, AAC, AIFF |
| OpenRouter Chat Completions | Same locally recognized formats, subject to the selected model/provider |

Model-specific `provider_capabilities.audio_input.formats` can narrow that list.
The engine checks declared support and known format restrictions before calling
a provider; the provider remains authoritative about codec and model-specific
limits. Unsupported or unmeasurable files are rejected, without conversion.

This feature does not enable speech synthesis, realtime sessions, dedicated ASR
endpoints or asynchronous Batch jobs. Ordinary OpenRouter `:batch` routes reject
inline audio even when the underlying model advertises audio input. Native Gemini
Batch needs a separate asynchronous job lifecycle; choosing an audio model does
not enable it. Existing text-only provider spend policies continue to reject audio.

## QA and regeneration

Set `include_input_audio: true` on a QA layer whose verdict depends on the
recording. Every evaluator assigned to that layer must support its audio format.
Other layers remain text-only. Missing recordings or incompatible evaluators
reject the request; an acoustic layer never silently becomes a text evaluation
or algorithmic bypass. Auto-QA validates this before accepting its generated plan.
Final verification preserves the same evidence requirement.

If `gran_sabio_fallback` is enabled, its regeneration model must also accept the
audio. Regeneration and JSON repair use the original resolved recording. The
Long Text execution mode does not currently accept audio.

## Limits, usage and failures

At most five audio references are accepted, also subject to
`ATTACHMENTS.max_files_per_request`. Existing attachment size/storage/rate limits
apply. The combined raw audio bytes per generation must fit
`ATTACHMENTS.max_size_bytes` (10 MiB by default). Larger recordings must be split
by the caller. The engine checks file identity, MIME/container, SHA-256 and
duration locally; it does not trust caller-supplied duration.

Context admission reserves a conservative 50 tokens per second of audio plus
the ordinary text/output budget. This is an admission estimate, not measured
usage or a billing formula. The stateless `/context/preflight` endpoint cannot
resolve uploads and explicitly rejects audio references; `/generate` resolves
and admits the actual payload.

Detailed usage records carry safe audio provenance (hash, upload ID, filename,
format, duration, size and optional source offset). `audio_input_tokens`, when
reported, is a subset of input tokens and is never added to the total again.
Provider-billed cost takes precedence. Catalog cost estimates require a reported
audio/text split and a verified `audio_input_per_million` price; otherwise cost
is unknown. Cached audio without a trustworthy modality split is also unknown.
Encoded recordings are excluded from usage records and provider-error diagnostics.

Input rejection uses HTTP 400 with `detail.code`, including
`audio_input_unsupported`, `audio_capability_unknown`, `audio_transport_unsupported`,
`audio_format_unsupported`, `audio_invalid`, `audio_integrity_error`,
`audio_reference_requires_audios`, `audio_required` and `audio_mode_unsupported`.
Ownership/missing references use 403/404; aggregate size overflow uses 413 with
`audio_too_large`. Schema validation can return 422. Rejections during execution
carry a failed terminal result with `error_type="audio_input"`, a specific
`error_code`, and `retryable=false`.
