# Gran Sabio LLM Client

Python client library for the Gran Sabio LLM Engine API.

## Installation

The client is included in the Gran Sabio LLM repository. No separate installation required.

Dependencies:
```bash
pip install aiohttp requests
```

## Environment ownership

SDK 1.3.0 includes original-request session evidence (`get_session_details`) and the application-owned environment contract below. SDK 1.3.1 adds idempotent generation submissions (`generate(request_id=...)`), the submission lookup `get_request_status()`, `list_sessions()`, `admission_rejection_error()` (the preflight/Auto-QA rejection rule of `generate()` for stored admission responses), and `get_session_details()` on the synchronous client (see [Idempotent submissions](#idempotent-submissions)). These repository changes are distributed from a compatible local/editable checkout; they do not imply a PyPI release or installation.

Importing the SDK does not load `.env` or mutate the process environment. Both clients read `GRANSABIO_BASE_URL` and `GRANSABIO_API_KEY` when instantiated; explicit constructor arguments take precedence. Provider credentials are forwarded only through explicit `provider_keys`.

Both clients accept optional `request_headers` for a gateway's per-request
metadata. The constructor copies the mapping for that client instance and sends
it on normal requests, streams and multipart uploads. `Authorization`,
`Content-Type`, and headers already supplied through `provider_keys` are rejected
case-insensitively. Set `api_key=""` to explicitly omit the engine Bearer header;
`api_key=None` retains the environment fallback. Keep private headers in the
calling server and create a separate client when the request scope changes.

Applications own environment-file loading. If your executable uses a file, load an explicit path before importing configuration-dependent modules and keep process variables authoritative:

```python
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
from client import GranSabioClient
```

`python-dotenv` is an application dependency when using this optional pattern; it is not required by the SDK. The twelve repository demo scripts explicitly load the repository `.env` and require the repository's demo dependencies. Embedded hosts supply their own environment and do not inherit the SDK repository's `.env`. Server/SaaS settings retain their own explicit bootstrap.

## Quick Start

Both clients accept `prompt_cache` in `generate()` and `generate_streaming()`:

```python
result = client.generate(existing_prompt, prompt_cache={"mode": "off"})
result = client.generate(
    existing_prompt,
    prompt_cache={"mode": "adaptive", "breakpoints": [stable_prefix_end]},
    long_text_mode="off",
)
```

Omitting the policy or using `{"mode": "auto"}` preserves existing behavior.
`off` suppresses provider writes throughout the request. `explicit` writes only at
the supplied generator-prefix endpoints and disables automatic writes elsewhere.
`adaptive` applies available prefix controls for each model, including the source
boundaries already built by internal phases, and otherwise preserves provider defaults.
It never changes models/routing or retries a failed call with another cache policy.
Inspect `GET /models/{model_name}` → `prompt_cache_capabilities` for route support.
Endpoints count Unicode code points from the start of the original `prompt`
(`prompt[:n]`, exclusive), not bytes or UTF-16 units; there may be up to four,
positive, sorted and unique. The text, message order and models stay unchanged.
See the [full policy, whitespace rules and supported providers](../README.md#per-request-prompt-cache-policy).

### Synchronous Client

```python
from client import GranSabioClient

# Create client
client = GranSabioClient()

# Quick generation (no QA)
result = client.generate_fast("Write a haiku about coding")
print(result["content"])

# Generation with QA
result = client.generate(
    prompt="Write a professional article about AI",
    content_type="article",
    qa_models=["gpt-4o-mini"],
    qa_layers=[{
        "name": "Clarity",
        "description": "Writing must be clear",
        "criteria": "Sentences should be easy to understand",
        "min_score": 7.5,
        "is_mandatory": True,
        "order": 1
    }],
    min_global_score=7.5
)
print(result["content"])
```

### Asynchronous Client

```python
import asyncio
from client import AsyncGranSabioClient

async def main():
    async with AsyncGranSabioClient() as client:
        # Quick generation
        result = await client.generate_fast("Write a haiku")
        print(result["content"])

        # Parallel generation
        prompts = ["Haiku about rain", "Haiku about sun", "Haiku about stars"]
        results = await client.generate_parallel(prompts)
        for r in results:
            print(r["content"])

asyncio.run(main())
```

## Features

### Content Generation

```python
result = client.generate(
    prompt="Your prompt here",
    content_type="article",         # article, creative, json, etc.
    generator_model="gpt-4o",       # AI model to use
    temperature=0.7,                # Creativity (0.0-2.0)
    max_tokens=12000,               # Optional; omit to use model max from server specs

    # QA Configuration
    qa_models=["gpt-4o-mini"],
    qa_layers=[...],                # Empty list = bypass QA
    min_global_score=8.0,
    max_iterations=3,

    # Escalation
    gran_sabio_model="claude-opus-4-6",
    gran_sabio_fallback=True,

    # Optional
    reasoning_effort="medium",      # For OpenAI reasoning models
    thinking_budget_tokens=8000,    # For Claude thinking models
    show_query_costs=2,             # Token/cost summary
    show_query_stats=3,             # Timing and per-call stats
)
```

`show_query_stats` controls prompt-free execution visibility: `1` returns summary aggregates, `2` adds phase/iteration/layer detail, and `3` adds per-call rows with duration, tokens, provider, model, and estimated cost.

### JSON Structured Output

```python
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"}
    },
    "required": ["name", "age"]
}

result = client.generate_json(
    prompt="Extract info: John is 30 years old",
    schema=schema
)
print(result["parsed_content"])  # {"name": "John", "age": 30}
```

### Text Analysis

```python
# Lexical diversity
diversity = client.analyze_lexical_diversity(
    text="Your text here...",
    metrics="all"
)
print(diversity["decision"])  # GREEN, AMBER, or RED

# Repetition analysis
repetition = client.analyze_repetition(
    text="Your text here...",
    min_n=2,
    max_n=5
)
```

### Project Management

```python
# Group related sessions
project_id = client.reserve_project()

result1 = client.generate(prompt="Part 1", project_id=project_id)
result2 = client.generate(prompt="Part 2", project_id=project_id)

# Cancel all sessions in project
client.stop_project(project_id)
```

### Attachments

```python
import base64

# Upload as base64
content = base64.b64encode(b"File content").decode()
result = client.upload_attachment_base64(
    username="demo_user",
    content_base64=content,
    filename="document.txt",
    content_type="text/plain"
)
upload_id = result["upload_id"]
```

## Configuration

### Environment Variables

```bash
GRANSABIO_BASE_URL=http://localhost:8000  # API URL
GRANSABIO_API_KEY=your_api_key            # Optional API key
```

### Custom Configuration

```python
client = GranSabioClient(
    base_url="http://custom-host:8000",
    api_key="your_key",
    timeout=(30, 12000)  # (connect, read) in seconds
)
```

### Per-Request Timeouts

The SDK forwards timeout controls to the API and also uses them for local wait/stream windows. Defaults are `12000` seconds.

```python
result = client.generate(
    prompt="Write a long technical report...",
    timeout_seconds=12000,
    timeouts={
        "generation_seconds": 12000,
        "qa_model_seconds": 12000,
        "gran_sabio_seconds": 12000,
    },
    qa_timeout_retries=0,
)
```

`timeout=(connect, read)` remains the HTTP transport timeout. Use `timeout_seconds` / `timeouts` for Gran Sabio request-process deadlines.

## Error Handling

```python
from client import GranSabioClient, GranSabioClientError, TransientGranSabioError

client = GranSabioClient()

try:
    result = client.generate(prompt="...")
except TransientGranSabioError as e:
    print(f"Temporary network error: {e}")
except GranSabioClientError as e:
    print(f"Error: {e}")
    print(f"Status code: {e.status_code}")
    print(f"Details: {e.details}")
```

## Retry & Error Classes

`GranSabioClientError` is the base exception for client-side failures, HTTP
errors, rejected generations, and malformed responses.

For native preflight and Auto-QA rejections, `provider_outcome` is `"rejected"`
and `details["response"]` preserves the complete generation-init response,
including project identity and feedback. The existing `preflight_feedback` or
`auto_qa_feedback` detail remains available. This is evidence of a terminal
request; preflight and Auto-QA may already have incurred provider costs. Other
client errors leave `provider_outcome` unset (`None`) unless explicitly supplied.

`TransientGranSabioError` is raised for temporary transport failures such as
connection drops, timeouts, truncated chunked bodies, or compressed-body decode
errors, and for unreadable JSON on a successful polling GET. Both
`wait_for_result()` and the final polling after streaming retry the complete
GET, including JSON decoding, using the shared backoff curve:

```python
(5, 10, 20, 40, 60, 60, 60, 60, 60, 60)
```

HTTP 202 from `get_result()` returns `status="in_progress"` and keeps either
polling flow waiting, even if the status endpoint has already reported a
terminal state. HTTP 425, 429, and 503 are transient polling failures and use
the backoff above, independently of the error body. Other HTTP errors,
including 400, fail immediately. HTTP 202 from `get_status()` uses the same
transient backoff because there is no usable status yet.

The retry sleeps count against the caller's `max_wait` budget, or the remaining
activity-aware window when fetching the final result after streaming. Non-idempotent
POST operations, including `generate`, project start/stop, session stop, and
attachment uploads, are not automatically retried because the server may have
accepted the operation before the response body failed. For `generate`, pass a
`request_id` to make your own retries safe (next section).

## Idempotent submissions

`generate(..., request_id="...")` (sync and async, keyword-only) sends the key as
the `X-GranSabio-SaaS-Request-ID` header of that POST only; the client's default
headers are unchanged. The key is 1-128 ASCII letters, digits or `._:-`; an
invalid key raises `ValueError` before anything is sent. The server keeps one
entry per key for 24 hours in a process-local registry (lost on a server
restart):

- same key, same payload, first admission finished: the original admission
  response is replayed, so the same `session_id` comes back and no second
  session starts. Stored and replayed the same way for the key's lifetime,
  so a new attempt after one needs a new key: an HTTP 400, 409 or 422 the
  engine answered from the payload, and the HTTP 200 preflight or Auto-QA
  rejection (`status` `preflight_rejected` / `auto_qa_rejected`, no
  `session_id`), which the SDK returns as a normal result, not as an error.
  Not stored, because capacity rather than the payload refused the admission
  and no generation was admitted: the admission-gate HTTP 503 (`Retry-After`
  sent), the HTTP 503 of an unavailable local runtime (no `Retry-After`) and
  the HTTP 503 "idempotency registry is temporarily full". Re-send the
  identical payload with the same key, after `retry_after_seconds` when it was
  sent and otherwise after a short backoff of your own, and the server admits
  it at most once;
- same key, first admission still running: the server waits for it (bounded),
  then answers HTTP 202, which the SDK raises as `TransientGranSabioError`
  with `status_code=202`, `details["retry_after_seconds"]` (when sent) and
  `details["request_id"]`. Re-send the same payload with the same key later;
- same key, different payload: HTTP 409. Never reuse a key for another request.

So re-sending the identical payload with the same key is safe, and the right
move, after a transport failure, after a proxy or gateway answer
(`details["json_error_detail"]` is `False`), after HTTP 202 and after one of
the capacity HTTP 503s listed above: the server admits it at most once. Every non-200 answer of `POST /generate` carries
`details["json_error_detail"]`: `True` when the engine itself answered with a
JSON `{"detail": ...}` body, `False` for a proxy or gateway page (for example a
Cloudflare 52x). It also carries `retry_after_seconds` when the server sent
`Retry-After`, and `request_id` when one was given. The body itself is never
added to `details`.

Reconciliation reads (idempotent GETs with the polling backoff within
`max_wait`, default 60 s; HTTP 425/429/503 and transport failures are retried,
other HTTP errors such as 400/404 raise `GranSabioClientError`):

```python
snapshot = client.get_request_status("my-app:job-42")
# {"found": False, "request_id": ...} when the server holds no entry, or
# {"found": True, "state": "pending" | "completed" | "error", "status",
#  "session_id", "project_id", "init_response"[, "error": {"status_code", "detail"}]}

page = client.list_sessions(limit=100, offset=0, project_id=None)
# {"sessions": [{"session_id", "created_at", "updated_at", "status", "phase",
#                "project_id", "request_name"}, ...], "project_id": None}
```

`list_sessions()` reads the debugger listing, ordered by `updated_at`
descending, with UTC `YYYY-MM-DD HH:MM:SS` timestamps. The server caps `limit`
(100 by default) and answers 404 when its debugger is disabled. It is meant as
secondary reconciliation evidence when a lookup finds nothing, together with
`get_session_details()`. The `error.detail` of a lookup snapshot can contain
request-derived text: keep it out of logs and persistent receipts.

A completed snapshot without a `session_id` can hold an admission that preflight
or Auto-QA rejected. `client.admission_rejection_error(init_response)` applies
the same rule `generate()` uses: it returns the `GranSabioClientError`
(`provider_outcome="rejected"`, feedback in `details`) that `generate()` would
have raised for that admission response, or `None` for any other response.

```python
from client import admission_rejection_error

rejection = admission_rejection_error(snapshot.get("init_response"))
if rejection is not None:
    raise rejection
```

## Polling and Callbacks

### Manual Polling

```python
# Start without waiting
result = client.generate(
    prompt="Long task...",
    wait_for_completion=False
)
session_id = result["session_id"]

# Check status manually
status = client.get_status(session_id)
print(status["status"])  # initializing, generating, completed, etc.

# Get result when ready
final = client.get_result(session_id)
```

For recovery that must verify the original request, both clients also expose
`get_session_details(session_id)` (awaitable on the asynchronous client). Pass the raw session ID.
This performs an authenticated GET only; it does not submit a generation. The
returned request evidence may contain private source text and belongs in private
operation storage, not public logs.

### Status Callbacks

```python
def on_status(status):
    print(f"Status: {status['status']}, Iteration: {status.get('current_iteration')}")

result = client.generate(
    prompt="...",
    on_status=on_status
)
```

## Available Models

```python
models = client.get_models()
for provider, model_list in models.items():
    print(f"\n{provider}:")
    for model in model_list:
        print(f"  - {model['key']}: {model['description']}")
```

## See Also

- [Demos](../demos/README.md) - Example scripts demonstrating various features
- [Engine README](../README.md) - Public behavior and configuration
- API docs at `http://localhost:8000/api-docs`
