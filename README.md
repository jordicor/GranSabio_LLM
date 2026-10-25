# Gran Sabio LLM Engine

### Multi-Layer AI Quality Assurance for Content Generation

**Your AI generates content but you can't trust it blindly?** Gran Sabio LLM coordinates generation and configurable review by multiple AI models. Results report the actual quality verdict; a completed generation or a run without QA does not imply reviewer approval.

---

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#license)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![Self-Hosted](https://img.shields.io/badge/Self--Hosted-BYOK-orange.svg)](#self-hosting-bring-your-own-api-keys)

---

## The Problem

Every developer using AI for content generation faces the same challenges:

| Problem | What Happens | The Cost |
|---------|--------------|----------|
| **Hallucinations** | AI invents facts, dates, or events | Credibility destroyed, corrections needed |
| **Quality inconsistency** | Sometimes great, sometimes terrible | Manual review of every output |
| **No validation** | You get content, hope it's good | Time wasted on unusable content |
| **Single point of failure** | One model, one opinion | Bias and blind spots undetected |
| **Format violations** | JSON that doesn't match your schema | Parsing errors, retry loops |
| **Repetitive vocabulary** | Same phrases appearing everywhere | Unprofessional, robotic text |

**Traditional solution:** Review everything manually or accept the risk.

**Gran Sabio LLM solution:** Let multiple AI models evaluate every output with configurable quality criteria, automatic retry on failure, and a "Great Sage" arbiter for final decisions.

---

## Shared source data

`POST /generate` and the sync/async SDK accept optional `shared_context` text.
Use it for a stable document or transcript and keep the changing task in `prompt`.
The source stays at user level, before the task; it is reference data and cannot
replace system instructions. Generation, retries, QA, preflight, grounding and
escalation retain that evidence. Admission counts the full source. Omitting the
field or sending `null` adds no shared-source message and preserves request identity.
A supplied source changes request and feedback identity; moving an existing source
from `prompt` into `shared_context` starts a new identity and may incur a cold paid
run. No historical result or accounting record is migrated automatically.

This contract supports standard generation, streaming, tool loops and vision.
Enabled Long Text Mode currently rejects `shared_context` before model dispatch;
set `long_text_mode="off"` when supplying shared context. Provider prompt caching
is an optimization, not a promise of reuse, lower costs or identical answers.
The shared source block receives an explicit cache boundary on supported OpenAI,
Anthropic and OpenRouter routes. Keep the source and preceding instructions
identical across related requests; put changing questions and run metadata after
the shared source. The first request or a changed source may still incur cache
write costs.

```python
result = await client.generate(
    prompt="Return the launch date stated in the document.",
    shared_context=document_text,
    long_text_mode="off",
    # Keep the usual model, QA and output settings for your application.
)
```

GranSabio also marks existing attachment text and the stable
request prefix before retry feedback, without adding API or SDK fields. QA and
preflight put existing `source_text` before changing task/round metadata; Auto-QA
preserves its existing source-length limit. Attachment previews, source scope,
output schemas and idempotency rules remain unchanged. Ordinary one-shot text
requests without attachments keep their original transport when no retry is
configured. These boundaries reuse complete prefixes, not independent arbitrary
chunks or previous answers.

### Per-request prompt cache policy

`POST /generate`, `generate()` and `generate_streaming()` accept `prompt_cache`.
This changes cache controls only: the engine keeps the existing prompt rendering,
message roles/order, selected models and provider routing.

| Policy | Behavior |
| --- | --- |
| Omitted, `null`, or `{"mode": "auto"}` | Existing automatic boundaries and provider defaults. |
| `{"mode": "off"}` | No cache writes anywhere in this request, including preflight, QA, retries and tools. |
| `{"mode": "explicit", "breakpoints": [1200, 2400]}` | Write only at these endpoints in the existing generator `prompt`; suppress all engine-added boundaries and automatic suffix writes. |
| `{"mode": "adaptive", "breakpoints": [1200, 2400]}` | Use supported prefix controls on each actual route; otherwise send no cache controls and retain provider defaults. Internal phases use their own existing source boundaries. |

Breakpoints are **exclusive Unicode code-point offsets in the original `prompt`**:
`n` ends the prefix `prompt[:n]` in Python. Use `len(prefix)` in Python or
`Array.from(prefix).length` in JavaScript, not UTF-8 byte counts or JavaScript
string `.length` (UTF-16 units). Supply at most four positive, strictly increasing,
unique integers, each no larger than the original prompt length. Positions are
not offsets in `shared_context`, system instructions, attachments or token IDs.
The existing renderer trims outer whitespace; leading-whitespace-only endpoints
are rejected, trailing-whitespace endpoints map to the trimmed end, and resulting
duplicate endpoints merge. No whitespace is newly trimmed by the policy.

```python
# Keep the application's existing prompt and model/QA settings.
result = await client.generate(prompt=existing_prompt, prompt_cache={"mode": "off"})

result = await client.generate(
    prompt=existing_prompt,
    prompt_cache={"mode": "explicit", "breakpoints": [stable_prefix_end]},
    long_text_mode="off",
)
```

Each endpoint caches the **complete preceding provider prefix**, including earlier
system instructions, tools and any existing context messages; it is not an
independent text chunk. Only generator prompts use the supplied offsets. In strict
`explicit` mode, other phases inherit the write policy with no automatic boundary. An empty explicit
breakpoint list has the same write behavior as `off`. Provider minimum sizes,
expiry and prefix equality still determine
whether a marked prefix is written or reused.

Strict `off` and `explicit` require a verified route contract: native Claude,
the curated OpenAI GPT-5.6/6/6.1 families, and those models through OpenRouter.
OpenAI (including through OpenRouter) receives
`prompt_cache_options={"mode":"explicit","ttl":"30m"}` even with **no**
breakpoints, preventing implicit writes of the rest. Claude receives only the
requested `cache_control` blocks, with no automatic request-level cache control.
Unsupported routes, including older OpenAI models, Gemini and generic router
aliases, are rejected before their provider call rather than silently ignoring
the policy. Predictable generation, QA, fallback and admission routes are checked
before generation starts; conditional later routes are checked at dispatch.
TypeSafe's non-token-based evaluation transport is unaffected. Enabled Long Text
rejects nonempty breakpoints in both `explicit` and `adaptive`; use standard generation for these offsets.

#### Adaptive policy and capability discovery

`adaptive` is opt-in and does not change omitted/`auto` requests. The application
still chooses its generator prefix endpoints. `shared_context` retains its existing
user role and position and may receive a boundary on a compatible route. Preflight,
Auto-QA and QA use only the stable source boundaries their own builders already
produce; generator offsets are never applied to these prompts. No prompt is
reordered, padded or interpreted to discover a prefix. Without any selected or
existing source boundary, a route with explicit-only controls writes no prefix;
a route without those controls keeps its provider defaults, which can include writes.

The existing `GET /models/{model_name}` response includes `prompt_cache_capabilities`:
independent `automatic`, `explicit_prefix`, `explicit_only` and `off` support states,
their evidence sources/dates, marker transport, minimum prefix size when known,
boundary limit/selection and implemented TTL. Unknown and conditional support are
not treated as confirmed support. `explicit_prefix` means inline prefix controls;
it does not mean support for a separate cache-object lifecycle. `transport_available`
only describes the implemented wire format; capability conditions must also hold.

| Route | `adaptive` behavior | Strict `off` / `explicit` |
| --- | --- | --- |
| Curated OpenAI GPT-5.6/6/6.1, native or OpenRouter | Explicit prefixes, 30-minute TTL; implicit suffix writes disabled | Supported |
| Claude, native or OpenRouter | Explicit `cache_control` blocks with default 5-minute TTL | Supported |
| OpenRouter Gemini 2.5 Flash/Pro | Last user-content prefix only; implicit caching stays provider-managed | Rejected: no verified disable/exclusive contract |
| Native Gemini, older OpenAI, OpenRouter Moonshot/Kimi | Provider defaults; no extra cache controls | Rejected |
| Selected OpenRouter Qwen/Alibaba models | Provider defaults: endpoint-specific support is conditional and routing is not changed | Rejected |
| Other/new models, generic router aliases, local routes | Provider defaults; unknown controls are omitted | Rejected until verified |

Caller endpoints take priority over an additional `shared_context` boundary when
all four available slots are used. Gemini's last-boundary rule is applied explicitly
and reported; the request API still accepts at most four caller endpoints. Minimum
sizes are provider token counts, not character offsets: the engine does not add a
token-count request, pad a short prefix or promise a cache hit.

Each non-default transport records `requested`, `effective`, `reason`, applied
boundary count and adaptations in the existing logger and matching usage callback's
`prompt_cache` metadata. Effective values are `explicit`, `off`, `prefix` (implicit
cache may remain active), or `provider_default`. These describe controls, not measured
cache hits; use the normal read/write usage counters for actual reuse. No source text
is logged. A provider failure never triggers a cache-policy fallback or extra generation.
Normal retry/idempotency behavior is unchanged; `adaptive` and its endpoints are part
of the request fingerprint, so changed policy requires a new request identity.

`model_cache.curated.json` is a tracked, reviewed overlay loaded by the existing
capability registry over optional `provider_capabilities.prompt_cache` catalog data.
It reloads with the existing model-catalog reload; there are no runtime network
lookups or background refresh services. Model sync preserves separate cache evidence
dates rather than presenting its sync time as verification. Curated values win over
catalog hints. Missing parameters or cache prices are not proof of unsupported or
supported controls. Update bounded model rules from the corresponding official route
contract; new wire protocols also require transport code and tests. Legacy `auto`
retains its prior marker behavior for compatibility; it is not a capability certificate.

Protocol references: [OpenAI prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching),
[OpenRouter prompt caching](https://openrouter.ai/docs/guides/best-practices/prompt-caching),
[Anthropic prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching).

Long Text Mode requires its optional controller package. When it is absent,
`long_text_mode="auto"` uses standard generation with the requested word limits
and an advisory if long-text activation would otherwise apply. Explicit
`long_text_mode="on"` requests return HTTP 400 before model dispatch.

## How It Works

```
Your Request
     |
     v
[Preflight Validation] --> Detects contradictions before wasting tokens
     |
     v
[Content Generation] --> Your chosen AI model generates content
     |
     v
[Multi-Layer QA] --> Multiple AI models evaluate different aspects
     |                - Historical accuracy
     |                - Literary quality
     |                - Format compliance
     |                - Custom criteria you define
     v
[Consensus Engine] --> Calculates scores across all evaluators
     |
     v
[Pass?] --No--> [Iterate with Feedback] --> Back to generation
     |
    Yes
     |
     v
[Deal Breaker?] --Yes--> [Gran Sabio Escalation] --> Premium model decides
     |
    No
     |
     v
[Approved Content] --> Delivered with confidence scores
```

---

## Output Quality

Want to see what Gran Sabio LLM actually produces? Every post on [Atagia Journal](https://atagia.org/blog) was generated by AI models using Gran Sabio as the QA engine. No cherry-picking, no manual edits — what you read is what the system delivered after multi-model evaluation, consensus scoring, and Gran Sabio arbitration.

---

## See It In Action

> **Note:** Gran Sabio LLM is fundamentally an **API-first tool** designed to integrate into your content generation pipelines. The web interface below is a development/demo UI to help visualize and test the API capabilities - not a production-ready application. Think of it as a reference implementation showing what's possible when you build on top of this API.

### Web Interface (Demo)

Access the interactive demo at `http://localhost:8000/` - configure your generation, select models, define QA layers, and watch results in real-time.

![Main Interface](screenshots/main-interface.png)
*Configure prompts, models, QA layers, and quality thresholds from an intuitive web UI*

---

### Live Matrix: Real-Time Generation Monitoring

Click **"Live Matrix"** to watch the entire process unfold:

- Content chunks streaming as they're generated
- QA evaluations appearing for each layer and model
- Scores updating as consensus is calculated
- Deal-breaker escalations and Gran Sabio decisions

![Live Matrix](screenshots/live-matrix.png)
*Watch content generation, QA evaluation, and scoring happen in real-time*

---

### 200+ Models via OpenRouter

Beyond direct API connections (OpenAI, Anthropic, Google, xAI, MiniMax, and Moonshot/Kimi), you can access **all models available on OpenRouter** - including Mistral, DeepSeek, LLaMA, Qwen, and many more.

![OpenRouter Models](screenshots/openrouter-models.png)
*Access hundreds of models through OpenRouter integration*

### Admin Models Catalog

Use `/admin/models` to manage the local model catalog without hand-editing `model_specs.json`:

- Toggle models enabled/disabled for runtime use
- Delete local catalog entries
- Fetch remote provider catalogs
- Run **Sync All** for the selected provider/model entries

Context admission prefers exact per-model limits from `model_limits.curated.json`
or provider metadata cached by model sync. If a newly discovered direct-provider
text-generation model has no trusted numeric capacity, the versioned and expiring
`model_family_floors.json` policy may supply a conservative family floor. It
never applies to OpenRouter, embeddings, realtime/audio/image surfaces, unknown
families, lookalike names, or expired rules. Machine-specific overrides belong
in ignored `model_limits.curated.local.json` and
`model_family_floors.local.json` files. For capacity without any usable numeric
limit, the risk bands count serialized input plus reserved output and thinking
tokens.

`POST /context/preflight` sizes every participating model without generating
content. There is no global context ceiling: `GRANSABIO_CONTEXT_POLICY_MAX_TOKENS`
is retired and ignored. To use each model's full native window, send this policy:

```json
{
  "schema_version": "gransabio_context_budget_policy.v1",
  "context_fraction": 1.0,
  "minimum_working_context_tokens": 1,
  "safety_margin_fraction": 0.05
}
```

An omitted or `null` `maximum_working_context_tokens` follows the model capacity;
an explicit integer remains an optional per-request cap. Output/reasoning
reservations, native input limits and the safety margin still reduce the usable
input budget. The combined budget is limited by the smallest participating
model. `GRANSABIO_CONTEXT_POLICY_SAFETY_MARGIN` sets the service's minimum margin
(default `0.05`); callers may request a larger margin, but cannot reduce it.

---

### Session Debugger: Full Transparency

Every generation is logged in detail. Access `/debugger` to inspect:

- Complete request payloads and parameters
- Every iteration with content and scores
- QA evaluations per layer and model
- Consensus calculations
- Gran Sabio escalations and decisions
- Token usage and costs per phase

![Session Debugger](screenshots/debugger.png)
*Inspect every detail of your generation sessions*

---

## Key Features

### Multi-Model Quality Assurance

Define what "quality" means for YOUR use case:

```json
{
  "qa_layers": [
    {
      "name": "Factual Accuracy",
      "criteria": "Verify all dates, names, and events are historically correct",
      "min_score": 8.5,
      "deal_breaker_criteria": "invents facts or presents false information"
    },
    {
      "name": "Narrative Flow",
      "criteria": "Evaluate prose quality, transitions, and reader engagement",
      "min_score": 7.5
    }
  ],
  "qa_models": ["gpt-4o", "claude-sonnet-4", "gemini-2.0-flash"]
}
```

QA scheduling is configurable through `qa_execution_mode`, `on_qa_model_unavailable`, `on_qa_timeout`, `min_valid_qa_models`, and `min_valid_qa_model_ratio`. The default `auto` mode uses progressive quorum for deal-breaker layers, so it may stop once a reliable quorum is guaranteed, and bounded parallel execution for normal scoring layers. Retryable QA provider failures use the bounded `MAX_QA_PROVIDER_RETRIES` budget before the evaluator is marked as a technical failure.

The supervisor observes process health and reports recommendations; it does not
replace evaluators. The legacy `qa_replacement_policy` field is accepted only
with replacement and benching disabled. Enabling either is rejected during
request validation.

### Request Timeouts

Request process timeouts are centralized in `request_timeouts.default.json`. Create a local `request_timeouts.json` to override defaults without changing tracked files. These values are for model generation, streaming, QA, Gran Sabio, Long Text, SDK waits, and MCP waits; provider connection/health/model-sync timeouts remain separate.

OpenAI Chat and Responses both use the asynchronous client and
`SDK_OPENAI_TIMEOUT_SECONDS` (`sdk_model_timeouts.openai_seconds`). The unused
synchronous client and its `SDK_OPENAI_SYNC_TIMEOUT_SECONDS` /
`sdk_model_timeouts.openai_sync_seconds` settings have been removed.

The default process timeout is intentionally high: `12000` seconds. Per request you can override it globally or by phase:

```json
{
  "timeout_seconds": 12000,
  "timeouts": {
    "generation_seconds": 18000,
    "qa_model_seconds": 12000,
    "qa_comprehensive_seconds": 12000,
    "gran_sabio_seconds": 18000,
    "long_text_seconds": 12000
  },
  "qa_timeout_retries": 0,
  "qa_models_config": {
    "claude-sonnet-4": {
      "timeout_seconds": 18000
    }
  }
}
```

Precedence is: request phase override, request `timeout_seconds`, local timeout config, tracked defaults. Capability-based `recommended_timeout_seconds` is only a hint and does not shorten explicit user/config values.

---

### Final Verification

Enable a read-only final QA pass when you need a last check over the approved content snapshot:

```json
{
  "qa_final_verification_mode": "after_modifications",
  "qa_final_verification_strategy": "full_parallel"
}
```

Use `after_modifications` to verify only after QA/smart-edit/Gran Sabio rewrites content, or `always` to verify whenever a QA contract exists. Strategies are `full_parallel`, `full_sequential`, and `fast_global`.

---

### Deal Breakers: Stop Problems Immediately

Some issues are too serious to just lower the score:

- **Majority deal-breaker (>50% of models):** Forces immediate regeneration
- **Minority deal-breaker (<50%):** Escalates to Gran Sabio for arbitration
- **Tie (50%):** Gran Sabio decides if it's a real issue or false positive

**Why this matters:** You define what's unacceptable. "Invented facts" can be a deal-breaker while "slightly awkward phrasing" just lowers the score.

```json
{
  "deal_breaker_criteria": "uses offensive language or invents historical events"
}
```

---

### Gran Sabio: The Final Arbiter

When evaluators disagree or max iterations are reached, the "Great Sage" steps in:

- **Uses premium reasoning models** (Claude Opus 4.5 with 30K thinking tokens by default)
- **Analyzes the conflict:** Was it a real issue or false positive?
- **Can modify content:** Fixes minor issues without full regeneration
- **Tracks model reliability:** Learns which models produce more false positives
- **Flexible model choice:** Use GPT-5.2-Pro for maximum accuracy or Claude Opus 4.6 for deep reasoning

```json
{
  "gran_sabio_model": "claude-opus-4-6",
  "gran_sabio_call_limit_per_session": 15
}
```

Or use OpenAI's most powerful model:
```json
{
  "gran_sabio_model": "gpt-5.2-pro"
}
```

---

### Preflight Validation: Don't Waste Tokens

Before spending money on generation, the system checks if your request makes sense:

```
Request: "Write a fiction story about dragons"
QA Layer: "Verify historical accuracy of all events"

Preflight Response:
{
  "decision": "reject",
  "issues": [{
    "code": "contradiction_detected",
    "severity": "critical",
    "message": "Fiction content cannot be validated for historical accuracy"
  }]
}
```

Requests without an active validation surface skip the LLM preflight query and proceed immediately. Adding QA layers, Auto-QA, word/phrase/lexical validators, evidence grounding, accent guard, images, context/source text, or Gran Sabio fallback keeps the normal preflight gate.

**No tokens wasted on impossible requests.**

---

### Word Count Enforcement

AI models are notoriously bad at hitting word targets. Gran Sabio LLM solves this:

```json
{
  "min_words": 800,
  "max_words": 1200,
  "word_count_enforcement": {
    "enabled": true,
    "flexibility_percent": 15,
    "direction": "both",
    "severity": "deal_breaker"
  }
}
```

The system automatically injects a QA layer that counts words and triggers regeneration if the target isn't met.

---

### Lexical Diversity Guard

Detect and prevent repetitive vocabulary:

- **MTLD, HD-D, Yule's K, Herdan's C** metrics calculated automatically
- **GREEN/AMBER/RED grading** based on configurable thresholds
- **Window analysis** finds exactly where repetition clusters appear
- **Top words report** shows which words are overused

```json
{
  "lexical_diversity": {
    "enabled": true,
    "metrics": "auto",
    "decision": {
      "deal_breaker_on_red": true,
      "deal_breaker_on_amber": false
    }
  }
}
```

---

### Phrase Frequency Guard

Block specific phrases or patterns:

```json
{
  "phrase_frequency": {
    "enabled": true,
    "rules": [
      {
        "name": "no_then_went_to",
        "phrase": "then went to",
        "max_repetitions": 1,
        "severity": "deal_breaker"
      },
      {
        "name": "short_phrases",
        "min_length": 3,
        "max_length": 6,
        "max_repetitions": 3,
        "severity": "warn"
      }
    ]
  }
}
```

---

### Generation System Context

`include_default_system_context` defaults to `true`: generation appends the
automatic same-language instruction and current date to the system prompt.
Set it to `false` when supplying a fixed generation system prompt. This removes
only that automatic suffix; JSON/tool instructions, validation prompts, and
other generation constraints still apply. The flag also covers Long Text
generation and JSON correction retries.

Backends supporting this flag advertise
`capabilities.include_default_system_context: true` in `/health`. Clients that
require a fixed prompt should check this capability before requesting generation.

---

### Deterministic Validation and Structural Smart Edit

Two request flags control the new deterministic path without changing existing iteration or token budgets:

```json
{
  "generation_tools_mode": "auto",
  "smart_edit_locator_mode": "ids"
}
```

- **`generation_tools_mode`**: `auto` or `never`. In `auto`, supported providers (`OpenAI`, `Claude`, `Gemini`, `xAI`, and `OpenRouter`) can call a local `validate_draft` tool when measurable constraints are active, such as word count, JSON target-field extraction, phrase frequency, lexical diversity, or cumulative repetition.
- **`smart_edit_locator_mode`**: `ids` or `legacy`. `ids` is now the default and targets paragraph/sentence IDs like `p1` or `p2s3`; `legacy` preserves the previous phrase-marker plus word-index fallback behavior.

---

### Evidence Grounding (Confabulation Detection)

Detect when AI models claim to use evidence but actually ignore it:

```json
{
  "evidence_grounding": {
    "enabled": true,
    "model": "gpt-4o-mini",
    "budget_gap_threshold": 0.5,
    "on_flag": "deal_breaker",
    "max_flagged_claims": 2
  }
}
```

**How it works:**
1. Extracts verifiable claims from generated content
2. Measures P(claim | evidence) vs P(claim | no evidence) using logprobs
3. Flags claims where confidence doesn't drop when evidence is removed

**This catches:**
- "According to the sources, Marie Curie was born in Paris" (context says Warsaw)
- Claims that sound referenced but ignore the actual context

**Configuration modes:**

| Mode | `on_flag` | When to use |
|------|-----------|-------------|
| Verification-only | `"warn"` | General content, informational logging |
| Fail-fast | `"deal_breaker"` | Critical factual content, medical/legal |
| Regenerate | `"regenerate"` | Auto-fix on detection |

**Cost:** ~$0.003 per request for 10 claims (2-6% overhead)

---

### Vision/Image Support

Process images alongside text for multimodal content generation:

```json
{
  "prompt": "Describe these product images in detail",
  "generator_model": "gpt-4o",
  "username": "your_username",
  "images": [
    {"upload_id": "img_001", "username": "your_username", "detail": "high"},
    {"upload_id": "img_002", "username": "your_username", "detail": "auto"}
  ],
  "image_detail": "auto",
  "qa_layers": []
}
```

**Supported vision models:**
- **OpenAI:** GPT-4o, GPT-5, GPT-5 Pro, O1, O3, O3-Pro
- **Anthropic:** Claude Sonnet 4, Claude Opus 4.5, Haiku 4
- **Google:** Gemini 2.0 Flash, Gemini 2.5 Pro/Flash
- **xAI:** Grok 2+ models

**Detail levels (OpenAI-style):**
| Level | Tokens | Use Case |
|-------|--------|----------|
| `low` | ~85 fixed | Quick classification, thumbnails |
| `high` | Variable (tiles) | Detailed analysis, text extraction |
| `auto` | Model decides | General use (default) |

**QA with vision** - Let QA models see input images for accuracy validation:

```json
{
  "prompt": "Describe this architectural diagram",
  "generator_model": "claude-sonnet-4-20250514",
  "images": [{"upload_id": "diagram", "username": "user1"}],
  "qa_with_vision": true,
  "qa_layers": [
    {
      "name": "Visual Accuracy",
      "criteria": "Verify description matches the actual diagram elements",
      "include_input_images": true,
      "min_score": 8.5
    },
    {
      "name": "Writing Quality",
      "criteria": "Evaluate clarity and technical accuracy",
      "include_input_images": false,
      "min_score": 7.5
    }
  ]
}
```

**Limits and auto-processing:**
- Default: 20 images per request (configurable up to 100)
- Auto-resize to optimal dimensions per provider
- Automatic format conversion (HEIC/HEIF to JPEG)
- Preflight validation rejects requests when model lacks vision capability

---

### JSON Schema Structured Outputs

**100% format guarantee** across all major providers:

```json
{
  "generator_model": "gpt-5",
  "json_output": true,
  "json_schema": {
    "type": "object",
    "properties": {
      "title": {"type": "string"},
      "summary": {"type": "string"},
      "key_points": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["title", "summary"]
  }
}
```

**Supported providers:**
- **OpenAI:** GPT-4o, GPT-5, GPT-5.2-Pro, O1/O3 series
- **Anthropic:** Claude 4 Sonnet, Claude Opus 4.5
- **Google:** Gemini 2.0+, Gemini 2.5
- **xAI:** Grok 4
- **OpenRouter:** All compatible models (Mistral, DeepSeek, LLaMA, Qwen, and 200+ more)

---

### Reasoning Models Support

OpenAI and Anthropic requests omit `temperature` by default, including calls through
OpenRouter. Only known older compatible models receive the configured value; the
allowlists in `ai_runtime/parameters.py` recognize their dated snapshots and OpenAI
fine-tunes. Catalog restrictions can disable temperature for an older model but
cannot enable it for a newer or unrecognized one. This policy applies to generation,
streaming, tool calls, and logprob probes. Other providers keep their own policies.

For complex tasks, enable deep thinking:

**OpenAI Reasoning:**
```json
{
  "generator_model": "gpt-5",
  "reasoning_effort": "high"
}
```

**Claude Thinking Mode:**
```json
{
  "generator_model": "claude-sonnet-4-20250514",
  "thinking_budget_tokens": 8000
}
```

**Both work for QA evaluation too** - your evaluators can "think" before scoring.

---

### QA Bypass for Rapid Prototyping

Need fast generation without QA? Just send empty layers:

```json
{
  "prompt": "Write a quick draft",
  "qa_layers": []
}
```

When no other validation surface is active, content is approved immediately. Perfect for testing, bulk generation, or content that will be manually edited.

---

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/jordicor/GranSabio_LLM.git
cd GranSabio_LLM
python -m pip install --upgrade "pip>=26.1.2"
python -m pip install -r requirements.txt
python quick_start.py
```

> **Tip:** No Docker image is provided at this time. If you prefer assisted setup, AI coding tools like [Claude Code](https://docs.anthropic.com/en/docs/claude-code) or [Codex CLI](https://github.com/openai/codex) can handle the full installation and configuration for you — just point them at this repo.

### 2. Configure Your API Keys

Create `.env` file with **your own API keys** from each provider:

```env
# Get your keys from each provider's dashboard:
# - OpenAI: https://platform.openai.com/api-keys
# - Anthropic: https://console.anthropic.com/
# - Google: https://aistudio.google.com/apikey
# - xAI: https://console.x.ai/
# - OpenRouter: https://openrouter.ai/keys
# - MiniMax: https://platform.minimax.io/
# - Moonshot/Kimi: https://platform.kimi.ai/console/api-keys

OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
XAI_API_KEY=xai-...
OPENROUTER_API_KEY=sk-or-...
MINIMAX_API_KEY=...
MOONSHOT_API_KEY=...
PEPPER=any-random-string-here
```

> **Note:** You only need keys for the providers you want to use. At minimum, configure one provider.

#### Optional accumulated provider budget

An isolated runtime can enable a durable USD ceiling for one OpenRouter text
model. Leave `PROVIDER_BUDGET_LEDGER_PATH` empty for normal behavior. When enabled,
configure every bound explicitly and use the same model for generation and
preflight. This scope supports ordinary and streaming text/JSON calls, including
the normal generation loop's local `validate_draft` tool and its final turn.
Other providers, external tools, accent auditing, images, embeddings and explicit
separate thinking budgets fail before dispatch. Preflight and generation
validation retain their normal behavior.

```env
PROVIDER_BUDGET_LEDGER_PATH=/absolute/private/provider-spend.sqlite
PROVIDER_BUDGET_LIMIT_USD=60
PROVIDER_BUDGET_OPENROUTER_MODEL=google/gemini-3.1-flash-lite-preview
PROVIDER_BUDGET_INPUT_USD_PER_MILLION=0.25
PROVIDER_BUDGET_OUTPUT_USD_PER_MILLION=1.5
PROVIDER_BUDGET_MAX_INPUT_TOKENS=1048576
PROVIDER_BUDGET_MAX_OUTPUT_TOKENS=65536
```

The example's [model limits and tariffs](https://openrouter.ai/api/v1/models/google/gemini-3.1-flash-lite-preview/endpoints)
were checked on September 11, 2026;
verify the selected model's current contract before configuring another runtime.
The bound assumes output `max_tokens` includes all billed reasoning and no
billable dimensions beyond text and a zero request fee. Every HTTP attempt
reserves twice the full input-window quote plus its actual output limit. The
request enforces prompt/completion price ceilings, a zero request price and no
provider fallback through [OpenRouter routing options](https://openrouter.ai/docs/guides/routing/provider-selection).
SDK retries are disabled in this mode; each ordinary GranSabio retry, tool turn
and reasoning-parameter retry reserves separately. Tool streaming settles raw
usage before its parser or event callbacks and closes the transport on cancellation.
All processes sharing this ledger share one ceiling.

TypeSafe System One may share that same ledger through an explicit optional
model and verified input tariff. These settings default to disabled:

```env
PROVIDER_BUDGET_TYPESAFE_MODEL=jev-latest
PROVIDER_BUDGET_TYPESAFE_INPUT_USD_PER_MILLION=0.042
PROVIDER_BUDGET_TYPESAFE_MAX_INPUT_TOKENS=32768
```

The [TypeSafe launch pricing](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
quotes USD 0.042 per million input tokens and free output (checked September 21,
2026). No chat output allowance is reserved or sent to TypeSafe. Each typed
attempt reserves twice the entire input-window quote and settles from valid
native input usage at the configured tariff, marked `conservative` rather than
invoice cost. Failed or malformed attempts keep the full reservation; usage
over the input ceiling halts further dispatch. The configured wire model is
allowed only for standalone System One, semantic preflight and QA evaluation.
Adding it to an existing ledger requires the documented offline
`transition_policy` operation, preserving the same path, cumulative ceiling and
all prior attempts; changing environment variables alone fails closed.

Settlement uses complete raw [OpenRouter usage metadata](https://openrouter.ai/docs/cookbook/administration/usage-accounting)
before normalization or optional callbacks. Explicit non-BYOK usage counts the
account cost; BYOK adds known upstream cost. When BYOK is unspecified but both
costs exist, their sum is labeled conservative. Missing metadata, interruption
or failure retains the reservation as uncertain; a process crash can leave an
unsettled reserved row, which also continues to count in full. Nothing expires
or refunds automatically. Actual and conservative settled amounts are distinct
from outstanding reserves in `ProviderBudget.summary()`. The ledger contains
only accounting metadata, never prompts, outputs or credentials.

Keep the ledger outside releases and preserve it across restarts. Configuration
changes against the same ledger are rejected. If reported usage exceeds its
contracted bound, future dispatch stops pending accounting verification. This
ceiling covers this runtime's attempts; unrelated provider calls need their own
accounting within the owner's overall budget.

For local Ollama models, `OLLAMA_MAX_CONCURRENT_REQUESTS` defaults to `1` so local generation and QA model calls are queued instead of hitting the same GPU in parallel. Increase it only if your local hardware can handle concurrent Ollama requests. `OLLAMA_ENDPOINT_ALIAS` gives that administrator-configured endpoint a stable audit identity; requests cannot supply or override its URL.

All local calls use registered endpoints, including those configured through
the legacy environment settings. QA retains its global Ollama admission limit;
each endpoint also applies its own queue. When both refer to the same legacy
queue, a QA call reuses the slot it already holds. Embedders should initialize
`AIService` normally; the old private provider helpers and the transport fallback
for partially initialized instances have been removed.

For a separate `llama.cpp` server exposing OpenAI-compatible Chat Completions,
set `LLAMACPP_BASE_URL` (including `/v1`), optionally set `LLAMACPP_API_KEY`,
and give it an operator-facing `LLAMACPP_ENDPOINT_ALIAS`. Its queue is isolated
from Ollama and defaults to one request through `LLAMACPP_MAX_CONCURRENT_REQUESTS`.
Use `LLAMACPP_MAX_OUTPUT_TOKENS` as the catalog fallback when the server does not
report its own generation limit.
The admin model sync performs only read-only discovery (`/health`, `/v1/models`,
`/props`, `/slots`) and will not remove llama.cpp catalog entries when discovery
is unavailable. Tool support is capability-driven: a local model may use tools
only when its catalog entry declares `tool_calling`.

For more than one local API service, set `LOCAL_ENDPOINTS_CONFIG_PATH` to an
administrative JSON file. It registers API endpoints—not machines or model
lifecycles—and routes identify a model as `(endpoint_id, model_id)`. URLs remain
server configuration and cannot be supplied by a request. The default policy is
`"availability": "loaded"`, so Ollama discovery uses `/api/ps`; use
`"available"` only when the server's own autoload policy is desired. Start
from [`local_endpoints.example.json`](local_endpoints.example.json) and keep the
deployment-specific copy out of version control.

```json
{
  "local_endpoints": {
    "spark_ollama": {
      "adapter": "ollama",
      "base_url": "http://spark-a.local:11434",
      "max_concurrent_requests": 1
    },
    "reviewer": {
      "adapter": "openai_compatible",
      "base_url": "http://spark-b.local:8000/v1",
      "api_key_env": "REVIEWER_API_KEY",
      "availability": "loaded"
    }
  }
}
```

Supported adapters are `ollama`, `llamacpp`, and `openai_compatible`. A new
registry is intentionally rejected if it conflicts with non-default legacy
`OLLAMA_HOST` or `LLAMACPP_BASE_URL` settings; it is never silently merged.
The JSON accepts only `api_key_env`, never an inline `api_key`.

An endpoint is one API service, not one machine: two services on the same host
but different ports are two endpoints, while one Ollama endpoint can announce
many models and the route selects the desired `model` under the same URL. Give
each route its stable `endpoint_id`, including each QA entry:

```json
{
  "calls": {
    "generation.main": {"model": "writer-model", "endpoint_id": "spark_ollama"},
    "qa.evaluate_layer": {"models": [{"model": "review-model", "endpoint_id": "reviewer"}]}
  }
}
```

`loaded` is the default availability policy: GranSabio verifies that the model
is already resident when a backend distinguishes it. It never loads or unloads
models, schedules GPUs, balances requests, or fails over to another endpoint.

### 3. Start the Server

```bash
python main.py
```

Server starts at `http://localhost:8000`

On Windows, the PowerShell launcher provides a local start/stop menu. Pass the
Python executable from your virtual environment:

```powershell
powershell -NoProfile -File .\tools\windows_launcher.ps1 -PythonPath .\venv\Scripts\python.exe -Port 8000
```

Use `-Port 8001` or the menu to select another port; `.env` is never rewritten.

The server opens in a separate console with file logging and extra verbosity.
Exiting the menu leaves it running. Reopening the menu also detects servers
started manually in Windows once they are listening. Status comes from the OS
listener, process identity, `/api`, and `/health`; no lock/PID file is needed.
An unidentified occupied port cannot be started over or stopped. Stopping a
verified instance is immediate and interrupts its running analyses. Instances
whose `/api` does not respond must be stopped from their own console.

### 4. Generate Your First Content

```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a 500-word biography of Marie Curie",
    "content_type": "biography",
    "generator_model": "gpt-4o",
    "qa_models": ["gpt-4o", "claude-sonnet-4"],
    "qa_layers": [
      {
        "name": "Accuracy",
        "criteria": "Verify historical facts",
        "min_score": 8.0,
        "deal_breaker_criteria": "invents facts"
      }
    ],
    "min_global_score": 8.0,
    "max_iterations": 3
  }'
```

### Model Routing

Runtime model defaults live in `llm_routing.default.json`. Requests can still
override exact call sites through `llm_routing.calls` without changing platform
configuration or `model_specs.json`.

```json
{
  "prompt": "Write a 500-word technical summary of the attached notes",
  "content_type": "technical",
  "llm_routing": {
    "calls": {
      "generation.main": {
        "model": "your-generator-model"
      },
      "qa.evaluate_layer": {
        "models": [
          {"model": "your-qa-model"}
        ]
      }
    }
  },
  "qa_layers": [
    {
      "name": "Completeness",
      "description": "Checks whether the answer covers the requested points",
      "criteria": "Verify that the summary covers all important points from the prompt",
      "min_score": 8.0
    }
  ]
}
```

### Default model roles and Jev preflight

The default routes in `llm_routing.default.json` use GPT-6 Luna with low
reasoning for drafts and auxiliary work, GPT-6 Luna / Claude Sonnet 5 /
Gemini 3.8 Flash for QA, and GPT-6 Sol with medium reasoning for arbitration
and Gran Sabio decisions. Model selection is per role.

Jev is a TypeSafe evaluator for typed decisions, not a text generator.
Semantic preflight now defaults to the pinned `jev-1.13.0`. One System One
request selects `proceed` or one of three rejection reasons:
`conflicting_requirements`, `qa_incompatible_with_task`, or
`missing_required_basis`. Each rejection has a fixed message; no chat model
is called to explain it. Parallel questions identify redundant length-only
QA. Technical failures retain separate error codes and fail closed.

The decision checks the intended result after instruction precedence and
conditional branches. A repair task is not rejected merely because its input
contains the defect to repair. Optional missing sources and ambitious quality
targets are not hard blockers. Context admission remains enforced, with no
silent fallback or source truncation.

Configure `TYPESAFE_API_KEY` in your local `.env` and register the default
`jev-1.13.0` under the `typesafe` provider in the ignored `model_specs.json`
catalog. Sync/register the other default model IDs above on each host too.
The tracked curated token limits do not register models or credentials.

To override preflight or explicitly opt into Jev QA, use the request routing fields:

```json
{
  "llm_routing": {
    "calls": {
      "preflight.validate": {"model": "jev-latest"},
      "qa.evaluate_layer": {"models": [{"model": "jev-latest"}]}
    }
  }
}
```

Each override is independent; omit a call to keep its normal model. Jev QA
receives the original prompt and complete source text, uses six descriptive
levels converted to the engine's 0–10 scale, and does not produce SmartEdit
instructions. Oversized state or questions fail context admission before the
provider call when `CONTEXT_ADMISSION_MODE=enforce`.

The deterministic `/context/preflight` contract identifies typed QA participants
with `call_kind: "system_one"`, zero output/reasoning reservation and the full
future candidate reservation. It counts the largest actual per-layer typed
payload, including the original prompt, complete source and typed questions.
Ordinary participants retain `call_kind: "chat"`. Consumers must preserve this
distinction when validating a mixed jury's output allowances.

For a standalone decision, `POST /systemone` requires an explicit `model`:

```json
{
  "model": "jev-latest",
  "state": {"message": "The delivery is late."},
  "questions": {
    "delivery_issue": {
      "type": "noul",
      "instructions": "Does the message report a delivery problem?"
    }
  }
}
```

Both Python clients expose
`evaluate_system_one(state, questions, model="jev-latest")`. Score questions
accept 2–10 ordered levels; the standalone response retains TypeSafe's raw
0-based level scale. See the [TypeSafe API](https://docs.typesafe.ai/api) and
[Score contract](https://docs.typesafe.ai/primitives/score).

### Fail-closed local execution

Set `execution_policy.scope` to `local_only` when every AI dispatch in a
generation must stay on an administrator-configured Ollama, llama.cpp, or
OpenAI-compatible local endpoint. GranSabio resolves the routes that request can reach, checks the
endpoints and advertised models through read-only live discovery, and then
checks the exact call ID, QA ordinal, provider, endpoint and model again before
each transport attempt. A local timeout or routing change fails the session; it
never falls back to a cloud provider.

```json
{
  "prompt": "Write a short factual summary from the supplied context.",
  "execution_policy": {"scope": "local_only"},
  "feedback_memory_enabled": false,
  "llm_routing": {
    "calls": {
      "generation.main": {"model": "your-local-model"},
      "preflight.validate": {"model": "your-local-model"}
    }
  },
  "qa_layers": []
}
```

Feedback Memory must currently be disabled because its embeddings backends are
external. If QA, Auto-QA, GranSabio fallback, Arbiter, Long Text, Evidence
Grounding, accent audit or JSON output is enabled, configure each reachable
call in `llm_routing` to a live local model with the required catalogued
capabilities. Successful and terminal session results expose a
`local_only_attestation` derived from the session dispatch ledger; `/status`
also exposes the current attestation while a session is active.

Run the API on `0.0.0.0` for LAN access. Other PCs should call the Gran Sabio
server; Ollama only needs to be reachable from the machine running the API.
An enabled `llm_accent_guard` is accepted only when its `accent.audit` route is
also admitted on a capable local model.

---

## Demos & Examples

The [`demos/`](demos/) folder contains 11 ready-to-run scripts showcasing different capabilities. Here are the highlights:

| Demo | Description | Complexity |
|------|-------------|------------|
| **[YouTube Script Generator](demos/03_youtube_script_generator.py)** | Multi-phase pipeline: topic analysis, script, scenes, thumbnails. Uses JSON Schema, lexical diversity, and project grouping. | Advanced |
| **[Code Analyzer](demos/11_code_analyzer.py)** | Dynamic JSON output for code review. Detects security issues, performance problems. Shows when to use flexible JSON vs strict schemas. | Advanced |
| **[Reasoning Models](demos/10_reasoning_models.py)** | GPT-5 reasoning effort, Claude thinking mode. Complex analysis with deep thinking. | Advanced |
| **[JSON Structured Output](demos/05_json_structured_output.py)** | 100% format guarantee with `json_schema`. Multi-provider support. | Intermediate |
| **[Text Quality Analyzer](demos/04_text_quality_analyzer.py)** | Analyze existing text without generating. Lexical diversity, AI pattern detection. | Intermediate |
| **[Parallel Generation](demos/08_parallel_generation.py)** | Bulk content creation with async parallel execution. | Advanced |

**Quick start:**
```bash
# Start the API server
python main.py

# Run any demo
python demos/03_youtube_script_generator.py --topic "How AI is Changing Music"
```

See the complete list and documentation: **[demos/README.md](demos/README.md)**

---

## API Documentation

OpenRouter error handling distinguishes content-policy rejections, errors inside
HTTP 200/SSE responses, and native finish reasons.

Full interactive documentation available at:
- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`
- **Custom Docs:** `http://localhost:8000/api-docs` *(recommended)*

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/generate` | POST | Start content generation with QA |
| `/status/{session_id}` | GET | Check session status |
| `/stream-content/{session_id}` | GET | Observe existing generation content as SSE |
| `/stream/project/{project_id}` | GET | Real-time SSE progress stream (project_id = session_id when not explicit) |
| `/result/{session_id}` | GET | Get final approved content |
| `/stop/{session_id}` | POST | Cancel active generation |
| `/models` | GET | List available AI models |
| `/debugger` | GET | Session history and inspection UI |
| `/admin/models` | GET | Manage local model catalog and provider sync |

`GET /stream-content-direct/{session_id}` is retired and returns HTTP 410. It
started a separate generation outside the normal input and QA pipeline. Use
`POST /generate` to start a session, then `GET /stream-content/{session_id}`
to observe its content, or `GET /stream/{session_id}` and
`GET /stream/project/{project_id}` for labelled events. Observing a session
does not start another generation. References to the direct route in older
audit and test reports describe historical behavior.

The `generation.direct_stream` routing call ID is also retired. Before
upgrading, remove that entry from operator routing files and request overlays;
unknown call IDs are rejected. The local-only policy admits only calls the
normal generation can reach and needs no separate route for observation.

### Analysis Endpoints

Standalone text analysis tools (no generation required):

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/analysis/lexical-diversity` | POST | Vocabulary richness metrics (MTLD, HD-D, etc.) |
| `/analysis/repetition` | POST | N-gram repetition analysis with clustering |

### Project Management

Group multiple sessions under a single project ID:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/project/new` | POST | Reserve a new project ID |
| `/project/start/{id}` | POST | Activate a project |
| `/project/stop/{id}` | POST | Cancel all project sessions |
| `/stream/project/{id}` | GET | Stream all project events |

---

### Admin Model Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/admin/models/catalog` | GET | Read the local model catalog |
| `/api/admin/models/providers/{provider}/remote` | GET | Fetch remote provider models |
| `/api/admin/models/providers/{provider}/sync` | POST | Sync selected provider models |
| `/api/admin/models/catalog/{provider}` | PATCH | Enable or disable one local catalog model |
| `/api/admin/models/catalog/{provider}` | DELETE | Remove one local catalog model |
| `/api/admin/models/sync-all` | POST | Sync the provider/model selection supplied in the request body |

Provider sync covers OpenAI, Anthropic, Google, xAI, OpenRouter, MiniMax,
Moonshot/Kimi, and the legacy local `OLLAMA_HOST`/`LLAMACPP_BASE_URL` settings.
For `LOCAL_ENDPOINTS_CONFIG_PATH`, local-only admission uses read-only live
discovery per registered endpoint: Ollama uses `/api/ps` for `loaded` (or
`/api/tags` for `available`), while llama.cpp and OpenAI-compatible services use
their served-model API. Discovery never loads models or mutates the catalog.

---

## Python Client SDK

A ready-to-use Python client is available for easy integration:

```python
from gransabio_client import GranSabioClient

client = GranSabioClient("http://localhost:8000")

# Simple generation
result = client.generate(
    prompt="Write a product description",
    generator_model="gpt-4o",
    qa_layers=[{"name": "Quality", "criteria": "...", "min_score": 8.0}]
)

print(result.content)
print(f"Score: {result.final_score}")
```

Stream progress:
```python
for event in client.stream_generate(prompt="...", qa_layers=[...]):
    print(f"[{event.phase}] {event.message}")
```

The SDK raises `TransientGranSabioError` for temporary transport failures and retries idempotent polling GETs with backoff. It does not automatically retry POST operations such as `generate`, `stop_session`, project mutations, or attachment uploads, because the server may have accepted the operation before the response failed.

---

## MCP Integration (Claude Code, Gemini CLI, Codex CLI)

Gran Sabio LLM includes a **Model Context Protocol (MCP) server** that integrates directly with AI coding assistants. Get multi-model code review and analysis without leaving your terminal.

### What You Get

| Tool | Description |
|------|-------------|
| `gransabio_analyze_code` | Analyze code for bugs, security issues, and best practices |
| `gransabio_review_fix` | Validate a proposed fix before applying it |
| `gransabio_generate_with_qa` | Generate content with multi-model QA |
| `gransabio_check_health` | Verify Gran Sabio LLM API connectivity |
| `gransabio_list_models` | List available AI models |

### Quick Setup

**1. Install MCP dependencies:**
```bash
pip install -r mcp_server/requirements.txt
```

**2. Run the installer script:**

**Windows:**
```cmd
install_mcp.bat
```

**Linux/macOS:**
```bash
./install_mcp.sh
```

The scripts automatically detect paths and register the MCP server with Claude Code.

**Manual installation** (if you prefer):

```bash
# Use absolute paths - relative paths won't work!
claude mcp add gransabio-llm -- python /path/to/GranSabio_LLM/mcp_server/gransabio_mcp_server.py
```

**Gemini CLI** (`~/.gemini/settings.json`):
```json
{
  "mcpServers": {
    "gransabio-llm": {
      "command": "python",
      "args": ["/path/to/GranSabio_LLM/mcp_server/gransabio_mcp_server.py"]
    }
  }
}
```

**Codex CLI** (`~/.codex/config.toml`):
```toml
[mcp_servers.gransabio-llm]
command = "python"
args = ["/path/to/GranSabio_LLM/mcp_server/gransabio_mcp_server.py"]
```

### Example Usage

```
You: Analyze this code for security issues using Gran Sabio

Claude: [Calls gransabio_analyze_code]

Gran Sabio Analysis (Score: 8.2/10):
- [CRITICAL] SQL injection at line 45
- [HIGH] Hardcoded credentials at line 12
- [MEDIUM] Missing input validation at line 30

Reviewed by: GPT-5-Codex, Claude Opus 4.5, GLM-4.7
Consensus: 3/3 models agree
```

### Remote Engine Configuration

For a remote engine behind your own authenticated proxy (the engine itself
does not validate this API key):
```bash
claude mcp add gransabio-llm \
  --env GRANSABIO_API_URL=https://api.gransabio.example.com \
  --env GRANSABIO_API_KEY=your-api-key \
  -- python /path/to/gransabio_mcp_server.py
```

See full documentation: **[mcp_server/README.md](mcp_server/README.md)**

---

## Self-Hosting (Bring Your Own API Keys)

Gran Sabio LLM is currently a **self-hosted solution**. You deploy it on your infrastructure and use your own API keys from each AI provider.

### What This Means

| Aspect | Self-Hosted |
|--------|-------------|
| **API Keys** | You obtain and configure keys from OpenAI, Anthropic, Google, xAI, MiniMax, Moonshot/Kimi, and/or OpenRouter |
| **Billing** | Each provider bills you directly based on your usage |
| **Infrastructure** | You host and maintain the server |
| **Data Privacy** | You control storage and routing; selected remote providers receive the inputs needed for their calls |
| **Models Available** | All models your API keys have access to, plus 200+ via OpenRouter |

### Why Self-Hosting?

- **Full control** over your data and costs
- **No intermediaries** - direct connection to AI providers
- **Use your existing accounts** - no new subscriptions needed
- **Enterprise compliance** - deploy in your own cloud/datacenter
- **Unlimited usage** - no rate limits beyond provider limits

### Requirements

- Python 3.10+
- **API keys for at least one provider** (OpenAI, Anthropic, Google, xAI, MiniMax, Moonshot/Kimi, or OpenRouter)
- ~500MB disk space for dependencies
- Recommended: 4GB RAM minimum
- Pillow library (auto-installed, required for vision/image processing)

**Quality assurance:** Automated tests cover API endpoints, engines, the client
SDK, and integrations. Live provider tests are opt-in; a passing offline suite
does not certify every provider or deployment.

### Production Deployment

```bash
# With uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1

# With gunicorn + uvicorn workers
gunicorn main:app -w 1 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

Use one worker for this deployment model: active sessions and generation tasks
are held in process memory. Independent workers do not share that state, and
restarting the process does not resume an in-flight generation.

### Security Model: No Built-In Authentication

Gran Sabio LLM does not include authentication, user management, or access control — **this is a deliberate design decision**, not an oversight.

This is a self-hosted engine meant to run on your own infrastructure, behind your own security perimeter (firewall, VPN, reverse proxy). Adding application-level auth would duplicate what your infrastructure already provides and add complexity that self-hosting users don't need.

Multi-user product access, billing, and account management belong to the
separately maintained **[hosted product](#hosted-gran-sabio)**.

### A Note on the Codebase

This repository documents the engine's implemented generation, QA and
orchestration features. Architecture proposals and historical audits are not
commitments that an unimplemented option is available today.

---

## Hosted Gran Sabio

The hosted product is maintained separately from this open-source engine.
Access, available features and commercial terms are announced through the
product's own channels. The engine's HTTP API and MCP configuration do not
promise a public commercial API, an active subscription offer or open signup.

Follow development on [GitHub](https://github.com/jordicor).

*Self-hosting will always remain available for those who prefer full control.*

---

## Cost Tracking

Every request can include cost breakdown:

```json
{
  "show_query_costs": 2,
  "prompt": "..."
}
```

Returns detailed token usage and costs:
```json
{
  "content": "Generated content...",
  "costs": {
    "grand_totals": {
      "input_tokens": 4370,
      "output_tokens": 2156,
      "cost": 0.018765
    },
    "phases": {
      "generation": {"cost": 0.008234},
      "qa": {"cost": 0.003456},
      "gran_sabio": {"cost": 0.005678}
    }
  }
}
```

For execution timing and per-call visibility, use `show_query_stats`:

```json
{
  "show_query_stats": 3,
  "prompt": "..."
}
```

`show_query_stats=1` returns phase/model/provider aggregates, `2` adds iteration and layer detail, and `3` adds prompt-free per-call rows with duration, tokens, provider, model, finish reason, and cost. A per-call `cost_basis` says whether `estimated_cost_usd` is the amount the provider reported as billed (`provider_billed`, OpenRouter's `usage.cost`, plus the upstream charge for BYOK requests) or a `catalog_estimate` from the model's `pricing` block; a call with neither leaves the cost `null`, and any such call keeps the aggregate cost `null` as well. The OpenRouter sync writes each catalog rate (`input_per_million`, `output_per_million`, `cached_input_per_million`, `cache_write_per_million`, and the `long_context` override above its `threshold_tokens`) only when OpenRouter lists it; a router model priced per routed request (OpenRouter lists `-1`) therefore has no catalog rate and stays `null` unless the call reported its billed cost. Local spans such as consensus and smart-edit timing are included without increasing the LLM call count.

Unreported token counts and costs are `null` (`unknown` in the text cost block), rather than zero. An aggregate is unavailable when a contributing call lacks the corresponding measurement; measured zero remains zero. A provider-reported total can still be available when its input/output breakdown is not.

---

## About the Name

### From BioAI to Gran Sabio

This project was originally called **BioAI Unified** - "Bio" for biography (its first use case was validating AI-generated biographies) and "Unified" because it brought together multiple AI providers into a single, coherent QA system.

However, "BioAI" consistently caused confusion. People assumed this was a biomedical or bioinformatics tool, expecting features for DNA analysis or drug discovery. The name created friction before the tool could even be evaluated.

### Why "Gran Sabio LLM"?

The new name directly reflects what makes this engine unique:

**"Gran Sabio"** (Spanish for "Great Sage") is not just a brand - it's a core architectural component. When multiple AI models disagree during quality evaluation, a premium reasoning model called the **Gran Sabio** (the wise arbiter) steps in to make the final decision. This concept of a "council of sages" deliberating on content quality is central to how the system works.

**"LLM"** (Large Language Model) clarifies that this is AI infrastructure for text generation - not a fantasy game, not biomedicine, but a practical tool for orchestrating language models.

The result: a name that immediately tells you what you're getting - an AI content pipeline with a wise, multi-model arbitration system at its heart.

> *Previous name: BioAI Unified (2024). Rebranded to Gran Sabio LLM in January 2025.*

---

## Stay Updated

This project is actively developed. If you find it useful:

- **Star this repo** to follow updates and new features
- **Follow me on social media** for development insights, AI tips, and product announcements

Find my social links on my [GitHub profile](https://github.com/jordicor).

---

## License

The project declares MIT licensing. This source snapshot does not include a
standalone license text.

---

<p align="center">
  <strong>Trust your AI output.</strong><br>
  <em>Let multiple models validate before you ship.</em>
</p>
