# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture Overview

Gran Sabio LLM Engine is a sophisticated content generation API that uses multiple AI providers (OpenAI, Anthropic, Google, xAI) with a multi-layer quality assurance system. The architecture follows a service-oriented pattern:

- **Preflight Validator** (`preflight_validator.py`): Intelligent feasibility analysis before content generation starts
- **AI Service Layer** (`ai_service.py`): Unified interface for multiple AI providers with model-specific parameter handling. Use `get_ai_service()` to reuse the shared connector instead of instantiating `AIService` directly.
- **QA Engine** (`qa_engine.py`): Multi-layer evaluation system with configurable criteria and deal-breaker detection
- **QA Scheduler** (`qa_scheduler.py`): Executes QA evaluator slots with bounded parallelism, progressive quorum for deal-breaker layers, and explicit technical-failure policy
- **QA Bypass Engine** (`qa_bypass_engine.py`): Alternative QA system for specific use cases
- **Arbiter** (`arbiter.py`): Per-layer intelligent conflict resolver for smart-edit operations. Detects and resolves conflicts between edits proposed by different QA evaluators, maintains edit history per layer, and verifies edit alignment with original request intent.
- **Consensus Engine** (`consensus_engine.py`): Calculates consensus scores across multiple evaluator models for final approval/rejection decision
- **Gran Sabio Engine** (`gran_sabio.py`): Final escalation system for conflict resolution and content modification
- **Gran Sabio Supervisor** (`gran_sabio_supervisor.py`): Observe-first process health reviewer for repeated false-positive and technical-failure patterns
- **Attachment System** (`attachments_router.py`): Handle file uploads and context ingestion with security controls
- **Session Management**: UUID-based async session tracking with real-time streaming
- **Text Analysis** (`text_analysis.py`): Word counting and content analysis utilities
- **JSON Optimization** (`json_utils.py`): Fast JSON processing using orjson (3.6x faster than standard json)

The generation workflow: Client request → **Preflight validation** → Content generation → Multi-layer QA evaluation → **Arbiter** (edit conflict resolution per layer) → Smart-edit application → Consensus calculation → Approval/retry → Gran Sabio escalation (if needed).

## Development Commands

```bash
# Initial setup
python quick_start.py

# Start development server  
python main.py
# OR with reload
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Run comprehensive test
python example_request.py

# Run simple tests (from dev_tests directory)
python dev_tests/simple_test.py

# Run specific tests (all test files are in dev_tests/)
python dev_tests/model_fixes_test.py
python dev_tests/test_word_count_enforcement.py
python dev_tests/test_preflight_endpoint.py
python dev_tests/test_preflight_simple.py

# Test attachments functionality
python dev_tests/test_attachment_functionality.py
python dev_tests/test_attachment_url.py
python dev_tests/test_attachment_cleanup.py

# Test analysis endpoints
python dev_tests/test_analysis_api.py

# Test specific AI models
python dev_tests/test_gpt5_minimal.py
python dev_tests/test_o3_pro.py
python dev_tests/test_claude_opus4.py
python dev_tests/test_xai_integration.py

# Attachment management CLI
python tools/attachments_cli.py cleanup --username <user>
python tools/attachments_cli.py list --username <user>
python tools/attachments_cli.py delete --username <user> --upload-id <id> --commit

# Environment setup for development (Windows)
start2.bat  # Sets up conda environment

# Install dependencies
pip install -r requirements.txt
```

## Configuration Requirements

Required environment variables in `.env`:
```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
XAI_API_KEY=xai-...
APP_HOST=0.0.0.0
APP_PORT=8000
DEFAULT_MIN_GLOBAL_SCORE=8.0
DEFAULT_MAX_ITERATIONS=5
SESSION_CLEANUP_INTERVAL=900  # Session cleanup interval in seconds
ATTACHMENTS_RETENTION_DAYS=32850  # Override attachment retention period

# Smart Edit Configuration (optional)
MAX_PARAGRAPHS_PER_INCREMENTAL_RUN=12  # Max paragraphs edited per smart edit round
MAX_EDIT_ROUNDS_PER_LAYER=5  # Max smart-edit rounds per QA layer before moving to next layer
MAX_JSON_RECURSION_DEPTH=3  # Max depth for automatic JSON text field discovery
SMART_EDIT_MAX_PHRASE_LENGTH=64  # Max words to check for unique n-grams (default 64, avoids word_map fallback)

# Evidence Grounding Configuration (ECONOMY mode - optimized for cost)
EVIDENCE_GROUNDING_EXTRACTION_MODEL=gpt-5-nano   # Cheap ($0.05/$0.40/M), no logprobs needed
EVIDENCE_GROUNDING_SCORING_MODEL=gpt-4o-mini    # Requires logprobs support
```

Model specifications are centralized in `model_specs.json` with provider-specific configurations, token limits, and pricing tiers.

## Key Architectural Patterns

**Multi-Provider AI Management**: Each provider has specific parameter requirements (GPT-5 uses `max_completion_tokens`, O3 doesn't support temperature, Claude thinking mode requires `thinking_budget_tokens`). The AI service handles these differences transparently. Always obtain the service via `ai_service.get_ai_service()` so all engines reuse the same connector pool; direct instantiation should be reserved for explicit test doubles.

**JSON Optimization**: The system uses `json_utils.py` with orjson for JSON processing, providing 3.6x performance improvement over standard JSON. All imports should use `import json_utils as json` instead of the standard library.

**QA Layer System**: Configurable evaluation layers with:
- Deal-breaker criteria detection (specific issues that immediately reject content via `deal_breaker_criteria` field)
- Concise feedback mode (`concise_on_pass: true` by default) returns "Passed" when score meets min_score (saves tokens)
- Minimum score thresholds per layer
- Concurrent evaluation by multiple models
- `QAScheduler` execution modes: `auto`, `sequential`, `parallel`, and `progressive_quorum`. `auto` uses progressive quorum for layers with deal-breaker criteria and bounded parallelism for normal scoring layers.
- Technical QA failures are tracked separately from semantic deal-breakers. `on_qa_model_unavailable`, `on_qa_timeout`, `min_valid_qa_models`, and `min_valid_qa_model_ratio` control quorum/fail-open behavior.
- `qa_replacement_policy` is opt-in. `GranSabioSupervisor` can recommend benching repeated bad slots, but replacements require explicit user-provided substitutes.
- Pre-defined templates for biography, script, novel content types
- **QA Bypass**: Use empty `qa_layers: []` to bypass semantic QA layers for rapid generation. The LLM preflight gate still runs for every `/generate` request and fails closed if no preflight-capable model is available.

**Session-Based Processing**: All generation is asynchronous with:
- UUID session tracking
- Background task processing
- Real-time progress streaming via Server-Sent Events
- Status management through enum states

**Debugger Storage**: The debugger SQLite DB (`debugger_history.db`) lives **outside** the repository, under local app/state storage. This is where to look when inspecting Gran Sabio sessions, QA timelines, or debugging live runs.

- **Windows**: `%LOCALAPPDATA%\GranSabio_LLM\debugger\debugger_history.db`
  - Typically resolves to `C:\Users\<user>\AppData\Local\GranSabio_LLM\debugger\debugger_history.db`
- **Unix**: `$XDG_STATE_HOME/gransabio/debugger/debugger_history.db`
  - Fallback: `~/.local/state/gransabio/debugger/debugger_history.db`
- **Override**: Set the `DEBUGGER_DB_PATH` env var only when an explicit custom location is required.

The first startup on each machine auto-migrates any legacy DB from the project root via `migrate_debugger_db_if_needed()` in `debug_logger.py`. No `debugger_history.db` should ever appear at the repo root again — if one does, it is stale and should be deleted. Path resolution logic lives in `debug_logger.py:resolve_debugger_db_path`.

**Preflight Validation**: Before starting content generation, the system analyzes request feasibility using GPT-4o to detect contradictions, incompatibilities, and impossible scenarios. This prevents wasted resources on requests that cannot succeed (e.g., asking for fiction with historical accuracy validation). Runs for every `/generate` request. There is no skip flag; requests that set `qa_layers: []` skip semantic QA but not the preflight gate.

**Final Verification**: Requests can enable a read-only final QA pass after normal generation/QA with:
- `qa_final_verification_mode`: `disabled`, `after_modifications`, or `always`
- `qa_final_verification_strategy`: `full_parallel`, `full_sequential`, or `fast_global`

`full_parallel` and `full_sequential` re-run the real effective QA layers without requesting smart-edit ranges. `fast_global` keeps deterministic/special guards separate and folds semantic layers into one global verification layer; it uses the request-level global score contract and fails closed if the synthetic criteria prompt exceeds `QA_FAST_GLOBAL_MAX_ESTIMATED_TOKENS`.

**LLM Accent Guard**: `llm_accent_guard` detects AI-accent/formulaic style with an LLM judge instead of regex or keyword lists.
- Modes: `off`, `inline`, `post`, and `inline_post`
- Inline modes use the generator tool loop and require `generation_tools_mode != "never"` plus a provider with tool calling
- Post mode runs an audit pass after generation using `AI_ACCENT_AUDIT_MODEL`, `AI_ACCENT_AUDIT_TIMEOUT_SECONDS`, and `AI_ACCENT_AUDIT_MAX_TOKENS`
- Not supported when `long_text_mode` is `on`

**Admin Models UI**: `/admin/models` exposes the local model catalog and provider sync flows.
- Use the catalog tab to enable/disable or delete local model entries.
- Provider tabs can fetch remote models and sync selected entries into `model_specs.json`.
- Bulk sync endpoint: `POST /api/admin/models/sync-all`.
- Mutating admin endpoints require same-origin headers via the CSRF dependency.

**Attachment Ingestion**: `/attachments`, `/attachments/base64`, and `/attachments/url` persist metadata through `AttachmentManager`. URL uploads require HTTPS, respect configurable redirect limits, enforce MIME checks, and cache successful downloads per usuario/URL via `AttachmentRecord.original_url`. Tune size limits, allowed schemes, host allow/block lists, timeouts, and cache TTL under `config.ATTACHMENTS`.
**Attachment Maintenance**: AttachmentManager.run_cleanup() valida firmas/sha, reconstruye uploads/index.json, respeta config.ATTACHMENTS.retention_days (override con ATTACHMENTS_RETENTION_DAYS) y dispone de CLI (python tools/attachments_cli.py cleanup|list|delete) para operaciones manuales (dry-run por defecto).

**Gran Sabio Escalation**: When max iterations reached without approval, the system escalates to a premium model for final decision and potential content modification.

**Arbiter System**: Per-layer intelligent arbitration for smart-edit operations:
- **Conflict Detection**: Detects incompatible edits (DELETE vs REPLACE on same fragment, EXPAND vs CONDENSE on same paragraph, severity mismatches, semantic redundancy, edit cycles)
- **Distribution Classification**: Classifies how edits are distributed among QA models (CONSENSUS, MAJORITY, MINORITY, CONFLICT, TIE, SINGLE_QA)
- **Model Escalation**: Uses economic model (gpt-4o-mini) for normal cases; escalates to GranSabio model for difficult cases (MINORITY, CONFLICT, TIE)
- **AI Verification**: ALWAYS calls AI to verify edit alignment with original request - can reject poorly-reasoned edits even without conflicts
- **Edit History**: Maintains per-layer history of applied/discarded edits, injected into QA prompts in subsequent rounds to prevent re-proposing discarded edits
- **Configuration**: request-level `arbiter_model` (default via `model_specs.json` `default_models.arbiter`), `ARBITER_MAX_TOKENS`, `ARBITER_TEMPERATURE`, `EDIT_HISTORY_MAX_ROUNDS`, `EDIT_HISTORY_MAX_CHARS`

**Smart Edit JSON Field Extraction**: When generating JSON output, smart-edit needs to work on the text content, not the JSON structure. The system automatically extracts text fields for editing:
- **`target_field`**: Specify which JSON field(s) contain the primary text using jmespath notation (e.g., `"generated_text"`, `"data.content"`, or `["chapter", "notes"]` for multiple fields)
- **`target_field_only`**: When `true`, AI QA receives only extracted text (saves tokens). When `false`, AI QA receives full JSON. Algorithmic guards always use extracted text when target_field is set.
- **`smart_edit_locator_mode`**: Defaults to `ids`, using stable paragraph/sentence IDs (`p1`, `p1s2`) for smart-edit targeting. Set to `legacy` to keep phrase markers plus automatic word-index fallback.
- **`generation_tools_mode`**: Controls generator-side deterministic validation tools (`auto`, `never`). `auto` enables the tool loop for supported providers (`OpenAI`, `Claude`, `Gemini`, `xAI`, `OpenRouter`) when measurable constraints such as word count, phrase frequency, lexical diversity, cumulative repetition, or JSON field extraction are active.
- **Auto-detection**: If `target_field` not specified, automatically finds the largest string field (errors if ambiguous)
- **Markdown extraction**: Automatically extracts JSON from markdown ```json code blocks
- **Reconstruction**: After smart-edit completes, the edited text is reconstructed back into the original JSON structure
- See `json_field_utils.py` for implementation details

**Evidence Grounding (Strawberry)**: Detects procedural hallucination by measuring if the model actually relied on cited evidence. Uses a dual-model architecture for cost optimization:
- **Claim Extraction** (`EVIDENCE_GROUNDING_EXTRACTION_MODEL`): Uses gpt-5-nano by default (cheapest: $0.05/$0.40 per M tokens). Does NOT require logprobs.
- **Budget Scoring** (`EVIDENCE_GROUNDING_SCORING_MODEL`): Uses gpt-4o-mini by default (requires logprobs support). Measures P(YES|evidence) vs P(YES|no_evidence) to detect confabulation.
- Alternative models: `grok-4-1-fast-non-reasoning` for speed (2.5x faster extraction, supports logprobs with top_logprobs=8 max)
- Benchmark script: `python dev_tests/test_grounding_model_comparison.py --save`

**Reusable Tool Loop infrastructure** (`ai_service.py:call_ai_with_validation_tools`, `tool_loop_models.py`, `deterministic_validation.DraftValidationResult`): Shared `validate_draft` tool loop reused by the Generator today (Phase 1) and the evaluator layers (QA, Arbiter, GranSabio) in later phases. Callers choose `loop_scope` (`GENERATOR` / `QA` / `ARBITER` / `GRAN_SABIO`), `output_contract` (`FREE_TEXT` / `JSON_LOOSE` / `JSON_STRUCTURED`), and `payload_scope` (`GENERATOR` shows approved/hard_failed to the LLM; `MEASUREMENT_ONLY` strips gate verdicts so evaluators see only objective metrics). `DraftValidationResult` is the internal validator return type; `DraftValidationResult.build_visible_payload(scope)` produces the filtered dict that reaches the model. Evaluators pass `initial_measurement_text=<text_under_review>` so the server computes `validate_draft` before the first turn and injects it into the system prompt as `<initial_measurement>...</initial_measurement>`; the generator keeps `initial_measurement_text=None` (model-driven). The entry point returns `Tuple[str, ToolLoopEnvelope]` with `tools_skipped_reason` populated for the centralized fallbacks (Responses API, unsupported providers, `context_too_large` fail-fast). Per-layer budgets live in `config.py` (`QA_MAX_TOOL_ROUNDS`, `ARBITER_MAX_TOOL_ROUNDS`, `GRAN_SABIO_*_MAX_TOOL_ROUNDS`, `TOOL_LOOP_MAX_PROMPT_CHARS`, `VALIDATE_DRAFT_MAX_LENGTH`); per-request activation flags are `generation_tools_mode`, `qa_tools_mode`, `arbiter_tools_mode`, `gransabio_tools_mode`.

**Generator Tool Loop** (`ai_service.py`): When the request has measurable constraints (word count, phrase frequency, lexical diversity, JSON output/schema, cumulative repetition) and the provider supports tool calling, the generator runs inside a validation tool loop instead of a single-shot call. The loop exposes one combined `validate_draft(text)` tool that returns approved/hard_failed/score/issues/metrics for every measurable check at once.
- **Activation**: `generation_tools_mode` request field (`auto`/`never`), default `auto`. `auto` enables when supported providers are used and any measurable validator is active (see `_should_use_generation_tools` in `core/generation_processor.py`).
- **Supported providers**: `openai`, `claude`/`anthropic`, `gemini`/`google`, `xai`, `openrouter`.
- **Excluded**: OpenAI Responses API models (`o3-pro`, `gpt-5-pro`) — different tool-call contract.
- **Rounds budget**: `max(2, min(5, max_iterations))` external rounds per generation (`core/generation_processor.py:744`).
- **Tool call budget**: `max(4, max_rounds * 2)` total calls before a force-finalize turn is injected (`ai_service.py:_get_tool_loop_call_budget`).
- **Long Text**: uses its own per-section budget `LONG_TEXT_MAX_TOOL_ROUNDS_PER_SECTION` (env override supported).
- **Design rationale**: a single combined `validate_draft` tool outperforms multiple specialized mechanical tools for medium/long drafts. This rule applies to deterministic (Python-cost) validators; AI-cost validators that require another model call are a separate architectural decision.

**Shared Tool Loop infrastructure** (`ai_service.call_ai_with_validation_tools`): The same validation-tool-loop entry point backs the four layers that need objective draft metrics — Generator, QA, Arbiter, GranSabio. The loop is parameterised by:
- `output_contract`: `FREE_TEXT` for prose, `JSON_LOOSE` for schema-free AI JSON extraction/validation, or `JSON_STRUCTURED` for schema-enforced JSON. Only `JSON_STRUCTURED` attaches the parsed dict in `envelope.payload`; loose JSON is normalized and validated but its payload remains `None`. Invalid caller combinations (`FREE_TEXT + response_format`, `JSON_LOOSE + response_format`, missing/empty/non-object `JSON_STRUCTURED` schema) raise `ToolLoopContractError`.
- `payload_scope`: `GENERATOR` (full `DraftValidationResult` visible to the model) or `MEASUREMENT_ONLY` (evaluators get only neutral metrics/checks/score/word_count/issues — no `approved`/`hard_failed` gate fields that would leak semantic judgement).
- `loop_scope`: `GENERATOR` / `QA` / `ARBITER` / `GRAN_SABIO` — stamped on the envelope for telemetry.
- `initial_measurement_text`: when set (evaluators), the server computes `validate_draft(text)` before the first turn and injects the filtered payload into the system prompt as `<initial_measurement>{json}</initial_measurement>`. Generator keeps it `None` (model-driven).
- `tool_event_callback`: per-turn hooks (`tool_call_start`, `tool_call_result`, `force_finalize`, `tool_loop_complete`) that each adapter pre-binds to `session_id` + phase label (`generation`/`qa`/`arbiter`/`gran_sabio`) so `/monitor` can render live progress. Persistence to the debugger DB is independent via `debug_event_callback` (Arbiter only, Side-quest 2).
- **Per-layer activation flags**: `qa_tools_mode`, `arbiter_tools_mode`, `gransabio_tools_mode` (all `Literal["auto","never"]`, default `"auto"`). Fail-fast when the prompt exceeds the estimated context window via `estimate_prompt_overflow(model_id, prompt_chars, max_tokens, thinking_budget)` (primary: `config.get_model_info`; paranoid fallback: `TOOL_LOOP_MAX_PROMPT_CHARS=200000`).
- **Internal vs visible payload**: `deterministic_validation.DraftValidationResult` is the internal dataclass the loop uses for gate decisions. `build_visible_payload(PayloadScope)` filters what the LLM sees in the `tool_response`. Legacy `DeterministicValidationReport` was removed (no shim). Schema `maxLength` on `validate_draft.text` plus a centralized server-side check in `_invoke_validation_callback` raise `ValidationToolInputTooLarge` if the model sends an oversized argument.
- **Arbiter**: `ARBITER_RESPONSE_SCHEMA` strictly enforces `APPLY|DISCARD` only (MERGE removed). `_parse_arbiter_response` is fail-closed on missing/duplicate/out-of-range indices or invalid decisions — NO default-apply fallback. The old `_parse_ai_response_json` helper is gone; the JSON contract is centralised in the tool loop.
- **GranSabio**: the 3 live methods (`review_minority_deal_breakers`, `regenerate_content`, `review_iterations`) use JSON_STRUCTURED contracts via `GRAN_SABIO_MINORITY_OVERRIDE_SCHEMA` and `GRAN_SABIO_ESCALATION_SCHEMA` (plus deliberate `FREE_TEXT` for regenerate). JSON content regenerated by GranSabio is enforced afterward by the shared JSON post-guard with the original request schema/expectations. `handle_model_conflict` was dead code and has been deleted. The legacy `_extract_*` / `_parse_gran_sabio_response` tag parsers are gone. The post-parse adapter preserves the invariant "`final_content` is the best content GranSabio knows" — empty only when literally nothing can be recovered.
- **QA**: `validation_context_factory.build_measurement_request_for_layer(request, layer)` builds a whitelisted `MeasurementRequest` (fail-closed — no semantic inference from `layer.criteria`). Tools activate only when the request carries structured request-level validators (min/max words, phrase_frequency, lexical_diversity, json_output/content_type=json/target_field) and are skipped for layers handled by `qa_bypass_engine` (deterministic bypass wins). Scheduler retries stay outside the loop (`retries_enabled=False`).

**JSON Output Contracts**: The system supports both flexible JSON and native JSON Schema structured outputs across all major AI providers. `json_output=True` and `content_type="json"` are equivalent request signals.
- **JSON_LOOSE**: Used when JSON is requested and `json_schema` is omitted/null. The first JSON object or array is extracted from common AI wrappers, markdown fences are unwrapped, and conservative repairs handle typical syntax slips such as trailing commas or JSON5-like keys/strings. Top-level scalars, missing JSON, truncation that cannot be repaired, and expectation failures are rejected.
- **JSON_STRUCTURED**: Used when a non-empty `json_schema` is provided. The schema must describe a top-level object for tool-loop structured outputs, and `envelope.payload` carries the parsed dict.
- **Public request validation**: `json_schema={}` is rejected. Omit `json_schema` or pass `null` for `JSON_LOOSE`; provide a non-empty object schema for `JSON_STRUCTURED`. A schema without effective JSON output is also rejected.
- **Grok (xAI)**: Uses `response_format` with `json_schema` type for all Grok 2-1212+ models
- **Gemini**: Uses `response_schema` parameter in both new and legacy SDKs (all active models supported)
- **Claude Sonnet 4.5 / Opus 4.1**: Uses beta Structured Outputs with `output_format` and header `anthropic-beta: structured-outputs-2025-11-13`. Thinking mode is supported alongside structured outputs (grammar only constrains final text, not thinking blocks)
- **Claude 3.x models**: Fallback to prompt engineering approach (no native structured outputs)
- **OpenAI (GPT-4o, GPT-5, O1/O3)**: Uses `response_format` with `json_schema` type for Chat Completions API
- **OpenAI (O3-Pro, GPT-5 Pro)**: Uses `text.format` with `json_schema` for Responses API
- **OpenRouter/Mistral**: Uses `response_format` with `json_schema` type for compatible models
- **Dual operation**: When `json_schema` is provided, models use native structured outputs; when omitted, the JSON_LOOSE contract uses provider JSON mode plus tolerant extraction followed by strict client-side validation of the recovered JSON payload.
- **Validation**: `json_schema` is used both for model-side generation guarantees AND client-side validation via `validate_ai_json()`; JSON_LOOSE uses `validate_loose_json()`.
- **QA evaluations**: QA uses dedicated simple/editable schemas in `qa_response_schemas.py`; parse failures must raise `QAResponseParseError` so multi-model QA can skip only the invalid evaluator instead of manufacturing deal-breakers.
- **Audit helper note**: `_audit_model_supports_structured_outputs("openai", "o3-pro")` returns `True`; Responses API OpenAI Pro models are expected to support strict structured audit payloads through the Responses text format path.

## Language-Agnostic Intelligence (MANDATORY)

**Do NOT use regex, keyword lists, or hardcoded string patterns for semantic
tasks.** This is a hard rule with no exceptions.

Semantic tasks include: temporal expression detection, intent classification,
entity extraction, topic detection, need/signal classification, query
understanding, content categorization — anything where the input is natural
language and the output requires understanding meaning.

**Why:** Keyword lists only work for one language, break on paraphrases, and
silently miss edge cases. LLMs have billions of parameters trained on every
language and every phrasing variant — they ARE the evolved, complete keyword
dictionary. Rebuilding a fraction of that with regex is anachronistic and
creates silent bugs that only show up in production on inputs nobody tested.

**What to use instead:** LLM calls (provider-agnostic). For latency-sensitive
paths, use small/fast models or structured output with constrained schemas.
For truly zero-latency needs, use pre-computed LLM results stored in the
database, not runtime regex.

**Acceptable uses of regex:** Parsing structured/mechanical formats ONLY —
SQL sanitization, UUID validation, FTS query syntax cleanup, log parsing,
config file parsing. If the input is machine-generated with a known grammar,
regex is fine. If the input is human language, use an LLM.

Mechanical IR cleanup for retrieval is allowed when it does not perform
semantic interpretation (tokenization, FTS-safe sanitization, duplicate-token
collapse, operator stripping). What is forbidden is hardcoded semantic
understanding tied to specific words, languages, or benchmark phrasing.

**When reviewing or modifying existing code:** If you encounter regex or
keyword lists used for semantic understanding, flag it as tech debt to be
replaced with an LLM call. Do not extend or "improve" keyword lists — that
deepens the problem.

## Important Development Considerations

**Model Handling**: The system automatically applies safety margins to token limits and handles model-specific requirements. New models should be added to `model_specs.json` with proper configuration.

**Error Handling**: Deal-breakers can terminate evaluation early. Always check for `deal_breaker` flags in QA results. Preflight validation can reject requests before generation starts with specific feedback.

**Preflight Validation**: The system includes `preflight_validator.py` which analyzes requests for feasibility before expensive generation iterations. Key considerations:
- Detects prompt/QA layer incompatibilities (fiction vs historical accuracy)
- Identifies internal contradictions in requests
- Returns structured feedback with decision (proceed/reject/manual_review)
- Prevents resource waste on impossible generation scenarios
- Runs for every `/generate` request. There is no skip flag; requests that set `qa_layers: []` skip semantic QA but not the preflight gate.

**Testing**: Use `verbose=true` in requests for full logging. The `example_request.py` demonstrates complete workflows including biography, script, and article generation.

**Shared AI Service usage**: When writing new modules or tests, import `get_ai_service` and pass the returned instance into engines (`QAEngine`, `ConsensusEngine`, `GranSabioEngine`). Patch `get_ai_service` in tests instead of constructing `AIService()` directly to avoid hitting real network clients.

**Test Files Organization**: All test files, benchmarks, and development utilities should be created in the `dev_tests/` directory. This includes:
- Unit tests (`test_*.py`)
- Integration tests
- Performance benchmarks (`*_benchmark.py`)
- Development utilities and temporary test scripts
- Model-specific test files
- Preflight validation tests (`test_preflight_*.py`, `debug_preflight_*.py`)

When creating new test files, always place them in `dev_tests/` to keep the root directory clean and organized.

**Session Cleanup**: Sessions are stored in memory (`active_sessions` dict). A startup background task now prunes expired sessions every `config.SESSION_CLEANUP_INTERVAL` seconds and trims verbose logs to `config.VERBOSE_MAX_ENTRIES`. Override the interval via `SESSION_CLEANUP_INTERVAL` env var; legacy `CLEANUP_INTERVAL` remains supported.

## API Endpoints

Core endpoints follow RESTful patterns:
- `POST /generate` - Start content generation with preflight validation (returns session_id and preflight_feedback)
- `GET /status/{session_id}` - Get current status
- `GET /stream/project/{project_id}` - Real-time progress stream (unified: project_id = session_id when not explicitly provided)
- `GET /result/{session_id}` - Final approved content
- `POST /stop/{session_id}` - Cancel/stop active content generation session
- `GET /models` - Available AI models with specifications
- `GET /api-docs` - Interactive API documentation

**Attachment Endpoints**:
- `POST /attachments` - Upload files via multipart/form-data
- `POST /attachments/base64` - Upload files as base64 JSON payload
- `POST /attachments/url` - Download files from remote URLs with security validation
- All attachment endpoints support username scoping and metadata persistence

**Analysis Endpoints** (Text Analysis Tools):
- `POST /analysis/lexical-diversity` - Analyze vocabulary richness with metrics (MTLD, HD-D, Yule's K, Herdan's C)
  - Returns GREEN/AMBER/RED grades based on configurable thresholds
  - Supports auto metric selection by text length
  - Optional window-based analysis for detecting local repetition zones
  - Optional top words frequency analysis
- `POST /analysis/repetition` - Analyze n-gram repetition patterns
  - Detects exact phrase repetitions with configurable n-gram length (min_n to max_n)
  - Computes distance metrics between repetitions
  - Optional cluster detection for grouped repetitions
  - Optional diagnostics for over-repetition detection
  - Optional positional bias analysis (sentence/paragraph/block starts/ends)
  - Multi-process support for large texts (>50k tokens)
- `GET /analysis/health` - Health check for analysis endpoints

**Project Streaming Endpoint**:
- `GET /stream/project/{project_id}` - Unified real-time SSE stream for project events
  - Query param `phases`: CSV of phases to subscribe (default: `all`)
  - Valid phases: `preflight`, `generation`, `qa`, `consensus`, `gransabio`/`gran_sabio`, `status`
  - All events include `phase` field for easy client-side filtering
  - If `status` included, sends initial `status_snapshot` after connection
  - Heartbeat every 15 seconds keeps connection alive
  - Examples: `/stream/project/abc?phases=generation,qa`, `/stream/project/abc?phases=status`

**Important**: The `/generate` endpoint now returns a `GenerationInitResponse` that includes preflight validation feedback. Sessions can be rejected at the preflight stage with status `"preflight_rejected"` if contradictions are detected.

The web interface at `/` provides a complete UI for testing all functionality with real-time progress monitoring and cancellation controls.

## Development Environment Notes

- Always write your code, variables, and comments in US English. Talk to the user in Spanish from Spain.

**Windows-Specific Considerations**:
- Never use emojis in Python code that outputs to console (causes encoding errors on Windows)
- Use `start2.bat` to set up conda environment automatically
- Emojis are acceptable in web templates and FastAPI responses

**Performance Optimizations**:
- All JSON operations use orjson via `json_utils.py` (3.6x faster than standard json)
- AI service connection pooling reduces latency for repeated requests
- Session cleanup background task prevents memory leaks

### GitHub Sync

When asked to sync to GitHub ("sincroniza github", "sync github",
"actualiza github"), read and follow `SYNC_GITHUB.md` step by step.

NOTE: "guarda en git", "commit", "guarda los cambios" etc. mean a normal
git commit in the dev repo — NOT the GitHub sync. Only trigger the sync
workflow when "github" is explicitly mentioned.

The sync tool lives in `tools/sync_github/` (scan.py, apply.py, config.py,
tests/test_transforms.py) and is itself excluded from the public mirror.
