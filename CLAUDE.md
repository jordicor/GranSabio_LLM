# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture Overview

Gran Sabio LLM Engine is a sophisticated content generation API that uses multiple AI providers (OpenAI, Anthropic, Google, xAI) with a multi-layer quality assurance system. The architecture follows a service-oriented pattern:

- **Preflight Validator** (`preflight_validator.py`): Intelligent feasibility analysis before content generation starts
- **AI Service Layer** (`ai_service.py`): Unified interface for multiple AI providers with model-specific parameter handling. Use `get_ai_service()` to reuse the shared connector instead of instantiating `AIService` directly.
- **QA Engine** (`qa_engine.py`): Multi-layer evaluation system with configurable criteria and deal-breaker detection
- **QA Bypass Engine** (`qa_bypass_engine.py`): Alternative QA system for specific use cases
- **Arbiter** (`arbiter.py`): Per-layer intelligent conflict resolver for smart-edit operations. Detects and resolves conflicts between edits proposed by different QA evaluators, maintains edit history per layer, and verifies edit alignment with original request intent.
- **Consensus Engine** (`consensus_engine.py`): Calculates consensus scores across multiple evaluator models for final approval/rejection decision
- **Gran Sabio Engine** (`gran_sabio.py`): Final escalation system for conflict resolution and content modification
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
- Pre-defined templates for biography, script, novel content types
- **QA Bypass**: Use empty qa_layers [] to completely bypass QA and preflight validation for rapid generation

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

**Preflight Validation**: Before starting content generation, the system analyzes request feasibility using GPT-4o to detect contradictions, incompatibilities, and impossible scenarios. This prevents wasted resources on requests that cannot succeed (e.g., asking for fiction with historical accuracy validation).

**Attachment Ingestion**: `/attachments`, `/attachments/base64`, and `/attachments/url` persist metadata through `AttachmentManager`. URL uploads require HTTPS, respect configurable redirect limits, enforce MIME checks, and cache successful downloads per usuario/URL via `AttachmentRecord.original_url`. Tune size limits, allowed schemes, host allow/block lists, timeouts, and cache TTL under `config.ATTACHMENTS`.
**Attachment Maintenance**: AttachmentManager.run_cleanup() valida firmas/sha, reconstruye uploads/index.json, respeta config.ATTACHMENTS.retention_days (override con ATTACHMENTS_RETENTION_DAYS) y dispone de CLI (python tools/attachments_cli.py cleanup|list|delete) para operaciones manuales (dry-run por defecto).

**Gran Sabio Escalation**: When max iterations reached without approval, the system escalates to a premium model for final decision and potential content modification.

**Arbiter System**: Per-layer intelligent arbitration for smart-edit operations:
- **Conflict Detection**: Detects incompatible edits (DELETE vs REPLACE on same fragment, EXPAND vs CONDENSE on same paragraph, severity mismatches, semantic redundancy, edit cycles)
- **Distribution Classification**: Classifies how edits are distributed among QA models (CONSENSUS, MAJORITY, MINORITY, CONFLICT, TIE, SINGLE_QA)
- **Model Escalation**: Uses economic model (gpt-4o-mini) for normal cases; escalates to GranSabio model for difficult cases (MINORITY, CONFLICT, TIE)
- **AI Verification**: ALWAYS calls AI to verify edit alignment with original request - can reject poorly-reasoned edits even without conflicts
- **Edit History**: Maintains per-layer history of applied/discarded edits, injected into QA prompts in subsequent rounds to prevent re-proposing discarded edits
- **Configuration**: `ARBITER_MODEL`, `ARBITER_MAX_TOKENS`, `ARBITER_TEMPERATURE`, `EDIT_HISTORY_MAX_ROUNDS`, `EDIT_HISTORY_MAX_CHARS`

**Smart Edit JSON Field Extraction**: When generating JSON output, smart-edit needs to work on the text content, not the JSON structure. The system automatically extracts text fields for editing:
- **`target_field`**: Specify which JSON field(s) contain the primary text using jmespath notation (e.g., `"generated_text"`, `"data.content"`, or `["chapter", "notes"]` for multiple fields)
- **`target_field_only`**: When `true`, AI QA receives only extracted text (saves tokens). When `false`, AI QA receives full JSON. Algorithmic guards always use extracted text when target_field is set.
- **`smart_edit_locator_mode`**: Defaults to `ids`, using stable paragraph/sentence IDs (`p1`, `p1s2`) for smart-edit targeting. Set to `legacy` to keep phrase markers plus automatic word-index fallback.
- **`generation_tools_mode`**: Controls generator-side deterministic validation tools (`auto`, `always`, `never`). `auto` enables the tool loop for supported providers (`OpenAI`, `Claude`, `Gemini`, `xAI`, `OpenRouter`) when measurable constraints such as word count, phrase frequency, lexical diversity, cumulative repetition, or JSON field extraction are active.
- **Auto-detection**: If `target_field` not specified, automatically finds the largest string field (errors if ambiguous)
- **Markdown extraction**: Automatically extracts JSON from markdown ```json code blocks
- **Reconstruction**: After smart-edit completes, the edited text is reconstructed back into the original JSON structure
- See `json_field_utils.py` for implementation details

**Evidence Grounding (Strawberry)**: Detects procedural hallucination by measuring if the model actually relied on cited evidence. Uses a dual-model architecture for cost optimization:
- **Claim Extraction** (`EVIDENCE_GROUNDING_EXTRACTION_MODEL`): Uses gpt-5-nano by default (cheapest: $0.05/$0.40 per M tokens). Does NOT require logprobs.
- **Budget Scoring** (`EVIDENCE_GROUNDING_SCORING_MODEL`): Uses gpt-4o-mini by default (requires logprobs support). Measures P(YES|evidence) vs P(YES|no_evidence) to detect confabulation.
- Alternative models: `grok-4-1-fast-non-reasoning` for speed (2.5x faster extraction, supports logprobs with top_logprobs=8 max)
- Benchmark script: `python dev_tests/test_grounding_model_comparison.py --save`

**Generator Tool Loop** (`ai_service.py`): When the request has measurable constraints (word count, phrase frequency, lexical diversity, JSON schema, cumulative repetition) and the provider supports tool calling, the generator runs inside a validation tool loop instead of a single-shot call. The loop exposes one combined `validate_draft(text)` tool that returns approved/hard_failed/score/issues/metrics for every measurable check at once.
- **Activation**: `generation_tools_mode` request field (`auto`/`always`/`never`), default `auto`. `auto` enables when supported providers are used and any measurable validator is active (see `_should_use_generation_tools` in `core/generation_processor.py`).
- **Supported providers**: `openai`, `claude`/`anthropic`, `gemini`/`google`, `xai`, `openrouter`.
- **Excluded**: OpenAI Responses API models (`o3-pro`, `gpt-5-pro`) — different tool-call contract.
- **Rounds budget**: `max(2, min(5, max_iterations))` external rounds per generation (`core/generation_processor.py:744`).
- **Tool call budget**: `max(4, max_rounds * 2)` total calls before a force-finalize turn is injected (`ai_service.py:_get_tool_loop_call_budget`).
- **Long Text**: uses its own per-section budget `LONG_TEXT_MAX_TOOL_ROUNDS_PER_SECTION` (env override supported).
- **Design rationale**: a single combined `validate_draft` tool outperforms multiple specialized mechanical tools for medium/long drafts. This rule applies to deterministic (Python-cost) validators; AI-cost validators that require another model call are a separate architectural decision.

**JSON Schema Structured Outputs**: The system supports native JSON Schema structured outputs across all major AI providers (implemented Nov 2025). Key behaviors:
- **Grok (xAI)**: Uses `response_format` with `json_schema` type for all Grok 2-1212+ models
- **Gemini**: Uses `response_schema` parameter in both new and legacy SDKs (all active models supported)
- **Claude Sonnet 4.5 / Opus 4.1**: Uses beta Structured Outputs with `output_format` and header `anthropic-beta: structured-outputs-2025-11-13`. Thinking mode is supported alongside structured outputs (grammar only constrains final text, not thinking blocks)
- **Claude 3.x models**: Fallback to prompt engineering approach (no native structured outputs)
- **OpenAI (GPT-4o, GPT-5, O1/O3)**: Uses `response_format` with `json_schema` type for Chat Completions API
- **OpenAI (O3-Pro, GPT-5 Pro)**: Uses `text.format` with `json_schema` for Responses API
- **OpenRouter/Mistral**: Uses `response_format` with `json_schema` type for compatible models
- **Dual operation**: When `json_schema` is provided, models use native structured outputs; when omitted, uses flexible JSON mode (backward compatible)
- **Validation**: `json_schema` is used both for model-side generation guarantees AND client-side validation via `validate_ai_json()`
- **QA evaluations**: QA uses dedicated simple/editable schemas in `qa_response_schemas.py`; parse failures must raise `QAResponseParseError` so multi-model QA can skip only the invalid evaluator instead of manufacturing deal-breakers.

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
