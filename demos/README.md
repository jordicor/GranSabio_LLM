# Gran Sabio LLM - Demos

This folder contains demonstration scripts showcasing the capabilities of Gran Sabio LLM.

## Quick Start

1. **Start the API server** (from project root):
   ```bash
   python main.py
   ```

2. **Run a demo**:
   ```bash
   python demos/01_simple_article.py
   ```

## Demo Overview

| # | Demo | Description | Complexity |
|---|------|-------------|------------|
| 01 | [Simple Article](#01-simple-article) | Basic generation without QA | Beginner |
| 02 | [LinkedIn Post](#02-linkedin-post) | Content with QA validation | Intermediate |
| 03 | [YouTube Script Generator](#03-youtube-script-generator) | Multi-phase pipeline | Advanced |
| 04 | [Text Quality Analyzer](#04-text-quality-analyzer) | Analyze text without generating | Intermediate |
| 05 | [JSON Structured Output](#05-json-structured-output) | Schema-enforced JSON | Intermediate |
| 06 | [Content with Sources](#06-content-with-sources) | Attachments and context | Advanced |
| 07 | [Creative Story](#07-creative-story) | Fiction with QA | Intermediate |
| 08 | [Parallel Generation](#08-parallel-generation) | Bulk content creation | Advanced |
| 09 | [Preflight Validation](#09-preflight-validation) | Request validation | Intermediate |
| 10 | [Reasoning Models](#10-reasoning-models) | Deep thinking models | Advanced |
| 11 | [Code Analyzer](#11-code-analyzer) | Dynamic JSON (no strict schema) | Advanced |

---

## Demo Details

### 01: Simple Article
**File:** `01_simple_article.py`

Demonstrates the most basic API usage - generate content without quality validation.

```bash
python demos/01_simple_article.py
```

**Features:**
- `qa_layers: []` for bypass mode
- Fast generation, single iteration
- Perfect for prototyping

**Use cases:** Rapid prototyping, bulk drafts, content that will be manually edited.

---

### 02: LinkedIn Post
**File:** `02_linkedin_post.py`

Generate professional social media content with quality validation.

```bash
python demos/02_linkedin_post.py
```

**Features:**
- Word count enforcement
- Two QA layers (Professional Tone, Clarity)
- Multiple evaluator models

**Use cases:** Business communications, social media content, professional writing.

---

### 03: YouTube Script Generator
**File:** `03_youtube_script_generator.py`

**STAR DEMO** - Complete multi-phase content generation pipeline.

```bash
python demos/03_youtube_script_generator.py

# Custom topic:
python demos/03_youtube_script_generator.py --topic "How to Learn Programming"

# Skip optional phases:
python demos/03_youtube_script_generator.py --skip-thumbnails
```

**Phases:**
1. **Topic Analysis** (JSON) - Structured content planning
2. **Script Generation** (Text + QA) - Full video script with validation
3. **Scene Breakdown** (JSON) - Visual production guide
4. **Thumbnail Ideas** (JSON) - Clickable thumbnail concepts

**Features:**
- Multi-call sequential pipeline
- Project ID for grouping
- JSON Schema structured outputs
- Lexical diversity monitoring
- QA for engagement and structure

**Use cases:** YouTube automation, video production, content agencies.

---

### 04: Text Quality Analyzer
**File:** `04_text_quality_analyzer.py`

Analyze existing text without generating new content.

```bash
python demos/04_text_quality_analyzer.py

# Analyze a file:
python demos/04_text_quality_analyzer.py --file path/to/text.txt

# Compare AI vs human writing:
python demos/04_text_quality_analyzer.py --sample both
```

**Features:**
- Lexical diversity metrics (MTLD, HD-D, Yule's K)
- Phrase repetition analysis
- AI pattern detection
- Writing improvement recommendations

**Use cases:** Content editing, AI detection, quality control.

---

### 05: JSON Structured Output
**File:** `05_json_structured_output.py`

Guaranteed JSON format compliance using schemas.

```bash
python demos/05_json_structured_output.py

# Simple schema:
python demos/05_json_structured_output.py --simple

# Different model:
python demos/05_json_structured_output.py --model claude-sonnet-4-20250514
```

**Features:**
- `json_schema` for 100% format guarantee
- Multi-provider support (GPT, Claude, Gemini, Grok)
- Complex nested schemas
- Zero parsing errors

**Use cases:** API integrations, data extraction, structured pipelines.

---

### 06: Content with Sources
**File:** `06_content_with_sources.py`

Generate content based on reference documents.

```bash
python demos/06_content_with_sources.py

# Custom source:
python demos/06_content_with_sources.py --source research.txt
```

**Features:**
- Attachment upload system
- Context document injection
- Source fidelity QA layer
- Accurate citation

**Use cases:** Research-based content, document summarization, knowledge bases.

---

### 07: Creative Story
**File:** `07_creative_story.py`

Generate creative fiction with appropriate quality control.

```bash
python demos/07_creative_story.py

# Different genre:
python demos/07_creative_story.py --genre sci-fi

# Custom premise:
python demos/07_creative_story.py --premise "A time traveler opens a bakery"
```

**Features:**
- Fiction-oriented QA layers
- Higher temperature for creativity
- Phrase frequency guard
- Lexical diversity monitoring

**Available genres:** fantasy, sci-fi, mystery, romance, horror, comedy, literary

**Use cases:** Short story generation, creative writing assistance, entertainment content.

---

### 08: Parallel Generation
**File:** `08_parallel_generation.py`

Launch multiple generation requests simultaneously.

```bash
python demos/08_parallel_generation.py

# More variations:
python demos/08_parallel_generation.py --count 5

# Timeout limit:
python demos/08_parallel_generation.py --timeout 60
```

**Features:**
- Async parallel execution
- Project ID grouping
- Progress monitoring
- Session cancellation

**Use cases:** Bulk content creation, A/B testing, content calendars.

---

### 09: Preflight Validation
**File:** `09_preflight_validation.py`

See how the system catches impossible or contradictory requests.

```bash
python demos/09_preflight_validation.py

# Only invalid examples:
python demos/09_preflight_validation.py --invalid-only
```

**Test cases:**
- Valid request (accepted)
- Fiction vs historical accuracy (rejected)
- Impossible word count (rejected)
- Contradictory tone requirements (rejected)

**Use cases:** Understanding validation, debugging requests, building integrations.

---

### 10: Reasoning Models
**File:** `10_reasoning_models.py`

Use advanced models that "think" before responding.

```bash
python demos/10_reasoning_models.py

# Specific model:
python demos/10_reasoning_models.py --model gpt-5

# High reasoning effort:
python demos/10_reasoning_models.py --effort high

# Different problem:
python demos/10_reasoning_models.py --problem ethical_analysis
```

**Supported models:**
| Model | Parameter | Values |
|-------|-----------|--------|
| GPT-5, O1, O3 | `reasoning_effort` | low, medium, high |
| Claude 3.7/4 | `thinking_budget_tokens` | 1024-16000+ |

**Problems available:** logic_puzzle, ethical_analysis, technical_design

**Use cases:** Complex analysis, multi-step reasoning, high-stakes content.

---

### 11: Code Analyzer
**File:** `11_code_analyzer.py`

**DYNAMIC JSON DEMO** - Analyze code and extract issues using flexible JSON output.

```bash
python demos/11_code_analyzer.py

# Analyze custom code:
python demos/11_code_analyzer.py --code "def foo(): pass"

# Analyze a file:
python demos/11_code_analyzer.py --file path/to/code.py

# Test with clean code (no issues):
python demos/11_code_analyzer.py --clean
```

**Key concept: Dynamic JSON vs Strict Schema**

This demo shows when to use `json_output=True` WITHOUT `json_schema`:

| Approach | When to Use |
|----------|-------------|
| `json_schema` (Demo 05) | Fixed structure, known fields, strict validation |
| Dynamic JSON (Demo 11) | Variable structure, unknown field count, conditional fields |

**Why dynamic JSON here?**
- Number of issues is unpredictable (0 to N)
- Issue types depend on what's found in the code
- Some fields are conditional (e.g., `cwe_id` only for security issues)
- AI providers don't allow `additionalProperties: true` in strict schemas

**Implementation pattern:**
1. Describe expected format IN the prompt (not as API parameter)
2. Set `json_output=True` (ensures valid JSON)
3. Omit `json_schema` (allows flexibility)
4. Validate response with ai-json-cleanroom (supports dynamic fields)

**Features:**
- Security vulnerability detection (SQL injection, hardcoded secrets, etc.)
- Performance and style issue identification
- CWE ID mapping for security issues
- QA validation for analysis quality

**Use cases:** Code review automation, CI/CD pipelines, security scanning.

---

## Client Library

For production use, we provide a full-featured client library in the `client/` folder:

```python
# Synchronous client (for scripts)
from client import GranSabioClient

client = GranSabioClient()
result = client.generate_fast("Write a haiku")
print(result["content"])

# Asynchronous client (for web apps, demos)
from client import AsyncGranSabioClient

async with AsyncGranSabioClient() as client:
    # Check API health
    info = await client.get_info()

    # Generate content
    result = await client.generate(
        prompt="Your prompt here",
        content_type="article",
        generator_model="gpt-4o"
    )

    # Wait for completion
    final = await client.wait_for_result(result["session_id"])
    print(final["content"])
```

See `client/README.md` for full documentation.

---

## Requirements

- Python 3.9+
- Gran Sabio LLM API running on localhost:8000
- Dependencies: `aiohttp` (installed with main requirements)

Optional:
- `pyperclip` for clipboard support in demo 04

---

## Tips

1. **Start with demo 01** to verify your setup works
2. **Demo 03** is the most comprehensive example of the API capabilities
3. Use `--help` with any demo for available options
4. Check the debugger at `/debugger` to see detailed logs
5. Use `project_id` to group related requests

---

## Troubleshooting

**"Cannot connect to Gran Sabio API"**
- Make sure the server is running: `python main.py`
- Check the server is on port 8000

**"Model not available"**
- Verify your API keys are configured in `.env`
- Check `/models` endpoint for available models

**Slow response times**
- Reasoning models (GPT-5, O3, Claude thinking) are slower by design
- Use faster models (gpt-4o-mini) for quick tests
