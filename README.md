# Gran Sabio LLM Engine

### Multi-Layer AI Quality Assurance for Content Generation

**Your AI generates content but you can't trust it blindly?** Put Gran Sabio LLM in the middle. Multiple AI models evaluate, score, and approve every piece of content before it reaches you.

---

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
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

Beyond direct API connections (OpenAI, Anthropic, Google, xAI), you can access **all models available on OpenRouter** - including Mistral, DeepSeek, LLaMA, Qwen, and many more.

![OpenRouter Models](screenshots/openrouter-models.png)
*Access hundreds of models through OpenRouter integration*

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

**Each layer is evaluated by ALL configured QA models.** If GPT-4o passes but Claude finds an issue, you'll know. Consensus is calculated automatically.

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
- **Flexible model choice:** Use GPT-5.2-Pro for maximum accuracy or Claude Opus 4.5 for deep reasoning

```json
{
  "gran_sabio_model": "claude-opus-4-5-20251101",
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

Content is approved immediately. Perfect for testing, bulk generation, or content that will be manually edited.

---

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/jordicor/Gran_Sabio_LLM.git
cd Gran_Sabio_LLM
python quick_start.py
```

### 2. Configure Your API Keys

Create `.env` file with **your own API keys** from each provider:

```env
# Get your keys from each provider's dashboard:
# - OpenAI: https://platform.openai.com/api-keys
# - Anthropic: https://console.anthropic.com/
# - Google: https://aistudio.google.com/apikey
# - xAI: https://console.x.ai/
# - OpenRouter: https://openrouter.ai/keys

OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
XAI_API_KEY=xai-...
OPENROUTER_API_KEY=sk-or-...
PEPPER=any-random-string-here
```

> **Note:** You only need keys for the providers you want to use. At minimum, configure one provider.

### 3. Start the Server

```bash
python main.py
```

Server starts at `http://localhost:8000`

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

Full interactive documentation available at:
- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`
- **Custom Docs:** `http://localhost:8000/api-docs` *(recommended)*

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/generate` | POST | Start content generation with QA |
| `/status/{session_id}` | GET | Check session status |
| `/stream/project/{project_id}` | GET | Real-time SSE progress stream (project_id = session_id when not explicit) |
| `/result/{session_id}` | GET | Get final approved content |
| `/stop/{session_id}` | POST | Cancel active generation |
| `/models` | GET | List available AI models |
| `/debugger` | GET | Session history and inspection UI |

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
pip install -r mcp/requirements.txt
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
claude mcp add gransabio-llm -- python /path/to/Gran_Sabio_LLM/mcp_server/gransabio_mcp_server.py
```

**Gemini CLI** (`~/.gemini/settings.json`):
```json
{
  "mcpServers": {
    "gransabio-llm": {
      "command": "python",
      "args": ["/path/to/Gran_Sabio_LLM/mcp_server/gransabio_mcp_server.py"]
    }
  }
}
```

**Codex CLI** (`~/.codex/config.toml`):
```toml
[mcp_servers.gransabio-llm]
command = "python"
args = ["/path/to/Gran_Sabio_LLM/mcp_server/gransabio_mcp_server.py"]
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

### Remote/SaaS Configuration

For hosted Gran Sabio LLM instances:
```bash
claude mcp add gransabio-llm \
  --env GRANSABIO_API_URL=https://api.gransabio.example.com \
  --env GRANSABIO_API_KEY=your-api-key \
  -- python /path/to/gransabio_mcp_server.py
```

See full documentation: **[mcp/README.md](mcp/README.md)**

---

## Self-Hosting (Bring Your Own API Keys)

Gran Sabio LLM is currently a **self-hosted solution**. You deploy it on your infrastructure and use your own API keys from each AI provider.

### What This Means

| Aspect | Self-Hosted |
|--------|-------------|
| **API Keys** | You obtain and configure keys from OpenAI, Anthropic, Google, xAI, and/or OpenRouter |
| **Billing** | Each provider bills you directly based on your usage |
| **Infrastructure** | You host and maintain the server |
| **Data Privacy** | Your prompts and content stay on your infrastructure |
| **Models Available** | All models your API keys have access to, plus 200+ via OpenRouter |

### Why Self-Hosting?

- **Full control** over your data and costs
- **No intermediaries** - direct connection to AI providers
- **Use your existing accounts** - no new subscriptions needed
- **Enterprise compliance** - deploy in your own cloud/datacenter
- **Unlimited usage** - no rate limits beyond provider limits

### Requirements

- Python 3.10+
- **API keys for at least one provider** (OpenAI, Anthropic, Google, xAI, or OpenRouter)
- ~500MB disk space for dependencies
- Recommended: 4GB RAM minimum

### Production Deployment

```bash
# With uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4

# With gunicorn + uvicorn workers
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

---

## Coming Soon: Gran Sabio LLM Cloud

**Don't want to manage API keys and infrastructure?** A hosted version is in development.

### What's Coming

| Feature | Cloud Version |
|---------|---------------|
| **API Keys** | None required - we handle all provider connections |
| **Setup** | Zero - just sign up and start making API calls |
| **Billing** | Single subscription covers all AI providers |
| **Models** | All supported models, always up to date |
| **Features** | Everything in self-hosted, fully managed |
| **Web Interface** | Polished, production-ready UI for non-developers |

### Get Early Access

Want to be notified when the Cloud version launches? **Star this repo** and **follow me on [GitHub](https://github.com/jordicor)** or my social media channels - I'll announce early access there first.

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
- **Follow me on social media** for development insights, AI tips, and early announcements about the upcoming Cloud version

Find my social links on my [GitHub profile](https://github.com/jordicor).

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

<p align="center">
  <strong>Trust your AI output.</strong><br>
  <em>Let multiple models validate before you ship.</em>
</p>
