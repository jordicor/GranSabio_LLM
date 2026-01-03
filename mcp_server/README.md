# Gran Sabio LLM MCP Server

Model Context Protocol (MCP) server that integrates Gran Sabio LLM's multi-model QA capabilities with AI coding assistants.

## What is This?

This MCP server allows AI coding tools like **Claude Code**, **Gemini CLI**, and **Codex CLI** to use Gran Sabio LLM for:

- **Code Analysis**: Get multi-model consensus on code quality, security issues, and bugs
- **Fix Validation**: Review proposed fixes before applying them
- **Content Generation**: Generate code or documentation with built-in quality assurance
- **Reasoning Control**: Configure thinking depth for complex analysis tasks

Instead of trusting a single AI's analysis, Gran Sabio LLM runs your code through multiple AI models and only approves results when they reach consensus.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r mcp_server/requirements.txt
```

### 2. Ensure Gran Sabio LLM is Running

```bash
# In the project root
python main.py
# Server starts at http://localhost:8000
```

### 3. Add to Your AI Coding Tool

#### Automatic Installation (Recommended)

Use the installer scripts in the project root:

**Windows:**
```cmd
install_mcp.bat
```

**Linux/macOS:**
```bash
chmod +x install_mcp.sh
./install_mcp.sh
```

The scripts automatically detect the correct absolute path and register the MCP server with Claude Code.

To uninstall:
```bash
./install_mcp.sh --uninstall   # Linux/macOS
install_mcp.bat --uninstall    # Windows
```

#### Manual Installation

> **Important**: MCP servers require **absolute paths**. Variables like `${PROJECT_ROOT}` are NOT supported by Claude Code. The only supported variables are system environment variables (`${HOME}`, `${USER}`, etc.) and `${CLAUDE_PLUGIN_ROOT}` (for plugins only).
>
> Relative paths will fail because the working directory is not guaranteed when Claude launches the server.

**Claude Code CLI:**
```bash
# Replace /path/to/Gran_Sabio_LLM with your actual path
claude mcp add gransabio-llm -- python /path/to/Gran_Sabio_LLM/mcp_server/gransabio_mcp_server.py

# Windows example:
claude mcp add gransabio-llm -- python C:\Projects\Gran_Sabio_LLM\mcp_server\gransabio_mcp_server.py

# Linux/macOS example:
claude mcp add gransabio-llm -- python /home/user/Gran_Sabio_LLM/mcp_server/gransabio_mcp_server.py
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
[features]
rmcp_client = true

[mcp_servers.gransabio-llm]
command = "python"
args = ["/path/to/Gran_Sabio_LLM/mcp_server/gransabio_mcp_server.py"]
```

> **Note**: Replace `/path/to/Gran_Sabio_LLM` with your actual installation path.

## Available Tools

### `gransabio_analyze_code`

Analyze code for bugs, security issues, performance problems, and best practices violations.

```
Arguments:
  - code (required): The code to analyze
  - language: Programming language (auto-detected if not specified)
  - context: Additional context about the code
  - focus_areas: Array of areas to focus on (security, performance, bugs, style, all)

  # Reasoning Configuration (optional):
  - generator_model: Override generator model
  - qa_models: Override QA models list
  - reasoning_effort: "none" | "low" | "medium" | "high" (for GPT-5/O1/O3)
  - thinking_budget_tokens: integer >= 1024 (for Claude models)
  - qa_reasoning_effort: Reasoning effort for QA models

Returns:
  - Analysis with issues, severity levels, and recommendations
  - Overall quality score (1-10)
  - Consensus from multiple AI models
```

**Example usage in Claude Code:**
```
> Analyze this code for security issues using Gran Sabio with high reasoning

[Claude calls gransabio_analyze_code with reasoning_effort: "high"]

Gran Sabio Analysis Results:
- Score: 7.5/10
- Issues Found: 3
  - [CRITICAL] SQL Injection vulnerability at line 45
  - [HIGH] Hardcoded API key at line 12
  - [MEDIUM] Missing input validation at line 30
```

### `gransabio_review_fix`

Review a proposed code fix before applying it.

```
Arguments:
  - original_code (required): The original code with the issue
  - proposed_fix (required): The proposed fixed code
  - issue_description (required): What issue is being fixed
  - language: Programming language

  # Reasoning Configuration (optional):
  - generator_model: Override generator model
  - reasoning_effort: Reasoning depth for analysis
  - qa_reasoning_effort: Reasoning depth for QA evaluation

Returns:
  - Verdict: approve / reject / needs_changes
  - Whether fix solves the issue
  - Whether fix introduces new issues
  - Security and performance impact assessment
```

**Example usage:**
```
> Before applying this fix, validate it with Gran Sabio

[Claude calls gransabio_review_fix]

Gran Sabio Review:
- Verdict: APPROVE
- Score: 8.5/10
- Solves Issue: Yes
- New Issues: None
- Security Impact: Positive (fixes vulnerability)
```

### `gransabio_generate_with_qa`

Generate content with multi-model quality assurance.

```
Arguments:
  - prompt (required): The generation prompt
  - content_type: Type of content (code, documentation, article, json)
  - qa_criteria: Custom QA criteria array
  - min_score: Minimum score to approve (default: 7.5)
  - max_iterations: Maximum generation attempts (default: 3)

  # Reasoning Configuration (optional):
  - generator_model: Override generator model
  - qa_models: Override QA models list
  - reasoning_effort: Reasoning depth for generation
  - thinking_budget_tokens: Thinking tokens for Claude
  - qa_reasoning_effort: Reasoning depth for QA

Returns:
  - Generated content (only if approved)
  - Final score
  - QA summary
```

### `gransabio_check_health`

Check if Gran Sabio LLM API is available.

### `gransabio_list_models`

List available AI models organized by provider.

### `gransabio_get_config`

Get current MCP server configuration including default models and reasoning settings.

## Configuration

All configuration is via environment variables:

### Connection Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `GRANSABIO_API_URL` | `http://localhost:8000` | Gran Sabio LLM API URL |
| `GRANSABIO_API_KEY` | (empty) | API key for authenticated access |
| `GRANSABIO_TIMEOUT` | `300` | Request timeout in seconds |
| `GRANSABIO_POLL_INTERVAL` | `2.0` | Polling interval for results |

### Model Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `GRANSABIO_GENERATOR_MODEL` | `gpt-5.2` | Default generator model |
| `GRANSABIO_QA_MODELS` | `claude-opus-4-5-20251101,z-ai/glm-4.7,gemini-3-pro-preview` | Comma-separated QA models |
| `GRANSABIO_ARBITER_MODEL` | `claude-opus-4-5-20251101` | Model for conflict resolution |

### Reasoning Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `GRANSABIO_GENERATOR_REASONING` | `medium` | Default reasoning effort for generator (`none`, `low`, `medium`, `high`) |
| `GRANSABIO_QA_REASONING` | `medium` | Default reasoning effort for QA models |
| `GRANSABIO_ARBITER_REASONING` | `high` | Default reasoning effort for arbiter |
| `GRANSABIO_THINKING_BUDGET` | `0` | Default thinking budget for Claude models (0 = auto) |

### Reasoning Effort Levels

| Level | Description | Use Case |
|-------|-------------|----------|
| `none` | No extended reasoning | Simple tasks, fast responses |
| `low` | Light reasoning | Routine code reviews |
| `medium` | Balanced reasoning | Most analysis tasks (default) |
| `high` | Deep reasoning | Complex security analysis, critical fixes |

**Example with custom configuration:**
```bash
GRANSABIO_API_URL=https://api.gransabio.example.com \
GRANSABIO_API_KEY=sk-xxx \
GRANSABIO_GENERATOR_MODEL=gpt-5.2 \
GRANSABIO_GENERATOR_REASONING=high \
GRANSABIO_THINKING_BUDGET=8192 \
python gransabio_mcp_server.py
```

## Using with Remote Gran Sabio (SaaS)

If you're using a hosted Gran Sabio LLM instance (run from Gran_Sabio_LLM directory):

```bash
claude mcp add gransabio-llm \
  --env GRANSABIO_API_URL=https://api.gransabio.example.com \
  --env GRANSABIO_API_KEY=your-api-key \
  --env GRANSABIO_GENERATOR_REASONING=high \
  -- python mcp_server/gransabio_mcp_server.py
```

## Per-Call Reasoning Override

The AI client can override reasoning settings per-call:

```json
{
  "tool": "gransabio_analyze_code",
  "arguments": {
    "code": "def vulnerable(): ...",
    "focus_areas": ["security"],
    "reasoning_effort": "high",
    "qa_reasoning_effort": "high"
  }
}
```

This allows the AI to request deeper analysis for complex or security-critical code.

## Windows Notes

On Windows, use the `install_mcp.bat` script for automatic setup.

For manual installation, use absolute paths:

```cmd
claude mcp add gransabio-llm -- python C:\path\to\Gran_Sabio_LLM\mcp_server\gransabio_mcp_server.py
```

If Python is not in your PATH, use the full Python path:

```cmd
claude mcp add gransabio-llm -- C:\Python311\python.exe C:\path\to\Gran_Sabio_LLM\mcp_server\gransabio_mcp_server.py
```

## Troubleshooting

### "Connection refused" error

Ensure Gran Sabio LLM is running:
```bash
curl http://localhost:8000/
```

### "MCP SDK not installed"

Install the MCP package:
```bash
pip install mcp
```

### Timeout errors

Increase the timeout for complex analyses:
```bash
GRANSABIO_TIMEOUT=600 python gransabio_mcp_server.py
```

### Models not found

Check available models:
```bash
curl http://localhost:8000/models
```

### Check current configuration

Use the `gransabio_get_config` tool to see active settings.

## How It Works

```
+---------------------+     stdio      +---------------------+     HTTP      +---------------------+
|  Claude Code        | -------------> |  MCP Server         | ------------> |  Gran Sabio LLM     |
|  Gemini CLI         |                |  (this script)      |               |  localhost:8000     |
|  Codex CLI          | <------------- |                     | <------------ |  or remote URL      |
+---------------------+                +---------------------+               +---------------------+
         |                                      |                                      |
         |  "Analyze this code"                 |                                      |
         |  + reasoning_effort: high            |                                      |
         |                                      |            +-------------------------+
         |                                      |            |  1. GPT-5.2             |
         |                                      |            |     generates analysis  |
         |                                      |            |     (with reasoning)    |
         |                                      |            |                         |
         |                                      |            |  2. Claude Opus 4.5     |
         |                                      |            |     + GLM-4.7           |
         |                                      |            |     + Gemini 3 Pro      |
         |                                      |            |     evaluate (QA)       |
         |                                      |            |     (with reasoning)    |
         |                                      |            |                         |
         |                                      |            |  3. Consensus score     |
         |  Analysis with 8.5/10 score          |            |     calculated          |
         | <------------------------------------+------------+-------------------------+
```

## License

Same license as Gran Sabio LLM (MIT).
