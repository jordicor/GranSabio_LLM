#!/usr/bin/env python3
"""
Gran Sabio LLM MCP Server

Model Context Protocol server that exposes Gran Sabio LLM's code analysis
and content generation capabilities to AI coding assistants like Claude Code,
Gemini CLI, and Codex CLI.

This server acts as a bridge between MCP-compatible clients and the Gran Sabio
LLM API, providing structured tools for code review, analysis, and
AI-assisted content generation with multi-model QA.

Usage:
    # Local development
    python gransabio_mcp_server.py

    # With Claude Code
    claude mcp add gransabio-llm -- python /path/to/gransabio_mcp_server.py

    # With custom Gran Sabio URL
    GRANSABIO_API_URL=https://api.example.com python gransabio_mcp_server.py
"""

import asyncio
import json
import os
import sys
import time
from typing import Any, Optional, List, Dict

import httpx

# MCP SDK imports
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
except ImportError:
    print(
        "Error: MCP SDK not installed. Install with: pip install mcp",
        file=sys.stderr
    )
    sys.exit(1)


# =============================================================================
# Configuration from environment
# =============================================================================

GRANSABIO_API_URL = os.getenv("GRANSABIO_API_URL", "http://localhost:8000")
GRANSABIO_API_KEY = os.getenv("GRANSABIO_API_KEY", "")
REQUEST_TIMEOUT = int(os.getenv("GRANSABIO_TIMEOUT", "300"))
POLL_INTERVAL = float(os.getenv("GRANSABIO_POLL_INTERVAL", "2.0"))

# Default models configuration
DEFAULT_GENERATOR_MODEL = os.getenv("GRANSABIO_GENERATOR_MODEL", "gpt-5.2")
DEFAULT_QA_MODELS = os.getenv(
    "GRANSABIO_QA_MODELS",
    "claude-opus-4-5-20251101,z-ai/glm-4.7,gemini-3-pro-preview"
).split(",")
DEFAULT_ARBITER_MODEL = os.getenv(
    "GRANSABIO_ARBITER_MODEL",
    "claude-opus-4-5-20251101"
)

# Reasoning configuration defaults
DEFAULT_GENERATOR_REASONING = os.getenv("GRANSABIO_GENERATOR_REASONING", "medium")
DEFAULT_QA_REASONING = os.getenv("GRANSABIO_QA_REASONING", "medium")
DEFAULT_ARBITER_REASONING = os.getenv("GRANSABIO_ARBITER_REASONING", "high")

# Thinking budget for Claude models (0 = auto/disabled)
_thinking_budget_env = os.getenv("GRANSABIO_THINKING_BUDGET", "0")
DEFAULT_THINKING_BUDGET = int(_thinking_budget_env) if _thinking_budget_env.isdigit() else 0


# =============================================================================
# Reasoning Configuration Helpers
# =============================================================================

# Common reasoning parameters schema for tool definitions
REASONING_PARAMS_SCHEMA = {
    "generator_model": {
        "type": "string",
        "description": "Override the generator model (e.g., gpt-5.2, claude-opus-4-5-20251101)"
    },
    "qa_models": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Override QA models list"
    },
    "reasoning_effort": {
        "type": "string",
        "enum": ["none", "low", "medium", "high"],
        "description": "Reasoning effort for generator (GPT-5/O1/O3 models). Higher = deeper analysis, slower response."
    },
    "thinking_budget_tokens": {
        "type": "integer",
        "minimum": 1024,
        "description": "Thinking tokens budget for Claude models (min 1024). Higher = more thorough reasoning."
    },
    "qa_reasoning_effort": {
        "type": "string",
        "enum": ["none", "low", "medium", "high"],
        "description": "Reasoning effort for QA evaluation models"
    }
}


def _parse_reasoning_effort(value: Optional[str]) -> Optional[str]:
    """Normalize reasoning effort value."""
    if not value:
        return None
    normalized = value.lower().strip()
    if normalized in ("none", "off", "disabled", "0", "false"):
        return None
    if normalized in ("low", "medium", "high"):
        return normalized
    return None


def _build_generator_config(args: Dict[str, Any]) -> Dict[str, Any]:
    """Build generator configuration from arguments."""
    config = {
        "generator_model": args.get("generator_model", DEFAULT_GENERATOR_MODEL)
    }

    # Reasoning effort (GPT-5, O1, O3)
    reasoning = args.get("reasoning_effort")
    if reasoning:
        parsed = _parse_reasoning_effort(reasoning)
        if parsed:
            config["reasoning_effort"] = parsed
    elif DEFAULT_GENERATOR_REASONING:
        parsed = _parse_reasoning_effort(DEFAULT_GENERATOR_REASONING)
        if parsed:
            config["reasoning_effort"] = parsed

    # Thinking budget (Claude models)
    thinking_budget = args.get("thinking_budget_tokens")
    if thinking_budget and isinstance(thinking_budget, int) and thinking_budget >= 1024:
        config["thinking_budget_tokens"] = thinking_budget
    elif DEFAULT_THINKING_BUDGET >= 1024:
        config["thinking_budget_tokens"] = DEFAULT_THINKING_BUDGET

    return config


def _build_qa_config(args: Dict[str, Any]) -> Dict[str, Any]:
    """Build QA models configuration from arguments."""
    config = {
        "qa_models": args.get("qa_models", DEFAULT_QA_MODELS)
    }

    # QA reasoning effort - apply globally to all QA models
    qa_reasoning = args.get("qa_reasoning_effort")
    if qa_reasoning:
        parsed = _parse_reasoning_effort(qa_reasoning)
        if parsed:
            config["qa_global_config"] = {"reasoning_effort": parsed}
    elif DEFAULT_QA_REASONING:
        parsed = _parse_reasoning_effort(DEFAULT_QA_REASONING)
        if parsed:
            config["qa_global_config"] = {"reasoning_effort": parsed}

    return config


def _build_arbiter_config(args: Dict[str, Any]) -> Dict[str, Any]:
    """Build arbiter (Gran Sabio) configuration."""
    # Arbiter doesn't support per-call reasoning config in the API yet,
    # but we prepare the model selection
    return {
        "gran_sabio_model": args.get("gran_sabio_model", DEFAULT_ARBITER_MODEL),
        "gran_sabio_fallback": True
    }


# Initialize MCP server
server = Server("gransabio-llm")


def _build_headers() -> dict[str, str]:
    """Build request headers with optional API key."""
    headers = {"Content-Type": "application/json"}
    if GRANSABIO_API_KEY:
        headers["Authorization"] = f"Bearer {GRANSABIO_API_KEY}"
    return headers


async def _wait_for_result(
    client: httpx.AsyncClient,
    session_id: str,
    timeout: float = REQUEST_TIMEOUT
) -> dict[str, Any]:
    """Poll for generation result until completion or timeout."""
    start_time = time.time()
    result_url = f"{GRANSABIO_API_URL}/result/{session_id}"

    while time.time() - start_time < timeout:
        try:
            response = await client.get(result_url, headers=_build_headers())

            if response.status_code == 200:
                return response.json()

            # Check if still processing
            if response.status_code in (202, 425):
                await asyncio.sleep(POLL_INTERVAL)
                continue

            # Handle other errors
            detail = ""
            try:
                data = response.json()
                detail = data.get("detail", "")
            except Exception:
                pass

            if "not finished" in detail.lower() or "in progress" in detail.lower():
                await asyncio.sleep(POLL_INTERVAL)
                continue

            raise Exception(f"API error: {response.status_code} - {response.text}")

        except httpx.RequestError as e:
            raise Exception(f"Connection error: {e}")

    raise Exception(f"Timeout waiting for result after {timeout}s")


async def _call_gransabio(payload: dict[str, Any]) -> dict[str, Any]:
    """Make a generation request to Gran Sabio LLM and wait for result."""
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        # Start generation
        response = await client.post(
            f"{GRANSABIO_API_URL}/generate",
            json=payload,
            headers=_build_headers()
        )

        if response.status_code != 200:
            raise Exception(f"Generation failed: {response.status_code} - {response.text}")

        data = response.json()

        # Check for preflight rejection
        if data.get("status") == "rejected":
            feedback = data.get("preflight_feedback", {})
            return {
                "success": False,
                "error": "Preflight validation rejected",
                "feedback": feedback.get("user_feedback", "Unknown reason"),
                "issues": feedback.get("issues", [])
            }

        session_id = data.get("session_id")
        if not session_id:
            raise Exception(f"No session_id in response: {data}")

        # Wait for completion
        result = await _wait_for_result(client, session_id)

        return {
            "success": result.get("approved", False),
            "content": result.get("content", ""),
            "score": result.get("final_score"),
            "status": result.get("status"),
            "qa_summary": result.get("qa_summary"),
            "iterations": result.get("iterations_used"),
            "session_id": session_id,
            "costs": result.get("costs")
        }


# =============================================================================
# MCP Tools Definition
# =============================================================================

@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available Gran Sabio LLM tools."""
    return [
        Tool(
            name="gransabio_analyze_code",
            description="""Analyze code for bugs, security issues, performance problems,
and best practices violations. Uses multiple AI models for consensus-based review.

Returns a detailed analysis with:
- Issues found (bugs, security, performance, style)
- Severity levels (critical, high, medium, low)
- Specific recommendations for each issue
- Overall quality score (1-10)

Supports reasoning configuration for deeper analysis on complex code.
Use this BEFORE implementing fixes to understand the full scope of issues.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The code to analyze"
                    },
                    "language": {
                        "type": "string",
                        "description": "Programming language (python, javascript, typescript, go, rust, etc.)",
                        "default": "auto-detect"
                    },
                    "context": {
                        "type": "string",
                        "description": "Additional context about the code (what it does, known issues, etc.)",
                        "default": ""
                    },
                    "focus_areas": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific areas to focus on: security, performance, bugs, style, all",
                        "default": ["all"]
                    },
                    **REASONING_PARAMS_SCHEMA
                },
                "required": ["code"]
            }
        ),
        Tool(
            name="gransabio_review_fix",
            description="""Review a proposed code fix before applying it.
Validates that the fix:
- Actually solves the problem
- Doesn't introduce new bugs or security issues
- Follows best practices
- Is the optimal solution

Supports reasoning configuration for thorough security reviews.
Use this AFTER proposing a fix but BEFORE applying it to get validation
from multiple AI models.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "original_code": {
                        "type": "string",
                        "description": "The original code with the issue"
                    },
                    "proposed_fix": {
                        "type": "string",
                        "description": "The proposed fixed code"
                    },
                    "issue_description": {
                        "type": "string",
                        "description": "Description of the issue being fixed"
                    },
                    "language": {
                        "type": "string",
                        "description": "Programming language",
                        "default": "auto-detect"
                    },
                    **REASONING_PARAMS_SCHEMA
                },
                "required": ["original_code", "proposed_fix", "issue_description"]
            }
        ),
        Tool(
            name="gransabio_generate_with_qa",
            description="""Generate content with multi-model quality assurance.
Useful for generating code, documentation, or any text that needs to meet
quality standards.

The content goes through:
1. Generation by a primary AI model (with optional reasoning)
2. QA evaluation by multiple reviewer models
3. Iteration if quality thresholds aren't met
4. Gran Sabio arbitration if there are conflicts

Supports full reasoning configuration for both generator and QA models.
Returns approved content only when it meets all quality criteria.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The generation prompt"
                    },
                    "content_type": {
                        "type": "string",
                        "enum": [
                            "analysis", "article", "biography", "comparison",
                            "creative", "essay", "evaluation", "json", "novel",
                            "opinion", "preference", "report", "review",
                            "script", "selection", "story", "technical", "vote"
                        ],
                        "description": "Type of content to generate",
                        "default": "technical"
                    },
                    "qa_criteria": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "criteria": {"type": "string"},
                                "min_score": {"type": "number", "default": 7.5}
                            }
                        },
                        "description": "Custom QA criteria to evaluate against",
                        "default": []
                    },
                    "min_score": {
                        "type": "number",
                        "description": "Minimum global score to approve (1-10)",
                        "default": 7.5
                    },
                    "max_iterations": {
                        "type": "integer",
                        "description": "Maximum generation attempts",
                        "default": 3
                    },
                    **REASONING_PARAMS_SCHEMA
                },
                "required": ["prompt"]
            }
        ),
        Tool(
            name="gransabio_check_health",
            description="""Check if Gran Sabio LLM API is available and responding.
Use this to verify connectivity before making other requests.""",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="gransabio_list_models",
            description="""List available AI models in Gran Sabio LLM.
Shows models organized by provider with their capabilities and pricing.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "provider": {
                        "type": "string",
                        "description": "Filter by provider: openai, anthropic, google, xai, openrouter, all",
                        "default": "all"
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="gransabio_get_config",
            description="""Get current MCP server configuration including default models
and reasoning settings. Useful for debugging or understanding current setup.""",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""

    try:
        if name == "gransabio_analyze_code":
            result = await _handle_analyze_code(arguments)
        elif name == "gransabio_review_fix":
            result = await _handle_review_fix(arguments)
        elif name == "gransabio_generate_with_qa":
            result = await _handle_generate_with_qa(arguments)
        elif name == "gransabio_check_health":
            result = await _handle_check_health()
        elif name == "gransabio_list_models":
            result = await _handle_list_models(arguments)
        elif name == "gransabio_get_config":
            result = _handle_get_config()
        else:
            result = {"error": f"Unknown tool: {name}"}

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        error_result = {
            "error": str(e),
            "tool": name,
            "hint": "Ensure Gran Sabio LLM is running at " + GRANSABIO_API_URL
        }
        return [TextContent(type="text", text=json.dumps(error_result, indent=2))]


async def _handle_analyze_code(args: dict[str, Any]) -> dict[str, Any]:
    """Handle code analysis request."""
    code = args.get("code", "")
    language = args.get("language", "auto-detect")
    context = args.get("context", "")
    focus_areas = args.get("focus_areas", ["all"])

    focus_text = ", ".join(focus_areas) if focus_areas else "all areas"

    prompt = f"""Analyze the following {language} code for issues.

Focus areas: {focus_text}

Context: {context if context else "No additional context provided."}

Code to analyze:
```{language}
{code}
```

Provide a comprehensive analysis in JSON format with:
- language: detected programming language
- complexity_score: 1-10 rating of code complexity
- summary: object with total_issues, critical_count, high_count, verdict
- issues: array of issues, each with:
  - id: unique identifier (ISS-001, ISS-002, etc.)
  - type: security | performance | bug | style | maintainability
  - severity: critical | high | medium | low
  - line: approximate line number if applicable
  - description: clear description of the issue
  - recommendation: specific fix recommendation
  - cwe_id: CWE identifier for security issues (optional)
- positive_patterns: array of good practices found
- recommendations_summary: overall recommendations"""

    qa_layers = [
        {
            "name": "Analysis Accuracy",
            "description": "Validates accuracy of issue detection",
            "criteria": "All identified issues are real problems, not false positives. Line numbers are accurate.",
            "min_score": 7.5,
            "deal_breaker_criteria": "Reports a non-existent issue or misidentifies the problem type"
        },
        {
            "name": "Security Coverage",
            "description": "Validates security vulnerability detection",
            "criteria": "All security vulnerabilities are identified including SQL injection, XSS, hardcoded credentials, unsafe deserialization.",
            "min_score": 8.0,
            "deal_breaker_criteria": "Misses an obvious security vulnerability"
        },
        {
            "name": "Completeness",
            "description": "Validates analysis completeness",
            "criteria": "Analysis covers all focus areas requested and doesn't omit significant issues.",
            "min_score": 7.0
        }
    ]

    # Build configuration from arguments
    generator_config = _build_generator_config(args)
    qa_config = _build_qa_config(args)
    arbiter_config = _build_arbiter_config(args)

    payload = {
        "prompt": prompt,
        "content_type": "json",
        "json_output": True,
        **generator_config,
        **qa_config,
        **arbiter_config,
        "qa_layers": qa_layers,
        "min_global_score": 7.5,
        "max_iterations": 3,
        "temperature": 0.3,
        "show_query_costs": 2
    }

    result = await _call_gransabio(payload)

    # Parse the content if successful
    if result.get("success") and result.get("content"):
        try:
            analysis = json.loads(result["content"])
            result["analysis"] = analysis
            del result["content"]  # Replace raw content with parsed
        except json.JSONDecodeError:
            result["raw_content"] = result.get("content")

    return result


async def _handle_review_fix(args: dict[str, Any]) -> dict[str, Any]:
    """Handle fix review request."""
    original = args.get("original_code", "")
    proposed = args.get("proposed_fix", "")
    issue = args.get("issue_description", "")
    language = args.get("language", "auto-detect")

    prompt = f"""Review this proposed code fix.

## Issue Being Fixed
{issue}

## Original Code
```{language}
{original}
```

## Proposed Fix
```{language}
{proposed}
```

Analyze and respond in JSON format:
{{
  "verdict": "approve" | "reject" | "needs_changes",
  "score": <1-10 overall quality score>,
  "solves_issue": true | false,
  "introduces_new_issues": true | false,
  "new_issues": [<list of any new issues introduced>],
  "security_impact": "positive" | "neutral" | "negative",
  "performance_impact": "positive" | "neutral" | "negative",
  "improvements": [<suggested improvements if any>],
  "explanation": "<detailed explanation of the verdict>"
}}"""

    qa_layers = [
        {
            "name": "Fix Correctness",
            "description": "Validates the fix solves the issue",
            "criteria": "The fix actually solves the stated issue without introducing syntax errors or logic bugs.",
            "min_score": 8.0,
            "deal_breaker_criteria": "Fix doesn't solve the issue or introduces a new bug"
        },
        {
            "name": "Security Review",
            "description": "Security impact assessment",
            "criteria": "The fix doesn't introduce security vulnerabilities.",
            "min_score": 8.5,
            "deal_breaker_criteria": "Fix introduces a security vulnerability"
        },
        {
            "name": "Code Quality",
            "description": "Best practices evaluation",
            "criteria": "The fix follows best practices and is the optimal solution.",
            "min_score": 7.0
        }
    ]

    # Build configuration from arguments
    generator_config = _build_generator_config(args)
    qa_config = _build_qa_config(args)
    arbiter_config = _build_arbiter_config(args)

    payload = {
        "prompt": prompt,
        "content_type": "json",
        "json_output": True,
        **generator_config,
        **qa_config,
        **arbiter_config,
        "qa_layers": qa_layers,
        "min_global_score": 7.5,
        "max_iterations": 2,
        "temperature": 0.2,
        "show_query_costs": 2
    }

    result = await _call_gransabio(payload)

    # Parse the content if successful
    if result.get("success") and result.get("content"):
        try:
            review = json.loads(result["content"])
            result["review"] = review
            del result["content"]
        except json.JSONDecodeError:
            result["raw_content"] = result.get("content")

    return result


async def _handle_generate_with_qa(args: dict[str, Any]) -> dict[str, Any]:
    """Handle generation with QA request."""
    prompt = args.get("prompt", "")
    content_type = args.get("content_type", "technical")
    custom_qa = args.get("qa_criteria", [])
    min_score = args.get("min_score", 7.5)
    max_iterations = args.get("max_iterations", 3)

    # Build QA layers from custom criteria or use defaults
    if custom_qa:
        qa_layers = [
            {
                "name": c.get("name", f"Criteria {i+1}"),
                "description": c.get("name", f"Custom criteria {i+1}"),
                "criteria": c.get("criteria", ""),
                "min_score": c.get("min_score", 7.5)
            }
            for i, c in enumerate(custom_qa)
        ]
    else:
        qa_layers = [
            {
                "name": "Quality",
                "description": "Overall content quality",
                "criteria": "Content is high quality, accurate, and complete.",
                "min_score": min_score
            },
            {
                "name": "Correctness",
                "description": "Technical correctness",
                "criteria": "No errors, bugs, or inaccuracies.",
                "min_score": min_score
            }
        ]

    # Build configuration from arguments
    generator_config = _build_generator_config(args)
    qa_config = _build_qa_config(args)
    arbiter_config = _build_arbiter_config(args)

    payload = {
        "prompt": prompt,
        "content_type": content_type,
        **generator_config,
        **qa_config,
        **arbiter_config,
        "qa_layers": qa_layers,
        "min_global_score": min_score,
        "max_iterations": max_iterations,
        "show_query_costs": 2
    }

    return await _call_gransabio(payload)


async def _handle_check_health() -> dict[str, Any]:
    """Check Gran Sabio LLM API health."""
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            response = await client.get(
                f"{GRANSABIO_API_URL}/",
                headers=_build_headers()
            )
            return {
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "api_url": GRANSABIO_API_URL,
                "response_code": response.status_code
            }
        except Exception as e:
            return {
                "status": "unreachable",
                "api_url": GRANSABIO_API_URL,
                "error": str(e)
            }


async def _handle_list_models(args: dict[str, Any]) -> dict[str, Any]:
    """List available models."""
    provider_filter = args.get("provider", "all").lower()

    async with httpx.AsyncClient(timeout=30) as client:
        try:
            response = await client.get(
                f"{GRANSABIO_API_URL}/models",
                headers=_build_headers()
            )

            if response.status_code != 200:
                return {"error": f"Failed to fetch models: {response.status_code}"}

            models = response.json()

            # Filter by provider if requested
            if provider_filter != "all":
                if provider_filter in models:
                    models = {provider_filter: models[provider_filter]}
                else:
                    return {
                        "error": f"Provider '{provider_filter}' not found",
                        "available_providers": list(models.keys())
                    }

            # Summarize for readability
            summary = {}
            for provider, provider_models in models.items():
                summary[provider] = [
                    {
                        "key": m.get("key", m.get("model_id")),
                        "name": m.get("name"),
                        "context": m.get("context_window"),
                        "output": m.get("output_tokens")
                    }
                    for m in (provider_models if isinstance(provider_models, list)
                              else provider_models.values())
                ][:10]  # Limit to first 10 per provider

            return {
                "api_url": GRANSABIO_API_URL,
                "models": summary,
                "note": "Showing first 10 models per provider. Use the web UI for full list."
            }

        except Exception as e:
            return {"error": str(e)}


def _handle_get_config() -> dict[str, Any]:
    """Return current MCP server configuration."""
    return {
        "api_url": GRANSABIO_API_URL,
        "models": {
            "generator": DEFAULT_GENERATOR_MODEL,
            "qa_models": DEFAULT_QA_MODELS,
            "arbiter": DEFAULT_ARBITER_MODEL
        },
        "reasoning": {
            "generator_reasoning": DEFAULT_GENERATOR_REASONING,
            "qa_reasoning": DEFAULT_QA_REASONING,
            "arbiter_reasoning": DEFAULT_ARBITER_REASONING,
            "thinking_budget": DEFAULT_THINKING_BUDGET if DEFAULT_THINKING_BUDGET >= 1024 else "auto"
        },
        "timeouts": {
            "request_timeout": REQUEST_TIMEOUT,
            "poll_interval": POLL_INTERVAL
        }
    }


# =============================================================================
# Main Entry Point
# =============================================================================

async def main():
    """Run the MCP server."""
    print("Gran Sabio LLM MCP Server", file=sys.stderr)
    print(f"API URL: {GRANSABIO_API_URL}", file=sys.stderr)
    print("", file=sys.stderr)
    print("Models:", file=sys.stderr)
    print(f"  Generator: {DEFAULT_GENERATOR_MODEL}", file=sys.stderr)
    print(f"  QA: {', '.join(DEFAULT_QA_MODELS)}", file=sys.stderr)
    print(f"  Arbiter: {DEFAULT_ARBITER_MODEL}", file=sys.stderr)
    print("", file=sys.stderr)
    print("Reasoning Defaults:", file=sys.stderr)
    print(f"  Generator: {DEFAULT_GENERATOR_REASONING}", file=sys.stderr)
    print(f"  QA: {DEFAULT_QA_REASONING}", file=sys.stderr)
    print(f"  Arbiter: {DEFAULT_ARBITER_REASONING}", file=sys.stderr)
    if DEFAULT_THINKING_BUDGET >= 1024:
        print(f"  Thinking Budget: {DEFAULT_THINKING_BUDGET} tokens", file=sys.stderr)
    print("", file=sys.stderr)

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
