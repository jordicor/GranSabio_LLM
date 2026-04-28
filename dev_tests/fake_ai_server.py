"""
Fake AI Server for controlled testing.

This server mimics the OpenAI API and returns predefined responses
based on the model name requested. This allows mixing real AI calls
with controlled fake responses for testing deal-breakers, smart-edits, etc.

Usage:
    python dev_tests/fake_ai_server.py

Models available:
    - Generator-Dumb: Returns content from fake_ai/Generator-Dumb.txt
    - QA-Dumb: Returns evaluation from fake_ai/QA-Dumb.txt
    - GranSabio-Dumb: Returns review from fake_ai/GranSabio-Dumb.txt
    - Arbiter-Dumb: Returns arbitration from fake_ai/Arbiter-Dumb.txt

Action variants (use model:action syntax):
    - QA-Dumb:dealbreaker -> reads QA-Dumb_dealbreaker.txt
    - QA-Dumb:with-edits -> reads QA-Dumb_with-edits.txt
    - GranSabio-Dumb:reject -> reads GranSabio-Dumb_reject.txt

To use with GranSabio LLM:
    1. Start this server: python dev_tests/fake_ai_server.py
    2. In your request, use model names like "Generator-Dumb" for generation
       or "QA-Dumb" in qa_models list (or "QA-Dumb:dealbreaker" for variants)
    3. Configure FAKE_AI_BASE_URL in .env or set it programmatically
"""

import asyncio
import re
import sys
import time
import uuid
from pathlib import Path

# Add parent directory to path for json_utils import
sys.path.insert(0, str(Path(__file__).parent.parent))
from typing import Optional, Tuple

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

import json_utils as json

# Configuration
FAKE_AI_PORT = 8989
FAKE_AI_DIR = Path(__file__).parent / "fake_ai"

app = FastAPI(title="Fake AI Server", description="OpenAI-compatible fake AI for testing")

# Ensure fake_ai directory exists
FAKE_AI_DIR.mkdir(exist_ok=True)

# Security: Pattern for safe file names (alphanumeric, underscore, hyphen only)
SAFE_NAME_PATTERN = re.compile(r'^[a-zA-Z0-9_-]+$')


def validate_safe_name(name: str, param_name: str = "name", max_len: int = 64) -> str:
    """Validate name contains only safe characters for filesystem use.

    Prevents path traversal attacks by rejecting:
    - Empty names
    - Names longer than max_len
    - Names with ../, /, \\, or special characters

    Args:
        name: The name to validate
        param_name: Parameter name for error messages
        max_len: Maximum allowed length

    Returns:
        The validated name (unchanged if valid)

    Raises:
        ValueError: If name is invalid
    """
    if not name:
        raise ValueError(f"{param_name} cannot be empty")
    if len(name) > max_len:
        raise ValueError(f"{param_name} too long: {len(name)} > {max_len}")
    if not SAFE_NAME_PATTERN.match(name):
        raise ValueError(f"{param_name} contains invalid characters: '{name}'. Only [a-zA-Z0-9_-] allowed")
    return name


def parse_model_action(model: str) -> Tuple[str, Optional[str]]:
    """Parse model name and optional action from 'Model:action' format.

    Examples:
        'QA-Dumb' -> ('QA-Dumb', None)
        'QA-Dumb:dealbreaker' -> ('QA-Dumb', 'dealbreaker')
        'provider/QA-Dumb:action' -> ('QA-Dumb', 'action')

    Args:
        model: Model string, optionally with :action suffix

    Returns:
        Tuple of (model_name, action) where action may be None
    """
    # Handle provider/model format first
    if "/" in model:
        model = model.split("/")[-1]

    # Parse action suffix
    if ":" in model:
        parts = model.rsplit(":", 1)
        return parts[0], parts[1]

    return model, None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    stream: Optional[bool] = False
    response_format: Optional[dict] = None


def get_response_file(model: str, action: Optional[str] = None) -> Path:
    """Get the response file path for a model with optional action.

    Args:
        model: Model name (already parsed, without provider prefix)
        action: Optional action suffix for variant files

    Returns:
        Path to the response file

    Raises:
        ValueError: If model or action contain unsafe characters

    File naming convention:
        - No action: {model}.txt (e.g., QA-Dumb.txt)
        - With action: {model}_{action}.txt (e.g., QA-Dumb_dealbreaker.txt)
    """
    # Validate for path safety
    model = validate_safe_name(model, "model")

    if action:
        action = validate_safe_name(action, "action")
        filename = f"{model}_{action}.txt"
    else:
        filename = f"{model}.txt"

    # Build path and verify it's within FAKE_AI_DIR (defense in depth)
    target = (FAKE_AI_DIR / filename).resolve()
    fake_ai_resolved = FAKE_AI_DIR.resolve()

    if not str(target).startswith(str(fake_ai_resolved)):
        raise ValueError(f"Path traversal attempt blocked: {filename}")

    return target


def load_response(model: str, messages: list = None, action: Optional[str] = None) -> str:
    """Load response content from file.

    Priority order:
    1. Action-specific file: {model}_{action}.txt (if action specified)
    2. Layer-specific file: {model}.{layer}.txt (if layer detected in prompt)
    3. Default file: {model}.txt

    Args:
        model: Model name (already parsed, without provider prefix or action)
        messages: Request messages (used for layer detection)
        action: Optional action for variant files (e.g., 'dealbreaker')

    File naming conventions:
    - Default: QA-Dumb.txt
    - Action variant: QA-Dumb_dealbreaker.txt
    - Layer-specific: QA-Dumb.Accuracy.txt
    """
    # Priority 1: Action-specific file
    if action:
        try:
            action_file = get_response_file(model, action)
            if action_file.exists():
                content = action_file.read_text(encoding="utf-8")
                print(f"[FakeAI] Loaded ACTION response from: {action_file.name}")
                return content
            print(f"[FakeAI] Action file not found: {action_file.name}, trying layer/default")
        except ValueError as e:
            print(f"[FakeAI] Invalid action '{action}': {e}")

    # Priority 2: Layer-specific response (for QA models)
    layer_name = None
    if messages and "qa" in model.lower():
        # Extract layer name from the last user message
        for msg in reversed(messages):
            if msg.get("role") == "user" or msg.get("role") == "system":
                content = msg.get("content", "")
                # Look for "Layer: X" or "evaluating X layer" patterns
                match = re.search(r'(?:Layer[:\s]+|evaluating\s+)(["\']?)([^"\']+)\1', content, re.IGNORECASE)
                if match:
                    layer_name = match.group(2).strip()
                    break
                # Also check for layer name in criteria/context
                match = re.search(r'name["\']?\s*:\s*["\']([^"\']+)["\']', content)
                if match:
                    layer_name = match.group(1).strip()
                    break

    if layer_name:
        # Normalize layer name for filename (only alphanumeric)
        safe_layer = re.sub(r'[^\w\s-]', '', layer_name).replace(' ', '')
        if safe_layer and SAFE_NAME_PATTERN.match(safe_layer):
            layer_file = FAKE_AI_DIR / f"{model}.{safe_layer}.txt"
            if layer_file.exists():
                content = layer_file.read_text(encoding="utf-8")
                print(f"[FakeAI] Loaded LAYER-SPECIFIC response from: {layer_file.name}")
                return content
            print(f"[FakeAI] No layer-specific file for {layer_name}, using default")

    # Priority 3: Default file
    try:
        response_file = get_response_file(model)
    except ValueError as e:
        print(f"[FakeAI] Invalid model name: {e}")
        return create_default_response(model)

    if not response_file.exists():
        # Create default response file
        default_content = create_default_response(model)
        response_file.write_text(default_content, encoding="utf-8")
        print(f"[FakeAI] Created default response file: {response_file.name}")
        return default_content

    content = response_file.read_text(encoding="utf-8")
    print(f"[FakeAI] Loaded response from: {response_file.name}")
    return content


def create_default_response(model: str) -> str:
    """Create a default response based on model type."""
    model_lower = model.lower()

    if "generator" in model_lower:
        return json.dumps({
            "generated_text": "This is fake generated content from Generator-Dumb. Edit this file to customize the response.",
            "title": "Fake Title",
            "summary": "This is a fake summary for testing purposes."
        }, indent=2, ensure_ascii=False)

    elif "qa" in model_lower:
        return json.dumps({
            "score": 8.5,
            "feedback": "This is fake QA feedback. The content looks good overall.",
            "deal_breaker": False,
            "deal_breaker_reason": None,
            "proposed_edits": []
        }, indent=2, ensure_ascii=False)

    elif "gransabio" in model_lower or "gran_sabio" in model_lower:
        return json.dumps({
            "approved": True,
            "reason": "GranSabio-Dumb approved the content. This is a fake response.",
            "modifications_made": False,
            "final_content": None,
            "final_score": 9.0
        }, indent=2, ensure_ascii=False)

    elif "arbiter" in model_lower:
        return json.dumps({
            "approved_edits": [],
            "rejected_edits": [],
            "reasoning": "Arbiter-Dumb processed the edits. This is a fake response.",
            "conflicts_resolved": [],
            "alignment_verified": True
        }, indent=2, ensure_ascii=False)

    else:
        return f"Default fake response for model: {model}. Edit the file {FAKE_AI_DIR / f'{model}.txt'} to customize."


def create_chat_completion_response(model: str, content: str, stream: bool = False):
    """Create a chat completion response in OpenAI format."""
    response_id = f"chatcmpl-fake-{uuid.uuid4().hex[:8]}"
    created = int(time.time())

    if stream:
        return create_streaming_response(response_id, model, content, created)

    return {
        "id": response_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": len(content.split()),
            "total_tokens": 100 + len(content.split())
        }
    }


async def create_streaming_response(response_id: str, model: str, content: str, created: int):
    """Generate streaming response chunks."""
    # Split content into chunks (by words for more natural streaming)
    words = content.split()
    chunk_size = 5  # words per chunk

    for i in range(0, len(words), chunk_size):
        chunk_words = words[i:i + chunk_size]
        chunk_text = " ".join(chunk_words)
        if i > 0:
            chunk_text = " " + chunk_text

        chunk_data = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": chunk_text
                    },
                    "finish_reason": None
                }
            ]
        }
        yield f"data: {json.dumps(chunk_data)}\n\n"
        await asyncio.sleep(0.02)  # Small delay for realistic streaming

    # Final chunk with finish_reason
    final_chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }
        ]
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint."""
    raw_model = request.model

    # Parse model:action format
    model, action = parse_model_action(raw_model)

    if action:
        print(f"[FakeAI] Request for model: {model}, action: {action}")
    else:
        print(f"[FakeAI] Request for model: {model}")
    print(f"[FakeAI] Stream: {request.stream}")

    # Check if this is a "Dumb" model
    if not any(x in model.lower() for x in ["dumb", "fake", "test", "mock"]):
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": f"Model '{model}' is not a fake model. Use Generator-Dumb, QA-Dumb, GranSabio-Dumb, or Arbiter-Dumb.",
                    "type": "invalid_request_error",
                    "code": "model_not_found"
                }
            }
        )

    # Load response from file (pass messages for layer-specific, action for variants)
    content = load_response(model, request.messages, action)

    # Handle streaming
    if request.stream:
        return StreamingResponse(
            create_streaming_response(
                f"chatcmpl-fake-{uuid.uuid4().hex[:8]}",
                model,
                content,
                int(time.time())
            ),
            media_type="text/event-stream"
        )

    # Non-streaming response
    response = create_chat_completion_response(model, content, stream=False)
    return JSONResponse(content=response)


@app.get("/v1/models")
async def list_models():
    """List available fake models."""
    models = [
        {"id": "Generator-Dumb", "object": "model", "owned_by": "fake-ai"},
        {"id": "QA-Dumb", "object": "model", "owned_by": "fake-ai"},
        {"id": "GranSabio-Dumb", "object": "model", "owned_by": "fake-ai"},
        {"id": "Arbiter-Dumb", "object": "model", "owned_by": "fake-ai"},
    ]

    # Add any custom models from fake_ai directory
    for txt_file in FAKE_AI_DIR.glob("*.txt"):
        model_name = txt_file.stem
        if model_name not in [m["id"] for m in models]:
            models.append({"id": model_name, "object": "model", "owned_by": "fake-ai"})

    return {"object": "list", "data": models}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "server": "fake-ai", "port": FAKE_AI_PORT}


@app.get("/")
async def root():
    """Root endpoint with usage info."""
    return {
        "name": "Fake AI Server",
        "description": "OpenAI-compatible server for testing GranSabio LLM",
        "models": [
            "Generator-Dumb - For content generation",
            "QA-Dumb - For QA evaluation",
            "GranSabio-Dumb - For GranSabio reviews",
            "Arbiter-Dumb - For Arbiter conflict resolution",
        ],
        "action_syntax": {
            "format": "model:action",
            "examples": [
                "QA-Dumb:dealbreaker -> reads QA-Dumb_dealbreaker.txt",
                "QA-Dumb:with-edits -> reads QA-Dumb_with-edits.txt",
                "GranSabio-Dumb:reject -> reads GranSabio-Dumb_reject.txt",
            ],
            "file_naming": "{model}_{action}.txt"
        },
        "response_files": str(FAKE_AI_DIR),
        "usage": {
            "edit_responses": f"Edit .txt files in {FAKE_AI_DIR} to customize responses",
            "streaming": "Supports streaming responses (stream=true)",
            "mixing": "Mix with real AI by using Dumb models only where needed"
        }
    }


@app.post("/reload/{model}")
async def reload_model(model: str):
    """Reload a model's response from file (useful after editing)."""
    response_file = get_response_file(model)
    if response_file.exists():
        content = response_file.read_text(encoding="utf-8")
        return {
            "status": "reloaded",
            "model": model,
            "file": str(response_file),
            "preview": content[:200] + "..." if len(content) > 200 else content
        }
    return {"status": "not_found", "model": model}


def main():
    """Run the fake AI server."""
    print("=" * 60)
    print("FAKE AI SERVER")
    print("=" * 60)
    print(f"Port: {FAKE_AI_PORT}")
    print(f"Response files: {FAKE_AI_DIR}")
    print()
    print("Available models:")
    print("  - Generator-Dumb")
    print("  - QA-Dumb")
    print("  - GranSabio-Dumb")
    print("  - Arbiter-Dumb")
    print()
    print("Action variants (model:action syntax):")
    print("  - QA-Dumb:dealbreaker -> QA-Dumb_dealbreaker.txt")
    print("  - QA-Dumb:with-edits  -> QA-Dumb_with-edits.txt")
    print("  - GranSabio-Dumb:reject -> GranSabio-Dumb_reject.txt")
    print()
    print("Edit .txt files in fake_ai/ to customize responses")
    print("=" * 60)
    print()

    uvicorn.run(app, host="0.0.0.0", port=FAKE_AI_PORT, log_level="info")


if __name__ == "__main__":
    main()
