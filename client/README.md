# Gran Sabio LLM Client

Python client library for the Gran Sabio LLM Engine API.

## Installation

The client is included in the Gran Sabio LLM repository. No separate installation required.

Dependencies:
```bash
pip install aiohttp requests
```

## Quick Start

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
    max_tokens=4000,

    # QA Configuration
    qa_models=["gpt-4o-mini"],
    qa_layers=[...],                # Empty list = bypass QA
    min_global_score=8.0,
    max_iterations=3,

    # Escalation
    gran_sabio_model="claude-opus-4-5-20251101",
    gran_sabio_fallback=True,

    # Optional
    reasoning_effort="medium",      # For OpenAI reasoning models
    thinking_budget_tokens=8000,    # For Claude thinking models
)
```

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
    timeout=(30, 600)  # (connect, read) in seconds
)
```

## Error Handling

```python
from client import GranSabioClient, GranSabioClientError

client = GranSabioClient()

try:
    result = client.generate(prompt="...")
except GranSabioClientError as e:
    print(f"Error: {e}")
    print(f"Status code: {e.status_code}")
    print(f"Details: {e.details}")
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

- `demos/` - Example scripts demonstrating various features
- `CLAUDE.md` - Architecture documentation
- API docs at `http://localhost:8000/api-docs`
