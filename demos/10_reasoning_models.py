"""
Demo 10: Reasoning Models (Deep Thinking)
==========================================

This demo shows how to use advanced reasoning models that "think"
before responding. These models produce higher quality outputs for
complex tasks.

Supported reasoning modes:
- OpenAI GPT-5/O1/O3: reasoning_effort (low, medium, high)
- Claude 3.7/4: thinking_budget_tokens (min 1024)

Features demonstrated:
- reasoning_effort for OpenAI models
- thinking_budget_tokens for Claude models
- Complex problem solving
- Deep analysis tasks

This is ideal for:
- Complex analytical tasks
- Multi-step reasoning problems
- High-stakes content generation
- Tasks requiring careful consideration

Usage:
    python demos/10_reasoning_models.py

    # With specific model:
    python demos/10_reasoning_models.py --model gpt-5

    # With high reasoning:
    python demos/10_reasoning_models.py --effort high
"""

import asyncio
import sys
import argparse
from pathlib import Path
from typing import Dict, Any

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from client import AsyncGranSabioClient
from demos.common import run_demo, print_header, print_status, print_generation_result


# Complex problems that benefit from reasoning
REASONING_PROBLEMS = {
    "logic_puzzle": {
        "name": "Logic Puzzle",
        "prompt": """
Solve this logic puzzle step by step:

Five friends (Alice, Bob, Carol, David, Eve) each have a different pet
(cat, dog, fish, bird, hamster) and a different favorite color
(red, blue, green, yellow, purple).

Clues:
1. Alice has the fish and her favorite color is not red.
2. The person with the dog loves blue.
3. Carol's favorite color is green and she doesn't have the cat.
4. David has the hamster.
5. The person with the bird has yellow as their favorite color.
6. Bob doesn't like purple or green.
7. Eve's favorite color is red.

Determine each person's pet and favorite color.
Show your reasoning process.
        """,
        "expected_format": "table"
    },
    "ethical_analysis": {
        "name": "Ethical Analysis",
        "prompt": """
Analyze this ethical dilemma from multiple philosophical perspectives:

A self-driving car's AI must make a split-second decision: swerve left
(harming 1 pedestrian) or stay straight (harming 3 passengers). The
pedestrian is elderly, the passengers include a child.

Analyze this scenario from these perspectives:
1. Utilitarian ethics
2. Deontological (Kantian) ethics
3. Virtue ethics
4. Care ethics

For each perspective:
- State the core principle
- Apply it to this scenario
- Identify the recommended action
- Note any limitations of this approach

Conclude with a nuanced synthesis of these viewpoints.
        """,
        "expected_format": "essay"
    },
    "technical_design": {
        "name": "System Design",
        "prompt": """
Design a rate limiting system for a high-traffic API with these requirements:

Requirements:
- 1000 requests/minute per user
- 10,000 requests/minute globally
- Graceful degradation under load
- Must work across multiple server instances
- Sub-millisecond latency impact

Constraints:
- Redis available for distributed state
- 99.99% uptime requirement
- Must handle burst traffic (3x normal load)

Provide:
1. High-level architecture diagram (text-based)
2. Algorithm choice with justification
3. Data structures used
4. Failure scenarios and mitigations
5. Monitoring and alerting recommendations
6. Code pseudocode for the core rate limiter

Consider trade-offs and explain your design decisions.
        """,
        "expected_format": "technical"
    }
}

# Model configurations
MODEL_CONFIGS = {
    "gpt-5": {
        "generator_model": "gpt-5",
        "reasoning_effort": "high",
        "supports_thinking": False
    },
    "gpt-5-mini": {
        "generator_model": "gpt-5-mini",
        "reasoning_effort": "medium",
        "supports_thinking": False
    },
    # Top reasoning models (Dec 2025) - GPT-5.2 and Claude Opus 4.5 are above O3
    "gpt-5.2": {
        "generator_model": "gpt-5.2",
        "reasoning_effort": "high",
        "supports_thinking": False
    },
    "claude-opus": {
        "generator_model": "claude-opus-4-5-20251101",
        "thinking_budget_tokens": 32768,  # Max thinking budget
        "supports_thinking": True
    },
    "gemini-3-pro": {
        "generator_model": "gemini-3-pro-preview",
        "supports_thinking": True
    },
    # Legacy O3 models
    "o3": {
        "generator_model": "o3",
        "reasoning_effort": "high",
        "supports_thinking": False
    },
    "o3-mini": {
        "generator_model": "o3-mini",
        "reasoning_effort": "medium",
        "supports_thinking": False
    },
    # Other reasoning models
    "claude-sonnet": {
        "generator_model": "claude-sonnet-4-5",
        "thinking_budget_tokens": 16384,
        "supports_thinking": True
    },
    "grok-4-reasoning": {
        "generator_model": "grok-4-1-fast-reasoning",
        "supports_thinking": False
    },
    "kimi-k2-thinking": {
        "generator_model": "moonshotai/kimi-k2-thinking",
        "supports_thinking": True
    },
    "qwen3-thinking": {
        "generator_model": "qwen/qwen3-235b-a22b-thinking",
        "supports_thinking": True
    }
}


async def demo_reasoning_models():
    """Run the reasoning models demo."""

    parser = argparse.ArgumentParser(description="Reasoning Models Demo")
    parser.add_argument("--model", choices=list(MODEL_CONFIGS.keys()),
                        default="gpt-5.2",
                        help="Model to use")
    parser.add_argument("--effort", choices=["low", "medium", "high"],
                        default="high",
                        help="Reasoning effort (for OpenAI models)")
    parser.add_argument("--thinking", type=int, default=8000,
                        help="Thinking tokens (for Claude models, min 1024)")
    parser.add_argument("--problem", choices=list(REASONING_PROBLEMS.keys()),
                        default="logic_puzzle",
                        help="Problem type to solve")

    args, _ = parser.parse_known_args()

    # Validate thinking budget
    thinking_tokens = max(1024, args.thinking)

    async with AsyncGranSabioClient() as client:
        info = await client.get_info()
        print(f"Connected to: {info['service']} v{info['version']}")

        # Get model config
        model_config = MODEL_CONFIGS.get(args.model, MODEL_CONFIGS["gpt-5.2"]).copy()

        # Apply custom settings
        if model_config.get("supports_thinking"):
            model_config["thinking_budget_tokens"] = thinking_tokens
        elif "reasoning_effort" in model_config:
            model_config["reasoning_effort"] = args.effort

        # Get problem
        problem = REASONING_PROBLEMS[args.problem]

        print()
        print(f"Model: {model_config['generator_model']}")

        if model_config.get("reasoning_effort"):
            print(f"Reasoning Effort: {model_config['reasoning_effort']}")
        if model_config.get("thinking_budget_tokens"):
            print(f"Thinking Budget: {model_config['thinking_budget_tokens']} tokens")

        print()
        print(f"Problem: {problem['name']}")
        print("-" * 50)
        print(problem["prompt"][:300] + "...")

        # QA layer for reasoning quality
        qa_layers = [
            {
                "name": "Reasoning Quality",
                "description": "Evaluates the quality of reasoning",
                "criteria": """
                    The response should:
                    - Show clear step-by-step reasoning
                    - Justify each conclusion
                    - Consider multiple angles when appropriate
                    - Arrive at a coherent final answer
                    - Be logically consistent throughout
                """,
                "min_score": 7.5,
                "is_mandatory": True,
                "order": 1
            }
        ]

        print()
        print_header("Generating with Reasoning", "-")

        # Build request
        request_kwargs = {
            "prompt": problem["prompt"],
            "content_type": "technical",
            "temperature": 0.3,  # Lower for reasoning
            "max_tokens": 4000,
            "qa_models": ["gpt-5-mini"],
            "qa_layers": qa_layers,
            "min_global_score": 7.5,
            "max_iterations": 2,
            "gran_sabio_model": "claude-opus-4-5-20251101",
            "verbose": True,
            "request_name": f"Reasoning: {problem['name']}"
        }

        # Add model-specific params
        request_kwargs["generator_model"] = model_config["generator_model"]

        if model_config.get("reasoning_effort"):
            request_kwargs["reasoning_effort"] = model_config["reasoning_effort"]

        if model_config.get("thinking_budget_tokens"):
            request_kwargs["thinking_budget_tokens"] = model_config["thinking_budget_tokens"]

        request_kwargs["wait_for_completion"] = False  # Return immediately
        result = await client.generate(**request_kwargs)

        session_id = result["session_id"]
        print(f"Session ID: {session_id}")

        if result.get("status") == "rejected":
            feedback = result.get("preflight_feedback", {})
            print(f"[REJECTED] {feedback.get('user_feedback', 'Unknown')}")
            return

        # Monitor with detailed status
        print()
        print("Processing (reasoning models take longer)...")

        final = await client.wait_for_completion(
            session_id,
            poll_interval=3.0,  # Longer poll for reasoning
            timeout=300.0,  # 5 minute timeout for complex reasoning
            on_status=print_status
        )

        # Show full reasoning output
        print_generation_result(
            final,
            title="Reasoning Output",
            content_title=f"Reasoning: {problem['name']}"
        )

        # Summary
        print()
        print("-" * 50)
        print("Reasoning Model Usage Tips:")
        print()
        print("GPT-5/O3 (reasoning_effort):")
        print("  low    - Quick responses, basic reasoning")
        print("  medium - Balanced (recommended for most)")
        print("  high   - Deep analysis, slower but better")
        print()
        print("Claude (thinking_budget_tokens):")
        print("  1024-4000  - Light thinking")
        print("  4000-8000  - Medium depth")
        print("  8000-16000 - Deep analysis")
        print()
        print("Note: Thinking tokens are billed as output tokens")


if __name__ == "__main__":
    asyncio.run(run_demo(demo_reasoning_models, "Demo 10: Reasoning Models"))
