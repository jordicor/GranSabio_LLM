"""
Demo 02: LinkedIn Post with Basic QA
=====================================

This demo shows how to generate professional content with quality validation:
- Word count enforcement
- Two QA layers: Professional Tone and Clarity
- Multiple evaluator models for consensus

This is ideal for:
- Professional social media content
- Business communications
- Content requiring specific tone and length

Usage:
    python demos/02_linkedin_post.py
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from client import AsyncGranSabioClient
from demos.common import run_demo, print_status, print_generation_result


async def demo_linkedin_post():
    """Generate a LinkedIn post with QA validation."""

    async with AsyncGranSabioClient() as client:
        # Check API is available
        info = await client.get_info()
        print(f"Connected to: {info['service']} v{info['version']}")

        # Define the prompt for a LinkedIn post
        prompt = """
Write a LinkedIn post about the importance of continuous learning in the tech industry.

The post should:
- Start with a hook that grabs attention
- Share a personal insight or observation (can be hypothetical)
- Include 2-3 actionable tips
- End with a question to encourage engagement
- Use appropriate emojis sparingly (2-3 max)
- Be professional but conversational
        """.strip()

        print()
        print("Prompt:")
        print("-" * 40)
        print(prompt)

        # Define QA layers for professional content
        qa_layers = [
            {
                "name": "Professional Tone",
                "description": "Ensures appropriate professional language for LinkedIn",
                "criteria": """
                    Evaluate if the content:
                    - Uses professional but approachable language
                    - Avoids slang, jargon, or overly casual expressions
                    - Maintains a confident, knowledgeable tone
                    - Is appropriate for a business networking platform
                """,
                "min_score": 7.5,
                "is_mandatory": True,
                "deal_breaker_criteria": "Uses unprofessional language or inappropriate tone",
                "order": 1
            },
            {
                "name": "Clarity and Engagement",
                "description": "Checks for clear messaging and engagement potential",
                "criteria": """
                    Evaluate if the content:
                    - Has a clear main message
                    - Is easy to read and understand
                    - Includes actionable takeaways
                    - Ends with an engaging call-to-action or question
                    - Is well-structured with appropriate formatting
                """,
                "min_score": 7.0,
                "is_mandatory": False,
                "order": 2
            }
        ]

        # Word count enforcement for optimal LinkedIn post length
        word_count_enforcement = {
            "enabled": True,
            "flexibility_percent": 20.0,  # Allow some flexibility
            "direction": "both",  # Can be slightly more or less
            "severity": "important"  # Reduces score but doesn't reject
        }

        print()
        print("Starting generation with QA...")
        print(f"  Target: 150-250 words")
        print(f"  QA Layers: {len(qa_layers)}")
        print(f"  QA Models: gemini-3-flash-preview, claude-sonnet-4-5")

        result = await client.generate(
            prompt=prompt,
            content_type="article",
            generator_model="gpt-5.2",
            temperature=0.75,
            max_tokens=1000,
            min_words=150,
            max_words=250,
            word_count_enforcement=word_count_enforcement,
            qa_models=["gemini-3-flash-preview", "claude-sonnet-4-5"],
            qa_layers=qa_layers,
            min_global_score=7.5,
            max_iterations=3,
            gran_sabio_model="claude-opus-4-5-20251101",
            verbose=True,
            request_name="LinkedIn Post Demo",
            wait_for_completion=False  # Return immediately with session_id
        )

        session_id = result["session_id"]
        print(f"Session ID: {session_id}")

        if result.get("status") == "rejected":
            feedback = result.get("preflight_feedback", {})
            print(f"[REJECTED] {feedback.get('user_feedback', 'Unknown reason')}")
            return

        # Monitor progress
        print()
        print("Monitoring progress...")

        final = await client.wait_for_completion(
            session_id,
            poll_interval=2.0,
            on_status=print_status
        )

        # Show full generated LinkedIn post
        print_generation_result(
            final,
            title="LinkedIn Post Result",
            content_title="Generated LinkedIn Post"
        )


if __name__ == "__main__":
    asyncio.run(run_demo(demo_linkedin_post, "Demo 02: LinkedIn Post with QA"))
