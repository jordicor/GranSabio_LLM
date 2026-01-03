"""
Demo 07: Creative Story Generation
===================================

This demo shows how to generate creative fiction with quality control.
Unlike factual content, creative writing requires different QA criteria
focused on narrative quality, character consistency, and engagement.

Features demonstrated:
- Creative content type with higher temperature
- Fiction-oriented QA layers
- Phrase frequency guard for varied writing
- Lexical diversity monitoring

This is ideal for:
- Short story generation
- Creative writing assistance
- Content for entertainment platforms
- Story idea exploration

Usage:
    python demos/07_creative_story.py

    # With custom genre:
    python demos/07_creative_story.py --genre "sci-fi"

    # With specific premise:
    python demos/07_creative_story.py --premise "A robot learns to paint"
"""

import asyncio
import sys
import argparse
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from client import AsyncGranSabioClient
from demos.common import run_demo, print_header, print_status, print_generation_result


# Story premises by genre
STORY_PREMISES = {
    "fantasy": "A mapmaker discovers that the maps they draw become real places.",
    "sci-fi": "The last human in the galaxy runs a coffee shop for alien tourists.",
    "mystery": "A detective who can only solve crimes while sleepwalking.",
    "romance": "Two rival food truck owners compete for the same parking spot.",
    "horror": "A house that shows each visitor their worst memory in every mirror.",
    "comedy": "A time traveler keeps accidentally preventing minor inconveniences.",
    "literary": "An elderly lighthouse keeper receives letters from their future self.",
}


async def demo_creative_story():
    """Run the creative story generation demo."""

    parser = argparse.ArgumentParser(description="Creative Story Demo")
    parser.add_argument("--genre", choices=list(STORY_PREMISES.keys()),
                        default="literary", help="Story genre")
    parser.add_argument("--premise", help="Custom story premise")
    parser.add_argument("--words", type=int, default=800,
                        help="Target word count (600-1200)")

    args, _ = parser.parse_known_args()

    # Clamp word count
    target_words = max(600, min(1200, args.words))

    async with AsyncGranSabioClient() as client:
        info = await client.get_info()
        print(f"Connected to: {info['service']} v{info['version']}")

        # Get premise
        premise = args.premise or STORY_PREMISES.get(args.genre, STORY_PREMISES["literary"])

        print()
        print(f"Genre: {args.genre}")
        print(f"Premise: {premise}")
        print(f"Target: ~{target_words} words")

        prompt = f"""
Write a complete short story based on this premise:

"{premise}"

Requirements:
- Genre: {args.genre}
- Length: approximately {target_words} words
- Include vivid sensory details
- Have a clear beginning, middle, and end
- Create at least one memorable character
- End with something unexpected but satisfying

Write the story directly, no preamble or explanation needed.
        """.strip()

        # QA layers for creative writing
        qa_layers = [
            {
                "name": "Narrative Flow",
                "description": "Story structure and pacing",
                "criteria": """
                    Evaluate the story for:
                    - Clear beginning that hooks the reader
                    - Rising tension or development in the middle
                    - Satisfying resolution or ending
                    - Smooth transitions between scenes
                    - Appropriate pacing (not rushed, not dragging)
                """,
                "min_score": 7.5,
                "is_mandatory": True,
                "order": 1
            },
            {
                "name": "Character Voice",
                "description": "Character authenticity and consistency",
                "criteria": """
                    Check that:
                    - Characters feel distinct and authentic
                    - Actions match established personality
                    - Dialogue sounds natural for each character
                    - Character motivations are clear
                """,
                "min_score": 7.0,
                "is_mandatory": False,
                "order": 2
            },
            {
                "name": "Sensory Immersion",
                "description": "Descriptive quality and atmosphere",
                "criteria": """
                    The story should:
                    - Use multiple senses (not just visual)
                    - Create vivid imagery
                    - Establish clear atmosphere/mood
                    - Show rather than tell where appropriate
                """,
                "min_score": 7.0,
                "is_mandatory": False,
                "order": 3
            }
        ]

        # Phrase frequency to avoid repetitive storytelling
        phrase_frequency = {
            "enabled": True,
            "language": "en",
            "min_n": 3,
            "max_n": 6,
            "min_count": 2,
            "rules": [
                {
                    "name": "action_variety",
                    "min_length": 3,
                    "max_length": 5,
                    "max_ratio_tokens": 0.01,  # 1% of text
                    "severity": "warn",
                    "guidance": "Vary action descriptions to avoid repetitive phrasing"
                },
                {
                    "name": "dialogue_tags",
                    "phrase": "he said",
                    "min_length": 2,
                    "max_ratio_tokens": 0.008,
                    "severity": "warn",
                    "guidance": "Use varied dialogue tags or action beats"
                },
                {
                    "name": "dialogue_tags_she",
                    "phrase": "she said",
                    "min_length": 2,
                    "max_ratio_tokens": 0.008,
                    "severity": "warn",
                    "guidance": "Use varied dialogue tags or action beats"
                }
            ]
        }

        # Lexical diversity for rich vocabulary
        lexical_diversity = {
            "enabled": True,
            "metrics": "auto",
            "top_words_k": 30,
            "language": "en",
            "decision": {
                "deal_breaker_on_red": True,
                "deal_breaker_on_amber": False,
                "require_majority": 2
            }
        }

        print()
        print_header("Generating Story", "-")
        print("QA focuses on: Narrative Flow, Character Voice, Sensory Immersion")
        print("Guards: Phrase repetition, Lexical diversity")

        result = await client.generate(
            prompt=prompt,
            content_type="creative",
            generator_model="claude-opus-4-5-20251101",
            temperature=0.85,  # Higher for creativity
            max_tokens=3000,
            min_words=int(target_words * 0.8),
            max_words=int(target_words * 1.2),
            word_count_enforcement={
                "enabled": True,
                "flexibility_percent": 20,
                "direction": "both",
                "severity": "important"
            },
            qa_models=["gpt-5-mini", "deepseek/deepseek-v3.2-exp"],
            qa_layers=qa_layers,
            phrase_frequency=phrase_frequency,
            lexical_diversity=lexical_diversity,
            min_global_score=7.5,
            max_iterations=4,
            gran_sabio_model="gpt-5.2",
            verbose=True,
            request_name=f"Creative Story ({args.genre})",
            wait_for_completion=False  # Return immediately with session_id
        )

        session_id = result["session_id"]
        print(f"Session ID: {session_id}")

        if result.get("status") == "rejected":
            feedback = result.get("preflight_feedback", {})
            print(f"[REJECTED] {feedback.get('user_feedback', 'Unknown')}")
            return

        # Monitor progress
        print()
        print("Generating story with quality validation...")

        final = await client.wait_for_completion(
            session_id,
            poll_interval=2.0,
            on_status=print_status
        )

        # Show full generated story
        print_generation_result(
            final,
            title="Generated Story",
            content_title=f"Creative Story ({args.genre.title()})"
        )


if __name__ == "__main__":
    asyncio.run(run_demo(demo_creative_story, "Demo 07: Creative Story Generation"))
