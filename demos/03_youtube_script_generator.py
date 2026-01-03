"""
Demo 03: YouTube Script Generator (Multi-Call Pipeline)
========================================================

*** STAR DEMO - Showcases Multi-Phase Content Generation ***

This demo demonstrates a complete content generation pipeline with multiple
sequential API calls, similar to professional content production workflows:

Phase 1: Topic Analysis (JSON output)
    - Analyze the topic and generate structured planning data
    - Extract hook, main points, and estimated scenes

Phase 2: Full Script Generation (text with QA)
    - Use Phase 1 data as context
    - Generate complete video script
    - Apply QA for engagement and structure

Phase 3: Scene Breakdown (JSON output)
    - Parse script into visual scenes
    - Generate scene descriptions and timing

Phase 4: Thumbnail Ideas (JSON output)
    - Generate clickable thumbnail concepts
    - Include title variations

This is ideal for:
- YouTube content automation
- Video production pipelines
- Multi-step content workflows
- Content agencies

Usage:
    python demos/03_youtube_script_generator.py

    # With custom topic:
    python demos/03_youtube_script_generator.py --topic "How to Learn Programming in 2024"
"""

import asyncio
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from client import AsyncGranSabioClient
from demos.common import (
    run_demo,
    print_header,
    print_json_content,
    print_full_content,
    print_multi_phase_summary,
    display_phase_result,
    colorize,
    safe_print,
)


# JSON Schemas for structured outputs
# Note: OpenAI Structured Outputs requires ALL properties in 'required'
# and optional fields must use ["type", "null"] union
TOPIC_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "hook": {
            "type": "string",
            "description": "Attention-grabbing opening line (5-15 words)"
        },
        "main_thesis": {
            "type": "string",
            "description": "Core message of the video"
        },
        "target_audience": {
            "type": ["string", "null"],
            "description": "Who this video is for"
        },
        "main_points": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "key_insight": {"type": "string"},
                    "example": {"type": ["string", "null"]}
                },
                "required": ["title", "key_insight", "example"],
                "additionalProperties": False
            },
            "minItems": 3,
            "maxItems": 7
        },
        "call_to_action": {
            "type": "string",
            "description": "What viewers should do after watching"
        },
        "estimated_duration_minutes": {
            "type": "integer",
            "minimum": 1,
            "maximum": 60
        },
        "content_style": {
            "type": ["string", "null"],
            "enum": ["educational", "entertainment", "motivational", "tutorial", "listicle"]
        }
    },
    "required": ["hook", "main_thesis", "target_audience", "main_points", "call_to_action", "estimated_duration_minutes", "content_style"],
    "additionalProperties": False
}

SCENE_BREAKDOWN_SCHEMA = {
    "type": "object",
    "properties": {
        "total_scenes": {"type": "integer"},
        "estimated_total_duration_seconds": {"type": ["integer", "null"]},
        "scenes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "scene_number": {"type": "integer"},
                    "scene_type": {
                        "type": "string",
                        "enum": ["hook", "intro", "main_content", "transition", "example", "recap", "cta", "outro"]
                    },
                    "duration_seconds": {"type": "integer", "minimum": 5, "maximum": 300},
                    "script_excerpt": {"type": ["string", "null"]},
                    "visual_description": {"type": "string"},
                    "b_roll_suggestions": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "text_overlay": {"type": ["string", "null"]}
                },
                "required": ["scene_number", "scene_type", "duration_seconds", "script_excerpt", "visual_description", "b_roll_suggestions", "text_overlay"],
                "additionalProperties": False
            }
        }
    },
    "required": ["total_scenes", "estimated_total_duration_seconds", "scenes"],
    "additionalProperties": False
}

THUMBNAIL_IDEAS_SCHEMA = {
    "type": "object",
    "properties": {
        "thumbnails": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "concept_id": {"type": "integer"},
                    "title_text": {
                        "type": "string",
                        "description": "Short, punchy text for thumbnail (3-6 words max)"
                    },
                    "visual_description": {"type": "string"},
                    "emotion_to_convey": {
                        "type": "string",
                        "enum": ["surprise", "curiosity", "excitement", "urgency", "trust", "fear"]
                    },
                    "color_scheme": {"type": ["string", "null"]},
                    "face_expression": {"type": ["string", "null"]}
                },
                "required": ["concept_id", "title_text", "visual_description", "emotion_to_convey", "color_scheme", "face_expression"],
                "additionalProperties": False
            },
            "minItems": 3,
            "maxItems": 5
        },
        "recommended_thumbnail": {"type": "integer"}
    },
    "required": ["thumbnails", "recommended_thumbnail"],
    "additionalProperties": False
}


class YouTubeScriptPipeline:
    """Multi-phase YouTube script generation pipeline."""

    def __init__(self, client: AsyncGranSabioClient, project_id: str):
        self.client = client
        self.project_id = project_id
        self.results: Dict[str, Any] = {}

    async def phase_1_topic_analysis(self, topic: str, duration_minutes: int = 10) -> Dict[str, Any]:
        """Phase 1: Analyze topic and create structured plan."""
        print()
        print_header("Phase 1: Topic Analysis", "-")

        prompt = f"""
Analyze the following YouTube video topic and create a structured content plan.

TOPIC: {topic}
TARGET DURATION: {duration_minutes} minutes

Create a compelling content structure that will:
1. Hook viewers in the first 5 seconds
2. Deliver clear, actionable value
3. Keep viewers engaged throughout
4. End with a strong call to action

Consider what makes content viral and engaging on YouTube.
        """.strip()

        print(f"Analyzing topic: {topic}")
        print(f"Target duration: {duration_minutes} minutes")

        result = await self.client.generate(
            prompt=prompt,
            content_type="json",
            generator_model="gemini-3-pro-preview",
            temperature=0.7,
            max_tokens=2000,
            json_output=True,
            json_schema=TOPIC_ANALYSIS_SCHEMA,
            qa_layers=[],  # No QA for structured extraction
            qa_models=["gpt-5-mini"],
            project_id=self.project_id,
            request_name="Phase 1: Topic Analysis",
            verbose=True,
            wait_for_completion=False
        )

        session_id = result["session_id"]
        print(f"Session ID: {session_id}")

        final = await self.client.wait_for_completion(
            session_id,
            poll_interval=1.5,
            on_status=lambda s: print(f"  Status: {s['status']}")
        )

        # Parse JSON content
        import json
        try:
            content = final.get("content", "{}")
            if isinstance(content, str):
                self.results["topic_analysis"] = json.loads(content)
            else:
                self.results["topic_analysis"] = content
        except json.JSONDecodeError:
            print("[WARNING] Could not parse JSON response")
            self.results["topic_analysis"] = {"raw": content}

        # Display full topic analysis
        display_phase_result(
            "Topic Analysis",
            phase_num=1,
            total_phases=4,
            content=self.results["topic_analysis"],
            is_json=True
        )

        return self.results["topic_analysis"]

    async def phase_2_script_generation(self, topic: str) -> str:
        """Phase 2: Generate the full video script with QA."""
        print()
        print_header("Phase 2: Full Script Generation", "-")

        analysis = self.results.get("topic_analysis", {})

        # Build context from Phase 1
        context_parts = [
            f"TOPIC: {topic}",
            f"HOOK: {analysis.get('hook', 'Create an attention-grabbing opening')}",
            f"MAIN THESIS: {analysis.get('main_thesis', 'Deliver value on the topic')}",
            f"TARGET AUDIENCE: {analysis.get('target_audience', 'General audience')}",
            f"DURATION: {analysis.get('estimated_duration_minutes', 10)} minutes",
            "",
            "MAIN POINTS TO COVER:"
        ]

        for i, point in enumerate(analysis.get("main_points", []), 1):
            context_parts.append(f"  {i}. {point.get('title', 'Point')}: {point.get('key_insight', '')}")

        context_parts.extend([
            "",
            f"CALL TO ACTION: {analysis.get('call_to_action', 'Subscribe and like')}"
        ])

        context = "\n".join(context_parts)

        prompt = f"""
Write a complete YouTube video script based on the following content plan.

{context}

SCRIPT REQUIREMENTS:
- Write in a conversational, engaging tone
- Include speaker cues like [PAUSE], [EMPHASIS], [GESTURE]
- Mark section transitions clearly
- Include timestamps suggestions for key moments
- Write natural dialogue, not robotic narration
- Make it sound like a real person talking to the viewer

FORMAT:
[INTRO - 0:00]
Hook and introduction...

[SECTION 1 - ~X:XX]
First main point...

...continue for all sections...

[OUTRO - ~X:XX]
Recap and call to action...
        """.strip()

        # QA layers for script quality
        qa_layers = [
            {
                "name": "Engagement Hook",
                "description": "Evaluates if the opening grabs attention",
                "criteria": """
                    The first 10 seconds must:
                    - Create curiosity or excitement
                    - Promise clear value
                    - NOT start with 'Hey guys' or generic greetings
                    - Make viewers want to keep watching
                """,
                "min_score": 7.5,
                "is_mandatory": True,
                "deal_breaker_criteria": "Opening is boring or generic",
                "order": 1
            },
            {
                "name": "Structure Clarity",
                "description": "Checks for clear, logical structure",
                "criteria": """
                    The script must:
                    - Have clear section breaks
                    - Flow logically from point to point
                    - Include smooth transitions
                    - Build to a satisfying conclusion
                """,
                "min_score": 7.0,
                "is_mandatory": True,
                "order": 2
            },
            {
                "name": "Viewer Retention",
                "description": "Evaluates pacing and engagement throughout",
                "criteria": """
                    Check that the script:
                    - Maintains energy throughout
                    - Varies pace and tone
                    - Includes hooks before each section ('Coming up...')
                    - Doesn't have long monotonous sections
                """,
                "min_score": 7.0,
                "is_mandatory": False,
                "order": 3
            }
        ]

        # Lexical diversity to avoid repetitive language
        lexical_diversity = {
            "enabled": True,
            "metrics": "auto",
            "top_words_k": 30,
            "decision": {
                "deal_breaker_on_red": True,
                "deal_breaker_on_amber": False,
                "require_majority": 2
            }
        }

        print("Generating full script with QA validation...")
        print(f"  QA Layers: {len(qa_layers)}")

        result = await self.client.generate(
            prompt=prompt,
            content_type="script",
            generator_model="gemini-3-pro-preview",
            temperature=0.75,
            max_tokens=4000,
            min_words=800,
            max_words=1500,
            qa_models=["gpt-5-mini", "moonshotai/kimi-k2-thinking"],
            qa_layers=qa_layers,
            lexical_diversity=lexical_diversity,
            min_global_score=7.5,
            max_iterations=4,
            gran_sabio_model="claude-opus-4-5-20251101",
            project_id=self.project_id,
            request_name="Phase 2: Script Generation",
            verbose=True,
            wait_for_completion=False
        )

        session_id = result["session_id"]
        print(f"Session ID: {session_id}")

        if result.get("status") == "rejected":
            print(f"[REJECTED] {result.get('preflight_feedback', {}).get('user_feedback', 'Unknown')}")
            return ""

        final = await self.client.wait_for_completion(
            session_id,
            poll_interval=2.0,
            on_status=lambda s: print(f"  Status: {s['status']} | Iteration: {s.get('current_iteration', '?')}/{s.get('max_iterations', '?')}")
        )

        script = final.get("content", "")
        self.results["script"] = script

        # Display full script
        print()
        safe_print(colorize(f"  Final Score: {final.get('final_score', 'N/A')}", "green"))

        display_phase_result(
            "Full Script",
            phase_num=2,
            total_phases=4,
            content=script,
            is_json=False
        )

        return script

    async def phase_3_scene_breakdown(self) -> Dict[str, Any]:
        """Phase 3: Break down script into visual scenes."""
        print()
        print_header("Phase 3: Scene Breakdown", "-")

        script = self.results.get("script", "")
        if not script:
            print("[ERROR] No script available. Run Phase 2 first.")
            return {}

        prompt = f"""
Analyze this video script and break it down into discrete visual scenes.

SCRIPT:
{script[:3000]}

For each scene, specify:
- Scene type (hook, intro, main_content, transition, example, recap, cta, outro)
- Duration in seconds
- Visual description for video editor
- B-roll suggestions
- Any text overlays needed

Aim for 8-15 scenes depending on video length.
        """.strip()

        print("Breaking down script into scenes...")

        result = await self.client.generate(
            prompt=prompt,
            content_type="json",
            generator_model="gemini-3-pro-preview",
            temperature=0.5,
            max_tokens=3000,
            json_output=True,
            json_schema=SCENE_BREAKDOWN_SCHEMA,
            qa_layers=[],
            qa_models=["gpt-5-mini"],
            project_id=self.project_id,
            request_name="Phase 3: Scene Breakdown",
            verbose=True,
            wait_for_completion=False
        )

        session_id = result["session_id"]
        print(f"Session ID: {session_id}")

        final = await self.client.wait_for_completion(
            session_id,
            poll_interval=1.5,
            on_status=lambda s: print(f"  Status: {s['status']}")
        )

        import json
        try:
            content = final.get("content", "{}")
            if isinstance(content, str):
                self.results["scenes"] = json.loads(content)
            else:
                self.results["scenes"] = content
        except json.JSONDecodeError:
            print("[WARNING] Could not parse JSON response")
            self.results["scenes"] = {"raw": content}

        scenes = self.results["scenes"]

        # Display full scene breakdown
        display_phase_result(
            "Scene Breakdown",
            phase_num=3,
            total_phases=4,
            content=scenes,
            is_json=True
        )

        return scenes

    async def phase_4_thumbnail_ideas(self, topic: str) -> Dict[str, Any]:
        """Phase 4: Generate thumbnail concepts."""
        print()
        print_header("Phase 4: Thumbnail Ideas", "-")

        analysis = self.results.get("topic_analysis", {})

        prompt = f"""
Generate 3-5 clickable YouTube thumbnail concepts for this video.

TOPIC: {topic}
HOOK: {analysis.get('hook', 'N/A')}
CONTENT STYLE: {analysis.get('content_style', 'educational')}

Each thumbnail should:
- Be visually striking and scroll-stopping
- Use proven psychological triggers
- Have minimal, punchy text (3-6 words MAX)
- Suggest specific colors and composition
- Describe the ideal facial expression if a face is included

Think about what makes people click on YouTube.
        """.strip()

        print("Generating thumbnail concepts...")

        result = await self.client.generate(
            prompt=prompt,
            content_type="json",
            generator_model="gemini-3-pro-preview",
            temperature=0.8,  # Higher creativity for thumbnails
            max_tokens=2000,
            json_output=True,
            json_schema=THUMBNAIL_IDEAS_SCHEMA,
            qa_layers=[],
            qa_models=["gpt-5-mini"],
            project_id=self.project_id,
            request_name="Phase 4: Thumbnail Ideas",
            verbose=True,
            wait_for_completion=False
        )

        session_id = result["session_id"]
        print(f"Session ID: {session_id}")

        final = await self.client.wait_for_completion(
            session_id,
            poll_interval=1.5,
            on_status=lambda s: print(f"  Status: {s['status']}")
        )

        import json
        try:
            content = final.get("content", "{}")
            if isinstance(content, str):
                self.results["thumbnails"] = json.loads(content)
            else:
                self.results["thumbnails"] = content
        except json.JSONDecodeError:
            print("[WARNING] Could not parse JSON response")
            self.results["thumbnails"] = {"raw": content}

        thumbnails = self.results["thumbnails"]

        # Display full thumbnail ideas
        display_phase_result(
            "Thumbnail Ideas",
            phase_num=4,
            total_phases=4,
            content=thumbnails,
            is_json=True
        )

        return thumbnails

    def get_full_results(self) -> Dict[str, Any]:
        """Get all results from all phases."""
        return {
            "project_id": self.project_id,
            "phases": self.results
        }


async def demo_youtube_script_generator():
    """Run the complete YouTube script generation pipeline."""

    # Parse command line args
    parser = argparse.ArgumentParser(description="YouTube Script Generator Demo")
    parser.add_argument(
        "--topic",
        default="5 Habits That Changed My Life (Backed by Science)",
        help="Video topic to generate script for"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=10,
        help="Target video duration in minutes"
    )
    parser.add_argument(
        "--skip-scenes",
        action="store_true",
        help="Skip scene breakdown phase"
    )
    parser.add_argument(
        "--skip-thumbnails",
        action="store_true",
        help="Skip thumbnail ideas phase"
    )

    # Only parse known args to avoid issues with run_demo
    args, _ = parser.parse_known_args()

    async with AsyncGranSabioClient() as client:
        # Check API
        info = await client.get_info()
        print(f"Connected to: {info['service']} v{info['version']}")

        # Reserve a project ID to group all phases
        project_id = await client.reserve_project(None)
        print(f"Project ID: {project_id}")

        print()
        print(f"Topic: {args.topic}")
        print(f"Target Duration: {args.duration} minutes")
        print()
        print("This demo will execute 4 phases:")
        print("  1. Topic Analysis (JSON)")
        print("  2. Full Script Generation (Text + QA)")
        print("  3. Scene Breakdown (JSON)")
        print("  4. Thumbnail Ideas (JSON)")

        # Initialize pipeline
        pipeline = YouTubeScriptPipeline(client, project_id)

        # Execute phases
        await pipeline.phase_1_topic_analysis(args.topic, args.duration)
        await pipeline.phase_2_script_generation(args.topic)

        if not args.skip_scenes:
            await pipeline.phase_3_scene_breakdown()

        if not args.skip_thumbnails:
            await pipeline.phase_4_thumbnail_ideas(args.topic)

        # Summary of all phases
        results = pipeline.get_full_results()
        print_multi_phase_summary(results["phases"], project_id=project_id)

        print()
        safe_print(colorize("  All phases complete. Full content shown above.", "green"))
        print(f"  View in debugger: /debugger/project/{project_id}")


if __name__ == "__main__":
    asyncio.run(run_demo(
        demo_youtube_script_generator,
        "Demo 03: YouTube Script Generator (Multi-Phase Pipeline)"
    ))
