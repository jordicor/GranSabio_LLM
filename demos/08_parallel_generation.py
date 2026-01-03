"""
Demo 08: Parallel Content Generation
=====================================

This demo shows how to launch multiple generation requests simultaneously
and monitor their progress. Useful for bulk content creation.

Features demonstrated:
- Parallel async generation
- Project ID for grouping related requests
- SSE streaming for real-time progress
- Session management and cancellation

This is ideal for:
- Bulk content creation
- Generating variations
- A/B testing content
- Content calendars

Usage:
    python demos/08_parallel_generation.py

    # Generate more variations:
    python demos/08_parallel_generation.py --count 5

    # Cancel after N seconds:
    python demos/08_parallel_generation.py --timeout 30
"""

import asyncio
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from client import AsyncGranSabioClient
from demos.common import print_header, print_full_content, colorize, safe_print


# Content prompts for parallel generation
CONTENT_PROMPTS = [
    {
        "name": "Product Description - Tech",
        "prompt": "Write a compelling product description (100-150 words) for a smart water bottle that tracks hydration and syncs with fitness apps.",
        "content_type": "article"
    },
    {
        "name": "Product Description - Fashion",
        "prompt": "Write a compelling product description (100-150 words) for a sustainable bamboo watch with minimalist design.",
        "content_type": "article"
    },
    {
        "name": "Social Post - Motivation",
        "prompt": "Write an inspiring social media post (50-100 words) about starting the week with purpose. Include a call to action.",
        "content_type": "article"
    },
    {
        "name": "Social Post - Tips",
        "prompt": "Write a helpful social media post (50-100 words) with 3 quick productivity tips for remote workers.",
        "content_type": "article"
    },
    {
        "name": "Email Subject Lines",
        "prompt": "Generate 5 compelling email subject lines for a newsletter about AI trends. Each should be under 50 characters and create curiosity.",
        "content_type": "creative"
    },
]


class ParallelGenerationManager:
    """Manages multiple parallel generation requests."""

    def __init__(self, client: AsyncGranSabioClient, project_id: str):
        self.client = client
        self.project_id = project_id
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.start_time: datetime = datetime.now()

    async def launch_generation(self, name: str, prompt: str, content_type: str) -> str:
        """Launch a single generation request."""
        try:
            result = await self.client.generate(
                prompt=prompt,
                content_type=content_type,
                generator_model="gpt-5-mini",  # Fast model for demos
                temperature=0.7,
                max_tokens=500,
                qa_layers=[],  # No QA for speed
                qa_models=["gpt-5-mini"],
                max_iterations=1,
                verbose=True,
                project_id=self.project_id,
                request_name=name,
                wait_for_completion=False
            )

            session_id = result.get("session_id", "unknown")
            self.sessions[session_id] = {
                "name": name,
                "status": "running",
                "started_at": datetime.now(),
                "result": None
            }
            return session_id

        except Exception as e:
            print(f"  [ERROR] Failed to launch '{name}': {e}")
            return ""

    async def monitor_session(self, session_id: str) -> Dict[str, Any]:
        """Monitor a single session until completion."""
        session_info = self.sessions.get(session_id, {})
        name = session_info.get("name", "Unknown")

        try:
            while True:
                status = await self.client.get_status(session_id)
                current_status = status.get("status", "unknown")

                self.sessions[session_id]["status"] = current_status

                if current_status in ("completed", "failed", "cancelled"):
                    if current_status == "completed":
                        result = await self.client.get_result(session_id)
                        self.sessions[session_id]["result"] = result
                        self.sessions[session_id]["completed_at"] = datetime.now()
                        return {"success": True, "name": name, "result": result}
                    else:
                        return {"success": False, "name": name, "status": current_status}

                await asyncio.sleep(1.0)

        except Exception as e:
            return {"success": False, "name": name, "error": str(e)}

    async def run_parallel(self, prompts: List[Dict[str, Any]], timeout: float = 120.0) -> Dict[str, Any]:
        """Run multiple generations in parallel."""
        print(f"Launching {len(prompts)} parallel generations...")
        print()

        # Launch all generations
        tasks_to_launch = []
        for prompt_info in prompts:
            tasks_to_launch.append(
                self.launch_generation(
                    prompt_info["name"],
                    prompt_info["prompt"],
                    prompt_info["content_type"]
                )
            )

        session_ids = await asyncio.gather(*tasks_to_launch)
        valid_sessions = [sid for sid in session_ids if sid]

        print(f"Successfully launched: {len(valid_sessions)}/{len(prompts)}")
        for sid in valid_sessions:
            name = self.sessions[sid]["name"]
            print(f"  [{sid[:8]}...] {name}")

        print()
        print("Monitoring progress...")

        # Monitor all sessions with timeout
        try:
            monitor_tasks = [
                self.monitor_session(sid)
                for sid in valid_sessions
            ]

            results = await asyncio.wait_for(
                asyncio.gather(*monitor_tasks, return_exceptions=True),
                timeout=timeout
            )

            return {
                "completed": len([r for r in results if isinstance(r, dict) and r.get("success")]),
                "failed": len([r for r in results if isinstance(r, dict) and not r.get("success")]),
                "results": results
            }

        except asyncio.TimeoutError:
            print()
            print(f"[TIMEOUT] Generation exceeded {timeout}s limit")

            # Cancel remaining sessions
            for sid, info in self.sessions.items():
                if info["status"] == "running":
                    try:
                        await self.client.stop_session(sid)
                        print(f"  Cancelled: {info['name']}")
                    except Exception:
                        pass

            return {
                "completed": len([s for s in self.sessions.values() if s["status"] == "completed"]),
                "failed": len([s for s in self.sessions.values() if s["status"] != "completed"]),
                "timeout": True
            }

    def print_summary(self):
        """Print a summary of all generations."""
        print()
        print_header("Generation Summary", "-")

        total_time = (datetime.now() - self.start_time).total_seconds()

        completed = 0
        total_words = 0

        for sid, info in self.sessions.items():
            status_icon = "[OK]" if info["status"] == "completed" else "[X]"
            name = info["name"][:30]

            if info["status"] == "completed" and info.get("result"):
                completed += 1
                content = info["result"].get("content", "")
                words = len(content.split())
                total_words += words
                print(f"  {status_icon} {name}: {words} words")
            else:
                print(f"  {status_icon} {name}: {info['status']}")

        print()
        print(f"Total: {completed}/{len(self.sessions)} completed")
        print(f"Total Words Generated: {total_words}")
        print(f"Total Time: {total_time:.1f}s")
        print(f"Avg Time per Request: {total_time/max(1, len(self.sessions)):.1f}s")


async def demo_parallel_generation():
    """Run the parallel generation demo."""

    parser = argparse.ArgumentParser(description="Parallel Generation Demo")
    parser.add_argument("--count", type=int, default=3,
                        help="Number of parallel generations (1-5)")
    parser.add_argument("--timeout", type=float, default=120.0,
                        help="Maximum time to wait in seconds")

    args, _ = parser.parse_known_args()

    # Clamp count
    count = max(1, min(5, args.count))

    async with AsyncGranSabioClient() as client:
        info = await client.get_info()
        print(f"Connected to: {info['service']} v{info['version']}")

        # Reserve a project ID
        project_id = await client.reserve_project(None)
        print(f"Project ID: {project_id}")

        print()
        print(f"Configuration:")
        print(f"  Parallel requests: {count}")
        print(f"  Timeout: {args.timeout}s")

        # Select prompts
        prompts = CONTENT_PROMPTS[:count]

        print()
        print("Content to generate:")
        for p in prompts:
            print(f"  - {p['name']}")

        # Run parallel generation
        print()
        print_header("Parallel Generation", "-")

        manager = ParallelGenerationManager(client, project_id)
        results = await manager.run_parallel(prompts, timeout=args.timeout)

        # Print summary
        manager.print_summary()

        # Show full content for all completed generations
        print()
        print_header("All Generated Content", "=")

        for sid, info in manager.sessions.items():
            if info["status"] == "completed" and info.get("result"):
                content = info["result"].get("content", "")
                print_full_content(
                    content,
                    title=f"{info['name']}",
                    indent=2
                )

        print()
        safe_print(colorize(f"  All sessions grouped under project: {project_id}", "cyan"))
        print(f"  View in debugger: /debugger/project/{project_id}")


if __name__ == "__main__":
    print()
    print("=" * 60)
    print(" Demo 08: Parallel Content Generation")
    print("=" * 60)

    try:
        asyncio.run(demo_parallel_generation())
        print()
        print("[OK] Demo completed successfully")
    except KeyboardInterrupt:
        print()
        print("[CANCELLED] Demo cancelled by user")
    except Exception as e:
        print()
        print(f"[ERROR] Demo failed: {e}")
        raise
