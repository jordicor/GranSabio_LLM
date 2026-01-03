"""
Shared Utilities for Gran Sabio LLM Demos
==========================================

Helper functions for formatting and running demo scripts.
"""

import sys
import json
import textwrap
from typing import Any, Callable, Dict, Optional, List

import aiohttp


# ANSI color codes
COLORS = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "green": "\033[92m",
    "blue": "\033[94m",
    "yellow": "\033[93m",
    "red": "\033[91m",
    "cyan": "\033[96m",
    "magenta": "\033[95m",
    "white": "\033[97m",
    "gray": "\033[90m",
}


def colorize(text: str, color: str) -> str:
    """Apply ANSI color to text."""
    return f"{COLORS.get(color, '')}{text}{COLORS['reset']}"


def print_header(title: str, char: str = "="):
    """Print a formatted header."""
    width = 60
    print()
    print(char * width)
    print(f" {title}")
    print(char * width)


def print_section(title: str, char: str = "-"):
    """Print a section header."""
    print()
    safe_print(colorize(f"  {title}", "cyan"))
    print(f"  {char * 50}")


def print_status(status: Dict[str, Any]):
    """Print a status update."""
    print(f"  Status: {status['status']} | Iteration: {status.get('current_iteration', '?')}/{status.get('max_iterations', '?')}")

    if status.get("verbose_log"):
        for entry in status["verbose_log"][-3:]:
            # Handle Unicode encoding issues on Windows
            try:
                print(f"    > {entry}")
            except UnicodeEncodeError:
                # Remove problematic characters and retry
                safe_entry = entry.encode('ascii', 'ignore').decode('ascii')
                print(f"    > {safe_entry}")


def safe_print(text: str):
    """Print text, handling Unicode issues on Windows."""
    try:
        print(text)
    except UnicodeEncodeError:
        safe_text = text.encode('ascii', 'ignore').decode('ascii')
        print(safe_text)


def print_result(result: Dict[str, Any], max_content_length: int = 500):
    """Print a generation result (legacy - preview only)."""
    print()
    safe_print(f"  Final Score: {result.get('final_score', 'N/A')}")
    safe_print(f"  Final Iteration: {result.get('final_iteration', 'N/A')}")

    if result.get("project_id"):
        safe_print(f"  Project ID: {result['project_id']}")

    content = result.get("content", "")
    if content:
        preview = content[:max_content_length]
        if len(content) > max_content_length:
            preview += "..."
        print()
        safe_print("  Content Preview:")
        safe_print("  " + "-" * 40)
        for line in preview.split("\n")[:10]:
            safe_print(f"  {line}")

    if result.get("gran_sabio_reason"):
        print()
        safe_print(f"  Gran Sabio: {result['gran_sabio_reason']}")


def print_full_content(
    content: str,
    title: str = "Generated Content",
    indent: int = 2,
    wrap_width: int = 76
):
    """
    Print the full generated content in a readable format.

    Args:
        content: The text content to display
        title: Title for the content section
        indent: Number of spaces to indent
        wrap_width: Maximum line width (0 = no wrapping)
    """
    if not content:
        print()
        safe_print(colorize("  (No content generated)", "yellow"))
        return

    indent_str = " " * indent

    print()
    print(colorize(f"{indent_str}{title}", "green"))
    print(f"{indent_str}{'=' * 56}")
    print()

    # Print content line by line, preserving structure
    for line in content.split("\n"):
        if wrap_width > 0 and len(line) > wrap_width:
            # Wrap long lines
            wrapped = textwrap.wrap(line, width=wrap_width - indent)
            for wrapped_line in wrapped:
                safe_print(f"{indent_str}{wrapped_line}")
        else:
            safe_print(f"{indent_str}{line}")

    print()
    print(f"{indent_str}{'=' * 56}")

    # Show stats
    word_count = len(content.split())
    char_count = len(content)
    line_count = len(content.split("\n"))
    print(f"{indent_str}{colorize(f'Stats: {word_count} words | {char_count} chars | {line_count} lines', 'gray')}")


def print_json_content(
    data: Any,
    title: str = "JSON Output",
    indent: int = 2,
    max_string_length: int = 200
):
    """
    Print JSON data in a formatted, colored way.

    Args:
        data: JSON data (dict, list, or string)
        title: Title for the section
        indent: Number of spaces to indent
        max_string_length: Max length for string values (0 = no limit)
    """
    indent_str = " " * indent

    print()
    print(colorize(f"{indent_str}{title}", "green"))
    print(f"{indent_str}{'=' * 56}")
    print()

    # Parse if string
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            safe_print(f"{indent_str}{data}")
            return

    # Pretty print with colors
    formatted = json.dumps(data, indent=2, ensure_ascii=False)

    for line in formatted.split("\n"):
        # Colorize JSON elements
        colored_line = line
        if '": ' in line or '":' in line:
            # Keys in blue
            parts = line.split('": ', 1)
            if len(parts) == 2:
                key_part = parts[0] + '"'
                value_part = parts[1]
                colored_line = f"{colorize(key_part, 'blue')}: {value_part}"

        safe_print(f"{indent_str}{colored_line}")

    print()
    print(f"{indent_str}{'=' * 56}")


def print_generation_result(
    result: Dict[str, Any],
    title: str = "Generation Result",
    show_full_content: bool = True,
    content_title: str = "Generated Content",
    is_json: bool = False
):
    """
    Print a complete generation result with full content display.

    Args:
        result: The result dictionary from the API
        title: Title for the result section
        show_full_content: Whether to show full content (True) or preview
        content_title: Title for the content section
        is_json: Whether the content is JSON
    """
    print()
    print_header(title, "=")

    # Metadata
    print()
    safe_print(colorize("  Metadata:", "cyan"))
    print("  " + "-" * 40)
    safe_print(f"  Final Score: {colorize(str(result.get('final_score', 'N/A')), 'green')}")
    safe_print(f"  Iterations: {result.get('final_iteration', 'N/A')}")

    if result.get("project_id"):
        safe_print(f"  Project ID: {result['project_id']}")

    if result.get("session_id"):
        safe_print(f"  Session ID: {result['session_id'][:8]}...")

    # QA Summary if available
    qa_summary = result.get("qa_summary", {})
    if qa_summary and "layer_scores" in qa_summary:
        print()
        safe_print(colorize("  QA Scores:", "cyan"))
        print("  " + "-" * 40)
        for layer, score in qa_summary["layer_scores"].items():
            score_color = "green" if score >= 7.5 else ("yellow" if score >= 6 else "red")
            safe_print(f"  {layer}: {colorize(f'{score:.1f}/10', score_color)}")

    # Gran Sabio intervention
    if result.get("gran_sabio_reason"):
        print()
        safe_print(colorize("  Gran Sabio Intervention:", "magenta"))
        print("  " + "-" * 40)
        safe_print(f"  {result['gran_sabio_reason']}")

    # Content
    content = result.get("content", "")
    if content:
        if is_json:
            print_json_content(content, title=content_title)
        else:
            print_full_content(content, title=content_title)
    else:
        print()
        safe_print(colorize("  (No content in result)", "yellow"))


def print_multi_phase_summary(
    phases: Dict[str, Any],
    project_id: str = None
):
    """
    Print a summary of multi-phase generation results.

    Args:
        phases: Dictionary of phase_name -> result data
        project_id: Optional project ID
    """
    print()
    print_header("Multi-Phase Generation Summary", "=")

    if project_id:
        print()
        safe_print(f"  Project ID: {colorize(project_id, 'cyan')}")

    print()
    safe_print(colorize("  Phases Completed:", "green"))
    print("  " + "-" * 50)

    for phase_name, phase_data in phases.items():
        if isinstance(phase_data, dict):
            # JSON phase
            item_count = len(phase_data) if isinstance(phase_data, dict) else "N/A"
            safe_print(f"  [+] {phase_name}: {colorize('JSON', 'blue')} ({item_count} keys)")
        elif isinstance(phase_data, str):
            # Text phase
            word_count = len(phase_data.split())
            safe_print(f"  [+] {phase_name}: {colorize('Text', 'green')} ({word_count} words)")
        else:
            safe_print(f"  [+] {phase_name}: {type(phase_data).__name__}")


def display_phase_result(
    phase_name: str,
    phase_num: int,
    total_phases: int,
    content: Any,
    is_json: bool = False
):
    """
    Display a single phase result from a multi-phase workflow.

    Args:
        phase_name: Name of the phase
        phase_num: Phase number (1-indexed)
        total_phases: Total number of phases
        content: The phase content
        is_json: Whether content is JSON
    """
    print()
    print(colorize(f"  PHASE {phase_num}/{total_phases}: {phase_name.upper()}", "magenta"))
    print("  " + "=" * 56)

    if is_json:
        print_json_content(content, title=f"Phase {phase_num} Output", indent=4)
    else:
        print_full_content(content, title=f"Phase {phase_num} Output", indent=4)


async def run_demo(demo_func: Callable, title: str):
    """Run a demo with error handling."""
    print_header(title)

    try:
        await demo_func()
        print()
        print("[OK] Demo completed successfully")
    except aiohttp.ClientConnectorError:
        print()
        print("[ERROR] Cannot connect to Gran Sabio API")
        print("        Make sure the server is running: python main.py")
        sys.exit(1)
    except Exception as e:
        print()
        print(f"[ERROR] Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
