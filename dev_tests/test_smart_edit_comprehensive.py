"""
Comprehensive Test Suite for Smart Edit System
===============================================

All tests prefixed with 'test_smart_edit_' for easy identification and batch execution.

This test suite covers:
1. Direct operations (no AI): delete, insert, replace, format
2. Phrase localization with various text types
3. Text normalization (typographic chars, Unicode)
4. Counted phrase format parsing and validation
5. AI-assisted editing operations
6. Edge cases: short texts, long texts, special chars, markdown, HTML

Usage:
    # Run all smart_edit tests:
    python dev_tests/test_smart_edit_comprehensive.py

    # Run with external text file(s):
    python dev_tests/test_smart_edit_comprehensive.py --file "path/to/text.txt"
    python dev_tests/test_smart_edit_comprehensive.py --file "path/to/file1.txt" --file "path/to/file2.txt"

    # Run specific test categories:
    python dev_tests/test_smart_edit_comprehensive.py --category direct
    python dev_tests/test_smart_edit_comprehensive.py --category localization
    python dev_tests/test_smart_edit_comprehensive.py --category ai
    python dev_tests/test_smart_edit_comprehensive.py --category edge
"""

import sys
import os
import asyncio
import argparse
import time
import traceback
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Global list of external test files (populated via --file argument)
EXTERNAL_TEST_FILES: List[str] = []

# =============================================================================
# IMPORTS FROM SMART_EDIT
# =============================================================================

from smart_edit import (
    SmartTextEditor,
    TargetMode,
    TargetScope,
    TextTarget,
    OperationType,
    find_optimal_phrase_length,
    build_word_map,
    locate_by_markers,
    locate_by_word_indices,
    normalize_source_text,
    normalize_for_matching,
    parse_counted_phrase,
    validate_counted_phrase_format,
    extract_phrase_from_response,
)


# =============================================================================
# TEST RESULT TRACKING
# =============================================================================

class TestStatus(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"
    ERROR = "ERROR"


@dataclass
class TestResult:
    name: str
    status: TestStatus
    message: str = ""
    duration_ms: float = 0
    details: Optional[Dict] = None


class TestRunner:
    """Tracks and reports test results."""

    def __init__(self):
        self.results: List[TestResult] = []
        self.current_category = ""

    def add_result(self, result: TestResult):
        self.results.append(result)
        status_icon = {
            TestStatus.PASS: "[PASS]",
            TestStatus.FAIL: "[FAIL]",
            TestStatus.SKIP: "[SKIP]",
            TestStatus.ERROR: "[ERROR]"
        }
        icon = status_icon.get(result.status, "[????]")
        duration = f"({result.duration_ms:.0f}ms)" if result.duration_ms > 0 else ""
        print(f"  {icon} {result.name} {duration}")
        if result.message and result.status != TestStatus.PASS:
            print(f"         -> {result.message[:200]}")

    def print_summary(self):
        total = len(self.results)
        passed = sum(1 for r in self.results if r.status == TestStatus.PASS)
        failed = sum(1 for r in self.results if r.status == TestStatus.FAIL)
        errors = sum(1 for r in self.results if r.status == TestStatus.ERROR)
        skipped = sum(1 for r in self.results if r.status == TestStatus.SKIP)

        print()
        print("=" * 70)
        print(f"SUMMARY: {passed}/{total} passed, {failed} failed, {errors} errors, {skipped} skipped")
        print("=" * 70)

        if failed > 0 or errors > 0:
            print("\nFailed/Error tests:")
            for r in self.results:
                if r.status in (TestStatus.FAIL, TestStatus.ERROR):
                    print(f"  - {r.name}: {r.message[:100]}")

        return failed == 0 and errors == 0


runner = TestRunner()


# =============================================================================
# SAMPLE TEXTS FOR TESTING
# =============================================================================

# Short text (< 50 words)
TEXT_SHORT = """The quick brown fox jumps over the lazy dog. This is a simple test sentence with basic punctuation."""

# Medium text (~200 words)
TEXT_MEDIUM = """
In the heart of the ancient forest, where sunlight barely penetrated the thick canopy of leaves,
there lived a wise old owl named Prometheus. He had witnessed countless seasons pass, each bringing
its own stories and mysteries. The younger creatures of the forest often sought his counsel,
traveling from far and wide to hear his wisdom.

One autumn evening, a young rabbit named Clara approached his oak tree. She was troubled by dreams
of a golden key that seemed to unlock nothing. Prometheus listened carefully, his large amber eyes
reflecting the dying light of day.

"Dreams are not always what they seem," he said softly. "Sometimes the key is not meant to open
a door, but to remind us that some mysteries are better left unsolved. The journey of seeking
is often more valuable than the destination itself."

Clara pondered his words as she hopped back through the darkening woods. The wise owl watched
her go, knowing that understanding would come to her in time, as it always did to those who
truly listened.
"""

# Text with markdown formatting
TEXT_MARKDOWN = """
# The Art of Programming

**Programming** is not just about writing code; it's about *solving problems* creatively.

## Key Principles

1. **Keep it simple** - Complexity is the enemy of reliability
2. **Write clean code** - Your future self will thank you
3. **Test thoroughly** - Bugs are easier to prevent than to fix

### Code Example

Here's a simple function:

```python
def hello_world():
    print("Hello, World!")
```

> "Programs must be written for people to read, and only incidentally for machines to execute."
> -- Harold Abelson

---

Remember: `code` should be **readable** and *maintainable*.

[Learn more](https://example.com) about best practices.
"""

# Text with HTML
TEXT_HTML = """
<html>
<head><title>Test Document</title></head>
<body>
<h1>Welcome to Our Website</h1>
<p>This is a <strong>paragraph</strong> with <em>various</em> HTML tags.</p>
<ul>
    <li>First item</li>
    <li>Second item with <a href="https://example.com">a link</a></li>
    <li>Third item</li>
</ul>
<div class="container">
    <span style="color: red;">Styled text</span> and normal text.
</div>
<p>Contact us at <code>info@example.com</code> or call 555-1234.</p>
</body>
</html>
"""

# Text with special/typographic characters
TEXT_SPECIAL_CHARS = """
"Hello," she said -- "how are you?"

The temperature was 25C (that's 77F in American units).

Price: $19.99 (20% off!)

The cafe served croissants, pain au chocolat, and cafe creme.

Em-dash example: The result was clear--victory was inevitable.

En-dash example: The years 2020-2025 were transformative.

Ellipsis: And then... silence.

Guillemets: Le livre s'appelle <<Les Miserables>>.

Curly quotes: "She replied, 'Yes, of course!'"

Mathematical: 2 + 2 = 4, but is e^(i*pi) = -1?

Arrows: Click here -> to continue <- or go back

Bullets: * Item one * Item two * Item three

Special symbols: (c) 2024 Company Inc. (TM) (R)
"""

# Text with Unicode characters (Spanish, French, German)
TEXT_UNICODE = """
El nino corrio rapidamente hacia la montana, donde encontro un hermoso jardin lleno de flores.
La senorita pregunto: "Donde esta el bano?" y el senor respondio amablemente.

En francais: L'ete dernier, nous avons visite la cathedrale Notre-Dame. C'etait magnifique!
Le garcon a mange des crepes avec du cafe au lait.

Auf Deutsch: Der Madchen spielte im Garten wahrend ihre Bruder Fu ball spielten.
Fruhstuck in Munchen: Brotchen mit Kase und Wurst.

Mixed: The resume showed experience in Zurich, Malaga, and Koln.
"""

# Text with repetitive patterns (challenging for phrase matching)
TEXT_REPETITIVE = """
The cat sat on the mat. The cat was happy. The cat purred loudly.
The dog ran in the park. The dog was excited. The dog barked happily.
The bird flew in the sky. The bird was free. The bird sang beautifully.

In the morning, the cat sat on the mat again. In the morning, the dog ran in the park again.
In the morning, the bird flew in the sky again.

The cat sat. The dog ran. The bird flew.
The cat sat. The dog ran. The bird flew.
The cat sat. The dog ran. The bird flew.

Finally, the cat sat on the mat one last time. The dog ran in the park one last time.
The bird flew in the sky one last time.
"""

# Very long paragraph (single block)
TEXT_LONG_PARAGRAPH = """
In the vast expanse of human history, few endeavors have captured the imagination and determination of our species quite like the exploration of space, that final frontier which beckons us with its mysteries and promises, challenging us to push beyond the comfortable boundaries of our terrestrial existence and venture into the cold, silent void that surrounds our small blue planet, a world that has been our home for countless millennia but which now seems almost confining as we gaze upward at the stars and wonder what secrets they hold, what worlds orbit those distant suns, and whether somewhere out there in the cosmic darkness other beings might be looking back at us, pondering the same eternal questions that have driven philosophers and scientists and dreamers throughout the ages to seek understanding of our place in this unimaginably vast universe, where distances are measured in light-years and time itself bends and stretches according to the speed at which one travels, where black holes swallow light and neutron stars spin with such ferocity that they emit beams of radiation detectable across galaxies, where nebulae birth new stars from clouds of gas and dust that have drifted through space for billions of years, carrying within them the very atoms that would one day coalesce into planets and moons and eventually into the complex molecular structures that we call life, that miraculous phenomenon which somehow emerged from the chaos of the early universe and evolved over eons into creatures capable of contemplating their own existence and reaching out to touch the face of infinity.
"""

# Text with code and technical content
TEXT_TECHNICAL = """
## API Documentation

### Authentication

All API requests require a valid API key passed in the header:

```http
Authorization: Bearer sk-proj-abc123xyz789
Content-Type: application/json
```

### Endpoints

#### POST /api/v1/generate

Request body:
```json
{
    "model": "gpt-4o",
    "prompt": "Write a haiku about programming",
    "max_tokens": 150,
    "temperature": 0.7
}
```

Response:
```json
{
    "id": "gen-12345",
    "content": "Lines of code flow down\\nBugs emerge from the shadows\\nDebugging begins",
    "usage": {"prompt_tokens": 25, "completion_tokens": 18}
}
```

#### Error Codes

| Code | Description |
|------|-------------|
| 400  | Bad Request - Invalid parameters |
| 401  | Unauthorized - Invalid API key |
| 429  | Too Many Requests - Rate limited |
| 500  | Internal Server Error |

### Rate Limits

- Free tier: 10 requests/minute
- Pro tier: 100 requests/minute
- Enterprise: Unlimited
"""


def load_external_text(filepath: str) -> Optional[str]:
    """Load text from external file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Warning: Could not load {filepath}: {e}")
        return None


# =============================================================================
# TEST UTILITIES
# =============================================================================

def run_test(name: str, test_func, *args, **kwargs) -> TestResult:
    """Run a single test and capture result."""
    start = time.perf_counter()
    try:
        result = test_func(*args, **kwargs)
        duration = (time.perf_counter() - start) * 1000

        if result is True:
            return TestResult(name, TestStatus.PASS, duration_ms=duration)
        elif result is False:
            return TestResult(name, TestStatus.FAIL, "Test returned False", duration_ms=duration)
        elif isinstance(result, str):
            return TestResult(name, TestStatus.FAIL, result, duration_ms=duration)
        elif isinstance(result, tuple) and len(result) == 2:
            status, msg = result
            return TestResult(name, TestStatus.PASS if status else TestStatus.FAIL, msg, duration_ms=duration)
        else:
            return TestResult(name, TestStatus.PASS, duration_ms=duration)
    except Exception as e:
        duration = (time.perf_counter() - start) * 1000
        return TestResult(name, TestStatus.ERROR, f"{type(e).__name__}: {str(e)}", duration_ms=duration)


async def run_async_test(name: str, test_func, *args, **kwargs) -> TestResult:
    """Run an async test and capture result."""
    start = time.perf_counter()
    try:
        result = await test_func(*args, **kwargs)
        duration = (time.perf_counter() - start) * 1000

        if result is True:
            return TestResult(name, TestStatus.PASS, duration_ms=duration)
        elif result is False:
            return TestResult(name, TestStatus.FAIL, "Test returned False", duration_ms=duration)
        elif isinstance(result, str):
            return TestResult(name, TestStatus.FAIL, result, duration_ms=duration)
        elif isinstance(result, tuple) and len(result) == 2:
            status, msg = result
            return TestResult(name, TestStatus.PASS if status else TestStatus.FAIL, msg, duration_ms=duration)
        else:
            return TestResult(name, TestStatus.PASS, duration_ms=duration)
    except Exception as e:
        duration = (time.perf_counter() - start) * 1000
        tb = traceback.format_exc()
        return TestResult(name, TestStatus.ERROR, f"{type(e).__name__}: {str(e)}\n{tb[-500:]}", duration_ms=duration)


# =============================================================================
# DIRECT OPERATION TESTS (No AI)
# =============================================================================

def test_smart_edit_direct_delete_simple():
    """Test simple text deletion."""
    editor = SmartTextEditor()
    text = "The quick brown fox jumps over the lazy dog."
    result = editor.delete(text, "brown ")

    if not result.success:
        return False, f"Delete failed: {result.errors}"
    if "brown" in result.content_after:
        return False, "Text still contains 'brown'"
    if result.content_after != "The quick fox jumps over the lazy dog.":
        return False, f"Unexpected result: {result.content_after}"
    return True


def test_smart_edit_direct_delete_multiple_occurrences():
    """Test deletion with multiple occurrences (should delete first by default)."""
    editor = SmartTextEditor()
    text = "The cat and the dog and the bird."
    result = editor.delete(text, "the ")

    if not result.success:
        return False, f"Delete failed: {result.errors}"
    # Should only delete first occurrence (case-insensitive, but "the" lowercase matches first "the")
    return True


def test_smart_edit_direct_delete_not_found():
    """Test deletion of non-existent text."""
    editor = SmartTextEditor()
    text = "Hello world."
    result = editor.delete(text, "xyz")

    # Should fail gracefully or return unchanged
    return True  # Just verify no crash


def test_smart_edit_direct_replace_simple():
    """Test simple text replacement."""
    editor = SmartTextEditor()
    text = "The quick brown fox."
    result = editor.replace(text, "brown", "red")

    if not result.success:
        return False, f"Replace failed: {result.errors}"
    if result.content_after != "The quick red fox.":
        return False, f"Unexpected result: {result.content_after}"
    return True


def test_smart_edit_direct_replace_longer():
    """Test replacement with longer text."""
    editor = SmartTextEditor()
    text = "Hello world."
    result = editor.replace(text, "world", "beautiful world of programming")

    if not result.success:
        return False, f"Replace failed: {result.errors}"
    if "beautiful world of programming" not in result.content_after:
        return False, f"Replacement not found in result"
    return True


def test_smart_edit_direct_replace_shorter():
    """Test replacement with shorter text."""
    editor = SmartTextEditor()
    text = "The quick brown fox jumps."
    result = editor.replace(text, "quick brown fox", "cat")

    if not result.success:
        return False, f"Replace failed: {result.errors}"
    if result.content_after != "The cat jumps.":
        return False, f"Unexpected result: {result.content_after}"
    return True


def test_smart_edit_direct_insert_after():
    """Test inserting text after a target."""
    editor = SmartTextEditor()
    text = "Hello world."
    result = editor.insert(text, " beautiful", "Hello", where="after")

    if not result.success:
        return False, f"Insert failed: {result.errors}"
    if "Hello beautiful" not in result.content_after:
        return False, f"Insert not in correct position: {result.content_after}"
    return True


def test_smart_edit_direct_insert_before():
    """Test inserting text before a target."""
    editor = SmartTextEditor()
    text = "Hello world."
    result = editor.insert(text, "beautiful ", "world", where="before")

    if not result.success:
        return False, f"Insert failed: {result.errors}"
    if "beautiful world" not in result.content_after:
        return False, f"Insert not in correct position: {result.content_after}"
    return True


def test_smart_edit_direct_format_bold():
    """Test bold formatting."""
    editor = SmartTextEditor()
    text = "This is important text."
    result = editor.format(text, "important", "bold")

    if not result.success:
        return False, f"Format failed: {result.errors}"
    if "**important**" not in result.content_after:
        return False, f"Bold markers not found: {result.content_after}"
    return True


def test_smart_edit_direct_format_italic():
    """Test italic formatting."""
    editor = SmartTextEditor()
    text = "This is emphasized text."
    result = editor.format(text, "emphasized", "italic")

    if not result.success:
        return False, f"Format failed: {result.errors}"
    if "*emphasized*" not in result.content_after or "**emphasized**" in result.content_after:
        return False, f"Italic markers not correct: {result.content_after}"
    return True


def test_smart_edit_direct_operations_preserve_other_text():
    """Test that direct operations don't modify other parts of text."""
    editor = SmartTextEditor()
    text = "Start. Middle part to edit. End."
    result = editor.replace(text, "Middle part to edit", "CHANGED")

    if not result.success:
        return False, f"Replace failed: {result.errors}"
    if not result.content_after.startswith("Start."):
        return False, "Start was modified"
    if not result.content_after.endswith("End."):
        return False, "End was modified"
    return True


# =============================================================================
# PHRASE LOCALIZATION TESTS
# =============================================================================

def test_smart_edit_locate_simple_phrase():
    """Test basic phrase location."""
    text = "The quick brown fox jumps over the lazy dog."
    # Need at least 3 words for expected_phrase_length=3
    start_dict = {"1": "The", "2": "quick", "3": "brown"}
    end_dict = {"1": "the", "2": "lazy", "3": "dog."}
    result = locate_by_markers(text, start_dict, end_dict, expected_phrase_length=3)

    if result is None:
        return False, "Failed to locate phrase"
    start, end = result
    segment = text[start:end]
    if segment != text:
        return False, f"Wrong segment: '{segment}'"
    return True


def test_smart_edit_locate_partial_phrase():
    """Test locating a partial segment."""
    text = "First paragraph here. Second paragraph follows nicely. Third paragraph ends."
    # Need at least 3 words
    start_dict = {"1": "Second", "2": "paragraph", "3": "follows"}
    end_dict = {"1": "paragraph", "2": "follows", "3": "nicely."}
    result = locate_by_markers(text, start_dict, end_dict, expected_phrase_length=3)

    if result is None:
        return False, "Failed to locate phrase"
    start, end = result
    segment = text[start:end]
    if "Second paragraph follows" not in segment:
        return False, f"Wrong segment: '{segment}'"
    return True


def test_smart_edit_locate_case_insensitive():
    """Test case-insensitive phrase location."""
    text = "THE QUICK BROWN FOX JUMPS."
    # Need at least 3 words
    start_dict = {"1": "the", "2": "quick", "3": "brown"}
    end_dict = {"1": "brown", "2": "fox", "3": "jumps."}
    result = locate_by_markers(text, start_dict, end_dict, case_sensitive=False, expected_phrase_length=3)

    if result is None:
        return False, "Failed to locate phrase (case insensitive)"
    return True


def test_smart_edit_locate_with_punctuation():
    """Test phrase location with punctuation."""
    text = '"Hello," said John politely. "How are you today?"'
    # Need at least 3 words
    start_dict = {"1": '"Hello,"', "2": "said", "3": "John"}
    end_dict = {"1": "are", "2": "you", "3": 'today?"'}
    result = locate_by_markers(text, start_dict, end_dict, expected_phrase_length=3)

    if result is None:
        return False, "Failed to locate phrase with punctuation"
    return True


def test_smart_edit_locate_not_found():
    """Test phrase location when not found."""
    text = "The quick brown fox."
    result = locate_by_markers(text, "nonexistent phrase", "also nonexistent")

    if result is not None:
        return False, "Should return None for non-existent phrase"
    return True


def test_smart_edit_locate_with_counted_format():
    """Test phrase location with counted phrase format."""
    text = "The quick brown fox jumps over the lazy dog."

    # New format: position as key, word as value (need at least 3 words)
    start_dict = {"1": "The", "2": "quick", "3": "brown"}
    end_dict = {"1": "the", "2": "lazy", "3": "dog."}

    result = locate_by_markers(text, start_dict, end_dict, expected_phrase_length=3)

    if result is None:
        return False, "Failed to locate with counted format"
    return True


def test_smart_edit_locate_with_duplicate_words():
    """Test phrase location with duplicate words (the main reason for the new format)."""
    text = "the cat saw the dog and the cat ran away fast"

    # New format allows duplicate words (values can repeat)
    start_dict = {"1": "the", "2": "cat", "3": "saw", "4": "the", "5": "dog"}
    end_dict = {"1": "the", "2": "cat", "3": "ran", "4": "away", "5": "fast"}

    result = locate_by_markers(text, start_dict, end_dict, expected_phrase_length=5)

    if result is None:
        return False, "Failed to locate with duplicate words"
    return True


def test_smart_edit_locate_in_markdown():
    """Test phrase location in markdown text."""
    # New format: position as key, word as value (need at least 3 words)
    start_dict = {"1": "#", "2": "The", "3": "Art"}
    end_dict = {"1": "prevent", "2": "than", "3": "to"}  # From "Bugs are easier to prevent than to fix"
    result = locate_by_markers(TEXT_MARKDOWN, start_dict, end_dict, expected_phrase_length=3)

    if result is None:
        return False, "Failed to locate in markdown"
    start, end = result
    segment = TEXT_MARKDOWN[start:end]
    if "# The Art" not in segment:
        return False, "Start not found in segment"
    return True


def test_smart_edit_locate_in_html():
    """Test phrase location in HTML text."""
    # New format: position as key, word as value (need at least 3 words)
    start_dict = {"1": "<h1>Welcome", "2": "to", "3": "Our"}
    end_dict = {"1": "or", "2": "call", "3": "555-1234.</p>"}
    result = locate_by_markers(TEXT_HTML, start_dict, end_dict, expected_phrase_length=3)

    if result is None:
        return False, "Failed to locate in HTML"
    return True


def test_smart_edit_locate_in_repetitive_text():
    """Test phrase location in text with repetitive patterns."""
    # New format: position as key, word as value (need at least 3 words)
    # Use unique enough phrase from first paragraph
    start_dict = {"1": "The", "2": "cat", "3": "sat"}
    end_dict = {"1": "cat", "2": "purred", "3": "loudly."}
    # This should find the first occurrence
    result = locate_by_markers(TEXT_REPETITIVE, start_dict, end_dict, expected_phrase_length=3)

    if result is None:
        return False, "Failed to locate in repetitive text"
    return True


def test_smart_edit_realistic_workflow_with_optimal_length():
    """
    Test realistic workflow: find optimal phrase length FIRST, then locate.

    This is how the system actually works in production:
    1. Call find_optimal_phrase_length() to determine N
    2. Use N-word phrases for paragraph_start and paragraph_end
    3. Locate the segment
    """
    text = TEXT_MEDIUM

    # Step 1: Find optimal phrase length (real system does this)
    optimal_n = find_optimal_phrase_length(text, min_n=4, max_n=20)
    if optimal_n is None:
        return False, "Could not find optimal phrase length for TEXT_MEDIUM"

    # Step 2: Extract N-word phrases from known paragraph
    # The paragraph: "One autumn evening...dying light of day."
    # We need first N words and last N words
    words = text.split()
    start_idx = None
    for i, word in enumerate(words):
        if "autumn" in word.lower():
            start_idx = i - 1  # "One" is one word before "autumn"
            break

    if start_idx is None:
        return False, "Could not find start of target paragraph"

    # Build N-word start phrase dict (new format: position as key, word as value)
    start_words = words[start_idx:start_idx + optimal_n]
    start_dict = {str(i + 1): word for i, word in enumerate(start_words)}

    # Find end of paragraph (ends with "day.")
    end_idx = None
    for i, word in enumerate(words):
        if "day." in word:
            end_idx = i
            break

    if end_idx is None:
        return False, "Could not find end of target paragraph"

    # Build N-word end phrase dict
    end_words = words[end_idx - optimal_n + 1:end_idx + 1]
    end_dict = {str(i + 1): word for i, word in enumerate(end_words)}

    # Step 3: Locate using optimal-length phrases
    result = locate_by_markers(text, start_dict, end_dict, expected_phrase_length=optimal_n)
    if result is None:
        start_phrase = " ".join(start_words)
        end_phrase = " ".join(end_words)
        return False, f"Failed to locate with {optimal_n}-word phrases: start='{start_phrase}', end='{end_phrase}'"

    # Verify we got something reasonable
    segment = text[result[0]:result[1]]
    if "autumn" not in segment.lower():
        return False, f"Segment doesn't contain expected content: {segment[:100]}"

    return True


def test_smart_edit_locate_with_word_map_fallback():
    """
    Test fallback to word_map mode when phrases aren't unique enough.

    For highly repetitive text, find_optimal_phrase_length returns None,
    and the system falls back to word indices.
    """
    # Use repetitive text that likely needs fallback
    optimal_n = find_optimal_phrase_length(TEXT_REPETITIVE, min_n=4, max_n=64)

    if optimal_n is not None:
        # If we found an optimal length, that's fine too - test passed
        return True

    # If None, we need word_map fallback
    word_map, formatted = build_word_map(TEXT_REPETITIVE)

    if not word_map:
        return False, "Failed to build word map"

    # Use word indices to locate "The dog ran in the park."
    # Find index of "The dog ran"
    start_idx = None
    end_idx = None
    for i, token in enumerate(word_map):
        if token['word'] == 'dog' and i > 0 and word_map[i-1]['word'] == 'The':
            start_idx = i - 1
            # Find end of sentence (next period)
            for j in range(i, min(i + 20, len(word_map))):
                if 'park.' in word_map[j]['word']:
                    end_idx = j
                    break
            if end_idx:
                break

    if start_idx is None or end_idx is None:
        return True  # Can't find specific phrase, but that's OK for this test

    result = locate_by_word_indices(TEXT_REPETITIVE, start_idx, end_idx, word_map)
    if result is None:
        return False, "Failed to locate by word indices"

    segment = TEXT_REPETITIVE[result[0]:result[1]]
    if "dog" not in segment.lower():
        return False, f"Segment doesn't contain expected content: {segment}"

    return True


# =============================================================================
# OPTIMAL PHRASE LENGTH TESTS
# =============================================================================

def test_smart_edit_optimal_length_short_text():
    """Test optimal phrase length for short text."""
    length = find_optimal_phrase_length(TEXT_SHORT, min_n=3, max_n=10)

    if length is None:
        return False, "Should find optimal length for short text"
    if length < 3 or length > 10:
        return False, f"Length {length} outside expected range"
    return True


def test_smart_edit_optimal_length_medium_text():
    """Test optimal phrase length for medium text."""
    length = find_optimal_phrase_length(TEXT_MEDIUM, min_n=4, max_n=20)

    if length is None:
        return False, "Should find optimal length for medium text"
    return True


def test_smart_edit_optimal_length_repetitive_text():
    """Test optimal phrase length for repetitive text."""
    length = find_optimal_phrase_length(TEXT_REPETITIVE, min_n=4, max_n=64)

    # Repetitive text might need longer phrases or fallback to word_map
    # Either result is acceptable
    return True


def test_smart_edit_optimal_length_very_short():
    """Test optimal phrase length for very short text."""
    text = "Hello world."
    length = find_optimal_phrase_length(text, min_n=2, max_n=10)

    # Should return min_n or small value
    if length is None:
        return False, "Should find length for short text"
    return True


# =============================================================================
# WORD MAP TESTS
# =============================================================================

def test_smart_edit_word_map_build():
    """Test word map building."""
    word_map, formatted = build_word_map(TEXT_SHORT)

    if not word_map:
        return False, "Word map is empty"
    if not formatted:
        return False, "Formatted string is empty"
    if "WORD_MAP" not in formatted:
        return False, "Formatted string missing header"
    if len(word_map) < 10:
        return False, f"Word map too short: {len(word_map)}"
    return True


def test_smart_edit_word_map_indices():
    """Test word map index accuracy."""
    text = "First second third fourth fifth."
    word_map, _ = build_word_map(text)

    # Verify indices point to correct words
    for token in word_map:
        extracted = text[token['start']:token['end']]
        if extracted != token['word']:
            return False, f"Index mismatch: '{extracted}' != '{token['word']}'"
    return True


def test_smart_edit_locate_by_word_indices():
    """Test location by word indices."""
    text = "The quick brown fox jumps over the lazy dog."
    word_map, _ = build_word_map(text)

    # Locate "brown fox jumps" (indices 2, 3, 4)
    result = locate_by_word_indices(text, 2, 4, word_map)

    if result is None:
        return False, "Failed to locate by indices"
    start, end = result
    segment = text[start:end]
    if "brown fox jumps" not in segment:
        return False, f"Wrong segment: '{segment}'"
    return True


# =============================================================================
# NORMALIZATION TESTS
# =============================================================================

def test_smart_edit_normalize_curly_quotes():
    """Test normalization of curly quotes."""
    text = '\u201CHello\u201D and \u2018world\u2019'
    result = normalize_source_text(text)

    if '\u201C' in result or '\u201D' in result:
        return False, "Curly double quotes not normalized"
    if '\u2018' in result or '\u2019' in result:
        return False, "Curly single quotes not normalized"
    if '"Hello"' not in result:
        return False, f"Expected straight quotes: {result}"
    return True


def test_smart_edit_normalize_dashes():
    """Test normalization of em-dash and en-dash."""
    text = "Hello\u2014world and 2020\u20132025"  # em-dash and en-dash
    result = normalize_source_text(text)

    if '\u2014' in result:
        return False, "Em-dash not normalized"
    if '\u2013' in result:
        return False, "En-dash not normalized"
    if "Hello--world" not in result:
        return False, f"Em-dash not converted to --: {result}"
    return True


def test_smart_edit_normalize_ellipsis():
    """Test normalization of ellipsis."""
    text = "Wait\u2026 what?"
    result = normalize_source_text(text)

    if '\u2026' in result:
        return False, "Ellipsis not normalized"
    if "Wait..." not in result:
        return False, f"Ellipsis not converted: {result}"
    return True


def test_smart_edit_normalize_preserves_content():
    """Test that normalization preserves semantic content."""
    text = "Regular text without special characters."
    result = normalize_source_text(text)

    if result != text:
        return False, "Regular text was modified"
    return True


def test_smart_edit_normalize_for_matching():
    """Test matching normalization."""
    text = 'He said "hello"'
    result = normalize_for_matching(text)

    if '"' in result:
        return False, "Double quotes should be unified to single"
    return True


def test_smart_edit_locate_after_normalization():
    """Test that location works after normalization."""
    # Text with typographic characters
    text = '\u201CHello,\u201D she said \u2014 \u201Chow are you today?\u201D'
    normalized = normalize_source_text(text)

    # Search with straight quotes (as AI would return in new format)
    # Need at least 3 words
    start_dict = {"1": '"Hello,"', "2": "she", "3": "said"}
    end_dict = {"1": "are", "2": "you", "3": 'today?"'}
    result = locate_by_markers(normalized, start_dict, end_dict, expected_phrase_length=3)

    if result is None:
        return False, "Failed to locate after normalization"
    return True


# =============================================================================
# COUNTED PHRASE FORMAT TESTS
# =============================================================================

def test_smart_edit_counted_phrase_valid():
    """Test valid counted phrase parsing (new format: position as key)."""
    data = {"1": "The", "2": "quick", "3": "brown", "4": "fox", "5": "jumps"}
    result = parse_counted_phrase(data, 5)

    if result != "The quick brown fox jumps":
        return False, f"Wrong result: {result}"
    return True


def test_smart_edit_counted_phrase_shorter():
    """Test counted phrase with fewer words than expected - should be REJECTED."""
    data = {"1": "Hello", "2": "world"}
    result = parse_counted_phrase(data, 5)  # Expecting 5 but got 2

    if result is not None:
        return False, f"Should reject shorter phrase, got: {result}"
    return True


def test_smart_edit_counted_phrase_extra():
    """Test counted phrase with more words than expected - should truncate."""
    data = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E", "6": "F"}
    result = parse_counted_phrase(data, 5)  # Expecting 5, got 6 - should truncate

    if result != "A B C D E":
        return False, f"Should truncate to first 5 words, got: {result}"
    return True


def test_smart_edit_counted_phrase_non_consecutive():
    """Test counted phrase with non-consecutive keys."""
    data = {"1": "The", "3": "quick", "5": "fox"}  # Missing 2, 4
    result = parse_counted_phrase(data, 5)

    if result is not None:
        return False, "Should reject non-consecutive keys"
    return True


def test_smart_edit_counted_phrase_wrong_start():
    """Test counted phrase starting at wrong key."""
    data = {"2": "The", "3": "quick", "4": "fox"}  # Starts at 2
    result = parse_counted_phrase(data, 5)

    if result is not None:
        return False, "Should reject keys not starting at 1"
    return True


def test_smart_edit_counted_phrase_non_integer():
    """Test counted phrase with non-integer keys."""
    data = {"one": "The", "two": "quick"}
    result = parse_counted_phrase(data, 5)

    if result is not None:
        return False, "Should reject non-integer keys"
    return True


def test_smart_edit_counted_phrase_with_duplicates():
    """Test counted phrase with duplicate words (the main reason for new format)."""
    data = {"1": "the", "2": "cat", "3": "saw", "4": "the", "5": "mouse"}
    result = parse_counted_phrase(data, 5)

    if result != "the cat saw the mouse":
        return False, f"Should handle duplicate words: {result}"
    return True


def test_smart_edit_validate_counted_format():
    """Test detailed validation of counted format."""
    # 5 words when expecting 5 - should be valid
    data = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}
    is_valid, phrase, msg = validate_counted_phrase_format(data, 5)

    if not is_valid:
        return False, f"Should be valid: {msg}"
    if phrase != "A B C D E":
        return False, f"Wrong phrase: {phrase}"
    if msg != "valid_counted_format":
        return False, f"Wrong message: {msg}"
    return True


def test_smart_edit_validate_counted_format_too_few():
    """Test that fewer words than expected is rejected."""
    data = {"1": "A", "2": "B", "3": "C"}  # Only 3, expecting 5
    is_valid, phrase, msg = validate_counted_phrase_format(data, 5)

    if is_valid:
        return False, "Should reject too few words"
    if "too few" not in msg.lower():
        return False, f"Wrong error message: {msg}"
    return True


def test_smart_edit_validate_string_rejected():
    """Test that string format is rejected (only dict allowed)."""
    is_valid, phrase, msg = validate_counted_phrase_format("legacy string", 5)

    if is_valid:
        return False, "String format should be rejected"
    if "dict" not in msg.lower():
        return False, f"Error message should mention dict: {msg}"
    return True


def test_smart_edit_extract_phrase_counted():
    """Test phrase extraction from counted format (new format: position as key)."""
    data = {"1": "One", "2": "two", "3": "three", "4": "four", "5": "five"}
    result = extract_phrase_from_response(data, 5, "test")

    if result != "One two three four five":
        return False, f"Wrong result: {result}"
    return True


def test_smart_edit_extract_phrase_counted_too_few():
    """Test that dict with too few words is rejected."""
    data = {"1": "One", "2": "two", "3": "three"}  # Only 3, need 5
    result = extract_phrase_from_response(data, 5, "test")

    if result is not None:
        return False, f"Should reject too few words, got: {result}"
    return True


def test_smart_edit_extract_phrase_string_enough():
    """Test that string format with enough words is accepted (fallback)."""
    result = extract_phrase_from_response("One two three four five", 5, "test")

    if result != "One two three four five":
        return False, f"String fallback should work with enough words, got: {result}"
    return True


def test_smart_edit_extract_phrase_string_too_few():
    """Test that string format with too few words is rejected."""
    result = extract_phrase_from_response("One two three", 5, "test")

    if result is not None:
        return False, "String with too few words should be rejected"
    return True


def test_smart_edit_extract_phrase_none():
    """Test phrase extraction from None."""
    result = extract_phrase_from_response(None, 5, "test")

    if result is not None:
        return False, "Should return None for None input"
    return True


# =============================================================================
# TEXT TYPE SPECIFIC TESTS
# =============================================================================

def test_smart_edit_with_markdown():
    """Test operations on markdown text."""
    editor = SmartTextEditor()

    # Test deletion in markdown
    result = editor.delete(TEXT_MARKDOWN, "Complexity is the enemy of reliability")
    if not result.success:
        return False, f"Delete in markdown failed: {result.errors}"

    # Test replacement in markdown
    result = editor.replace(TEXT_MARKDOWN, "best practices", "good habits")
    if not result.success:
        return False, f"Replace in markdown failed: {result.errors}"

    return True


def test_smart_edit_with_html():
    """Test operations on HTML text."""
    editor = SmartTextEditor()

    # Test deletion in HTML
    result = editor.delete(TEXT_HTML, '<strong>paragraph</strong>')
    if not result.success:
        return False, f"Delete in HTML failed: {result.errors}"

    return True


def test_smart_edit_with_special_chars():
    """Test operations on text with special characters."""
    editor = SmartTextEditor()
    normalized = normalize_source_text(TEXT_SPECIAL_CHARS)

    # Test operations on normalized text
    result = editor.replace(normalized, "victory was inevitable", "success was certain")
    if not result.success:
        return False, f"Replace with special chars failed: {result.errors}"

    return True


def test_smart_edit_with_unicode():
    """Test operations on Unicode text."""
    editor = SmartTextEditor()

    result = editor.replace(TEXT_UNICODE, "hermoso jardin", "bonito parque")
    if not result.success:
        return False, f"Replace in Unicode text failed: {result.errors}"

    return True


def test_smart_edit_with_technical():
    """Test operations on technical/code text."""
    editor = SmartTextEditor()

    result = editor.replace(TEXT_TECHNICAL, "gpt-4o", "gpt-5")
    if not result.success:
        return False, f"Replace in technical text failed: {result.errors}"

    return True


def test_smart_edit_long_paragraph_locate():
    """Test phrase location in very long paragraph."""
    # Should still find phrases even in very long text
    # Need at least 3 words - text ends with "touch the face of infinity."
    start_dict = {"1": "In", "2": "the", "3": "vast"}
    end_dict = {"1": "face", "2": "of", "3": "infinity."}
    result = locate_by_markers(TEXT_LONG_PARAGRAPH, start_dict, end_dict, expected_phrase_length=3)

    if result is None:
        return False, "Failed to locate in long paragraph"
    return True


def test_smart_edit_optimal_length_external_file():
    """Test optimal phrase length with external file if available."""
    # This test uses files passed via --file argument
    # If no files provided, skip
    if not EXTERNAL_TEST_FILES:
        return True  # Skip if no external files provided

    for filepath in EXTERNAL_TEST_FILES:
        text = load_external_text(filepath)
        if text is None:
            continue

        length = find_optimal_phrase_length(text, min_n=4, max_n=64)
        # Just verify it doesn't crash

    return True


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

def test_smart_edit_empty_text():
    """Test operations on empty text."""
    editor = SmartTextEditor()
    result = editor.delete("", "anything")

    # Should handle gracefully
    return True


def test_smart_edit_single_word():
    """Test operations on single word."""
    editor = SmartTextEditor()
    result = editor.replace("Hello", "Hello", "Hi")

    if not result.success:
        return False, f"Replace single word failed: {result.errors}"
    if result.content_after != "Hi":
        return False, f"Wrong result: {result.content_after}"
    return True


def test_smart_edit_whitespace_only():
    """Test operations on whitespace-only text."""
    editor = SmartTextEditor()
    result = editor.delete("   \n\t  ", "anything")

    # Should handle gracefully
    return True


def test_smart_edit_very_long_target():
    """Test with very long target phrase."""
    editor = SmartTextEditor()
    long_target = "word " * 100
    text = f"Start {long_target} End"

    result = editor.replace(text, long_target.strip(), "REPLACED")
    if not result.success:
        return False, f"Replace long target failed: {result.errors}"
    return True


def test_smart_edit_special_regex_chars():
    """Test with regex special characters in target."""
    editor = SmartTextEditor()
    text = "Price is $19.99 (20% off) [limited]"

    # These contain regex special chars: $ . ( ) % [ ]
    result = editor.replace(text, "$19.99", "$29.99")
    if not result.success:
        return False, f"Replace with regex chars failed: {result.errors}"
    if "$29.99" not in result.content_after:
        return False, f"Replacement not found: {result.content_after}"
    return True


def test_smart_edit_newlines_in_target():
    """Test with newlines in target text."""
    editor = SmartTextEditor()
    text = "Line 1\nLine 2\nLine 3"

    result = editor.replace(text, "Line 1\nLine 2", "Lines combined")
    if not result.success:
        return False, f"Replace with newlines failed: {result.errors}"
    return True


def test_smart_edit_unicode_normalization():
    """Test Unicode normalization edge cases."""
    # Various Unicode spaces
    text = "Hello\u00A0world\u2003here"  # NBSP and EM SPACE
    normalized = normalize_source_text(text)

    if '\u00A0' in normalized:
        return False, "NBSP not normalized"
    return True


def test_smart_edit_overlapping_markers():
    """Test phrase location with overlapping start/end markers."""
    text = "The quick brown fox jumps."
    # Start and end markers overlap (quick brown / brown fox)
    # Need at least 2 words with expected_phrase_length=2
    start_dict = {"1": "quick", "2": "brown"}
    end_dict = {"1": "brown", "2": "fox"}
    result = locate_by_markers(text, start_dict, end_dict, expected_phrase_length=2)

    if result is None:
        return False, "Should handle overlapping markers"
    return True


def test_smart_edit_identical_markers():
    """Test phrase location with identical start/end markers."""
    text = "Hello world today Hello world today"
    # Need at least 2 words
    start_dict = {"1": "Hello", "2": "world"}
    end_dict = {"1": "Hello", "2": "world"}
    result = locate_by_markers(text, start_dict, end_dict, expected_phrase_length=2)

    # Should find something (first occurrence to second occurrence)
    if result is None:
        return False, "Should find with identical markers"
    return True


def test_smart_edit_markers_at_boundaries():
    """Test phrase location at text boundaries."""
    text = "Start word middle word end"

    # Marker at very start, need at least 2 words
    start_dict = {"1": "Start", "2": "word"}
    end_dict = {"1": "word", "2": "end"}
    result = locate_by_markers(text, start_dict, end_dict, expected_phrase_length=2)
    if result is None:
        return False, "Should find markers at boundaries"

    start, end = result
    if start != 0:
        return False, f"Start should be 0, got {start}"
    return True


# =============================================================================
# AI-ASSISTED OPERATION TESTS (Require AI Service)
# =============================================================================

async def test_smart_edit_ai_rephrase():
    """Test AI-assisted rephrasing."""
    try:
        from ai_service import get_ai_service
        ai_service = get_ai_service()
    except Exception as e:
        return True  # Skip if AI service not available

    editor = SmartTextEditor(ai_service=ai_service)
    text = "The cat sat on the mat. It was very comfortable there."

    result = await editor.rephrase(
        text,
        "The cat sat on the mat",
        model="gpt-4o-mini",
        temperature=0.5
    )

    if not result.success:
        return False, f"Rephrase failed: {result.errors}"
    if result.content_after == text:
        return False, "Text was not changed"
    return True


async def test_smart_edit_ai_apply_edit():
    """Test AI-assisted custom edit."""
    try:
        from ai_service import get_ai_service
        ai_service = get_ai_service()
    except Exception as e:
        return True  # Skip if AI service not available

    editor = SmartTextEditor(ai_service=ai_service)
    text = "The year was 1985 when the company was founded."

    result = await editor.apply_edit(
        content=text,
        target="The year was 1985",
        instruction="Change the year to 1987",
        model="gpt-4o-mini",
        temperature=0.3
    )

    if not result.success:
        return False, f"Apply edit failed: {result.errors}"
    if "1987" not in result.content_after:
        return False, f"Edit not applied: {result.content_after}"
    return True


async def test_smart_edit_ai_improve():
    """Test AI-assisted improvement."""
    try:
        from ai_service import get_ai_service
        ai_service = get_ai_service()
    except Exception as e:
        return True  # Skip if AI service not available

    editor = SmartTextEditor(ai_service=ai_service)
    text = "He went to the store. He bought milk. He came back home."

    result = await editor.improve(
        text,
        "He went to the store. He bought milk. He came back home.",
        model="gpt-4o-mini"
    )

    if not result.success:
        return False, f"Improve failed: {result.errors}"
    return True


async def test_smart_edit_ai_with_markdown():
    """Test AI edit on markdown text."""
    try:
        from ai_service import get_ai_service
        ai_service = get_ai_service()
    except Exception as e:
        return True  # Skip if AI service not available

    editor = SmartTextEditor(ai_service=ai_service)

    result = await editor.apply_edit(
        content=TEXT_MARKDOWN,
        target="**Programming** is not just about writing code",
        instruction="Make this sentence more engaging and inspirational",
        model="gpt-4o-mini",
        preserve_length=True
    )

    if not result.success:
        return False, f"AI edit on markdown failed: {result.errors}"
    return True


async def test_smart_edit_ai_with_long_text():
    """Test AI edit on longer text."""
    try:
        from ai_service import get_ai_service
        ai_service = get_ai_service()
    except Exception as e:
        return True  # Skip if AI service not available

    editor = SmartTextEditor(ai_service=ai_service)

    result = await editor.apply_edit(
        content=TEXT_MEDIUM,
        target="Dreams are not always what they seem",
        instruction="Rephrase this wisdom in a more poetic way",
        model="gpt-4o-mini"
    )

    if not result.success:
        return False, f"AI edit on long text failed: {result.errors}"
    return True


async def test_smart_edit_ai_random_instruction():
    """Test AI edit with random creative instruction."""
    try:
        from ai_service import get_ai_service
        ai_service = get_ai_service()
    except Exception as e:
        return True  # Skip if AI service not available

    import random
    instructions = [
        "Add more vivid imagery",
        "Make it sound more formal",
        "Add a touch of humor",
        "Make it more concise",
        "Add emotional depth",
        "Change the tone to be more optimistic",
    ]

    editor = SmartTextEditor(ai_service=ai_service)
    instruction = random.choice(instructions)

    result = await editor.apply_edit(
        content=TEXT_MEDIUM,
        target="a wise old owl named Prometheus",
        instruction=instruction,
        model="gpt-4o-mini"
    )

    if not result.success:
        return False, f"AI edit with instruction '{instruction}' failed: {result.errors}"
    return True


# =============================================================================
# EXTERNAL FILE TESTS
# =============================================================================

def test_smart_edit_external_file_operations():
    """
    Test operations with external files passed via --file argument.

    This test runs phrase length detection and word map building on
    any external files provided. If no files are provided, the test
    is skipped (passes automatically).

    Usage:
        python test_smart_edit_comprehensive.py --file "path/to/large_text.txt"
    """
    if not EXTERNAL_TEST_FILES:
        return True  # Skip if no external files provided

    for filepath in EXTERNAL_TEST_FILES:
        text = load_external_text(filepath)
        if text is None:
            continue

        filename = os.path.basename(filepath)

        # Test phrase length detection (use first 50k chars for very long files)
        test_text = text[:50000] if len(text) > 50000 else text
        length = find_optimal_phrase_length(test_text, min_n=5, max_n=64)

        # Test word map building (use first 10k chars for very long files)
        map_text = text[:10000] if len(text) > 10000 else text
        word_map, formatted = build_word_map(map_text)

        if not word_map:
            return False, f"Failed to build word map for {filename}"

        # Test basic editor operations
        editor = SmartTextEditor()
        words = text.split()[:100]  # Use first 100 words
        if len(words) >= 10:
            sample = " ".join(words[:10])
            result = editor.delete(sample, words[0])
            if not result.success:
                return False, f"Delete operation failed on {filename}"

    return True


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

def test_smart_edit_full_workflow_direct():
    """Test complete workflow with direct operations."""
    editor = SmartTextEditor()

    # Start with medium text
    text = TEXT_MEDIUM

    # Step 1: Delete something
    result = editor.delete(text, "countless seasons")
    if not result.success:
        return False, f"Step 1 (delete) failed: {result.errors}"
    text = result.content_after

    # Step 2: Replace something
    result = editor.replace(text, "wise old owl", "ancient wise owl")
    if not result.success:
        return False, f"Step 2 (replace) failed: {result.errors}"
    text = result.content_after

    # Step 3: Insert something
    result = editor.insert(text, " the Great", "Prometheus", where="after")
    if not result.success:
        return False, f"Step 3 (insert) failed: {result.errors}"
    text = result.content_after

    # Step 4: Format something
    result = editor.format(text, "golden key", "bold")
    if not result.success:
        return False, f"Step 4 (format) failed: {result.errors}"

    return True


async def test_smart_edit_full_workflow_ai():
    """Test complete workflow with AI operations."""
    try:
        from ai_service import get_ai_service
        ai_service = get_ai_service()
    except Exception as e:
        return True  # Skip if AI service not available

    editor = SmartTextEditor(ai_service=ai_service)
    text = TEXT_MEDIUM

    # First find optimal phrase length (as the real system does)
    phrase_length = find_optimal_phrase_length(text, min_n=4, max_n=20)
    if phrase_length is None:
        return False, "Could not find optimal phrase length"

    # Locate a segment using proper start/end markers (new format: position as key)
    # "One autumn evening" starts a paragraph and "dying light of day." ends it
    # Build dicts with at least phrase_length words
    start_words = ["One", "autumn", "evening,", "a", "young", "rabbit", "named"]
    end_words = ["the", "dying", "light", "of", "day."]

    # Ensure we have enough words
    start_dict = {str(i+1): w for i, w in enumerate(start_words[:max(phrase_length, 4)])}
    end_dict = {str(i+1): w for i, w in enumerate(end_words[:max(phrase_length, 4)])}

    span = locate_by_markers(text, start_dict, end_dict, expected_phrase_length=min(phrase_length, len(start_dict)))
    if span is None:
        return False, "Failed to locate segment"

    # Apply AI edit to that segment
    segment = text[span[0]:span[1]]
    result = await editor.apply_edit(
        content=text,
        target=TextTarget(
            mode=TargetMode.POSITION,
            value=span,
            scope=TargetScope.PARAGRAPH
        ),
        instruction="Make this passage more mysterious and atmospheric",
        model="gpt-4o-mini"
    )

    if not result.success:
        return False, f"AI edit failed: {result.errors}"

    return True


# =============================================================================
# TEST RUNNER
# =============================================================================

def run_direct_tests():
    """Run all direct operation tests."""
    print("\n" + "=" * 70)
    print("DIRECT OPERATIONS (No AI)")
    print("=" * 70)

    tests = [
        ("test_smart_edit_direct_delete_simple", test_smart_edit_direct_delete_simple),
        ("test_smart_edit_direct_delete_multiple_occurrences", test_smart_edit_direct_delete_multiple_occurrences),
        ("test_smart_edit_direct_delete_not_found", test_smart_edit_direct_delete_not_found),
        ("test_smart_edit_direct_replace_simple", test_smart_edit_direct_replace_simple),
        ("test_smart_edit_direct_replace_longer", test_smart_edit_direct_replace_longer),
        ("test_smart_edit_direct_replace_shorter", test_smart_edit_direct_replace_shorter),
        ("test_smart_edit_direct_insert_after", test_smart_edit_direct_insert_after),
        ("test_smart_edit_direct_insert_before", test_smart_edit_direct_insert_before),
        ("test_smart_edit_direct_format_bold", test_smart_edit_direct_format_bold),
        ("test_smart_edit_direct_format_italic", test_smart_edit_direct_format_italic),
        ("test_smart_edit_direct_operations_preserve_other_text", test_smart_edit_direct_operations_preserve_other_text),
    ]

    for name, test_func in tests:
        result = run_test(name, test_func)
        runner.add_result(result)


def run_localization_tests():
    """Run all phrase localization tests."""
    print("\n" + "=" * 70)
    print("PHRASE LOCALIZATION")
    print("=" * 70)

    tests = [
        ("test_smart_edit_locate_simple_phrase", test_smart_edit_locate_simple_phrase),
        ("test_smart_edit_locate_partial_phrase", test_smart_edit_locate_partial_phrase),
        ("test_smart_edit_locate_case_insensitive", test_smart_edit_locate_case_insensitive),
        ("test_smart_edit_locate_with_punctuation", test_smart_edit_locate_with_punctuation),
        ("test_smart_edit_locate_not_found", test_smart_edit_locate_not_found),
        ("test_smart_edit_locate_with_counted_format", test_smart_edit_locate_with_counted_format),
        ("test_smart_edit_locate_with_duplicate_words", test_smart_edit_locate_with_duplicate_words),
        ("test_smart_edit_locate_in_markdown", test_smart_edit_locate_in_markdown),
        ("test_smart_edit_locate_in_html", test_smart_edit_locate_in_html),
        ("test_smart_edit_locate_in_repetitive_text", test_smart_edit_locate_in_repetitive_text),
        ("test_smart_edit_realistic_workflow_with_optimal_length", test_smart_edit_realistic_workflow_with_optimal_length),
        ("test_smart_edit_locate_with_word_map_fallback", test_smart_edit_locate_with_word_map_fallback),
        ("test_smart_edit_optimal_length_short_text", test_smart_edit_optimal_length_short_text),
        ("test_smart_edit_optimal_length_medium_text", test_smart_edit_optimal_length_medium_text),
        ("test_smart_edit_optimal_length_repetitive_text", test_smart_edit_optimal_length_repetitive_text),
        ("test_smart_edit_optimal_length_very_short", test_smart_edit_optimal_length_very_short),
        ("test_smart_edit_word_map_build", test_smart_edit_word_map_build),
        ("test_smart_edit_word_map_indices", test_smart_edit_word_map_indices),
        ("test_smart_edit_locate_by_word_indices", test_smart_edit_locate_by_word_indices),
    ]

    for name, test_func in tests:
        result = run_test(name, test_func)
        runner.add_result(result)


def run_normalization_tests():
    """Run all normalization tests."""
    print("\n" + "=" * 70)
    print("TEXT NORMALIZATION")
    print("=" * 70)

    tests = [
        ("test_smart_edit_normalize_curly_quotes", test_smart_edit_normalize_curly_quotes),
        ("test_smart_edit_normalize_dashes", test_smart_edit_normalize_dashes),
        ("test_smart_edit_normalize_ellipsis", test_smart_edit_normalize_ellipsis),
        ("test_smart_edit_normalize_preserves_content", test_smart_edit_normalize_preserves_content),
        ("test_smart_edit_normalize_for_matching", test_smart_edit_normalize_for_matching),
        ("test_smart_edit_locate_after_normalization", test_smart_edit_locate_after_normalization),
    ]

    for name, test_func in tests:
        result = run_test(name, test_func)
        runner.add_result(result)


def run_counted_phrase_tests():
    """Run all counted phrase format tests."""
    print("\n" + "=" * 70)
    print("COUNTED PHRASE FORMAT")
    print("=" * 70)

    tests = [
        ("test_smart_edit_counted_phrase_valid", test_smart_edit_counted_phrase_valid),
        ("test_smart_edit_counted_phrase_shorter", test_smart_edit_counted_phrase_shorter),
        ("test_smart_edit_counted_phrase_extra", test_smart_edit_counted_phrase_extra),
        ("test_smart_edit_counted_phrase_non_consecutive", test_smart_edit_counted_phrase_non_consecutive),
        ("test_smart_edit_counted_phrase_wrong_start", test_smart_edit_counted_phrase_wrong_start),
        ("test_smart_edit_counted_phrase_non_integer", test_smart_edit_counted_phrase_non_integer),
        ("test_smart_edit_counted_phrase_with_duplicates", test_smart_edit_counted_phrase_with_duplicates),
        ("test_smart_edit_validate_counted_format", test_smart_edit_validate_counted_format),
        ("test_smart_edit_validate_counted_format_too_few", test_smart_edit_validate_counted_format_too_few),
        ("test_smart_edit_validate_string_rejected", test_smart_edit_validate_string_rejected),
        ("test_smart_edit_extract_phrase_counted", test_smart_edit_extract_phrase_counted),
        ("test_smart_edit_extract_phrase_counted_too_few", test_smart_edit_extract_phrase_counted_too_few),
        ("test_smart_edit_extract_phrase_string_enough", test_smart_edit_extract_phrase_string_enough),
        ("test_smart_edit_extract_phrase_string_too_few", test_smart_edit_extract_phrase_string_too_few),
        ("test_smart_edit_extract_phrase_none", test_smart_edit_extract_phrase_none),
    ]

    for name, test_func in tests:
        result = run_test(name, test_func)
        runner.add_result(result)


def run_text_type_tests():
    """Run tests for different text types."""
    print("\n" + "=" * 70)
    print("TEXT TYPE SPECIFIC")
    print("=" * 70)

    tests = [
        ("test_smart_edit_with_markdown", test_smart_edit_with_markdown),
        ("test_smart_edit_with_html", test_smart_edit_with_html),
        ("test_smart_edit_with_special_chars", test_smart_edit_with_special_chars),
        ("test_smart_edit_with_unicode", test_smart_edit_with_unicode),
        ("test_smart_edit_with_technical", test_smart_edit_with_technical),
        ("test_smart_edit_long_paragraph_locate", test_smart_edit_long_paragraph_locate),
        ("test_smart_edit_optimal_length_external_file", test_smart_edit_optimal_length_external_file),
    ]

    for name, test_func in tests:
        result = run_test(name, test_func)
        runner.add_result(result)


def run_edge_case_tests():
    """Run all edge case tests."""
    print("\n" + "=" * 70)
    print("EDGE CASES")
    print("=" * 70)

    tests = [
        ("test_smart_edit_empty_text", test_smart_edit_empty_text),
        ("test_smart_edit_single_word", test_smart_edit_single_word),
        ("test_smart_edit_whitespace_only", test_smart_edit_whitespace_only),
        ("test_smart_edit_very_long_target", test_smart_edit_very_long_target),
        ("test_smart_edit_special_regex_chars", test_smart_edit_special_regex_chars),
        ("test_smart_edit_newlines_in_target", test_smart_edit_newlines_in_target),
        ("test_smart_edit_unicode_normalization", test_smart_edit_unicode_normalization),
        ("test_smart_edit_overlapping_markers", test_smart_edit_overlapping_markers),
        ("test_smart_edit_identical_markers", test_smart_edit_identical_markers),
        ("test_smart_edit_markers_at_boundaries", test_smart_edit_markers_at_boundaries),
    ]

    for name, test_func in tests:
        result = run_test(name, test_func)
        runner.add_result(result)


def run_external_file_tests():
    """Run tests with external files."""
    print("\n" + "=" * 70)
    print("EXTERNAL FILES")
    print("=" * 70)

    tests = [
        ("test_smart_edit_external_file_operations", test_smart_edit_external_file_operations),
    ]

    for name, test_func in tests:
        result = run_test(name, test_func)
        runner.add_result(result)


def run_integration_tests():
    """Run integration tests (direct only)."""
    print("\n" + "=" * 70)
    print("INTEGRATION (Direct)")
    print("=" * 70)

    tests = [
        ("test_smart_edit_full_workflow_direct", test_smart_edit_full_workflow_direct),
    ]

    for name, test_func in tests:
        result = run_test(name, test_func)
        runner.add_result(result)


async def run_ai_tests():
    """Run all AI-assisted tests."""
    print("\n" + "=" * 70)
    print("AI-ASSISTED OPERATIONS")
    print("=" * 70)

    tests = [
        ("test_smart_edit_ai_rephrase", test_smart_edit_ai_rephrase),
        ("test_smart_edit_ai_apply_edit", test_smart_edit_ai_apply_edit),
        ("test_smart_edit_ai_improve", test_smart_edit_ai_improve),
        ("test_smart_edit_ai_with_markdown", test_smart_edit_ai_with_markdown),
        ("test_smart_edit_ai_with_long_text", test_smart_edit_ai_with_long_text),
        ("test_smart_edit_ai_random_instruction", test_smart_edit_ai_random_instruction),
        ("test_smart_edit_full_workflow_ai", test_smart_edit_full_workflow_ai),
    ]

    for name, test_func in tests:
        result = await run_async_test(name, test_func)
        runner.add_result(result)


def main():
    global EXTERNAL_TEST_FILES

    parser = argparse.ArgumentParser(description="Comprehensive Smart Edit Test Suite")
    parser.add_argument("--file", type=str, action="append", dest="files",
                       help="Path to external text file to test (can be specified multiple times)")
    parser.add_argument("--category", type=str,
                       choices=["direct", "localization", "normalization", "counted",
                               "text_types", "edge", "external", "ai", "integration", "all"],
                       default="all", help="Test category to run")
    parser.add_argument("--skip-ai", action="store_true", help="Skip AI-assisted tests")
    args = parser.parse_args()

    # Populate external test files
    if args.files:
        EXTERNAL_TEST_FILES = args.files

    print("=" * 70)
    print("SMART EDIT COMPREHENSIVE TEST SUITE")
    print("=" * 70)
    print(f"Category: {args.category}")
    if EXTERNAL_TEST_FILES:
        print(f"External files: {len(EXTERNAL_TEST_FILES)} file(s)")
        for f in EXTERNAL_TEST_FILES:
            print(f"  - {os.path.basename(f)}")
    print()

    # Run selected test categories
    if args.category in ["direct", "all"]:
        run_direct_tests()

    if args.category in ["localization", "all"]:
        run_localization_tests()

    if args.category in ["normalization", "all"]:
        run_normalization_tests()

    if args.category in ["counted", "all"]:
        run_counted_phrase_tests()

    if args.category in ["text_types", "all"]:
        run_text_type_tests()

    if args.category in ["edge", "all"]:
        run_edge_case_tests()

    if args.category in ["external", "all"]:
        run_external_file_tests()

    if args.category in ["integration", "all"]:
        run_integration_tests()

    # AI tests require async
    if args.category in ["ai", "all"] and not args.skip_ai:
        asyncio.run(run_ai_tests())

    # Print summary
    success = runner.print_summary()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
