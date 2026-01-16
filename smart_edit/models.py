"""
Smart Edit Models - Data structures for text editing operations.

This module defines the core data models used throughout the smart_edit system:

Core Types:
- OperationType: Types of edit operations (direct and AI-assisted)
- TargetMode: How to locate text (exact, regex, position, marker)
- TargetScope: Granularity of edit (word, sentence, paragraph, document)
- TextTarget: Complete specification of what text to target
- EditOperation: A single edit operation to perform
- EditResult: The result of executing an operation

QA Integration Types (for AI-assisted quality evaluation):
- SeverityLevel: Problem severity (critical, major, minor)
- MarkerMode: How paragraphs are identified (phrase markers vs word indices)
- MarkerConfig: Configuration for paragraph markers
- TextEditRange: A text range identified by QA for editing
- EditContext: Context for applying edits
- EditDecision: Strategic decision about edit approach
- QAEvaluationWithRanges: Extended QA evaluation with edit ranges

This module is designed to be self-contained for use as a standalone package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, AliasChoices


class OperationType(str, Enum):
    """Types of edit operations supported by SmartTextEditor."""

    # Direct operations (no AI required, instant execution)
    DELETE = "delete"
    INSERT_BEFORE = "insert_before"
    INSERT_AFTER = "insert_after"
    REPLACE = "replace"
    MOVE = "move"
    DUPLICATE = "duplicate"
    FORMAT = "format"

    # AI-assisted operations (require LLM calls)
    REPHRASE = "rephrase"
    IMPROVE = "improve"
    FIX_GRAMMAR = "fix_grammar"
    FIX_STYLE = "fix_style"
    EXPAND = "expand"
    CONDENSE = "condense"

    @property
    def requires_ai(self) -> bool:
        """Check if this operation type requires AI assistance."""
        return self in {
            OperationType.REPHRASE,
            OperationType.IMPROVE,
            OperationType.FIX_GRAMMAR,
            OperationType.FIX_STYLE,
            OperationType.EXPAND,
            OperationType.CONDENSE,
        }

    @property
    def is_direct(self) -> bool:
        """Check if this operation can be executed directly without AI."""
        return not self.requires_ai


class TargetMode(str, Enum):
    """How to locate text for editing."""

    EXACT = "exact"  # Exact string match
    REGEX = "regex"  # Regular expression pattern
    POSITION = "position"  # Character positions (start, end)
    MARKER = "marker"  # N-word phrase markers (paragraph_start, paragraph_end)
    WORD_INDEX = "word_index"  # Word map indices (start_word_index, end_word_index)
    PARAGRAPH = "paragraph"  # Paragraph by index (0-based)
    SENTENCE = "sentence"  # Sentence by index within paragraph


class TargetScope(str, Enum):
    """Granularity of the edit target."""

    WORD = "word"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    DOCUMENT = "document"


class FormatType(str, Enum):
    """Types of markdown formatting that can be applied."""

    BOLD = "bold"  # **text**
    ITALIC = "italic"  # *text*
    CODE = "code"  # `text`
    STRIKETHROUGH = "strikethrough"  # ~~text~~
    UNDERLINE = "underline"  # <u>text</u> (HTML)


@dataclass
class TextTarget:
    """
    Specification for locating text to edit.

    Examples:
        # Exact string match
        TextTarget(mode=TargetMode.EXACT, value="hello world")

        # Position-based (characters 10-20)
        TextTarget(mode=TargetMode.POSITION, value=(10, 20))

        # Regex pattern
        TextTarget(mode=TargetMode.REGEX, value=r"\\d{4}-\\d{2}-\\d{2}")

        # Paragraph index
        TextTarget(mode=TargetMode.PARAGRAPH, value=2)  # Third paragraph

        # Phrase markers (for QA integration)
        TextTarget(
            mode=TargetMode.MARKER,
            value={"start": "The quick brown", "end": "lazy dog."}
        )

        # Word indices (fallback for repetitive text)
        TextTarget(
            mode=TargetMode.WORD_INDEX,
            value={"start_idx": 0, "end_idx": 8}  # word_map can be passed separately
        )
    """

    mode: TargetMode
    value: Union[str, int, Tuple[int, int], Dict[str, Any]]
    scope: TargetScope = TargetScope.DOCUMENT
    occurrence: int = 1  # Which occurrence to target (1 = first, -1 = all)
    case_sensitive: bool = True
    word_map: Optional[List[Dict[str, Any]]] = None  # For WORD_INDEX mode

    def __post_init__(self):
        """Validate target configuration."""
        if self.mode == TargetMode.POSITION:
            if not isinstance(self.value, (tuple, list)) or len(self.value) != 2:
                raise ValueError(
                    "POSITION mode requires value to be (start, end) tuple"
                )
        elif self.mode in (TargetMode.PARAGRAPH, TargetMode.SENTENCE):
            if not isinstance(self.value, int):
                raise ValueError(f"{self.mode.value} mode requires integer index")
        elif self.mode == TargetMode.MARKER:
            if not isinstance(self.value, dict):
                raise ValueError(
                    "MARKER mode requires value to be dict with 'start' and 'end' keys"
                )
            if "start" not in self.value or "end" not in self.value:
                raise ValueError(
                    "MARKER mode value must have 'start' and 'end' keys "
                    "(paragraph_start and paragraph_end phrases)"
                )
        elif self.mode == TargetMode.WORD_INDEX:
            if not isinstance(self.value, dict):
                raise ValueError(
                    "WORD_INDEX mode requires value to be dict with 'start_idx' and 'end_idx' keys"
                )
            if "start_idx" not in self.value or "end_idx" not in self.value:
                raise ValueError(
                    "WORD_INDEX mode value must have 'start_idx' and 'end_idx' keys"
                )


@dataclass
class ChangeDetail:
    """Details about a single change made to the text."""

    position_start: int
    position_end: int
    removed_text: str
    inserted_text: str

    @property
    def char_delta(self) -> int:
        """Net change in character count."""
        return len(self.inserted_text) - len(self.removed_text)


@dataclass
class EditOperation:
    """
    A single edit operation to be executed.

    Attributes:
        operation_type: The type of operation (DELETE, REPLACE, etc.)
        target: How to locate the text to edit
        content: New content for INSERT/REPLACE operations
        instruction: Instructions for AI-assisted operations
        description: Human-readable description of the operation
        category: Category for grouping (e.g., "redundancy", "grammar")
        priority: Execution priority (lower = higher priority)
        metadata: Additional operation-specific data
    """

    operation_type: OperationType
    target: Union[str, TextTarget, Tuple[int, int]]
    content: Optional[str] = None
    instruction: Optional[str] = None
    description: str = ""
    category: str = ""
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = ""

    def __post_init__(self):
        """Generate ID if not provided and normalize target."""
        if not self.id:
            import uuid

            self.id = f"op-{uuid.uuid4().hex[:8]}"

        # Convert simple string/tuple targets to TextTarget
        if isinstance(self.target, str):
            self.target = TextTarget(mode=TargetMode.EXACT, value=self.target)
        elif isinstance(self.target, tuple) and len(self.target) == 2:
            self.target = TextTarget(mode=TargetMode.POSITION, value=self.target)

    @property
    def requires_ai(self) -> bool:
        """Check if this operation requires AI assistance."""
        return self.operation_type.requires_ai

    @property
    def estimated_ms(self) -> int:
        """Estimate execution time in milliseconds."""
        if self.requires_ai:
            return 2500  # AI operations ~2.5s average
        return 10  # Direct operations ~10ms


@dataclass
class EditResult:
    """
    Result of executing one or more edit operations.

    Attributes:
        success: Whether the operation(s) succeeded
        content_before: Original content before edits
        content_after: Content after edits applied
        changes: List of individual changes made
        errors: List of error messages if any
        diff: Unified diff string showing changes
        execution_time_ms: Time taken to execute in milliseconds
        operation_id: ID of the operation (for single operations)
    """

    success: bool
    content_before: str
    content_after: str
    changes: List[ChangeDetail] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    diff: str = ""
    execution_time_ms: int = 0
    operation_id: str = ""

    @property
    def char_delta(self) -> int:
        """Total net change in character count."""
        return len(self.content_after) - len(self.content_before)

    @property
    def word_delta(self) -> int:
        """Approximate net change in word count."""
        before_words = len(self.content_before.split())
        after_words = len(self.content_after.split())
        return after_words - before_words

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "content_before": self.content_before,
            "content_after": self.content_after,
            "changes": [
                {
                    "position": {
                        "start": c.position_start,
                        "end": c.position_end,
                    },
                    "removed": c.removed_text,
                    "inserted": c.inserted_text,
                }
                for c in self.changes
            ],
            "errors": self.errors,
            "diff": self.diff,
            "execution_time_ms": self.execution_time_ms,
            "operation_id": self.operation_id,
            "stats": {
                "char_delta": self.char_delta,
                "word_delta": self.word_delta,
            },
        }


# Type alias for flexible target specification
TargetSpec = Union[str, Tuple[int, int], TextTarget]

# Alias for backward compatibility (EditType was the old name)
EditType = OperationType


# =============================================================================
# QA INTEGRATION MODELS
# =============================================================================
# These models support integration with QA evaluation systems that identify
# specific text ranges for editing.


class SeverityLevel(str, Enum):
    """Problem severity levels for QA-identified issues."""
    CRITICAL = "critical"  # Serious content errors requiring immediate fix
    MAJOR = "major"        # Important problems affecting quality
    MINOR = "minor"        # Minor improvements or polish


class MarkerMode(str, Enum):
    """Marker identification mode for smart edit paragraph location."""
    PHRASE = "phrase"          # Use n-word phrase markers (default)
    WORD_INDEX = "word_index"  # Use word map indices (fallback for repetitive content)


class EditBaseModel(BaseModel):
    """Base model enabling population by field name while preserving Pydantic's namespace safeguards."""
    model_config = {"populate_by_name": True}


class MarkerConfig(EditBaseModel):
    """
    Configuration for paragraph markers in smart edit.

    This is computed via pre-scan before QA evaluation and determines
    how paragraphs will be identified for editing.
    """
    mode: MarkerMode = Field(
        default=MarkerMode.PHRASE,
        description="How paragraphs are identified: 'phrase' uses text markers, 'word_index' uses indices"
    )
    phrase_length: Optional[int] = Field(
        default=None,
        ge=4,
        le=80,
        description="Number of words for phrase markers (4-80), None if word_index mode"
    )
    word_map: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Word map tokens for word_index mode (from build_word_map)"
    )
    word_map_formatted: Optional[str] = Field(
        default=None,
        description="Formatted word map string for QA prompt inclusion"
    )


class TextEditRange(EditBaseModel):
    """
    Identifies a specific text range to edit, as identified by QA evaluation.

    Supports two identification modes:
    - phrase mode: Uses paragraph_start/paragraph_end text markers
    - word_index mode: Uses start_word_index/end_word_index from word map
    """

    # Marker mode selection
    marker_mode: str = Field(
        default="phrase",
        description="Identification mode: 'phrase' (text markers) or 'word_index' (word map indices)"
    )

    # Paragraph identification - phrase mode (optional for word_index mode)
    paragraph_start: str = Field(
        default="",
        description="First N words of the paragraph (phrase mode)"
    )
    paragraph_end: str = Field(
        default="",
        description="Last N words of the paragraph (phrase mode)"
    )

    # Paragraph identification - word_index mode
    start_word_index: Optional[int] = Field(
        default=None,
        ge=0,
        description="Start word index from word map (word_index mode)"
    )
    end_word_index: Optional[int] = Field(
        default=None,
        ge=0,
        description="End word index from word map, inclusive (word_index mode)"
    )

    # Specific fragment to edit (optional, used as fallback)
    exact_fragment: str = Field(
        default="",
        description="Exact text fragment to modify (5-20 words)"
    )

    # Edit details
    edit_type: OperationType = Field(
        default=OperationType.REPLACE,
        description="Type of edit operation"
    )
    new_content: Optional[str] = Field(
        default=None,
        description="New content for replace/add operations"
    )
    edit_instruction: str = Field(
        default="",
        description="Specific instruction for what to change"
    )

    # Metadata
    issue_severity: SeverityLevel = Field(
        default=SeverityLevel.MINOR,
        description="Severity level of the issue"
    )
    issue_description: str = Field(
        default="",
        description="Clear description of the detected problem"
    )

    # Validation
    is_unique: bool = Field(
        default=True,
        description="Whether the fragment is unique in the text"
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence in problem identification (0-1)"
    )

    # Direct operation flag
    can_use_direct: bool = Field(
        default=False,
        description="True if this edit can be executed without AI (delete/replace with exact text)"
    )

    def can_execute_directly(self) -> bool:
        """
        Determine if this edit can be executed without AI.

        Direct operations are possible when:
        - DELETE: exact_fragment is provided (text to delete)
        - REPLACE: both exact_fragment AND new_content are provided
        """
        if self.edit_type == OperationType.DELETE:
            return bool(self.exact_fragment and self.exact_fragment.strip())

        if self.edit_type == OperationType.REPLACE:
            return bool(
                self.exact_fragment and self.exact_fragment.strip() and
                self.new_content is not None
            )

        # AI operations (rephrase, add_*, etc.) always need AI
        return False


class EditContext(EditBaseModel):
    """Adaptive context for applying edits efficiently."""

    full_text: Optional[str] = None
    context_window: Optional[str] = None
    window_size: int = 0
    total_length: int = 0
    strategy: str = "full_text"  # "full_text" or "windowed"

    # Additional metadata
    style_sample: Optional[str] = Field(
        None,
        description="Sample of writing style (first 200 words)"
    )
    tone_indicators: Optional[List[str]] = Field(
        None,
        description="Detected tone indicators"
    )


class EditDecision(EditBaseModel):
    """Strategic decision about correction approach."""

    strategy: str = Field(
        ...,
        description="'incremental_edit' or 'full_regeneration'"
    )
    reason: str = Field(
        ...,
        description="Justification for the chosen strategy"
    )

    # For incremental editing
    edit_ranges: Optional[List[TextEditRange]] = None
    total_issues: int = 0
    editable_issues: int = 0
    estimated_tokens_saved: int = 0

    # Applied thresholds
    applied_thresholds: Optional[Dict[str, Any]] = None


class QAEvaluationWithRanges(EditBaseModel):
    """Extended QA evaluation with identified text ranges."""

    score: float
    feedback: str
    has_deal_breakers: bool = False
    deal_breaker_reason: Optional[str] = None

    # Identified issue ranges
    identified_issues: Optional[List[TextEditRange]] = None

    # Metadata
    used_model: str = Field(
        ...,
        description="Model that produced this QA evaluation",
        validation_alias=AliasChoices("model_used", "used_model"),
        serialization_alias="model_used",
    )
    layer_name: str
    evaluation_time: float
