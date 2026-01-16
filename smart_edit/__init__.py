"""
Smart Edit Module - Standalone text editing with direct and AI-assisted operations.

This module provides a powerful, self-contained text editing system designed to be
used as a plugin in any application. It includes:

1. **Direct Operations** (no AI needed): DELETE, INSERT, REPLACE, FORMAT
2. **AI-Assisted Operations**: REPHRASE, IMPROVE, FIX, EXPAND, CONDENSE, apply_edit
3. **QA Integration**: Prompt builders and response parsers for QA evaluation systems

Core Editing:
    from smart_edit import SmartTextEditor, OperationType

    # Direct operations (instant, no AI)
    editor = SmartTextEditor()
    result = editor.delete(content, "text to delete")
    result = editor.replace(content, "old", "new")

    # AI-assisted operations (require ai_service)
    editor = SmartTextEditor(ai_service=ai_service)
    result = await editor.apply_edit(content, target, instruction)

QA Integration (for evaluation systems):
    from smart_edit import (
        build_qa_edit_prompt,
        parse_qa_edit_groups,
        TextEditRange,
        SeverityLevel,
    )

    # Build prompt format section for QA evaluation
    json_format = build_qa_edit_prompt(
        marker_mode="phrase",
        phrase_length=5,
        feedback_format_example="Passed",
    )

    # Parse AI response into TextEditRange objects
    edit_ranges = parse_qa_edit_groups(
        edit_groups=response["edit_groups"],
        marker_mode="phrase",
        marker_length=5,
    )

Version: 2.0.0 (Self-contained module with QA integration)
"""

from .models import (
    # Core operation types
    EditOperation,
    EditResult,
    TextTarget,
    TargetMode,
    TargetScope,
    OperationType,
    ChangeDetail,
    FormatType,
    TargetSpec,
    # Backward compatibility alias
    EditType,
    # QA integration models
    SeverityLevel,
    MarkerMode,
    EditBaseModel,
    MarkerConfig,
    TextEditRange,
    EditContext,
    EditDecision,
    QAEvaluationWithRanges,
)
from .operations import SmartTextEditor
from .locators import (
    find_optimal_phrase_length,
    build_word_map,
    locate_by_markers,
    locate_by_word_indices,
    validate_marker_uniqueness,
    analyze_text_for_markers,
    # Text normalization
    normalize_source_text,
    normalize_for_matching,
    # Counted phrase format support
    parse_counted_phrase,
    validate_counted_phrase_format,
    extract_phrase_from_response,
)
from .analyzer import (
    TextAnalyzer,
    calculate_stats,
)
from .qa_integration import (
    build_qa_edit_prompt,
    parse_qa_edit_groups,
    get_operation_type,
    OPERATION_TYPE_MAP,
    QA_EDIT_FORMAT_PHRASE_MODE,
    QA_EDIT_FORMAT_WORD_INDEX_MODE,
    QA_SIMPLE_FORMAT,
)

__all__ = [
    # Core classes
    "SmartTextEditor",
    "EditOperation",
    "EditResult",
    "TextTarget",
    "TargetMode",
    "TargetScope",
    "OperationType",
    "ChangeDetail",
    "FormatType",
    "TargetSpec",
    # Backward compatibility
    "EditType",
    # QA integration models
    "SeverityLevel",
    "MarkerMode",
    "EditBaseModel",
    "MarkerConfig",
    "TextEditRange",
    "EditContext",
    "EditDecision",
    "QAEvaluationWithRanges",
    # Localization utilities
    "find_optimal_phrase_length",
    "build_word_map",
    "locate_by_markers",
    "locate_by_word_indices",
    "validate_marker_uniqueness",
    "analyze_text_for_markers",
    # Text normalization
    "normalize_source_text",
    "normalize_for_matching",
    # Counted phrase format support
    "parse_counted_phrase",
    "validate_counted_phrase_format",
    "extract_phrase_from_response",
    # Text analysis
    "TextAnalyzer",
    "calculate_stats",
    # QA integration functions
    "build_qa_edit_prompt",
    "parse_qa_edit_groups",
    "get_operation_type",
    "OPERATION_TYPE_MAP",
    # QA prompt templates (for advanced customization)
    "QA_EDIT_FORMAT_PHRASE_MODE",
    "QA_EDIT_FORMAT_WORD_INDEX_MODE",
    "QA_SIMPLE_FORMAT",
]

__version__ = "2.0.0"  # Self-contained module with QA integration
