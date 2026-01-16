"""
Smart Edit Router - FastAPI endpoints for the Smart Edit system.

Endpoints:
- GET  /smart-edit/demo           - Interactive demo page
- POST /smart-edit/analyze        - Analyze text and suggest actions
- POST /smart-edit/execute        - Execute a single action
- POST /smart-edit/execute-batch  - Execute multiple actions
- GET  /smart-edit/samples        - Get sample texts for demo
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from .models import EditResult, OperationType, TargetMode
from .operations import SmartTextEditor

router = APIRouter(tags=["Smart Edit"])

# Templates for demo page
templates = Jinja2Templates(directory="templates")


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================


class TextTargetRequest(BaseModel):
    """Request model for text targeting."""

    mode: str = Field(
        default="exact",
        description="Target mode: exact, regex, position, paragraph, sentence",
    )
    value: Any = Field(
        description="Target value (string, position tuple, or index)"
    )
    occurrence: int = Field(
        default=1,
        description="Which occurrence to target (1=first, -1=all)",
    )
    case_sensitive: bool = Field(default=True)


class ActionRequest(BaseModel):
    """Request model for a single edit action."""

    id: str = Field(default="", description="Action ID (auto-generated if empty)")
    type: str = Field(description="Operation type: delete, insert, replace, etc.")
    target: TextTargetRequest = Field(description="Target specification")
    content: Optional[str] = Field(
        default=None,
        description="Content for insert/replace operations",
    )
    instruction: Optional[str] = Field(
        default=None,
        description="Instruction for AI operations",
    )
    description: str = Field(default="", description="Human-readable description")
    category: str = Field(default="", description="Category for grouping")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AnalyzeRequest(BaseModel):
    """Request for text analysis."""

    text: str = Field(description="Text to analyze", max_length=50000)
    analysis_model: str = Field(
        default="gpt-4o-mini",
        description="Model to use for analysis",
    )
    max_actions: int = Field(default=20, description="Maximum actions to return")
    action_types: Optional[List[str]] = Field(
        default=None,
        description="Filter by action types",
    )


class AnalyzeResponse(BaseModel):
    """Response from text analysis."""

    actions: List[Dict[str, Any]] = Field(description="Suggested edit actions")
    stats: Dict[str, Any] = Field(description="Analysis statistics")
    analysis_model: str = Field(description="Model used for analysis")
    analysis_time_ms: int = Field(description="Time taken for analysis")


class ExecuteRequest(BaseModel):
    """Request to execute a single action."""

    text: str = Field(description="Current text content", max_length=50000)
    action: ActionRequest = Field(description="Action to execute")


class ExecuteResponse(BaseModel):
    """Response from action execution."""

    success: bool
    text_before: str
    text_after: str
    change: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time_ms: int


class BatchExecuteRequest(BaseModel):
    """Request to execute multiple actions."""

    text: str = Field(description="Original text content", max_length=50000)
    actions: List[ActionRequest] = Field(description="Actions to execute")
    stop_on_error: bool = Field(
        default=False,
        description="Stop execution on first error",
    )


class BatchExecuteResponse(BaseModel):
    """Response from batch execution."""

    success: bool
    text_final: str
    results: List[Dict[str, Any]]
    total_time_ms: int
    stats: Dict[str, Any]


class SampleText(BaseModel):
    """A sample text for the demo."""

    id: str
    title: str
    description: str
    text: str
    word_count: int
    suggested_actions: List[Dict[str, Any]]


# =============================================================================
# DEMO PAGE
# =============================================================================


@router.get("/demo", response_class=HTMLResponse)
async def smart_edit_demo_page(request: Request):
    """
    Serve the interactive Smart Edit demo page.

    This page demonstrates:
    - Text analysis and action generation
    - Visual animations for each operation type
    - Real-time preview of changes
    """
    return templates.TemplateResponse(
        "smart_edit_demo.html",
        {"request": request},
    )


# =============================================================================
# SAMPLE TEXTS
# =============================================================================


@router.get("/samples", response_model=List[SampleText])
async def get_sample_texts():
    """
    Get sample texts for the demo.

    Returns pre-configured texts with suggested edit actions.
    """
    # Sample texts with pre-analyzed actions
    samples = [
        {
            "id": "article-errors",
            "title": "Article with errors",
            "description": "Common mistakes that need fixing",
            "text": (
                "Maria was born in Barcelona in the year 1985. She grew up in a very very "
                "loving family with three siblings. Her childhood was full of joy and "
                "learning and learning continuously.\n\n"
                "After graduating from university, Maria moved to Madrid to pursue her "
                "career in journalism. She worked hard and eventually became one of the "
                "most respected journalists in the country."
            ),
            "word_count": 68,
            "suggested_actions": [
                {
                    "id": "act-001",
                    "type": "delete",
                    "target": {"mode": "exact", "value": "very ", "occurrence": 1},
                    "description": "Remove duplicate 'very'",
                    "category": "redundancy",
                },
                {
                    "id": "act-002",
                    "type": "delete",
                    "target": {"mode": "exact", "value": "and learning"},
                    "description": "Remove duplicate phrase",
                    "category": "redundancy",
                },
                {
                    "id": "act-003",
                    "type": "replace",
                    "target": {"mode": "exact", "value": "in the year"},
                    "content": "in",
                    "description": "Simplify year reference",
                    "category": "conciseness",
                },
                {
                    "id": "act-004",
                    "type": "format",
                    "target": {"mode": "exact", "value": "Barcelona"},
                    "metadata": {"format_type": "bold"},
                    "description": "Emphasize city name",
                    "category": "formatting",
                },
            ],
        },
        {
            "id": "email-draft",
            "title": "Professional email",
            "description": "Email draft that needs polishing",
            "text": (
                "Dear Mr. Johnson,\n\n"
                "I am writing to you to inform you about the project status. The project "
                "is going well and we are making good progress. We have completed the "
                "first phase and are now working on the second phase.\n\n"
                "Please let me know if you have any questions or concerns.\n\n"
                "Best regards,\nJohn"
            ),
            "word_count": 58,
            "suggested_actions": [
                {
                    "id": "act-001",
                    "type": "replace",
                    "target": {"mode": "exact", "value": "I am writing to you to inform you about"},
                    "content": "I wanted to update you on",
                    "description": "Make opening more concise",
                    "category": "conciseness",
                },
                {
                    "id": "act-002",
                    "type": "replace",
                    "target": {"mode": "exact", "value": "is going well and we are making good progress"},
                    "content": "is progressing smoothly",
                    "description": "Remove redundancy",
                    "category": "conciseness",
                },
            ],
        },
        {
            "id": "repetitive-text",
            "title": "Repetitive text",
            "description": "Text with duplications and redundancies",
            "text": (
                "The new software system is very efficient and effective. The system "
                "processes data quickly and efficiently. Users find the system easy to "
                "use and user-friendly.\n\n"
                "The implementation was successful and achieved success. The team worked "
                "hard and put in a lot of hard work. Results exceeded expectations and "
                "went beyond what was expected."
            ),
            "word_count": 65,
            "suggested_actions": [
                {
                    "id": "act-001",
                    "type": "replace",
                    "target": {"mode": "exact", "value": "efficient and effective"},
                    "content": "highly efficient",
                    "description": "Remove near-synonym redundancy",
                    "category": "redundancy",
                },
                {
                    "id": "act-002",
                    "type": "replace",
                    "target": {"mode": "exact", "value": "easy to use and user-friendly"},
                    "content": "intuitive",
                    "description": "Consolidate similar concepts",
                    "category": "redundancy",
                },
                {
                    "id": "act-003",
                    "type": "replace",
                    "target": {"mode": "exact", "value": "successful and achieved success"},
                    "content": "successful",
                    "description": "Remove tautology",
                    "category": "redundancy",
                },
                {
                    "id": "act-004",
                    "type": "replace",
                    "target": {"mode": "exact", "value": "worked hard and put in a lot of hard work"},
                    "content": "worked diligently",
                    "description": "Remove repetition",
                    "category": "redundancy",
                },
                {
                    "id": "act-005",
                    "type": "replace",
                    "target": {"mode": "exact", "value": "exceeded expectations and went beyond what was expected"},
                    "content": "exceeded all expectations",
                    "description": "Remove explanation redundancy",
                    "category": "redundancy",
                },
            ],
        },
    ]

    return [SampleText(**s) for s in samples]


# =============================================================================
# ANALYSIS ENDPOINT
# =============================================================================


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_text(request: AnalyzeRequest):
    """
    Analyze text and return suggested edit actions.

    Uses AI to identify issues like:
    - Redundancies and repetitions
    - Grammar errors
    - Style improvements
    - Formatting suggestions

    Phase 6: Full AI analysis implementation.
    Falls back to basic pattern matching if AI service unavailable.
    """
    import time

    start_time = time.perf_counter()

    actions = []
    analysis_model_used = request.analysis_model

    try:
        # Try AI-powered analysis
        from ai_service import get_ai_service
        from .analyzer import TextAnalyzer, calculate_stats

        ai_service = get_ai_service()
        analyzer = TextAnalyzer(ai_service)

        # Parse categories filter if provided
        categories = None
        if request.action_types:
            # Map action types to categories
            type_to_category = {
                "DELETE": "redundancy",
                "REPLACE": "style",
                "FORMAT": "formatting",
                "REPHRASE": "style",
            }
            categories = [
                type_to_category.get(t.upper(), t.lower())
                for t in request.action_types
            ]

        actions = await analyzer.analyze(
            text=request.text,
            model=request.analysis_model,
            max_issues=request.max_actions,
            categories=categories,
        )

        stats = calculate_stats(actions)

    except Exception as e:
        # Fallback to basic pattern matching
        from .analyzer import TextAnalyzer, calculate_stats

        # Use dummy analyzer with basic method
        analyzer = TextAnalyzer.__new__(TextAnalyzer)
        actions = analyzer.analyze_basic(request.text, max_issues=request.max_actions)
        stats = calculate_stats(actions)
        analysis_model_used = "pattern-matching (fallback)"

        # Log the error for debugging
        import logging
        logging.getLogger(__name__).warning(
            f"AI analysis failed, using fallback: {e}"
        )

    analysis_time_ms = int((time.perf_counter() - start_time) * 1000)

    return AnalyzeResponse(
        actions=actions,
        stats=stats,
        analysis_model=analysis_model_used,
        analysis_time_ms=analysis_time_ms,
    )


# =============================================================================
# EXECUTE ENDPOINTS
# =============================================================================


@router.post("/execute", response_model=ExecuteResponse)
async def execute_single_action(request: ExecuteRequest):
    """
    Execute a single edit action on the provided text.

    Supports both direct operations (instant) and AI-assisted operations.
    """
    import time

    start_time = time.perf_counter()
    action = request.action

    try:
        # Map request to operation type
        op_type_map = {
            "delete": "delete",
            "insert": "insert_after",
            "insert_before": "insert_before",
            "insert_after": "insert_after",
            "replace": "replace",
            "format": "format",
            "rephrase": "rephrase",
            "improve": "improve",
            "fix_grammar": "fix_grammar",
            "fix_style": "fix_style",
            "expand": "expand",
            "condense": "condense",
        }

        op_type = op_type_map.get(action.type.lower())
        if not op_type:
            raise ValueError(f"Unknown action type: {action.type}")

        # Build target
        target = action.target.value
        if action.target.mode == "position" and isinstance(target, list):
            target = tuple(target)

        # Determine if this is an AI operation
        ai_operations = {"rephrase", "improve", "fix_grammar", "fix_style", "expand", "condense"}
        is_ai_op = op_type in ai_operations

        # Create editor (with ai_service for AI operations)
        if is_ai_op:
            from ai_service import get_ai_service
            editor = SmartTextEditor(ai_service=get_ai_service())
        else:
            editor = SmartTextEditor()

        # Execute based on operation type
        if op_type == "delete":
            result = editor.delete(request.text, target)
        elif op_type in ("insert_before", "insert_after"):
            where = "before" if op_type == "insert_before" else "after"
            result = editor.insert(request.text, action.content or "", target, where)
        elif op_type == "replace":
            result = editor.replace(request.text, target, action.content or "")
        elif op_type == "format":
            format_type = action.metadata.get("format_type", "bold")
            result = editor.format(request.text, target, format_type)
        elif op_type == "rephrase":
            result = await editor.rephrase(
                request.text, target, instruction=action.instruction
            )
        elif op_type == "improve":
            criteria = action.metadata.get("criteria", ["clarity", "conciseness"])
            result = await editor.improve(request.text, target, criteria=criteria)
        elif op_type == "fix_grammar":
            result = await editor.fix(request.text, target, fix_type="grammar")
        elif op_type == "fix_style":
            result = await editor.fix(request.text, target, fix_type="style")
        elif op_type == "expand":
            result = await editor.expand(request.text, target)
        elif op_type == "condense":
            result = await editor.condense(request.text, target)
        else:
            # Fallback for any unmapped operation
            return ExecuteResponse(
                success=False,
                text_before=request.text,
                text_after=request.text,
                error=f"Unknown operation type: {op_type}",
                execution_time_ms=int((time.perf_counter() - start_time) * 1000),
            )

        # Build response
        change = None
        if result.changes:
            c = result.changes[0]
            change = {
                "position": {"start": c.position_start, "end": c.position_end},
                "removed": c.removed_text,
                "inserted": c.inserted_text,
            }

        return ExecuteResponse(
            success=result.success,
            text_before=result.content_before,
            text_after=result.content_after,
            change=change,
            error=result.errors[0] if result.errors else None,
            execution_time_ms=int((time.perf_counter() - start_time) * 1000),
        )

    except Exception as e:
        return ExecuteResponse(
            success=False,
            text_before=request.text,
            text_after=request.text,
            error=str(e),
            execution_time_ms=int((time.perf_counter() - start_time) * 1000),
        )


@router.post("/execute-batch", response_model=BatchExecuteResponse)
async def execute_batch_actions(request: BatchExecuteRequest):
    """
    Execute multiple actions in sequence.

    Actions are applied in order, with each action operating on
    the result of the previous one.
    """
    import time

    start_time = time.perf_counter()

    current_text = request.text
    results = []
    errors_count = 0

    for action in request.actions:
        action_start = time.perf_counter()

        # Execute single action
        exec_request = ExecuteRequest(text=current_text, action=action)
        exec_response = await execute_single_action(exec_request)

        action_result = {
            "action_id": action.id or f"action-{len(results)+1}",
            "success": exec_response.success,
            "time_ms": exec_response.execution_time_ms,
        }

        if exec_response.success:
            current_text = exec_response.text_after
            action_result["change"] = exec_response.change
        else:
            errors_count += 1
            action_result["error"] = exec_response.error
            if request.stop_on_error:
                results.append(action_result)
                break

        results.append(action_result)

    total_time_ms = int((time.perf_counter() - start_time) * 1000)

    return BatchExecuteResponse(
        success=errors_count == 0,
        text_final=current_text,
        results=results,
        total_time_ms=total_time_ms,
        stats={
            "total_actions": len(request.actions),
            "successful": len(results) - errors_count,
            "failed": errors_count,
            "char_delta": len(current_text) - len(request.text),
        },
    )
