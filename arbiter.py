"""
Arbiter: Intelligent Conflict Resolver for Smart-Edit Operations
================================================================

The Arbiter is a per-layer arbitration system that:
- Detects conflicts between edits proposed by different QA evaluators
- Resolves conflicts intelligently using AI with original request context
- Maintains edit history per layer for informing subsequent rounds
- Prevents contradictory edits from degrading content quality

This module contains data models, enums, and prompt templates.
The actual Arbiter class implementation will be added in Phase 2.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Callable, Awaitable, TYPE_CHECKING
from enum import Enum
from pydantic import BaseModel, Field
from model_aliasing import PromptPart

if TYPE_CHECKING:
    from smart_edit import TextEditRange


# =============================================================================
# ENUMS
# =============================================================================


class ConflictType(str, Enum):
    """Types of conflicts between proposed edits."""
    OPPOSITE_OPERATIONS = "opposite_operations"      # DELETE vs REPLACE
    OPPOSITE_DIRECTIONS = "opposite_directions"      # EXPAND vs CONDENSE
    SEVERITY_MISMATCH = "severity_mismatch"          # critical vs minor
    SEMANTIC_REDUNDANCY = "semantic_redundancy"      # Same intent, different words
    CYCLE_DETECTED = "cycle_detected"                # Previously discarded edit
    STALE_FRAGMENT = "stale_fragment"                # exact_fragment no longer in content
    ALREADY_APPLIED = "already_applied"              # suggested_text already matches content


class ArbiterDecision(str, Enum):
    """Arbiter's decision for an edit."""
    APPLY = "apply"
    DISCARD = "discard"


class EditDistribution(str, Enum):
    """
    Classification of how edits are distributed among QA evaluators.

    This determines which model Arbiter uses for resolution:
    - CONSENSUS, MAJORITY, SINGLE_QA → Economic model (arbiter_model)
    - MINORITY, CONFLICT, TIE → Powerful model (gran_sabio_model)

    Logic by QA model count:
    - 1 QA model: SINGLE_QA (no comparison possible)
    - 2 QA models: CONSENSUS (both agree) or TIE (any disagreement)
    - 3+ QA models: CONSENSUS, MAJORITY, MINORITY, or CONFLICT
    """
    CONSENSUS = "consensus"       # ALL QA models propose same/compatible edits
    MAJORITY = "majority"         # >50% propose edit (only with 3+ models)
    MINORITY = "minority"         # <50% propose edit (only with 3+ models)
    CONFLICT = "conflict"         # Multiple incompatible edits for same target
    TIE = "tie"                   # 50-50 split or disagreement (2 models)
    SINGLE_QA = "single_qa"       # Only 1 QA model (no comparison possible)


# =============================================================================
# DATA MODELS
# =============================================================================


@dataclass
class ProposedEdit:
    """An edit proposed by a QA evaluator."""
    edit: "TextEditRange"           # The actual edit
    source_evaluator: str           # Blind evaluator alias (e.g., "Evaluator A")
    source_score: float             # Score given by that evaluator
    paragraph_key: str              # Unique key for the paragraph


@dataclass
class ConflictInfo:
    """Information about a detected conflict."""
    conflict_type: ConflictType
    paragraph_key: str
    involved_edits: List[ProposedEdit]
    description: str


@dataclass
class ArbiterEditDecision:
    """Arbiter's decision for a single edit."""
    edit: "TextEditRange"
    decision: ArbiterDecision
    reason: str
    source_evaluator: str
    conflict_resolved: Optional[ConflictInfo] = None


@dataclass
class EditRoundRecord:
    """Record of a single smart-edit round within a layer."""
    round_number: int
    proposed_edits: List[ProposedEdit]
    conflicts_detected: List[ConflictInfo]
    decisions: List[ArbiterEditDecision]

    @property
    def edits_applied(self) -> List[ArbiterEditDecision]:
        """Get all edits that were approved for application."""
        return [d for d in self.decisions if d.decision == ArbiterDecision.APPLY]

    @property
    def edits_discarded(self) -> List[ArbiterEditDecision]:
        """Get all edits that were discarded."""
        return [d for d in self.decisions if d.decision == ArbiterDecision.DISCARD]


def _enum_or_text_value(value: Any, default: str = "unknown") -> str:
    """Return enum .value when present, otherwise a stable string value."""
    if value is None:
        return default
    raw_value = value.value if hasattr(value, "value") else value
    text = str(raw_value).strip()
    return text or default


def _edit_type_value(edit: Any, default: str = "unknown") -> str:
    return _enum_or_text_value(getattr(edit, "edit_type", None), default)


def _get_paragraph_key_for_history(edit: "TextEditRange") -> str:
    """
    Generate a unique key for an edit based on its location markers.

    This is used by LayerEditHistory to track which paragraphs have been edited.
    """
    # Use start_marker if available, otherwise use word indices
    if hasattr(edit, 'start_marker') and edit.start_marker:
        return f"phrase:{edit.start_marker[:50]}"
    elif hasattr(edit, 'start_word_index') and edit.start_word_index is not None:
        return f"word_idx:{edit.start_word_index}"
    else:
        # Fallback to issue description hash
        desc = getattr(edit, 'issue_description', '') or ''
        return f"desc:{hash(desc[:100])}"


@dataclass
class LayerEditHistory:
    """
    Complete edit history for a single QA layer.

    This tracks all edits (applied and discarded) across multiple rounds
    within a single layer. The history is:
    - Used to inform QA evaluators in subsequent rounds
    - Used by Arbiter to detect edit cycles
    - Cleared when moving to a new layer
    """
    layer_name: str
    rounds: List[EditRoundRecord] = field(default_factory=list)

    def add_round(self, record: EditRoundRecord) -> None:
        """Add a new round record to the history."""
        self.rounds.append(record)

    def format_for_prompt(
        self,
        max_rounds: Optional[int] = None,
        max_chars: Optional[int] = None
    ) -> str:
        """
        Format history as concise summary for injection into QA/Arbiter prompts.

        Args:
            max_rounds: Maximum number of recent rounds to include.
                        If None, uses config.EDIT_HISTORY_MAX_ROUNDS.
            max_chars: Maximum total characters for the formatted output.
                       If None, uses config.EDIT_HISTORY_MAX_CHARS.

        Returns:
            Formatted string like:
            [PREVIOUS_EDITS_IN_LAYER]
            Round 1:
            - Applied: DELETE duplicate "and" (p3) - GPT-4o
            - Discarded: REPLACE "and"->"but" (p3) - Claude
              Reason: Conflicts with DELETE operation
            [/PREVIOUS_EDITS_IN_LAYER]
        """
        # Import config here to avoid circular imports
        from config import config
        if max_rounds is None:
            max_rounds = config.EDIT_HISTORY_MAX_ROUNDS
        if max_chars is None:
            max_chars = config.EDIT_HISTORY_MAX_CHARS
        if not self.rounds:
            return ""

        lines = ["[PREVIOUS_EDITS_IN_LAYER]"]

        for record in self.rounds[-max_rounds:]:
            lines.append(f"Round {record.round_number}:")

            for decision in record.edits_applied:
                op = _edit_type_value(decision.edit, "EDIT")
                desc = ""
                if hasattr(decision.edit, 'issue_description') and decision.edit.issue_description:
                    desc = decision.edit.issue_description[:50]
                elif hasattr(decision.edit, 'edit_instruction') and decision.edit.edit_instruction:
                    desc = decision.edit.edit_instruction[:50]
                lines.append(f"- Applied: {op.upper()} {desc} - {decision.source_evaluator}")

            for decision in record.edits_discarded:
                op = _edit_type_value(decision.edit, "EDIT")
                desc = ""
                if hasattr(decision.edit, 'issue_description') and decision.edit.issue_description:
                    desc = decision.edit.issue_description[:50]
                elif hasattr(decision.edit, 'edit_instruction') and decision.edit.edit_instruction:
                    desc = decision.edit.edit_instruction[:50]
                lines.append(f"- Discarded: {op.upper()} {desc} - {decision.source_evaluator}")
                lines.append(f"  Reason: {decision.reason[:80]}")

        lines.append("[/PREVIOUS_EDITS_IN_LAYER]")

        result = "\n".join(lines)
        if len(result) > max_chars and max_rounds > 1:
            # Truncate older rounds if too long
            return self.format_for_prompt(max_rounds - 1, max_chars)

        return result

    def get_discarded_edit_keys(self) -> set:
        """Get paragraph keys of all previously discarded edits."""
        keys = set()
        for record in self.rounds:
            for decision in record.edits_discarded:
                keys.add(_get_paragraph_key_for_history(decision.edit))
        return keys

    def get_applied_edit_keys(self) -> set:
        """Get paragraph keys of all previously applied edits."""
        keys = set()
        for record in self.rounds:
            for decision in record.edits_applied:
                keys.add(_get_paragraph_key_for_history(decision.edit))
        return keys


class ArbiterResult(BaseModel):
    """Result from Arbiter arbitration."""
    edits_to_apply: List[Any] = Field(default_factory=list, description="TextEditRange objects to apply")
    edits_discarded: List[Dict[str, Any]] = Field(default_factory=list, description="Discarded edits with reasons")
    edit_decisions: List[Any] = Field(default_factory=list, exclude=True, description="Internal ArbiterEditDecision records")
    conflicts_found: int = Field(default=0, description="Number of conflicts detected")
    conflicts_resolved: int = Field(default=0, description="Number of conflicts resolved")
    round_record: Optional[Dict[str, Any]] = Field(default=None, description="EditRoundRecord as dict for history")
    arbiter_reasoning: str = Field(default="", description="Arbiter's overall reasoning")
    # Distribution and escalation info
    distribution: str = Field(default="single_qa", description="Edit distribution classification")
    escalated_to_gran_sabio: bool = Field(default=False, description="Whether GranSabio model was used")
    model_used: str = Field(default="", description="Actual model used for arbitration")


# =============================================================================
# ARBITER CONTEXT (for Phase 2)
# =============================================================================


@dataclass
class ArbiterContext:
    """
    Context provided to Arbiter for making informed decisions.

    This mirrors the context that GranSabio receives, ensuring
    the Arbiter can make decisions aligned with the original request.
    """
    # From original request
    original_prompt: str           # User's original instructions
    content_type: str              # "biography", "script", etc.
    system_prompt: Optional[str]   # System prompt if provided

    # From current layer
    layer_name: str                # "Spelling", "Coherence", etc.
    layer_criteria: str            # Layer's evaluation criteria
    layer_min_score: float         # Minimum score required

    # From content
    current_content: str           # Content after previous edits
    content_excerpt: Optional[str] # Relevant fragment (to save tokens)

    # From evaluators
    proposed_edits: List[ProposedEdit]  # All proposed edits
    evaluator_scores: Dict[str, float]  # Scores by model

    # History
    layer_history: LayerEditHistory     # Previous edits in this layer

    # Model escalation (for minority/conflict/tie cases)
    gran_sabio_model: Optional[str] = None  # Powerful model for difficult cases
    qa_model_count: int = 1                 # Total number of QA models (for distribution calc)
    model_alias_registry: Optional[Any] = None

    # Tool-loop activation knob forwarded from ``ContentRequest.arbiter_tools_mode``.
    # ``"auto"`` lets the Arbiter activate the shared ``call_ai_with_validation_tools``
    # loop when the provider supports it; ``"never"`` forces the legacy single-shot
    # path. Declared here (not on the constructor) so tests and callers can vary
    # it per-arbitration without mutating Arbiter state.
    arbiter_tools_mode: str = "auto"


# =============================================================================
# PROMPT TEMPLATES
# =============================================================================


ARBITER_SYSTEM_PROMPT = """You are Arbiter, an intelligent conflict resolver for text editing operations.

Your role is to analyze proposed edits from multiple AI evaluators and determine which edits should be applied. You MUST verify that each edit aligns with the user's original request.

CRITICAL: Edits can be WRONG even without conflicts. A QA model may propose an edit that:
- Contradicts the user's explicit instructions
- Changes content the user specifically requested to keep
- Applies criteria beyond the current layer's scope
- Is based on flawed reasoning

DECISION CRITERIA (in order of importance):
1. ALIGNMENT WITH ORIGINAL REQUEST - Does this edit honor what the user asked for?
2. Appropriateness for the current QA layer's criteria ONLY
3. Whether the edit's reasoning is sound
4. Severity of the issue being addressed
5. Avoiding changes that alter meaning when not intended

CONFLICT TYPES TO DETECT:
- OPPOSITE_OPERATIONS: DELETE vs REPLACE/MODIFY on same text
- OPPOSITE_DIRECTIONS: EXPAND vs CONDENSE on same paragraph
- SEVERITY_MISMATCH: Same issue marked with different severities
- SEMANTIC_REDUNDANCY: Multiple edits asking for the same thing differently
- CYCLE_DETECTED: Edit was previously discarded and is being re-proposed
- MISALIGNMENT: Edit contradicts user's original request (FALSE POSITIVE)

YOU CAN AND SHOULD REJECT ALL EDITS if they are poorly reasoned or contradict the user's intent.

OUTPUT FORMAT:
Return valid JSON with your decisions for each proposed edit.
"""


ARBITER_USER_PROMPT_TEMPLATE = """
## ORIGINAL REQUEST CONTEXT
Content Type: {content_type}
Original Instructions: {original_prompt}
{system_prompt_section}

## CURRENT QA LAYER
Layer: {layer_name}
Criteria: {layer_criteria}
Minimum Score: {layer_min_score}

## CONTENT EXCERPT (relevant section)
{content_excerpt}

## EDIT HISTORY FOR THIS LAYER
{layer_history}

## PROPOSED EDITS FROM EVALUATORS
{proposed_edits_formatted}

## EDIT DISTRIBUTION
{distribution_info}

## DETECTED POTENTIAL CONFLICTS
{conflicts_formatted}

## YOUR TASK
Analyze EACH proposed edit and decide:
1. APPLY - The edit should be applied (well-reasoned and aligned with request)
2. DISCARD - The edit should NOT be applied (explain why - misalignment, poor reasoning, etc.)

CRITICAL VERIFICATION (check for EACH edit):
1. Does this edit CONTRADICT the user's original instructions? If yes → DISCARD
2. Is the edit's reasoning sound, or is it based on a misunderstanding? If flawed → DISCARD
3. Does the edit apply criteria BEYOND this layer's scope? If yes → DISCARD
4. If this is a MINORITY edit (only one model proposed it), be extra skeptical
5. Even if multiple models agree, verify the edit doesn't violate user intent

You CAN discard ALL edits if none are appropriate. An empty edit list is valid.

RESPONSE FORMAT (JSON):
{{
  "reasoning": "Your analysis - especially note any edits that contradict user intent...",
  "decisions": [
    {{
      "edit_index": 0,
      "decision": "APPLY|DISCARD",
      "reason": "Specific reason - if discarding, explain the misalignment or flaw"
    }}
  ],
  "conflicts_resolved": [
    {{
      "conflict_index": 0,
      "resolution": "How you resolved it"
    }}
  ]
}}
"""


# =============================================================================
# ARBITER RESPONSE SCHEMA (JSON Structured Outputs contract)
# =============================================================================

# Schema conventions (mirroring ``qa_response_schemas.py``):
# - No numeric ``minimum``/``maximum`` — OpenAI strict structured outputs
#   reject those keywords.
# - ``additionalProperties: false`` at every object level (root + items).
# - Every property listed in ``required`` (nullables are expressed via
#   ``"type": [..., "null"]`` inside ``properties``, not by omission).
# - MERGE decision removed — only ``APPLY`` / ``DISCARD`` survive.

ARBITER_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["decisions", "conflicts_resolved", "reasoning"],
    "properties": {
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["edit_index", "decision", "reason"],
                "properties": {
                    "edit_index": {"type": "integer"},
                    "decision": {"type": "string", "enum": ["APPLY", "DISCARD"]},
                    "reason": {"type": "string"},
                },
            },
        },
        "conflicts_resolved": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["conflict_index", "resolution"],
                "properties": {
                    "conflict_index": {"type": "integer"},
                    "resolution": {"type": "string"},
                },
            },
        },
        "reasoning": {"type": "string"},
    },
}


# =============================================================================
# ARBITER PARSE ERROR (fail-closed exception for the hardened parser)
# =============================================================================


class ArbiterParseError(Exception):
    """Raised when the Arbiter response violates the decision contract.

    Carries granular lists for telemetry so operators can diagnose which
    criterion failed (missing index, duplicate, out-of-range, or an invalid
    decision value). When raised inside the runtime arbitration flow the
    caller fail-closes the batch — no edit is applied.
    """

    def __init__(
        self,
        message: str,
        *,
        missing_indices: Optional[List[int]] = None,
        duplicate_indices: Optional[List[int]] = None,
        out_of_range: Optional[List[int]] = None,
        invalid_decisions: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        super().__init__(message)
        self.missing_indices: List[int] = list(missing_indices or [])
        self.duplicate_indices: List[int] = list(duplicate_indices or [])
        self.out_of_range: List[int] = list(out_of_range or [])
        self.invalid_decisions: List[Dict[str, Any]] = list(invalid_decisions or [])

    def to_event_payload(self) -> Dict[str, Any]:
        """Shape the exception data into a debug-event payload dict."""
        return {
            "message": str(self),
            "missing_indices": list(self.missing_indices),
            "duplicate_indices": list(self.duplicate_indices),
            "out_of_range": list(self.out_of_range),
            "invalid_decisions": list(self.invalid_decisions),
        }


# =============================================================================
# ARBITER CLASS (Phase 2 Implementation)
# =============================================================================


# Define operation compatibility groups for conflict detection
_DESTRUCTIVE_OPS = {"delete"}
_MODIFYING_OPS = {"replace", "rephrase", "improve", "fix_grammar", "fix_style"}
_EXPANDING_OPS = {"expand", "insert_before", "insert_after"}
_CONTRACTING_OPS = {"condense", "delete"}


class Arbiter:
    """
    Intelligent conflict resolver for smart-edit operations.

    The Arbiter acts at the per-layer level to:
    1. Receive all edits proposed by QA evaluators
    2. Classify edit distribution (consensus, majority, minority, conflict, tie)
    3. ALWAYS verify edits with AI - checking alignment with original request
    4. Detect and resolve conflicts between edits
    5. Reject edits that contradict user intent (even without conflicts)
    6. Generate history for informing subsequent rounds
    7. Return curated list of edits to apply

    Model escalation:
    - CONSENSUS/MAJORITY/SINGLE_QA: Use economic arbiter model
    - MINORITY/CONFLICT/TIE: Escalate to GranSabio model (more powerful)
    """

    def __init__(
        self,
        ai_service: Any,
        model: Optional[str] = None,
        stream_callback: Optional[callable] = None,
        debug_event_callback: Optional[Callable[[str, Dict[str, Any]], Awaitable[None]]] = None,
        tool_event_callback: Optional[Callable[[str, Dict[str, Any]], Awaitable[None]]] = None,
    ):
        """
        Initialize Arbiter.

        Args:
            ai_service: AI service instance for making API calls
            model: Model to use for conflict resolution (default from config)
            stream_callback: Optional callback for streaming AI responses.
                             Signature: async def callback(chunk: str, model: str, operation: str)
            debug_event_callback: Optional pre-bound callback that persists arbitration
                events to the debugger DB. Signature: async def cb(event_type, payload).
                The caller is responsible for binding session_id at construction time.
            tool_event_callback: Optional pre-bound callback for live tool-loop events
                pushed to /stream/project. Signature: async def cb(event_type, payload).
                Used when the Arbiter runs inside the shared tool loop.
        """
        self.ai_service = ai_service
        self._model = model
        self.stream_callback = stream_callback
        self._debug_event_callback = debug_event_callback
        self._tool_event_callback = tool_event_callback
        self._logger = __import__('logging').getLogger(__name__)

    async def _emit_debug_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Persist an arbiter event to the debugger DB if a callback is bound."""
        if self._debug_event_callback is None:
            return
        try:
            await self._debug_event_callback(event_type, payload)
        except Exception:
            self._logger.exception("Arbiter debug_event_callback failed for %s", event_type)

    @property
    def model(self) -> str:
        """Get the default economic model for arbitration."""
        if self._model:
            return self._model
        from config import get_default_models
        return get_default_models().get("arbiter", "gpt-5-mini")

    # =========================================================================
    # DISTRIBUTION CLASSIFICATION METHODS
    # =========================================================================

    def _classify_distribution(
        self,
        proposed_edits: List[ProposedEdit],
        qa_model_count: int,
        conflicts: List[ConflictInfo]
    ) -> EditDistribution:
        """
        Classify the distribution of edits among QA evaluators.

        This determines which model to use for arbitration:
        - CONSENSUS/MAJORITY/SINGLE_QA → Economic model
        - MINORITY/CONFLICT/TIE → GranSabio model (escalation)

        Args:
            proposed_edits: All proposed edits
            qa_model_count: Total number of QA models
            conflicts: Detected conflicts

        Returns:
            EditDistribution classification
        """
        if qa_model_count <= 0:
            return EditDistribution.SINGLE_QA

        # Count unique evaluators that proposed edits
        evaluators_with_edits = set(pe.source_evaluator for pe in proposed_edits)
        proposing_count = len(evaluators_with_edits)

        # Check for conflicts first (always escalate)
        if conflicts:
            # With 2 models, any conflict is a TIE (disagreement between the pair)
            if qa_model_count == 2:
                return EditDistribution.TIE
            return EditDistribution.CONFLICT

        # Single QA model - no comparison possible
        if qa_model_count == 1:
            return EditDistribution.SINGLE_QA

        # 2 QA models
        if qa_model_count == 2:
            if proposing_count == 0:
                return EditDistribution.CONSENSUS  # Both agree: no edits needed
            if proposing_count == 2:
                # Both proposed - check if for same paragraph (implicit conflict)
                paragraphs = set(pe.paragraph_key for pe in proposed_edits)
                if len(paragraphs) < len(proposed_edits):
                    return EditDistribution.TIE  # Multiple edits for same paragraph
                return EditDistribution.CONSENSUS  # Different paragraphs, compatible
            # Only 1 of 2 proposed - this is a TIE (disagreement)
            return EditDistribution.TIE

        # 3+ QA models
        if proposing_count == 0:
            return EditDistribution.CONSENSUS  # No edits proposed

        if proposing_count == qa_model_count:
            return EditDistribution.CONSENSUS  # All models proposed edits

        # Calculate ratio
        ratio = proposing_count / qa_model_count

        if ratio > 0.5:
            return EditDistribution.MAJORITY  # >50% proposed
        if proposing_count * 2 == qa_model_count:
            return EditDistribution.TIE  # Exactly 50% proposed
        return EditDistribution.MINORITY  # <50% proposed

    def _select_model_for_distribution(
        self,
        distribution: EditDistribution,
        gran_sabio_model: Optional[str]
    ) -> tuple:
        """
        Select appropriate model based on edit distribution.

        Args:
            distribution: Classified distribution
            gran_sabio_model: Model for escalation cases

        Returns:
            Tuple of (model_to_use, escalated_to_gran_sabio)
        """
        # Escalate for difficult cases
        if distribution in (EditDistribution.MINORITY, EditDistribution.CONFLICT, EditDistribution.TIE):
            if gran_sabio_model:
                self._logger.info(
                    f"Distribution={distribution.value}: Escalating to GranSabio model ({gran_sabio_model})"
                )
                return gran_sabio_model, True
            else:
                self._logger.warning(
                    f"Distribution={distribution.value}: Would escalate but no gran_sabio_model provided"
                )

        # Use economic model for consensus/majority/single_qa
        return self.model, False

    def _format_distribution_info(
        self,
        distribution: EditDistribution,
        proposed_edits: List[ProposedEdit],
        qa_model_count: int
    ) -> str:
        """
        Format distribution information for the prompt.

        Args:
            distribution: Classified distribution
            proposed_edits: All proposed edits
            qa_model_count: Total number of QA models

        Returns:
            Formatted string for prompt injection
        """
        evaluators_with_edits = set(pe.source_evaluator for pe in proposed_edits)
        proposing_count = len(evaluators_with_edits)

        lines = [
            f"Total QA evaluators: {qa_model_count}",
            f"Evaluators proposing edits: {proposing_count} ({', '.join(sorted(evaluators_with_edits)) if evaluators_with_edits else 'none'})",
            f"Distribution: {distribution.value.upper()}"
        ]

        if distribution == EditDistribution.MINORITY:
            lines.append("WARNING: MINORITY of evaluators proposed these edits. Be extra skeptical.")
        elif distribution == EditDistribution.TIE:
            lines.append("WARNING: TIE/DISAGREEMENT between evaluators. Careful analysis required.")
        elif distribution == EditDistribution.CONFLICT:
            lines.append("WARNING: CONFLICTING edits detected. Must resolve incompatibilities.")

        return "\n".join(lines)

    # =========================================================================
    # CONFLICT DETECTION METHODS
    # =========================================================================

    def _group_edits_by_paragraph(
        self,
        proposed_edits: List[ProposedEdit]
    ) -> Dict[str, List[ProposedEdit]]:
        """
        Group proposed edits by their paragraph key.

        Args:
            proposed_edits: List of ProposedEdit objects

        Returns:
            Dict mapping paragraph_key to list of edits affecting that paragraph
        """
        groups: Dict[str, List[ProposedEdit]] = {}
        for pe in proposed_edits:
            key = pe.paragraph_key
            if key not in groups:
                groups[key] = []
            groups[key].append(pe)
        return groups

    def _filter_stale_edits(
        self,
        proposed_edits: List[ProposedEdit],
        current_content: str
    ) -> Tuple[List[ProposedEdit], List[Dict[str, Any]], List[Tuple[int, ArbiterEditDecision]]]:
        """
        Filter out stale edits that are no longer applicable to current content.

        An edit is stale if:
        - STALE_FRAGMENT: exact_fragment no longer exists in content (was already modified)
        - ALREADY_APPLIED: suggested_text already matches content at the target location

        This prevents wasted AI calls and potential corruption from applying edits
        that reference outdated content state.

        Args:
            proposed_edits: List of proposed edits to filter
            current_content: Current content after previous edits

        Returns:
            Tuple of (valid_edits, discarded_edits_info, discarded_decisions_by_original_index)
        """
        valid_edits = []
        discarded_info = []
        discarded_decisions: List[Tuple[int, ArbiterEditDecision]] = []

        for original_index, pe in enumerate(proposed_edits):
            edit = pe.edit

            # Get exact_fragment and suggested_text from edit
            exact_fragment = getattr(edit, 'exact_fragment', None) or ""
            suggested_text = getattr(edit, 'new_content', None)

            # Skip edits without exact_fragment (can't verify staleness)
            if not exact_fragment:
                valid_edits.append(pe)
                continue

            # Check 1: Does exact_fragment still exist in content?
            if exact_fragment not in current_content:
                reason = "STALE_FRAGMENT: exact_fragment no longer exists in content"
                self._logger.info(
                    f"Stale edit detected (STALE_FRAGMENT): '{exact_fragment[:40]}...' "
                    f"no longer in content [evaluator: {pe.source_evaluator}]"
                )
                discarded_info.append({
                    "edit_type": _edit_type_value(edit),
                    "source_evaluator": pe.source_evaluator,
                    "reason": reason,
                    "paragraph_key": pe.paragraph_key[:80],
                    "conflict_type": ConflictType.STALE_FRAGMENT.value
                })
                discarded_decisions.append((
                    original_index,
                    ArbiterEditDecision(
                        edit=edit,
                        decision=ArbiterDecision.DISCARD,
                        reason=reason,
                        source_evaluator=pe.source_evaluator,
                    ),
                ))
                continue

            # Check 2: Is suggested_text already in place? (REPLACE/MODIFY only)
            # DELETE operations have new_content=None and are not covered here:
            # an already-applied DELETE leaves exact_fragment absent from content,
            # so Check 1 (STALE_FRAGMENT) catches that case.
            if suggested_text is not None:
                fragment_pos = current_content.find(exact_fragment)
                if fragment_pos != -1:
                    # suggested_text == exact_fragment means the edit is a noop
                    # (already applied or nothing to change).
                    if exact_fragment == suggested_text:
                        reason = "ALREADY_APPLIED: suggested_text equals exact_fragment (noop)"
                        self._logger.info(
                            f"Stale edit detected (ALREADY_APPLIED): suggested_text equals "
                            f"exact_fragment [evaluator: {pe.source_evaluator}]"
                        )
                        discarded_info.append({
                            "edit_type": _edit_type_value(edit),
                            "source_evaluator": pe.source_evaluator,
                            "reason": reason,
                            "paragraph_key": pe.paragraph_key[:80],
                            "conflict_type": ConflictType.ALREADY_APPLIED.value
                        })
                        discarded_decisions.append((
                            original_index,
                            ArbiterEditDecision(
                                edit=edit,
                                decision=ArbiterDecision.DISCARD,
                                reason=reason,
                                source_evaluator=pe.source_evaluator,
                            ),
                        ))
                        continue

            # Edit is valid
            valid_edits.append(pe)

        if discarded_info:
            self._logger.info(
                f"Filtered {len(discarded_info)} stale edit(s), "
                f"{len(valid_edits)} valid edit(s) remaining"
            )

        return valid_edits, discarded_info, discarded_decisions

    def _detect_opposite_operations(
        self,
        edits: List[ProposedEdit],
        paragraph_key: str
    ) -> Optional[ConflictInfo]:
        """
        Detect DELETE vs REPLACE/MODIFY conflicts on same fragment.

        Args:
            edits: List of edits on the same paragraph
            paragraph_key: The paragraph key

        Returns:
            ConflictInfo if conflict found, None otherwise
        """
        if len(edits) < 2:
            return None

        op_types = []
        for pe in edits:
            op = _edit_type_value(pe.edit).lower()
            op_types.append(op)

        has_destructive = any(op in _DESTRUCTIVE_OPS for op in op_types)
        has_modifying = any(op in _MODIFYING_OPS for op in op_types)

        if has_destructive and has_modifying:
            return ConflictInfo(
                conflict_type=ConflictType.OPPOSITE_OPERATIONS,
                paragraph_key=paragraph_key,
                involved_edits=edits,
                description=f"DELETE conflicts with MODIFY operations: {', '.join(op_types)}"
            )
        return None

    def _detect_opposite_directions(
        self,
        edits: List[ProposedEdit],
        paragraph_key: str
    ) -> Optional[ConflictInfo]:
        """
        Detect EXPAND vs CONDENSE conflicts on same paragraph.

        Args:
            edits: List of edits on the same paragraph
            paragraph_key: The paragraph key

        Returns:
            ConflictInfo if conflict found, None otherwise
        """
        if len(edits) < 2:
            return None

        op_types = []
        for pe in edits:
            op = _edit_type_value(pe.edit).lower()
            op_types.append(op)

        has_expanding = any(op in _EXPANDING_OPS for op in op_types)
        has_contracting = any(op in _CONTRACTING_OPS for op in op_types)

        if has_expanding and has_contracting:
            return ConflictInfo(
                conflict_type=ConflictType.OPPOSITE_DIRECTIONS,
                paragraph_key=paragraph_key,
                involved_edits=edits,
                description=f"EXPAND conflicts with CONTRACT operations: {', '.join(op_types)}"
            )
        return None

    def _detect_severity_mismatch(
        self,
        edits: List[ProposedEdit],
        paragraph_key: str
    ) -> Optional[ConflictInfo]:
        """
        Detect same issue with different severities from different evaluators.

        Args:
            edits: List of edits on the same paragraph
            paragraph_key: The paragraph key

        Returns:
            ConflictInfo if significant severity mismatch found, None otherwise
        """
        if len(edits) < 2:
            return None

        severities = []
        for pe in edits:
            sev = _enum_or_text_value(getattr(pe.edit, "issue_severity", None), "minor")
            severities.append(sev)

        unique_severities = set(severities)
        # Only flag if we have critical vs minor (significant disagreement)
        if "critical" in unique_severities and "minor" in unique_severities:
            return ConflictInfo(
                conflict_type=ConflictType.SEVERITY_MISMATCH,
                paragraph_key=paragraph_key,
                involved_edits=edits,
                description=f"Severity disagreement: {', '.join(unique_severities)}"
            )
        return None

    def _detect_semantic_redundancy(
        self,
        edits: List[ProposedEdit],
        paragraph_key: str
    ) -> Optional[ConflictInfo]:
        """
        Detect multiple edits asking for the same thing with different wording.

        This is a simple heuristic based on operation type and target.
        Full semantic analysis would require AI, so this is conservative.

        Args:
            edits: List of edits on the same paragraph
            paragraph_key: The paragraph key

        Returns:
            ConflictInfo if likely redundancy found, None otherwise
        """
        if len(edits) < 2:
            return None

        # Group by operation type - if multiple edits of same type on same paragraph,
        # they are likely redundant
        op_counts: Dict[str, int] = {}
        for pe in edits:
            op = _edit_type_value(pe.edit).lower()
            op_counts[op] = op_counts.get(op, 0) + 1

        redundant_ops = [op for op, count in op_counts.items() if count > 1]
        if redundant_ops:
            return ConflictInfo(
                conflict_type=ConflictType.SEMANTIC_REDUNDANCY,
                paragraph_key=paragraph_key,
                involved_edits=edits,
                description=f"Multiple {', '.join(redundant_ops)} operations on same paragraph"
            )
        return None

    def _detect_cycles(
        self,
        edits: List[ProposedEdit],
        paragraph_key: str,
        layer_history: LayerEditHistory
    ) -> List[ConflictInfo]:
        """
        Detect edits that were previously discarded being re-proposed.

        Args:
            edits: List of edits on the same paragraph
            paragraph_key: The paragraph key
            layer_history: Edit history for cycle detection

        Returns:
            List of ConflictInfo for cycle detections (may be empty)
        """
        conflicts = []
        discarded_keys = layer_history.get_discarded_edit_keys()

        for pe in edits:
            edit_key = _get_paragraph_key_for_history(pe.edit)
            if edit_key in discarded_keys:
                conflicts.append(ConflictInfo(
                    conflict_type=ConflictType.CYCLE_DETECTED,
                    paragraph_key=paragraph_key,
                    involved_edits=[pe],
                    description=f"Edit was previously discarded: {edit_key[:50]}"
                ))

        return conflicts

    def _detect_conflicts(
        self,
        proposed_edits: List[ProposedEdit],
        layer_history: LayerEditHistory
    ) -> List[ConflictInfo]:
        """
        Main conflict detection method that orchestrates all detection types.

        Args:
            proposed_edits: All proposed edits for this round
            layer_history: History for cycle detection

        Returns:
            List of all detected conflicts
        """
        all_conflicts: List[ConflictInfo] = []

        # Group edits by paragraph
        groups = self._group_edits_by_paragraph(proposed_edits)

        for paragraph_key, edits in groups.items():
            # Check for opposite operations
            conflict = self._detect_opposite_operations(edits, paragraph_key)
            if conflict:
                all_conflicts.append(conflict)

            # Check for opposite directions
            conflict = self._detect_opposite_directions(edits, paragraph_key)
            if conflict:
                all_conflicts.append(conflict)

            # Check for severity mismatch
            conflict = self._detect_severity_mismatch(edits, paragraph_key)
            if conflict:
                all_conflicts.append(conflict)

            # Check for semantic redundancy
            conflict = self._detect_semantic_redundancy(edits, paragraph_key)
            if conflict:
                all_conflicts.append(conflict)

            # Check for cycles
            cycle_conflicts = self._detect_cycles(edits, paragraph_key, layer_history)
            all_conflicts.extend(cycle_conflicts)

        return all_conflicts

    # =========================================================================
    # PROMPT BUILDING METHODS
    # =========================================================================

    def _format_edits_for_prompt(
        self,
        proposed_edits: List[ProposedEdit]
    ) -> str:
        """
        Format proposed edits for the Arbiter prompt.

        Args:
            proposed_edits: List of proposed edits

        Returns:
            Formatted string for prompt injection
        """
        if not proposed_edits:
            return "No edits proposed."

        lines = []
        for i, pe in enumerate(proposed_edits):
            op = _edit_type_value(pe.edit, "EDIT")
            sev = _enum_or_text_value(getattr(pe.edit, "issue_severity", None), "minor")
            desc = ""
            if hasattr(pe.edit, 'issue_description') and pe.edit.issue_description:
                desc = pe.edit.issue_description[:100]
            elif hasattr(pe.edit, 'edit_instruction') and pe.edit.edit_instruction:
                desc = pe.edit.edit_instruction[:100]

            # Get exact_fragment and suggested_text for verification
            exact_fragment = getattr(pe.edit, 'exact_fragment', None) or ""
            suggested_text = getattr(pe.edit, 'new_content', None)

            # Build edit info with fragment details
            edit_info = (
                f"[{i}] {op.upper()} (severity={sev}) by {pe.source_evaluator}\n"
                f"    Paragraph: {pe.paragraph_key[:60]}...\n"
                f"    Description: {desc}"
            )

            # Add fragment details if available (truncate for prompt size)
            if exact_fragment:
                fragment_preview = exact_fragment[:80] + ("..." if len(exact_fragment) > 80 else "")
                edit_info += f"\n    Find: \"{fragment_preview}\""

            if suggested_text is not None:
                suggested_preview = suggested_text[:80] + ("..." if len(suggested_text) > 80 else "")
                edit_info += f"\n    Replace with: \"{suggested_preview}\""

            lines.append(edit_info)

        return "\n\n".join(lines)

    def _format_conflicts_for_prompt(
        self,
        conflicts: List[ConflictInfo]
    ) -> str:
        """
        Format detected conflicts for the Arbiter prompt.

        Args:
            conflicts: List of detected conflicts

        Returns:
            Formatted string for prompt injection
        """
        if not conflicts:
            return "No conflicts detected."

        lines = []
        for i, conflict in enumerate(conflicts):
            involved_evaluators = [pe.source_evaluator for pe in conflict.involved_edits]
            lines.append(
                f"[Conflict {i + 1}] {conflict.conflict_type.value}\n"
                f"  Paragraph: {conflict.paragraph_key[:60]}...\n"
                f"  Involved evaluators: {', '.join(involved_evaluators)}\n"
                f"  Description: {conflict.description}"
            )

        return "\n\n".join(lines)

    def _build_arbiter_prompt(
        self,
        context: ArbiterContext,
        conflicts: List[ConflictInfo],
        distribution: EditDistribution
    ) -> str:
        """
        Build the complete Arbiter prompt with all context.

        Args:
            context: Full arbitration context
            conflicts: Detected conflicts
            distribution: Classified edit distribution

        Returns:
            Complete prompt string
        """
        system_prompt_section = ""
        if context.system_prompt:
            system_prompt_section = f"System Prompt: {context.system_prompt}"

        # Get history formatted
        history_str = context.layer_history.format_for_prompt()
        if not history_str:
            history_str = "No previous edits in this layer."

        # Use full content excerpt; context budget is enforced by the tool loop
        # (estimate_prompt_overflow + TOOL_LOOP_MAX_PROMPT_CHARS), fail-fast if overflow.
        content_excerpt = context.content_excerpt or context.current_content

        # Format distribution info
        distribution_info = self._format_distribution_info(
            distribution, context.proposed_edits, context.qa_model_count
        )

        return ARBITER_USER_PROMPT_TEMPLATE.format(
            content_type=context.content_type,
            original_prompt=context.original_prompt,
            system_prompt_section=system_prompt_section,
            layer_name=context.layer_name,
            layer_criteria=context.layer_criteria,
            layer_min_score=context.layer_min_score,
            content_excerpt=content_excerpt,
            layer_history=history_str,
            proposed_edits_formatted=self._format_edits_for_prompt(context.proposed_edits),
            distribution_info=distribution_info,
            conflicts_formatted=self._format_conflicts_for_prompt(conflicts)
        )

    # =========================================================================
    # AI RESOLUTION METHODS
    # =========================================================================

    @staticmethod
    def _should_use_arbiter_tools(mode: str, model: Optional[str]) -> bool:
        """Decide whether to route the Arbiter call through the shared tool loop.

        Returns ``False`` when:

        - ``mode == "never"`` (user explicitly disabled).
        - Model resolves to the OpenAI Responses API (single-shot only — the
          tool loop does not support that API today).
        - Provider is outside the supported matrix
          (``openai``, ``openrouter``, ``xai``, ``claude``, ``gemini``).

        Returns ``True`` otherwise. Unknown / unresolvable models fall back
        to ``False`` (fail-closed — no tools if we can't prove support).
        """
        if mode == "never":
            return False
        if not model:
            return False

        from ai_service import AIService
        from config import config as _config

        try:
            info = _config.get_model_info(model)
        except Exception:
            return False

        provider_raw = info.get("provider", "") if isinstance(info, dict) else ""
        model_id = info.get("model_id", model) if isinstance(info, dict) else model
        provider_key = AIService._normalize_tool_loop_provider(provider_raw)

        if provider_key == "openai" and AIService._is_openai_responses_api_model(model_id):
            return False

        if provider_key not in {"openai", "openrouter", "xai", "claude", "gemini"}:
            return False

        return True

    async def _resolve_with_ai(
        self,
        context: ArbiterContext,
        conflicts: List[ConflictInfo],
        distribution: EditDistribution,
        selected_model: str
    ) -> Dict[str, Any]:
        """
        Call AI to analyze and decide on proposed edits.

        ALWAYS called - verifies alignment with original request even without conflicts.

        When ``context.arbiter_tools_mode == "auto"`` and the provider/model
        support the shared tool loop, dispatches through
        :meth:`AIService.call_ai_with_validation_tools` with
        ``OutputContract.JSON_STRUCTURED`` + ``ARBITER_RESPONSE_SCHEMA`` so the
        returned ``envelope.payload`` is a parsed+validated dict. When the tool
        loop is not usable (mode ``"never"`` or unsupported provider), the
        legacy single-shot path runs through ``generate_content`` /
        ``generate_content_stream``.

        Args:
            context: Full arbitration context
            conflicts: Detected conflicts to resolve
            distribution: Classified edit distribution
            selected_model: Model to use (may be GranSabio for escalation)

        Returns:
            Parsed AI response as dict
        """
        from config import config

        prompt = self._build_arbiter_prompt(context, conflicts, distribution)
        prompt_safety_parts = None
        if context.model_alias_registry:
            prompt_safety_parts = [
                PromptPart(
                    text="\n\n".join(
                        [
                            context.layer_history.format_for_prompt(),
                            self._format_edits_for_prompt(context.proposed_edits),
                            self._format_distribution_info(
                                distribution,
                                context.proposed_edits,
                                context.qa_model_count,
                            ),
                            self._format_conflicts_for_prompt(conflicts),
                        ]
                    ),
                    source="system_generated",
                    label="arbiter.evaluator_context",
                )
            ]

        tool_loop_enabled = self._should_use_arbiter_tools(
            context.arbiter_tools_mode, selected_model
        )

        try:
            if tool_loop_enabled:
                parsed = await self._resolve_with_tool_loop(
                    context=context,
                    prompt=prompt,
                    selected_model=selected_model,
                    prompt_safety_parts=prompt_safety_parts,
                )
            else:
                parsed = await self._resolve_single_shot(
                    context=context,
                    prompt=prompt,
                    selected_model=selected_model,
                    prompt_safety_parts=prompt_safety_parts,
                )

            await self._emit_debug_event(
                "arbiter_ai_resolution",
                {
                    "model": selected_model,
                    "layer_name": context.layer_name,
                    "distribution": distribution.value,
                    "proposed_edit_count": len(context.proposed_edits),
                    "conflict_count": len(conflicts),
                    "has_decisions": bool(parsed.get("decisions")),
                    "tool_loop": tool_loop_enabled,
                },
            )
            return parsed

        except Exception as e:
            self._logger.error(f"Arbiter AI call failed: {e}")
            await self._emit_debug_event(
                "arbiter_ai_error",
                {
                    "model": selected_model,
                    "layer_name": context.layer_name,
                    "error": str(e),
                    "tool_loop": tool_loop_enabled,
                },
            )
            raise RuntimeError(f"Arbiter AI call failed: {e}") from e

    async def _resolve_with_tool_loop(
        self,
        *,
        context: ArbiterContext,
        prompt: str,
        selected_model: str,
        prompt_safety_parts: Optional[List[Any]],
    ) -> Dict[str, Any]:
        """Dispatch the Arbiter call through ``call_ai_with_validation_tools``.

        Uses ``OutputContract.JSON_STRUCTURED`` with ``ARBITER_RESPONSE_SCHEMA``
        so ``envelope.payload`` arrives already parsed and schema-validated.
        """
        from config import config
        from tool_loop_models import (
            LoopScope,
            OutputContract,
            PayloadScope,
        )

        # A neutral validation callback is required by the tool loop signature.
        # The Arbiter does not iterate on draft metrics — it arbitrates edits.
        # The initial measurement is the current content so the LLM sees the
        # deterministic picture before making a decision; subsequent turns
        # only happen if the model chooses to call ``validate_draft`` again.
        from deterministic_validation import DraftValidationResult
        from word_count_utils import count_words

        def _neutral_validation_callback(text: str) -> DraftValidationResult:
            wc = count_words(text or "")
            return DraftValidationResult(
                approved=True,
                hard_failed=False,
                score=10.0,
                word_count=wc,
                feedback="Arbiter measurement snapshot (no enforced constraints).",
                issues=[],
                metrics={"word_count": wc},
                checks={},
                stylistic_metrics=None,
                visible_payload={},
            )

        max_rounds = getattr(config, "ARBITER_MAX_TOOL_ROUNDS", 2)

        _, envelope = await self.ai_service.call_ai_with_validation_tools(
            prompt=prompt,
            model=selected_model,
            validation_callback=_neutral_validation_callback,
            output_contract=OutputContract.JSON_STRUCTURED,
            response_format=ARBITER_RESPONSE_SCHEMA,
            payload_scope=PayloadScope.MEASUREMENT_ONLY,
            stop_on_approval=False,
            loop_scope=LoopScope.ARBITER,
            retries_enabled=True,
            max_tool_rounds=max_rounds,
            initial_measurement_text=context.current_content,
            tool_event_callback=self._tool_event_callback,
            temperature=config.ARBITER_TEMPERATURE,
            max_tokens=config.ARBITER_MAX_TOKENS,
            system_prompt=ARBITER_SYSTEM_PROMPT,
            model_alias_registry=context.model_alias_registry,
            prompt_safety_parts=prompt_safety_parts,
        )

        payload = envelope.payload if envelope is not None else None
        if not isinstance(payload, dict):
            # Tool loop fell back (e.g. ``tools_skipped_reason``) without
            # producing a parsed dict. Fail-closed to the caller; the
            # arbitrate() layer will DISCARD the batch conservatively.
            from tool_loop_models import JsonContractError
            reason = (
                envelope.tools_skipped_reason
                if envelope is not None else "tool_loop_payload_missing"
            )
            raise JsonContractError(
                f"Arbiter tool-loop returned no parsed payload (reason={reason})."
            )
        return payload

    async def _resolve_single_shot(
        self,
        *,
        context: ArbiterContext,
        prompt: str,
        selected_model: str,
        prompt_safety_parts: Optional[List[Any]],
    ) -> Dict[str, Any]:
        """Legacy single-shot arbitration path (tools disabled / unsupported).

        Supports streaming when ``stream_callback`` is set. Parses the final
        JSON via :func:`tool_loop_models.parse_json_with_markdown_fences` so
        the strip-markdown logic lives in a single utility.
        """
        from config import config
        from tool_loop_models import parse_json_with_markdown_fences

        response_content = ""
        if self.stream_callback:
            async for chunk in self.ai_service.generate_content_stream(
                prompt=prompt,
                model=selected_model,
                system_prompt=ARBITER_SYSTEM_PROMPT,
                max_tokens=config.ARBITER_MAX_TOKENS,
                temperature=config.ARBITER_TEMPERATURE,
                json_output=True,
                model_alias_registry=context.model_alias_registry,
                prompt_safety_parts=prompt_safety_parts,
            ):
                if hasattr(chunk, 'text'):
                    chunk_text = chunk.text
                    is_thinking = getattr(chunk, 'is_thinking', False)
                else:
                    chunk_text = chunk
                    is_thinking = False

                if chunk_text:
                    if not is_thinking:
                        response_content += chunk_text
                    await self.stream_callback(chunk_text, selected_model, "arbitration")
        else:
            response_content = await self.ai_service.generate_content(
                prompt=prompt,
                model=selected_model,
                system_prompt=ARBITER_SYSTEM_PROMPT,
                max_tokens=config.ARBITER_MAX_TOKENS,
                temperature=config.ARBITER_TEMPERATURE,
                json_output=True,
                model_alias_registry=context.model_alias_registry,
                prompt_safety_parts=prompt_safety_parts,
            )

        return parse_json_with_markdown_fences(
            response_content,
            schema=ARBITER_RESPONSE_SCHEMA,
            context="Arbiter response",
        )

    def _parse_arbiter_response(
        self,
        ai_response: Dict[str, Any],
        proposed_edits: List[ProposedEdit],
        conflicts: List[ConflictInfo]
    ) -> List[ArbiterEditDecision]:
        """
        Convert AI response into ArbiterEditDecision objects.

        Fail-closed validation contract (§3.4.2 of the refactor proposal):

        1. **Coverage**: every ``edit_index`` in ``[0, N-1]`` must appear in
           ``ai_response["decisions"]`` exactly once (``N = len(proposed_edits)``).
        2. **Uniqueness**: no duplicate ``edit_index`` values.
        3. **Range**: every ``edit_index`` must fall inside ``[0, N-1]``.
        4. **Decision validity**: the ``decision`` field must be exactly
           ``"APPLY"`` or ``"DISCARD"`` (uppercase, no case-insensitive mapping,
           no ``"merge"`` survivors).

        Any violation raises :class:`ArbiterParseError` carrying the offending
        indices/values for telemetry. The caller (``arbitrate``) catches it
        and fail-closes the batch (DISCARD for every edit).

        Args:
            ai_response: Parsed AI response dict
            proposed_edits: Original proposed edits
            conflicts: Detected conflicts

        Returns:
            List of ArbiterEditDecision objects

        Raises:
            ArbiterParseError: if the response violates any of the 4 criteria.
        """
        total_edits = len(proposed_edits)
        raw_decisions_any = ai_response.get("decisions", [])
        if not isinstance(raw_decisions_any, list):
            raise ArbiterParseError(
                "Arbiter response 'decisions' must be a list, "
                f"got {type(raw_decisions_any).__name__}."
            )

        # Uniqueness + range + decision-validity passes -------------------------
        seen_indices: Dict[int, Dict[str, Any]] = {}
        duplicate_indices: List[int] = []
        out_of_range: List[int] = []
        invalid_decisions: List[Dict[str, Any]] = []

        valid_decision_values = {"APPLY", "DISCARD"}

        for raw in raw_decisions_any:
            if not isinstance(raw, dict):
                invalid_decisions.append(
                    {"edit_index": None, "decision": None, "issue": "non_object_entry"}
                )
                continue

            raw_index = raw.get("edit_index")
            if not isinstance(raw_index, int) or isinstance(raw_index, bool):
                invalid_decisions.append(
                    {"edit_index": raw_index, "decision": raw.get("decision"),
                     "issue": "edit_index_not_integer"}
                )
                continue

            raw_decision = raw.get("decision")
            if raw_decision not in valid_decision_values:
                # Exact uppercase match required — no case normalization so
                # "apply" / "merge" / etc. all fall through as invalid.
                invalid_decisions.append(
                    {"edit_index": raw_index, "decision": raw_decision,
                     "issue": "invalid_decision_value"}
                )
                continue

            if raw_index < 0 or raw_index >= total_edits:
                out_of_range.append(raw_index)
                continue

            if raw_index in seen_indices:
                duplicate_indices.append(raw_index)
                continue

            seen_indices[raw_index] = raw

        # Coverage pass ---------------------------------------------------------
        missing_indices = [i for i in range(total_edits) if i not in seen_indices]

        if missing_indices or duplicate_indices or out_of_range or invalid_decisions:
            raise ArbiterParseError(
                "Arbiter response failed fail-closed validation: "
                f"missing={missing_indices}, duplicates={duplicate_indices}, "
                f"out_of_range={out_of_range}, invalid={invalid_decisions}.",
                missing_indices=missing_indices,
                duplicate_indices=duplicate_indices,
                out_of_range=out_of_range,
                invalid_decisions=invalid_decisions,
            )

        # Map each proposed edit index to the resolved conflict (if any) -------
        edit_to_conflict: Dict[int, ConflictInfo] = {}
        for conflict in conflicts:
            for pe in conflict.involved_edits:
                try:
                    idx = proposed_edits.index(pe)
                    edit_to_conflict[idx] = conflict
                except ValueError:
                    pass

        decisions: List[ArbiterEditDecision] = []
        for i, pe in enumerate(proposed_edits):
            entry = seen_indices[i]
            decision_value = entry["decision"]
            reason = entry.get("reason") or "AI decision"
            decision_enum = (
                ArbiterDecision.APPLY if decision_value == "APPLY"
                else ArbiterDecision.DISCARD
            )
            decisions.append(ArbiterEditDecision(
                edit=pe.edit,
                decision=decision_enum,
                reason=reason,
                source_evaluator=pe.source_evaluator,
                conflict_resolved=edit_to_conflict.get(i),
            ))

        return decisions

    def _fail_closed_discard_all(
        self,
        proposed_edits: List[ProposedEdit],
        conflicts: List[ConflictInfo],
        reason: str,
    ) -> List[ArbiterEditDecision]:
        """Build a conservative DISCARD decision for every proposed edit.

        Used when the Arbiter parser raises :class:`ArbiterParseError` (any of
        the 4 criteria) or when the AI returns no decisions at all. No edit is
        applied — the QA feedback still exists but the batch cannot be
        arbitrated safely.
        """
        edit_to_conflict: Dict[int, ConflictInfo] = {}
        for conflict in conflicts:
            for pe in conflict.involved_edits:
                try:
                    idx = proposed_edits.index(pe)
                    edit_to_conflict[idx] = conflict
                except ValueError:
                    pass

        return [
            ArbiterEditDecision(
                edit=pe.edit,
                decision=ArbiterDecision.DISCARD,
                reason=reason,
                source_evaluator=pe.source_evaluator,
                conflict_resolved=edit_to_conflict.get(i),
            )
            for i, pe in enumerate(proposed_edits)
        ]

    # =========================================================================
    # MAIN ARBITRATION METHOD
    # =========================================================================

    async def arbitrate(
        self,
        context: ArbiterContext,
    ) -> ArbiterResult:
        """
        Arbitrate between proposed edits and return curated list.

        This is the main entry point for the Arbiter. It:
        1. Detects conflicts between proposed edits
        2. Classifies edit distribution (consensus, majority, minority, conflict, tie)
        3. Selects model based on distribution (escalate to GranSabio for difficult cases)
        4. ALWAYS calls AI to verify alignment with original request
        5. Returns curated list of edits with history record

        CRITICAL: AI is ALWAYS called to verify edits align with user intent.
        Even without conflicts, a single QA model can propose poorly-reasoned edits.

        Args:
            context: Full context for arbitration decision

        Returns:
            ArbiterResult with edits to apply and history record
        """
        original_proposed_edits = list(context.proposed_edits)
        proposed_edits = original_proposed_edits

        # If no edits proposed, nothing to do
        if not proposed_edits:
            return ArbiterResult(
                edits_to_apply=[],
                edits_discarded=[],
                conflicts_found=0,
                conflicts_resolved=0,
                arbiter_reasoning="No edits proposed",
                distribution=EditDistribution.CONSENSUS.value,
                escalated_to_gran_sabio=False,
                model_used=""
            )

        # Filter stale edits (exact_fragment missing or suggested_text already applied)
        proposed_edits, stale_discarded, stale_decision_entries = self._filter_stale_edits(
            proposed_edits, context.current_content
        )

        # If all edits were stale, nothing left to do
        if not proposed_edits:
            stale_decisions = [decision for _, decision in stale_decision_entries]
            return ArbiterResult(
                edits_to_apply=[],
                edits_discarded=stale_discarded,
                edit_decisions=stale_decisions,
                conflicts_found=0,
                conflicts_resolved=0,
                round_record={
                    "proposed_count": len(original_proposed_edits),
                    "conflicts_detected": 0,
                    "distribution": EditDistribution.CONSENSUS.value,
                    "model_used": "",
                    "escalated_to_gran_sabio": False,
                    "applied_count": 0,
                    "discarded_count": len(stale_discarded),
                    "decisions": [
                        {
                            "decision": d.decision.value,
                            "source_evaluator": d.source_evaluator,
                            "reason": d.reason[:100],
                        }
                        for d in stale_decisions
                    ],
                },
                arbiter_reasoning="All proposed edits were stale (fragments no longer in content)",
                distribution=EditDistribution.CONSENSUS.value,
                escalated_to_gran_sabio=False,
                model_used=""
            )

        # Detect conflicts
        conflicts = self._detect_conflicts(proposed_edits, context.layer_history)
        num_conflicts = len(conflicts)

        # Classify distribution
        distribution = self._classify_distribution(
            proposed_edits, context.qa_model_count, conflicts
        )

        # Select model based on distribution
        selected_model, escalated = self._select_model_for_distribution(
            distribution, context.gran_sabio_model
        )

        self._logger.info(
            f"Arbiter: {len(proposed_edits)} edit(s), {num_conflicts} conflict(s), "
            f"distribution={distribution.value}, model={selected_model}, escalated={escalated}"
        )

        # ALWAYS call AI to verify edits (no passthrough, no fallback)
        self._logger.info(f"Verifying {len(proposed_edits)} edit(s) with AI ({selected_model})...")
        ai_response = await self._resolve_with_ai(context, conflicts, distribution, selected_model)

        ai_decisions = ai_response.get("decisions")
        if ai_decisions is None:
            error_msg = ai_response.get("reasoning", "AI returned no decisions")
            self._logger.error(f"Arbiter AI verification failed: {error_msg}")
            raise RuntimeError(f"Arbiter cannot resolve edits: {error_msg}")
        if not ai_decisions:
            # AI returned an empty list. Preserve the reasoning as the
            # discard justification so the user sees why the batch was
            # rejected. With the schema enforcing uppercase enum values, we
            # must produce synthetic entries that match — lowercase would
            # fail-close at the parser boundary below.
            reasoning = ai_response.get("reasoning")
            if not reasoning:
                self._logger.error("Arbiter AI verification failed: empty decisions without reasoning")
                raise RuntimeError("Arbiter cannot resolve edits: AI returned no decisions")
            ai_response = {
                **ai_response,
                "decisions": [
                    {"edit_index": index, "decision": "DISCARD", "reason": reasoning}
                    for index, _ in enumerate(proposed_edits)
                ],
            }

        # Parse AI response with fail-closed contract
        try:
            decisions = self._parse_arbiter_response(
                ai_response, proposed_edits, conflicts
            )
            reasoning = ai_response.get("reasoning", "AI-verified edits")
        except ArbiterParseError as parse_error:
            # Fail-closed: DISCARD every edit conservatively. The batch has
            # QA-flagged issues but the AI failed to deliver valid arbitration,
            # so applying any edit is unsafe.
            self._logger.error(
                f"Arbiter parser fail-closed: {parse_error}"
            )
            await self._emit_debug_event(
                "arbiter_parse_error",
                {
                    "layer_name": context.layer_name,
                    "distribution": distribution.value,
                    "total_edits": len(proposed_edits),
                    **parse_error.to_event_payload(),
                },
            )
            decisions = self._fail_closed_discard_all(
                proposed_edits,
                conflicts,
                reason=f"Arbiter parse error: {parse_error}",
            )
            reasoning = f"Arbiter parse error (fail-closed DISCARD batch): {parse_error}"

        await self._emit_debug_event(
            "arbiter_decision",
            {
                "layer_name": context.layer_name,
                "distribution": distribution.value,
                "total_edits": len(proposed_edits),
                "applied": sum(1 for d in decisions if d.decision == ArbiterDecision.APPLY),
                "discarded": sum(1 for d in decisions if d.decision == ArbiterDecision.DISCARD),
                "decisions": [
                    {
                        "decision": d.decision.value,
                        "reason": d.reason[:200],
                        "source_evaluator": d.source_evaluator,
                    }
                    for d in decisions
                ],
            },
        )

        stale_decisions_by_index = dict(stale_decision_entries)
        expected_decision_count = len(original_proposed_edits) - len(stale_decision_entries)
        if len(decisions) != expected_decision_count:
            raise RuntimeError(
                "Arbiter AI returned a decision count that does not match the number of "
                f"non-stale proposed edits: expected {expected_decision_count}, got {len(decisions)} "
                f"(original proposals={len(original_proposed_edits)}, stale filtered={len(stale_decision_entries)})."
            )

        valid_decision_index = 0
        all_decisions: List[ArbiterEditDecision] = []
        for original_index, _ in enumerate(original_proposed_edits):
            stale_decision = stale_decisions_by_index.get(original_index)
            if stale_decision is not None:
                all_decisions.append(stale_decision)
                continue
            all_decisions.append(decisions[valid_decision_index])
            valid_decision_index += 1
        assert valid_decision_index == len(decisions), (
            f"Arbiter interleave invariant broken: consumed {valid_decision_index} "
            f"of {len(decisions)} non-stale decisions "
            f"(original={len(original_proposed_edits)}, stale={len(stale_decision_entries)})"
        )

        # Build result
        edits_to_apply = [d.edit for d in decisions if d.decision == ArbiterDecision.APPLY]
        ai_discarded = [
            {
                "edit_type": _edit_type_value(d.edit),
                "source_evaluator": d.source_evaluator,
                "reason": d.reason,
                "paragraph_key": _get_paragraph_key_for_history(d.edit)[:80]
            }
            for d in decisions if d.decision != ArbiterDecision.APPLY
        ]
        # Combine stale edits (filtered early) with AI-discarded edits
        edits_discarded = stale_discarded + ai_discarded

        # Build round record for history
        round_record = {
            "proposed_count": len(original_proposed_edits),
            "conflicts_detected": num_conflicts,
            "distribution": distribution.value,
            "model_used": selected_model,
            "escalated_to_gran_sabio": escalated,
            "applied_count": len(edits_to_apply),
            "discarded_count": len(edits_discarded),
            "decisions": [
                {
                    "decision": d.decision.value,
                    "source_evaluator": d.source_evaluator,
                    "reason": d.reason[:100]
                }
                for d in all_decisions
            ]
        }

        conflicts_resolved = sum(
            1 for d in decisions
            if d.conflict_resolved is not None and d.decision == ArbiterDecision.APPLY
        )

        return ArbiterResult(
            edits_to_apply=edits_to_apply,
            edits_discarded=edits_discarded,
            edit_decisions=all_decisions,
            conflicts_found=num_conflicts,
            conflicts_resolved=conflicts_resolved,
            round_record=round_record,
            arbiter_reasoning=reasoning,
            distribution=distribution.value,
            escalated_to_gran_sabio=escalated,
            model_used=selected_model
        )
