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
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from enum import Enum
from pydantic import BaseModel, Field

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


class ArbiterDecision(str, Enum):
    """Arbiter's decision for an edit."""
    APPLY = "apply"
    DISCARD = "discard"
    MERGE = "merge"  # Combine with another edit


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
    source_model: str               # Which model proposed it (e.g., "gpt-4o")
    source_score: float             # Score given by that model
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
    source_model: str
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
                op = decision.edit.edit_type.value if hasattr(decision.edit, 'edit_type') else "EDIT"
                desc = ""
                if hasattr(decision.edit, 'issue_description') and decision.edit.issue_description:
                    desc = decision.edit.issue_description[:50]
                elif hasattr(decision.edit, 'edit_instruction') and decision.edit.edit_instruction:
                    desc = decision.edit.edit_instruction[:50]
                lines.append(f"- Applied: {op.upper()} {desc} - {decision.source_model}")

            for decision in record.edits_discarded:
                op = decision.edit.edit_type.value if hasattr(decision.edit, 'edit_type') else "EDIT"
                desc = ""
                if hasattr(decision.edit, 'issue_description') and decision.edit.issue_description:
                    desc = decision.edit.issue_description[:50]
                elif hasattr(decision.edit, 'edit_instruction') and decision.edit.edit_instruction:
                    desc = decision.edit.edit_instruction[:50]
                lines.append(f"- Discarded: {op.upper()} {desc} - {decision.source_model}")
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
3. MERGE - Combine with another edit (specify which)

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
      "decision": "apply|discard|merge",
      "reason": "Specific reason - if discarding, explain the misalignment or flaw",
      "merge_with": null
    }}
  ],
  "conflicts_resolved": [
    {{
      "conflict_type": "opposite_operations",
      "resolution": "How you resolved it",
      "chosen_edit_index": 0
    }}
  ]
}}
"""


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
        model: Optional[str] = None
    ):
        """
        Initialize Arbiter.

        Args:
            ai_service: AI service instance for making API calls
            model: Model to use for conflict resolution (default from config)
        """
        self.ai_service = ai_service
        self._model = model
        self._logger = __import__('logging').getLogger(__name__)

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

        # Count unique models that proposed edits
        models_with_edits = set(pe.source_model for pe in proposed_edits)
        proposing_count = len(models_with_edits)

        # Check for conflicts first (always escalate)
        if conflicts:
            # Check if it's a tie situation (2 models, any disagreement)
            if qa_model_count == 2 and proposing_count >= 1:
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
        else:
            return EditDistribution.MINORITY  # ≤50% proposed (includes exactly 50%)

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
        models_with_edits = set(pe.source_model for pe in proposed_edits)
        proposing_count = len(models_with_edits)

        lines = [
            f"Total QA Models: {qa_model_count}",
            f"Models proposing edits: {proposing_count} ({', '.join(models_with_edits) if models_with_edits else 'none'})",
            f"Distribution: {distribution.value.upper()}"
        ]

        if distribution == EditDistribution.MINORITY:
            lines.append("WARNING: MINORITY of models proposed these edits. Be extra skeptical.")
        elif distribution == EditDistribution.TIE:
            lines.append("WARNING: TIE/DISAGREEMENT between models. Careful analysis required.")
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
            op = pe.edit.edit_type.value.lower() if hasattr(pe.edit, 'edit_type') else "unknown"
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
            op = pe.edit.edit_type.value.lower() if hasattr(pe.edit, 'edit_type') else "unknown"
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
            sev = pe.edit.issue_severity.value if hasattr(pe.edit, 'issue_severity') else "minor"
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
            op = pe.edit.edit_type.value.lower() if hasattr(pe.edit, 'edit_type') else "unknown"
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
            op = pe.edit.edit_type.value if hasattr(pe.edit, 'edit_type') else "EDIT"
            sev = pe.edit.issue_severity.value if hasattr(pe.edit, 'issue_severity') else "minor"
            desc = ""
            if hasattr(pe.edit, 'issue_description') and pe.edit.issue_description:
                desc = pe.edit.issue_description[:100]
            elif hasattr(pe.edit, 'edit_instruction') and pe.edit.edit_instruction:
                desc = pe.edit.edit_instruction[:100]

            lines.append(
                f"[{i}] {op.upper()} (severity={sev}) by {pe.source_model}\n"
                f"    Paragraph: {pe.paragraph_key[:60]}...\n"
                f"    Description: {desc}"
            )

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
            involved_models = [pe.source_model for pe in conflict.involved_edits]
            lines.append(
                f"[Conflict {i + 1}] {conflict.conflict_type.value}\n"
                f"  Paragraph: {conflict.paragraph_key[:60]}...\n"
                f"  Involved models: {', '.join(involved_models)}\n"
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
            system_prompt_section = f"System Prompt: {context.system_prompt[:500]}..."

        # Get history formatted
        history_str = context.layer_history.format_for_prompt()
        if not history_str:
            history_str = "No previous edits in this layer."

        # Get content excerpt
        content_excerpt = context.content_excerpt or context.current_content[:2000]
        if len(context.current_content) > 2000:
            content_excerpt += "\n[... content truncated ...]"

        # Format distribution info
        distribution_info = self._format_distribution_info(
            distribution, context.proposed_edits, context.qa_model_count
        )

        return ARBITER_USER_PROMPT_TEMPLATE.format(
            content_type=context.content_type,
            original_prompt=context.original_prompt[:1000],
            system_prompt_section=system_prompt_section,
            layer_name=context.layer_name,
            layer_criteria=context.layer_criteria[:500],
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

        try:
            response = await self.ai_service.generate_content(
                prompt=prompt,
                model=selected_model,
                system_prompt=ARBITER_SYSTEM_PROMPT,
                max_tokens=config.ARBITER_MAX_TOKENS,
                temperature=config.ARBITER_TEMPERATURE,
                response_format={"type": "json_object"}
            )

            # Parse JSON response
            return self._parse_ai_response_json(response)

        except Exception as e:
            self._logger.error(f"Arbiter AI call failed: {e}")
            # Return empty response - will fall back to algorithmic resolution
            return {
                "reasoning": f"AI resolution failed: {str(e)}",
                "decisions": [],
                "conflicts_resolved": []
            }

    def _parse_ai_response_json(self, response: str) -> Dict[str, Any]:
        """
        Parse AI response JSON string into dict.

        Args:
            response: JSON string from AI

        Returns:
            Parsed dict, or empty dict on error
        """
        import json_utils as json

        # Handle response that might be wrapped in markdown code blocks
        text = response.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        try:
            return json.loads(text)
        except Exception as e:
            self._logger.warning(f"Failed to parse Arbiter response as JSON: {e}")
            return {
                "reasoning": "Failed to parse AI response",
                "decisions": [],
                "conflicts_resolved": []
            }

    def _parse_arbiter_response(
        self,
        ai_response: Dict[str, Any],
        proposed_edits: List[ProposedEdit],
        conflicts: List[ConflictInfo]
    ) -> List[ArbiterEditDecision]:
        """
        Convert AI response into ArbiterEditDecision objects.

        Args:
            ai_response: Parsed AI response dict
            proposed_edits: Original proposed edits
            conflicts: Detected conflicts

        Returns:
            List of ArbiterEditDecision objects
        """
        decisions: List[ArbiterEditDecision] = []
        ai_decisions = ai_response.get("decisions", [])

        # Create a map of edit index to conflict
        edit_to_conflict: Dict[int, ConflictInfo] = {}
        for conflict in conflicts:
            for pe in conflict.involved_edits:
                try:
                    idx = proposed_edits.index(pe)
                    edit_to_conflict[idx] = conflict
                except ValueError:
                    pass

        # Process each proposed edit
        for i, pe in enumerate(proposed_edits):
            # Find AI decision for this edit
            ai_decision = None
            for d in ai_decisions:
                if d.get("edit_index") == i:
                    ai_decision = d
                    break

            if ai_decision:
                decision_str = ai_decision.get("decision", "apply").lower()
                reason = ai_decision.get("reason", "AI decision")

                if decision_str == "discard":
                    decision = ArbiterDecision.DISCARD
                elif decision_str == "merge":
                    decision = ArbiterDecision.MERGE
                else:
                    decision = ArbiterDecision.APPLY
            else:
                # No explicit AI decision - default to apply
                decision = ArbiterDecision.APPLY
                reason = "No conflict / default apply"

            decisions.append(ArbiterEditDecision(
                edit=pe.edit,
                decision=decision,
                reason=reason,
                source_model=pe.source_model,
                conflict_resolved=edit_to_conflict.get(i)
            ))

        return decisions

    def _resolve_algorithmically(
        self,
        proposed_edits: List[ProposedEdit],
        conflicts: List[ConflictInfo]
    ) -> List[ArbiterEditDecision]:
        """
        Resolve conflicts without AI using algorithmic rules.

        This is used when AI is not needed (no conflicts) or as fallback.

        Rules:
        1. Higher severity wins
        2. DELETE wins over MODIFY (safer to remove than change incorrectly)
        3. First proposer wins for redundancy

        Args:
            proposed_edits: All proposed edits
            conflicts: Detected conflicts

        Returns:
            List of ArbiterEditDecision objects
        """
        decisions: List[ArbiterEditDecision] = []
        processed_paragraphs: Dict[str, ArbiterEditDecision] = {}

        # Build conflict map
        conflicted_paragraphs: set = set()
        for conflict in conflicts:
            conflicted_paragraphs.add(conflict.paragraph_key)

        # Sort edits by severity (higher first) and confidence
        severity_order = {"critical": 3, "major": 2, "minor": 1}
        sorted_edits = sorted(
            proposed_edits,
            key=lambda pe: (
                severity_order.get(
                    pe.edit.issue_severity.value if hasattr(pe.edit, 'issue_severity') else "minor",
                    1
                ),
                getattr(pe.edit, 'confidence', 1.0)
            ),
            reverse=True
        )

        for pe in sorted_edits:
            key = pe.paragraph_key

            if key in processed_paragraphs:
                # Already processed - this is a conflicting edit
                decisions.append(ArbiterEditDecision(
                    edit=pe.edit,
                    decision=ArbiterDecision.DISCARD,
                    reason=f"Conflict with higher-priority edit from {processed_paragraphs[key].source_model}",
                    source_model=pe.source_model,
                    conflict_resolved=None
                ))
            else:
                # First edit for this paragraph - apply it
                decision = ArbiterEditDecision(
                    edit=pe.edit,
                    decision=ArbiterDecision.APPLY,
                    reason="Highest priority edit for this paragraph",
                    source_model=pe.source_model,
                    conflict_resolved=None
                )
                decisions.append(decision)
                processed_paragraphs[key] = decision

        return decisions

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
        proposed_edits = context.proposed_edits

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

        # ALWAYS call AI to verify edits (no passthrough)
        self._logger.info(f"Verifying {len(proposed_edits)} edit(s) with AI ({selected_model})...")
        ai_response = await self._resolve_with_ai(context, conflicts, distribution, selected_model)

        if ai_response.get("decisions"):
            # Parse AI response
            decisions = self._parse_arbiter_response(
                ai_response, proposed_edits, conflicts
            )
            reasoning = ai_response.get("reasoning", "AI-verified edits")
        else:
            # AI failed - fall back to algorithmic
            self._logger.warning("AI verification failed, using algorithmic fallback")
            decisions = self._resolve_algorithmically(proposed_edits, conflicts)
            reasoning = "Algorithmic resolution (AI fallback)"

        # Build result
        edits_to_apply = [d.edit for d in decisions if d.decision == ArbiterDecision.APPLY]
        edits_discarded = [
            {
                "edit_type": d.edit.edit_type.value if hasattr(d.edit, 'edit_type') else "unknown",
                "source_model": d.source_model,
                "reason": d.reason,
                "paragraph_key": _get_paragraph_key_for_history(d.edit)[:80]
            }
            for d in decisions if d.decision != ArbiterDecision.APPLY
        ]

        # Build round record for history
        round_record = {
            "proposed_count": len(proposed_edits),
            "conflicts_detected": num_conflicts,
            "distribution": distribution.value,
            "model_used": selected_model,
            "escalated_to_gran_sabio": escalated,
            "applied_count": len(edits_to_apply),
            "discarded_count": len(edits_discarded),
            "decisions": [
                {
                    "decision": d.decision.value,
                    "source_model": d.source_model,
                    "reason": d.reason[:100]
                }
                for d in decisions
            ]
        }

        conflicts_resolved = sum(
            1 for d in decisions
            if d.conflict_resolved is not None and d.decision == ArbiterDecision.APPLY
        )

        return ArbiterResult(
            edits_to_apply=edits_to_apply,
            edits_discarded=edits_discarded,
            conflicts_found=num_conflicts,
            conflicts_resolved=conflicts_resolved,
            round_record=round_record,
            arbiter_reasoning=reasoning,
            distribution=distribution.value,
            escalated_to_gran_sabio=escalated,
            model_used=selected_model
        )
