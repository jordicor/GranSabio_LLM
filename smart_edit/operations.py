"""
Smart Edit Operations - Core text editing functionality.

This module provides the SmartTextEditor class with:
- Direct operations: delete, insert, replace, format (no AI needed)
- AI-assisted operations: rephrase, improve, fix, apply_edit (require LLM)
- Batch operations: apply multiple edits in sequence

Phase 0: Skeleton implementation
Phase 1: Direct operations implementation
Phase 2: Core AI editing operations (apply_edit)
"""

from __future__ import annotations

import difflib
import re
import time
from typing import TYPE_CHECKING, Callable, List, Literal, Optional, Tuple, Union

from .models import (
    ChangeDetail,
    EditOperation,
    EditResult,
    FormatType,
    OperationType,
    TargetMode,
    TargetScope,
    TargetSpec,
    TextTarget,
)
from .prompts import build_edit_prompt, get_operation_instruction
from .text_utils import clean_edited_text, remove_common_ai_artifacts
from .locators import (
    locate_by_markers,
    locate_by_word_indices,
    build_word_map,
)

if TYPE_CHECKING:
    from ai_service import AIService


class SmartTextEditor:
    """
    Standalone text editor with direct and AI-assisted operations.

    Direct operations (no AI, instant):
        - delete(): Remove exact text or text at position
        - insert(): Add text before/after target
        - replace(): Replace text with new content
        - format(): Apply markdown formatting

    AI-assisted operations (require ai_service):
        - rephrase(): Reword maintaining meaning
        - improve(): Enhance based on criteria
        - fix(): Fix grammar, style, or tone

    Example:
        # Direct operations only
        editor = SmartTextEditor()
        result = editor.delete(content, "unwanted text")

        # With AI support
        editor = SmartTextEditor(ai_service=get_ai_service())
        result = await editor.rephrase(content, target_text)
    """

    def __init__(self, ai_service: Optional["AIService"] = None):
        """
        Initialize the editor.

        Args:
            ai_service: Optional AI service for AI-assisted operations.
                       Not required for direct operations.
        """
        self.ai_service = ai_service

    # =========================================================================
    # TEXT LOCATION UTILITIES
    # =========================================================================

    def locate(
        self, content: str, target: TargetSpec
    ) -> Optional[Tuple[int, int]]:
        """
        Find the position of target text in content.

        Args:
            content: The text to search in
            target: What to find (string, position tuple, or TextTarget)

        Returns:
            Tuple of (start, end) positions, or None if not found
        """
        # Normalize target to TextTarget
        if isinstance(target, str):
            target = TextTarget(mode=TargetMode.EXACT, value=target)
        elif isinstance(target, tuple):
            # Position tuple - return as-is if valid
            start, end = target
            if 0 <= start <= end <= len(content):
                return (start, end)
            return None

        # Handle different target modes
        if target.mode == TargetMode.EXACT:
            return self._locate_exact(content, target)
        elif target.mode == TargetMode.POSITION:
            start, end = target.value
            if 0 <= start <= end <= len(content):
                return (start, end)
            return None
        elif target.mode == TargetMode.REGEX:
            return self._locate_regex(content, target)
        elif target.mode == TargetMode.PARAGRAPH:
            return self._locate_paragraph(content, target.value)
        elif target.mode == TargetMode.SENTENCE:
            return self._locate_sentence(content, target)
        elif target.mode == TargetMode.MARKER:
            return self._locate_by_markers(content, target)
        elif target.mode == TargetMode.WORD_INDEX:
            return self._locate_by_word_indices(content, target)

        return None

    def _locate_exact(
        self, content: str, target: TextTarget
    ) -> Optional[Tuple[int, int]]:
        """Locate exact string match."""
        search_content = content if target.case_sensitive else content.lower()
        search_value = (
            target.value if target.case_sensitive else target.value.lower()
        )

        if target.occurrence == -1:
            # Find all occurrences - return first one
            # (batch operations should handle multiple)
            pos = search_content.find(search_value)
        elif target.occurrence > 0:
            # Find Nth occurrence
            pos = -1
            start = 0
            for _ in range(target.occurrence):
                pos = search_content.find(search_value, start)
                if pos == -1:
                    break
                start = pos + 1
        else:
            return None

        if pos == -1:
            return None

        return (pos, pos + len(target.value))

    def _locate_regex(
        self, content: str, target: TextTarget
    ) -> Optional[Tuple[int, int]]:
        """Locate text matching regex pattern."""
        flags = 0 if target.case_sensitive else re.IGNORECASE
        pattern = re.compile(target.value, flags)

        matches = list(pattern.finditer(content))
        if not matches:
            return None

        if target.occurrence == -1:
            # Return first match (for -1 = all, caller handles iteration)
            match = matches[0]
        elif 0 < target.occurrence <= len(matches):
            match = matches[target.occurrence - 1]
        else:
            return None

        return (match.start(), match.end())

    def _locate_paragraph(
        self, content: str, index: int
    ) -> Optional[Tuple[int, int]]:
        """Locate paragraph by index (0-based)."""
        # Split by double newlines or single newlines followed by blank line
        paragraphs = re.split(r"\n\s*\n", content)

        if index < 0 or index >= len(paragraphs):
            return None

        # Calculate position by reconstructing
        pos = 0
        for i, para in enumerate(paragraphs):
            if i == index:
                return (pos, pos + len(para))
            pos += len(para)
            # Account for separator
            if i < len(paragraphs) - 1:
                # Find actual separator length
                remaining = content[pos:]
                sep_match = re.match(r"\n\s*\n", remaining)
                if sep_match:
                    pos += sep_match.end()

        return None

    def _locate_sentence(
        self, content: str, target: TextTarget
    ) -> Optional[Tuple[int, int]]:
        """Locate sentence by index within scope."""
        # Simple sentence splitting (can be improved)
        sentence_pattern = r"[.!?]+\s+"
        sentences = re.split(sentence_pattern, content)

        index = target.value if isinstance(target.value, int) else 0
        if index < 0 or index >= len(sentences):
            return None

        # Calculate position
        pos = 0
        for i, sent in enumerate(sentences):
            if i == index:
                end = pos + len(sent)
                # Include trailing punctuation if present
                if end < len(content):
                    punct_match = re.match(r"[.!?]+\s*", content[end:])
                    if punct_match:
                        end += punct_match.end()
                return (pos, end)
            pos += len(sent)
            # Account for split pattern
            remaining = content[pos:]
            sep_match = re.match(sentence_pattern, remaining)
            if sep_match:
                pos += sep_match.end()

        return None

    def _locate_by_markers(
        self, content: str, target: TextTarget
    ) -> Optional[Tuple[int, int]]:
        """
        Locate text using phrase markers (paragraph_start/paragraph_end).

        This is the primary localization method used by the QA system.

        Args:
            content: Text to search in
            target: TextTarget with mode=MARKER and value={"start": ..., "end": ...}

        Returns:
            Tuple of (start, end) positions or None if not found
        """
        if not isinstance(target.value, dict):
            return None

        paragraph_start = target.value.get("start", "")
        paragraph_end = target.value.get("end", "")

        return locate_by_markers(
            content,
            paragraph_start,
            paragraph_end,
            case_sensitive=target.case_sensitive,
        )

    def _locate_by_word_indices(
        self, content: str, target: TextTarget
    ) -> Optional[Tuple[int, int]]:
        """
        Locate text using word indices from word_map.

        This is the fallback method when phrase markers aren't unique enough.

        Args:
            content: Text to search in
            target: TextTarget with mode=WORD_INDEX and
                   value={"start_idx": ..., "end_idx": ...}

        Returns:
            Tuple of (start, end) positions or None if not found
        """
        if not isinstance(target.value, dict):
            return None

        start_idx = target.value.get("start_idx")
        end_idx = target.value.get("end_idx")

        if start_idx is None or end_idx is None:
            return None

        # Use word_map from target if available, otherwise build it
        word_map = target.word_map
        if word_map is None:
            word_map, _ = build_word_map(content)

        return locate_by_word_indices(
            content,
            start_idx,
            end_idx,
            word_map,
        )

    # =========================================================================
    # DIRECT OPERATIONS (No AI Required) - Phase 1
    # =========================================================================

    def delete(
        self,
        content: str,
        target: TargetSpec,
    ) -> EditResult:
        """
        Delete text from content.

        Args:
            content: The text to edit
            target: Text to delete (string, position tuple, or TextTarget)

        Returns:
            EditResult with before/after content and change details

        Example:
            result = editor.delete("hello world", "world")
            # result.content_after == "hello "
        """
        start_time = time.perf_counter()

        # Find target position
        position = self.locate(content, target)
        if position is None:
            return EditResult(
                success=False,
                content_before=content,
                content_after=content,
                errors=["Target not found in content"],
                execution_time_ms=int((time.perf_counter() - start_time) * 1000),
            )

        start, end = position
        removed_text = content[start:end]

        # Apply deletion
        new_content = content[:start] + content[end:]

        # Generate diff
        diff = self._generate_diff(content, new_content)

        return EditResult(
            success=True,
            content_before=content,
            content_after=new_content,
            changes=[
                ChangeDetail(
                    position_start=start,
                    position_end=end,
                    removed_text=removed_text,
                    inserted_text="",
                )
            ],
            diff=diff,
            execution_time_ms=int((time.perf_counter() - start_time) * 1000),
        )

    def insert(
        self,
        content: str,
        text: str,
        target: TargetSpec,
        where: Literal["before", "after"] = "after",
    ) -> EditResult:
        """
        Insert text before or after target.

        Args:
            content: The text to edit
            text: Text to insert
            target: Where to insert (string to find, position, or TextTarget)
            where: "before" or "after" the target

        Returns:
            EditResult with before/after content and change details

        Example:
            result = editor.insert("hello world", " beautiful", "hello", "after")
            # result.content_after == "hello beautiful world"
        """
        start_time = time.perf_counter()

        # Find target position
        position = self.locate(content, target)
        if position is None:
            return EditResult(
                success=False,
                content_before=content,
                content_after=content,
                errors=["Target not found in content"],
                execution_time_ms=int((time.perf_counter() - start_time) * 1000),
            )

        start, end = position
        insert_pos = start if where == "before" else end

        # Apply insertion
        new_content = content[:insert_pos] + text + content[insert_pos:]

        # Generate diff
        diff = self._generate_diff(content, new_content)

        return EditResult(
            success=True,
            content_before=content,
            content_after=new_content,
            changes=[
                ChangeDetail(
                    position_start=insert_pos,
                    position_end=insert_pos,
                    removed_text="",
                    inserted_text=text,
                )
            ],
            diff=diff,
            execution_time_ms=int((time.perf_counter() - start_time) * 1000),
        )

    def replace(
        self,
        content: str,
        old: TargetSpec,
        new: str,
        count: int = -1,
    ) -> EditResult:
        """
        Replace text in content.

        Args:
            content: The text to edit
            old: Text to replace (string, position, or TextTarget)
            new: Replacement text
            count: Number of occurrences to replace (-1 = all)

        Returns:
            EditResult with before/after content and change details

        Example:
            result = editor.replace("the cat and the cat", "cat", "dog")
            # result.content_after == "the dog and the dog"
        """
        start_time = time.perf_counter()
        changes = []

        # Handle simple string replacement
        if isinstance(old, str) and count == -1:
            # Replace all occurrences
            new_content = content.replace(old, new)
            if new_content == content:
                return EditResult(
                    success=False,
                    content_before=content,
                    content_after=content,
                    errors=["Target not found in content"],
                    execution_time_ms=int(
                        (time.perf_counter() - start_time) * 1000
                    ),
                )

            # Track all changes
            pos = 0
            temp_content = content
            while old in temp_content:
                idx = temp_content.find(old)
                actual_pos = pos + idx
                changes.append(
                    ChangeDetail(
                        position_start=actual_pos,
                        position_end=actual_pos + len(old),
                        removed_text=old,
                        inserted_text=new,
                    )
                )
                # Adjust for next search accounting for replacement length diff
                pos = actual_pos + len(new)
                temp_content = temp_content[idx + len(old) :]

            diff = self._generate_diff(content, new_content)
            return EditResult(
                success=True,
                content_before=content,
                content_after=new_content,
                changes=changes,
                diff=diff,
                execution_time_ms=int((time.perf_counter() - start_time) * 1000),
            )

        # Handle single or counted replacements
        if isinstance(old, str):
            target = TextTarget(mode=TargetMode.EXACT, value=old)
        elif isinstance(old, tuple):
            target = TextTarget(mode=TargetMode.POSITION, value=old)
        else:
            target = old

        # For counted replacements, iterate
        new_content = content
        replacements_made = 0
        offset = 0

        while count == -1 or replacements_made < count:
            position = self.locate(new_content, target)
            if position is None:
                break

            start, end = position
            old_text = new_content[start:end]

            changes.append(
                ChangeDetail(
                    position_start=start + offset,
                    position_end=end + offset,
                    removed_text=old_text,
                    inserted_text=new,
                )
            )

            new_content = new_content[:start] + new + new_content[end:]
            replacements_made += 1
            offset += len(new) - len(old_text)

            # POSITION mode: single replacement at fixed location, exit immediately
            # Without this, the loop would run infinitely because locate() always
            # returns the same valid position for POSITION mode targets
            if isinstance(target, TextTarget) and target.mode == TargetMode.POSITION:
                break

            # For exact matches with occurrence=1, we're done after one
            if (
                isinstance(target, TextTarget)
                and target.mode == TargetMode.EXACT
                and target.occurrence == 1
            ):
                if count == -1:
                    # Continue finding more
                    target = TextTarget(
                        mode=TargetMode.EXACT,
                        value=target.value,
                        occurrence=1,
                        case_sensitive=target.case_sensitive,
                    )
                else:
                    break

        if not changes:
            return EditResult(
                success=False,
                content_before=content,
                content_after=content,
                errors=["Target not found in content"],
                execution_time_ms=int((time.perf_counter() - start_time) * 1000),
            )

        diff = self._generate_diff(content, new_content)
        return EditResult(
            success=True,
            content_before=content,
            content_after=new_content,
            changes=changes,
            diff=diff,
            execution_time_ms=int((time.perf_counter() - start_time) * 1000),
        )

    def format(
        self,
        content: str,
        target: TargetSpec,
        format_type: Union[str, FormatType],
    ) -> EditResult:
        """
        Apply markdown formatting to target text.

        Args:
            content: The text to edit
            target: Text to format
            format_type: Type of formatting ("bold", "italic", "code", "strikethrough")

        Returns:
            EditResult with before/after content and change details

        Example:
            result = editor.format("important word", "important", "bold")
            # result.content_after == "**important** word"
        """
        start_time = time.perf_counter()

        # Convert string to FormatType
        if isinstance(format_type, str):
            try:
                format_type = FormatType(format_type.lower())
            except ValueError:
                return EditResult(
                    success=False,
                    content_before=content,
                    content_after=content,
                    errors=[
                        f"Invalid format type: {format_type}. "
                        f"Valid types: {[f.value for f in FormatType]}"
                    ],
                    execution_time_ms=int(
                        (time.perf_counter() - start_time) * 1000
                    ),
                )

        # Find target position
        position = self.locate(content, target)
        if position is None:
            return EditResult(
                success=False,
                content_before=content,
                content_after=content,
                errors=["Target not found in content"],
                execution_time_ms=int((time.perf_counter() - start_time) * 1000),
            )

        start, end = position
        original_text = content[start:end]

        # Determine markers
        markers = {
            FormatType.BOLD: ("**", "**"),
            FormatType.ITALIC: ("*", "*"),
            FormatType.CODE: ("`", "`"),
            FormatType.STRIKETHROUGH: ("~~", "~~"),
            FormatType.UNDERLINE: ("<u>", "</u>"),
        }

        prefix, suffix = markers[format_type]
        formatted_text = f"{prefix}{original_text}{suffix}"

        # Apply formatting
        new_content = content[:start] + formatted_text + content[end:]

        diff = self._generate_diff(content, new_content)

        return EditResult(
            success=True,
            content_before=content,
            content_after=new_content,
            changes=[
                ChangeDetail(
                    position_start=start,
                    position_end=end,
                    removed_text=original_text,
                    inserted_text=formatted_text,
                )
            ],
            diff=diff,
            execution_time_ms=int((time.perf_counter() - start_time) * 1000),
        )

    # =========================================================================
    # AI-ASSISTED OPERATIONS - Phase 2
    # =========================================================================

    async def _call_ai(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.2,
        usage_callback: Optional[Callable] = None,
    ) -> str:
        """
        Call AI service to generate text.

        Args:
            prompt: The prompt to send
            model: AI model to use (defaults to gpt-4o-mini)
            temperature: Generation temperature
            usage_callback: Optional callback for tracking token usage

        Returns:
            AI response text

        Raises:
            ValueError: If AI service not configured
        """
        if self.ai_service is None:
            raise ValueError(
                "AI service required for this operation. "
                "Initialize SmartTextEditor with ai_service parameter."
            )

        model = model or "gpt-4o-mini"

        # Call AI service using generate_content (the actual method name)
        response = await self.ai_service.generate_content(
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=4096,
            usage_callback=usage_callback,
        )

        # generate_content returns the content string directly
        if isinstance(response, str):
            return response
        # Handle unexpected response types
        elif hasattr(response, "content"):
            return response.content
        elif hasattr(response, "text"):
            return response.text
        else:
            return str(response)

    async def apply_edit(
        self,
        content: str,
        target: TargetSpec,
        instruction: str,
        context: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.2,
        preserve_length: bool = True,
        clean_output: bool = True,
        usage_callback: Optional[Callable] = None,
    ) -> EditResult:
        """
        Apply an edit to target text using AI with a specific instruction.

        This is the CORE method for AI-assisted editing. Unlike rephrase/improve/fix
        which use generic prompts, this method uses YOUR instruction from external
        sources (QA evaluation, user input, etc.).

        Args:
            content: Full text to edit
            target: Text segment to edit (string, position tuple, or TextTarget)
            instruction: Specific instruction for the edit (from QA, user, etc.)
            context: Optional full document context for style preservation
            model: AI model to use (defaults to gpt-4o-mini)
            temperature: Generation temperature (default 0.2 for consistency)
            preserve_length: Try to maintain similar length (default True)
            clean_output: Clean AI artifacts from output (default True)
            usage_callback: Optional callback for tracking token usage

        Returns:
            EditResult with edited content

        Example (standalone):
            result = await editor.apply_edit(
                content=biography,
                target="Maria was born in Barcelona in 1985",
                instruction="Correct: birth year should be 1987 per source document"
            )

        Example (from QA):
            result = await editor.apply_edit(
                content=current_content,
                target=TextTarget(mode=TargetMode.EXACT, value=paragraph_text),
                instruction=edit_range.edit_instruction
            )
        """
        start_time = time.perf_counter()

        # Check AI service
        if self.ai_service is None:
            return EditResult(
                success=False,
                content_before=content,
                content_after=content,
                errors=["AI service required for apply_edit operation"],
                execution_time_ms=int((time.perf_counter() - start_time) * 1000),
            )

        # Find target position
        position = self.locate(content, target)
        if position is None:
            return EditResult(
                success=False,
                content_before=content,
                content_after=content,
                errors=["Target not found in content"],
                execution_time_ms=int((time.perf_counter() - start_time) * 1000),
            )

        start, end = position
        target_text = content[start:end]

        # Build prompt
        prompt = build_edit_prompt(
            target_text=target_text,
            instruction=instruction,
            context=context or content,
            preserve_length=preserve_length,
        )

        try:
            # Call AI
            ai_response = await self._call_ai(
                prompt=prompt,
                model=model,
                temperature=temperature,
                usage_callback=usage_callback,
            )

            # Clean AI response
            edited_text = ai_response.strip()
            if clean_output:
                edited_text = remove_common_ai_artifacts(edited_text)
                edited_text = clean_edited_text(edited_text, original=target_text)

            # Apply edit to content
            new_content = content[:start] + edited_text + content[end:]

            # Generate diff
            diff = self._generate_diff(content, new_content)

            return EditResult(
                success=True,
                content_before=content,
                content_after=new_content,
                changes=[
                    ChangeDetail(
                        position_start=start,
                        position_end=end,
                        removed_text=target_text,
                        inserted_text=edited_text,
                    )
                ],
                diff=diff,
                execution_time_ms=int((time.perf_counter() - start_time) * 1000),
            )

        except Exception as e:
            return EditResult(
                success=False,
                content_before=content,
                content_after=content,
                errors=[f"AI error: {str(e)}"],
                execution_time_ms=int((time.perf_counter() - start_time) * 1000),
            )

    async def rephrase(
        self,
        content: str,
        target: TargetSpec,
        instruction: Optional[str] = None,
        preserve_length: bool = True,
        model: Optional[str] = None,
        temperature: float = 0.2,
    ) -> EditResult:
        """
        Rephrase target text using AI while maintaining meaning.

        Args:
            content: The text to edit
            target: Text to rephrase
            instruction: Additional instructions for rephrasing
            preserve_length: Try to maintain similar length
            model: AI model to use
            temperature: Generation temperature

        Returns:
            EditResult with rephrased content

        Note: Requires ai_service to be set during initialization.
        """
        # Get rephrase instruction
        rephrase_instruction = get_operation_instruction(
            operation="rephrase",
            custom_instruction=instruction,
        )

        return await self.apply_edit(
            content=content,
            target=target,
            instruction=rephrase_instruction,
            model=model,
            temperature=temperature,
            preserve_length=preserve_length,
        )

    async def improve(
        self,
        content: str,
        target: TargetSpec,
        criteria: Optional[List[str]] = None,
        model: Optional[str] = None,
        temperature: float = 0.3,
    ) -> EditResult:
        """
        Improve target text using AI based on criteria.

        Args:
            content: The text to edit
            target: Text to improve
            criteria: Improvement criteria (e.g., ["clarity", "conciseness"])
            model: AI model to use
            temperature: Generation temperature

        Returns:
            EditResult with improved content
        """
        # Get improve instruction
        improve_instruction = get_operation_instruction(
            operation="improve",
            criteria=criteria,
        )

        return await self.apply_edit(
            content=content,
            target=target,
            instruction=improve_instruction,
            model=model,
            temperature=temperature,
            preserve_length=False,  # Improvements may change length
        )

    async def fix(
        self,
        content: str,
        target: TargetSpec,
        fix_type: Literal["grammar", "style", "tone", "all"] = "all",
        model: Optional[str] = None,
        temperature: float = 0.2,
    ) -> EditResult:
        """
        Fix issues in target text using AI.

        Args:
            content: The text to edit
            target: Text to fix
            fix_type: Type of fixes to apply
            model: AI model to use
            temperature: Generation temperature

        Returns:
            EditResult with fixed content
        """
        # Map fix_type to operation
        operation_map = {
            "grammar": "fix_grammar",
            "style": "fix_style",
            "tone": "fix_tone",
            "all": "fix_all",
        }

        operation = operation_map.get(fix_type, "fix_all")
        fix_instruction = get_operation_instruction(operation=operation)

        return await self.apply_edit(
            content=content,
            target=target,
            instruction=fix_instruction,
            model=model,
            temperature=temperature,
            preserve_length=True,
        )

    async def expand(
        self,
        content: str,
        target: TargetSpec,
        model: Optional[str] = None,
        temperature: float = 0.4,
    ) -> EditResult:
        """
        Expand target text with more detail using AI.

        Args:
            content: The text to edit
            target: Text to expand
            model: AI model to use
            temperature: Generation temperature

        Returns:
            EditResult with expanded content
        """
        expand_instruction = get_operation_instruction(operation="expand")

        return await self.apply_edit(
            content=content,
            target=target,
            instruction=expand_instruction,
            model=model,
            temperature=temperature,
            preserve_length=False,  # Expansion increases length
        )

    async def condense(
        self,
        content: str,
        target: TargetSpec,
        model: Optional[str] = None,
        temperature: float = 0.2,
    ) -> EditResult:
        """
        Condense target text to be more concise using AI.

        Args:
            content: The text to edit
            target: Text to condense
            model: AI model to use
            temperature: Generation temperature

        Returns:
            EditResult with condensed content
        """
        condense_instruction = get_operation_instruction(operation="condense")

        return await self.apply_edit(
            content=content,
            target=target,
            instruction=condense_instruction,
            model=model,
            temperature=temperature,
            preserve_length=False,  # Condensing reduces length
        )

    # =========================================================================
    # BATCH OPERATIONS
    # =========================================================================

    def apply_batch(
        self,
        content: str,
        operations: List[EditOperation],
    ) -> EditResult:
        """
        Apply multiple direct operations in sequence.

        Args:
            content: The text to edit
            operations: List of operations to apply

        Returns:
            EditResult with cumulative changes

        Note: AI-assisted operations in the batch will be skipped.
              Use apply_batch_async for mixed operations.
        """
        start_time = time.perf_counter()
        current_content = content
        all_changes = []
        errors = []

        for op in sorted(operations, key=lambda x: x.priority):
            if op.requires_ai:
                errors.append(
                    f"Skipped AI operation {op.id}: use apply_batch_async"
                )
                continue

            result = self._execute_operation(current_content, op)
            if result.success:
                current_content = result.content_after
                all_changes.extend(result.changes)
            else:
                errors.extend(result.errors)

        diff = self._generate_diff(content, current_content)

        return EditResult(
            success=len(errors) == 0 or len(all_changes) > 0,
            content_before=content,
            content_after=current_content,
            changes=all_changes,
            errors=errors,
            diff=diff,
            execution_time_ms=int((time.perf_counter() - start_time) * 1000),
        )

    async def apply_batch_async(
        self,
        content: str,
        operations: List[EditOperation],
    ) -> EditResult:
        """
        Apply multiple operations including AI-assisted ones.

        Args:
            content: The text to edit
            operations: List of operations to apply

        Returns:
            EditResult with cumulative changes
        """
        start_time = time.perf_counter()
        current_content = content
        all_changes = []
        errors = []

        for op in sorted(operations, key=lambda x: x.priority):
            if op.requires_ai:
                result = await self._execute_ai_operation(current_content, op)
            else:
                result = self._execute_operation(current_content, op)

            if result.success:
                current_content = result.content_after
                all_changes.extend(result.changes)
            else:
                errors.extend(result.errors)

        diff = self._generate_diff(content, current_content)

        return EditResult(
            success=len(errors) == 0 or len(all_changes) > 0,
            content_before=content,
            content_after=current_content,
            changes=all_changes,
            errors=errors,
            diff=diff,
            execution_time_ms=int((time.perf_counter() - start_time) * 1000),
        )

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def preview(
        self,
        content: str,
        operations: List[EditOperation],
    ) -> str:
        """
        Preview changes as a diff without applying.

        Args:
            content: The text to edit
            operations: Operations to preview

        Returns:
            Unified diff string showing what would change
        """
        result = self.apply_batch(content, operations)
        return result.diff

    def validate(
        self,
        content: str,
        operations: List[EditOperation],
    ) -> List[str]:
        """
        Validate operations without executing.

        Args:
            content: The text to validate against
            operations: Operations to validate

        Returns:
            List of validation error messages (empty if all valid)
        """
        errors = []

        for op in operations:
            # Check if target can be found
            if op.target:
                position = self.locate(content, op.target)
                if position is None:
                    errors.append(f"Operation {op.id}: target not found")

            # Check AI requirements
            if op.requires_ai and self.ai_service is None:
                errors.append(
                    f"Operation {op.id}: requires AI service but none provided"
                )

            # Check content for insert/replace
            if op.operation_type in (
                OperationType.INSERT_BEFORE,
                OperationType.INSERT_AFTER,
                OperationType.REPLACE,
            ):
                if not op.content and not op.instruction:
                    errors.append(
                        f"Operation {op.id}: requires content or instruction"
                    )

        return errors

    def _execute_operation(
        self, content: str, operation: EditOperation
    ) -> EditResult:
        """Execute a single direct operation."""
        op_type = operation.operation_type

        if op_type == OperationType.DELETE:
            return self.delete(content, operation.target)

        elif op_type == OperationType.INSERT_BEFORE:
            return self.insert(
                content, operation.content or "", operation.target, "before"
            )

        elif op_type == OperationType.INSERT_AFTER:
            return self.insert(
                content, operation.content or "", operation.target, "after"
            )

        elif op_type == OperationType.REPLACE:
            return self.replace(content, operation.target, operation.content or "")

        elif op_type == OperationType.FORMAT:
            format_type = operation.metadata.get("format_type", "bold")
            return self.format(content, operation.target, format_type)

        else:
            return EditResult(
                success=False,
                content_before=content,
                content_after=content,
                errors=[f"Unknown operation type: {op_type}"],
            )

    async def _execute_ai_operation(
        self, content: str, operation: EditOperation
    ) -> EditResult:
        """Execute a single AI-assisted operation."""
        op_type = operation.operation_type

        if op_type == OperationType.REPHRASE:
            return await self.rephrase(
                content, operation.target, operation.instruction
            )

        elif op_type == OperationType.IMPROVE:
            criteria = operation.metadata.get("criteria", [])
            return await self.improve(content, operation.target, criteria)

        elif op_type in (
            OperationType.FIX_GRAMMAR,
            OperationType.FIX_STYLE,
        ):
            fix_type = "grammar" if op_type == OperationType.FIX_GRAMMAR else "style"
            return await self.fix(content, operation.target, fix_type)

        elif op_type == OperationType.EXPAND:
            return await self.expand(content, operation.target)

        elif op_type == OperationType.CONDENSE:
            return await self.condense(content, operation.target)

        elif operation.instruction:
            # Generic AI edit with custom instruction
            return await self.apply_edit(
                content, operation.target, operation.instruction
            )

        else:
            return EditResult(
                success=False,
                content_before=content,
                content_after=content,
                errors=[f"AI operation not implemented: {op_type}"],
            )

    def _generate_diff(self, before: str, after: str) -> str:
        """Generate unified diff between two texts."""
        before_lines = before.splitlines(keepends=True)
        after_lines = after.splitlines(keepends=True)

        diff = difflib.unified_diff(
            before_lines,
            after_lines,
            fromfile="before",
            tofile="after",
            lineterm="",
        )

        return "".join(diff)
