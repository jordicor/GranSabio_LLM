"""
Structural paragraph and sentence IDs for smart-edit targeting.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import re
import unicodedata
from typing import Dict, List, Optional, Sequence, Tuple


SENTENCE_RE = re.compile(r"[^.!?]+[.!?]+|[^.!?]+$", re.MULTILINE)


@dataclass(frozen=True)
class TextNode:
    """One addressable paragraph or sentence."""

    node_id: str
    kind: str
    start: int
    end: int
    text: str
    parent_id: Optional[str] = None


@dataclass(frozen=True)
class TargetSpan:
    """Resolved editable span for one or more IDs."""

    node_ids: List[str]
    start: int
    end: int
    text: str
    quote_match: bool
    quote_score: float
    evidence_quote: str = ""


@dataclass(frozen=True)
class TargetResolution:
    """Result of resolving target IDs against a draft."""

    spans: List[TargetSpan]
    invalid_ids: List[str]
    quote_match: bool
    quote_score: float
    rejected_reason: Optional[str] = None


class SegmentMap:
    """Map paragraph and sentence IDs to draft spans."""

    def __init__(self, text: str, nodes: Sequence[TextNode]) -> None:
        self.text = text
        self.nodes = list(nodes)
        self.by_id: Dict[str, TextNode] = {node.node_id: node for node in nodes}

    def render_for_prompt(self) -> str:
        """Render an XML-like view with stable structural IDs."""

        lines = ["<draft_map>"]
        paragraphs = [node for node in self.nodes if node.kind == "paragraph"]
        for paragraph in paragraphs:
            lines.append(f'  <p id="{paragraph.node_id}">')
            sentences = [
                node
                for node in self.nodes
                if node.kind == "sentence" and node.parent_id == paragraph.node_id
            ]
            if sentences:
                for sentence in sentences:
                    lines.append(
                        f'    <s id="{sentence.node_id}">{_escape_text(sentence.text.strip())}</s>'
                    )
            else:
                lines.append(f"    {_escape_text(paragraph.text.strip())}")
            lines.append("  </p>")
        lines.append("</draft_map>")
        return "\n".join(lines)

    def resolve_target_ids(
        self,
        target_ids: Sequence[str],
        evidence_quote: str = "",
    ) -> TargetResolution:
        """Resolve IDs and softly verify the evidence quote when present."""

        ids = _dedupe([target_id.strip() for target_id in target_ids if target_id and target_id.strip()])
        invalid_ids = [target_id for target_id in ids if target_id not in self.by_id]
        nodes = [self.by_id[target_id] for target_id in ids if target_id in self.by_id]
        quote = (evidence_quote or "").strip()

        if not nodes:
            return TargetResolution(
                spans=[],
                invalid_ids=invalid_ids,
                quote_match=False,
                quote_score=0.0,
                rejected_reason="no_valid_target_ids",
            )

        quote_score = _quote_score(quote, " ".join(node.text for node in nodes)) if quote else 1.0
        quote_match = quote_score >= 0.78
        if quote and not quote_match:
            return TargetResolution(
                spans=[],
                invalid_ids=invalid_ids,
                quote_match=False,
                quote_score=quote_score,
                rejected_reason="evidence_quote_mismatch",
            )

        spans = self._group_nodes(nodes, quote_match=quote_match, quote_score=quote_score, quote=quote)
        return TargetResolution(
            spans=spans,
            invalid_ids=invalid_ids,
            quote_match=quote_match,
            quote_score=quote_score,
        )

    def _group_nodes(
        self,
        nodes: Sequence[TextNode],
        *,
        quote_match: bool,
        quote_score: float,
        quote: str,
    ) -> List[TargetSpan]:
        paragraphs = [node for node in nodes if node.kind == "paragraph"]
        paragraph_ids = {paragraph.node_id for paragraph in paragraphs}
        sentences = [node for node in nodes if node.kind == "sentence"]

        spans: List[TargetSpan] = []
        for paragraph in paragraphs:
            spans.append(
                TargetSpan(
                    node_ids=[paragraph.node_id],
                    start=paragraph.start,
                    end=paragraph.end,
                    text=self.text[paragraph.start:paragraph.end],
                    quote_match=quote_match,
                    quote_score=quote_score,
                    evidence_quote=quote,
                )
            )

        by_parent: Dict[str, List[TextNode]] = {}
        for sentence in sentences:
            if sentence.parent_id in paragraph_ids:
                continue
            by_parent.setdefault(sentence.parent_id or "", []).append(sentence)

        for grouped in by_parent.values():
            ordered = sorted(grouped, key=lambda node: node.start)
            for contiguous_group in _contiguous_sentence_groups(self.text, ordered):
                start = min(node.start for node in contiguous_group)
                end = max(node.end for node in contiguous_group)
                spans.append(
                    TargetSpan(
                        node_ids=[node.node_id for node in contiguous_group],
                        start=start,
                        end=end,
                        text=self.text[start:end],
                        quote_match=quote_match,
                        quote_score=quote_score,
                        evidence_quote=quote,
                    )
                )

        return sorted(spans, key=lambda span: span.start, reverse=True)


def sentence_spans(text: str) -> List[Tuple[int, int, str]]:
    """Return approximate sentence spans."""

    spans: List[Tuple[int, int, str]] = []
    for match in SENTENCE_RE.finditer(text):
        raw = match.group(0)
        if raw and raw.strip():
            spans.append((match.start(), match.end(), raw))
    return spans


def paragraph_spans(text: str) -> List[Tuple[int, int, str]]:
    """Return paragraph spans while preserving original offsets."""

    spans: List[Tuple[int, int, str]] = []
    for match in re.finditer(r"[^\n]+(?:\n(?!\n)[^\n]+)*", text):
        raw = match.group(0)
        if raw.strip():
            spans.append((match.start(), match.end(), raw))
    return spans


def build_segment_map(text: str) -> SegmentMap:
    """Build paragraph and sentence IDs from text."""

    nodes: List[TextNode] = []
    for paragraph_index, (p_start, p_end, paragraph_text) in enumerate(paragraph_spans(text), start=1):
        paragraph_id = f"p{paragraph_index}"
        nodes.append(
            TextNode(
                node_id=paragraph_id,
                kind="paragraph",
                start=p_start,
                end=p_end,
                text=paragraph_text,
            )
        )
        for sentence_index, (s_start, s_end, _) in enumerate(sentence_spans(paragraph_text), start=1):
            absolute_start = p_start + s_start
            absolute_end = p_start + s_end
            raw_sentence = text[absolute_start:absolute_end]
            leading = len(raw_sentence) - len(raw_sentence.lstrip())
            trailing_end = absolute_start + len(raw_sentence.rstrip())
            absolute_start += leading
            absolute_end = max(absolute_start, trailing_end)
            nodes.append(
                TextNode(
                    node_id=f"{paragraph_id}s{sentence_index}",
                    kind="sentence",
                    start=absolute_start,
                    end=absolute_end,
                    text=text[absolute_start:absolute_end],
                    parent_id=paragraph_id,
                )
            )
    return SegmentMap(text=text, nodes=nodes)


def _contiguous_sentence_groups(text: str, ordered: Sequence[TextNode]) -> List[List[TextNode]]:
    groups: List[List[TextNode]] = []
    current: List[TextNode] = []
    previous: Optional[TextNode] = None
    for node in ordered:
        if previous is not None and text[previous.end:node.start].strip():
            groups.append(current)
            current = []
        current.append(node)
        previous = node
    if current:
        groups.append(current)
    return groups


def _normalize_for_quote(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text.lower())
    without_marks = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", re.sub(r"[^\w]+", " ", without_marks, flags=re.UNICODE)).strip()


def _tokens_in_order(needle: Sequence[str], haystack: Sequence[str]) -> bool:
    position = 0
    for token in haystack:
        if position < len(needle) and token == needle[position]:
            position += 1
    return position == len(needle)


def _quote_score(quote: str, target_text: str) -> float:
    normalized_quote = _normalize_for_quote(quote)
    normalized_target = _normalize_for_quote(target_text)
    if not normalized_quote or not normalized_target:
        return 0.0
    if normalized_quote in normalized_target:
        return 1.0

    quote_tokens = normalized_quote.split()
    target_tokens = normalized_target.split()
    if not quote_tokens or not target_tokens:
        return 0.0

    window_sizes = {len(quote_tokens)}
    if len(quote_tokens) > 3:
        window_sizes.update({len(quote_tokens) - 1, len(quote_tokens) + 1})

    best = 0.0
    for size in sorted(size for size in window_sizes if size > 0):
        if size > len(target_tokens):
            continue
        for index in range(0, len(target_tokens) - size + 1):
            window = " ".join(target_tokens[index:index + size])
            best = max(best, SequenceMatcher(None, normalized_quote, window).ratio())
            if best >= 0.92:
                return best

    if _tokens_in_order(quote_tokens, target_tokens):
        best = max(best, 0.82)
    return best


def _dedupe(values: Sequence[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _escape_text(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
