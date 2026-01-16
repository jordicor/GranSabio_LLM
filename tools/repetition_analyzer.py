#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Repetition Analyzer (exact n-gram repetition analytics with distances, clustering, diagnostics)
Version: 2.3.3
Author: ChatGPT (GPT-5 Pro)
License: MIT

What's new in 2.3.3
-------------------
- Feature: Added stopword filtering support via 'language' and 'filter_stop_words' config options.
  When enabled, stopwords are excluded from summaries (top_by_count, top_by_ratio).
  For n=1, pure stopwords are filtered. For n>1, phrases starting or ending with stopwords are filtered.
- New meta fields: language_hint, stop_words_filtered, stop_words_count.

What's new in 2.3.2
-------------------
- Fix: Paragraph/block detection now uses the exact same normalized text domain as token spans
  to avoid CRLF vs. LF mismatches. compute_token_segments_from_blanklines() now runs on
  _remove_invisible_control(text), aligning character coordinates with token_spans.

What's new in 2.3.1
-------------------
- Hardened config normalization & validation:
  * Clamps for negative/contradictory numeric values (min_n, max_n, min_count, thresholds, *top_k, etc.).
  * Safe defaults for invalid enum choices (tokenizer, punct_policy, summary_mode, details, clusters_top_by,
    sentence_mode, algo_mode, core_policy).
  * Warnings recorded in meta.config_warnings and printed to stderr in CLI mode.
- parse_len_bins now robust to malformed entries; clamps to meaningful ranges.
- Slicing guards: summary_top_k, top_windows_k are clamped to avoid negative slicing semantics.
- Minor: counts_only_limit_per_n applied after sorting (stable & predictable).
"""

from __future__ import annotations
import argparse
import json
import math
import re
import sys
import os
import platform
import subprocess
import unicodedata
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple, Optional, Set
from collections import Counter, defaultdict
import hashlib

from multiprocessing import get_context

from tools.stopwords_utils import get_stopwords_for_language, resolve_language_hint

# Optional psutil for physical core detection (best effort)
try:
    import psutil  # type: ignore
    _HAS_PSUTIL = True
except Exception:
    _HAS_PSUTIL = False

# Security: Maximum allowed worker processes to prevent DoS
MAX_WORKERS = int(os.getenv("MAX_ANALYSIS_WORKERS", "16"))

# -----------------------------
# Tokenization & normalization
# -----------------------------

def strip_accents(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))

def _remove_invisible_control(text: str) -> str:
    out: List[str] = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat == "Cf":
            continue
        if cat == "Cc" and ch not in ("\n", "\t"):
            out.append(" ")
            continue
        out.append(ch)
    return "".join(out)

# Words (Unicode-aware) OR runs of non-(word|space) as punctuation tokens.
WORD_PUNCT_RE = re.compile(r"\w+(?:['’]\w+)*|[^\w\s]+", re.UNICODE)
WORD_TOKEN_RE = re.compile(r"^\w+(?:['’]\w+)*$", re.UNICODE)

def tokenize_word_punct_with_spans(
    text: str,
    lowercase: bool = True,
    remove_accents_flag: bool = False,
) -> Tuple[List[str], List[Tuple[int, int]]]:
    clean = _remove_invisible_control(text)
    tokens: List[str] = []
    spans: List[Tuple[int, int]] = []
    for m in WORD_PUNCT_RE.finditer(clean):
        tok = m.group(0)
        if remove_accents_flag:
            tok = strip_accents(tok)
        if lowercase:
            tok = tok.lower()
        tokens.append(tok)
        spans.append((m.start(), m.end()))
    return tokens, spans

def tokenize_alnum_with_spans(
    text: str,
    lowercase: bool = True,
    remove_accents_flag: bool = False,
) -> Tuple[List[str], List[Tuple[int, int]]]:
    clean = _remove_invisible_control(text)
    tokens: List[str] = []
    spans: List[Tuple[int, int]] = []
    buf: List[str] = []
    start_idx: Optional[int] = None
    for i, ch in enumerate(clean):
        if ch.isalnum():
            if start_idx is None:
                start_idx = i
            buf.append(ch)
        else:
            if buf:
                surface = "".join(buf)
                if remove_accents_flag:
                    surface = strip_accents(surface)
                if lowercase:
                    surface = surface.lower()
                tokens.append(surface)
                spans.append((start_idx, i))
                buf = []
                start_idx = None
    if buf:
        surface = "".join(buf)
        if remove_accents_flag:
            surface = strip_accents(surface)
        if lowercase:
            surface = surface.lower()
        tokens.append(surface)
        spans.append((start_idx if start_idx is not None else 0, len(clean)))
    return tokens, spans

def tokenize_with_spans(
    text: str,
    tokenizer: str = "word_punct",
    lowercase: bool = True,
    remove_accents_flag: bool = False,
) -> Tuple[List[str], List[Tuple[int, int]]]:
    tokenizer = (tokenizer or "word_punct").lower()
    if tokenizer == "word_punct":
        return tokenize_word_punct_with_spans(text, lowercase=lowercase, remove_accents_flag=remove_accents_flag)
    elif tokenizer == "alnum":
        return tokenize_alnum_with_spans(text, lowercase=lowercase, remove_accents_flag=remove_accents_flag)
    else:
        raise ValueError(f"Unknown tokenizer: {tokenizer}")

# -----------------------------
# Phrase reconstruction (untokenize)
# -----------------------------

_NO_SPACE_BEFORE = {",", ".", ";", ":", "!", "?", "%", ")", "]", "}", "»", "”", "’", "›", "…"}
_NO_SPACE_AFTER  = {"(", "[", "{", "«", "“", "‘", "‹", "¡", "¿"}
_PUNCT_MULTI = {"...", "…"}  # multi-char punct treated as a single mark

def untokenize(tokens: List[str]) -> str:
    out: List[str] = []
    prev: Optional[str] = None
    for t in tokens:
        if not t:
            continue
        is_punct = (t in _PUNCT_MULTI) or (len(t) == 1 and not t.isalnum() and not t.isspace())
        if not out:
            out.append(t)
        else:
            if is_punct and (t in _NO_SPACE_BEFORE or t in _PUNCT_MULTI):
                out[-1] = out[-1] + t
            elif prev in _NO_SPACE_AFTER:
                out.append(t)
            else:
                out.append(" " + t)
        prev = t
    return "".join(out)

# -----------------------------
# Sentence boundaries (unambiguous)
# -----------------------------

DEFAULT_SENTENCE_TERMINATORS = ".,?!;:…"

@dataclass(frozen=True)
class TerminatorConfig:
    chars: frozenset
    multis: Tuple[str, ...]  # e.g., "...", "!!"

def _parse_multis_csv(csv: str) -> List[str]:
    if not csv:
        return []
    return [p.strip() for p in csv.split(",") if p.strip()]

def parse_sentence_terminators(
    spec: str,
    mode: str = "chars",           # 'chars' or 'list'
    multis_spec: str = ""          # extra multi-char terminators (CSV)
) -> TerminatorConfig:
    spec = (spec or "").strip()
    mode = (mode or "chars").lower()
    chars = set()
    multis: List[str] = []
    if mode == "list":
        parts = [p.strip() for p in spec.split(",") if p.strip()]
        for p in parts:
            if len(p) == 1:
                chars.add(p)
            else:
                multis.append(p)
    else:
        if "..." in spec:
            multis.append("...")
        for ch in spec:
            if ch:
                chars.add(ch)
    extra = _parse_multis_csv(multis_spec)
    for m in extra:
        if m:
            multis.append(m)
    # dedupe
    seen = set()
    uniq: List[str] = []
    for m in multis:
        if m not in seen:
            uniq.append(m); seen.add(m)
    return TerminatorConfig(chars=frozenset(chars), multis=tuple(uniq))

def is_sentence_boundary_token(tok: str, cfg: TerminatorConfig) -> bool:
    if not tok or tok.isalnum():
        return False
    for m in cfg.multis:
        if m and m in tok:
            return True
    return any((ch in cfg.chars) for ch in tok)

def compute_sentence_segments(tokens: List[str], cfg: TerminatorConfig) -> List[Tuple[int, int]]:
    segs: List[Tuple[int, int]] = []
    a = 0
    for i, t in enumerate(tokens):
        if is_sentence_boundary_token(t, cfg):
            if i > a:
                segs.append((a, i))
            a = i + 1
    if a < len(tokens):
        segs.append((a, len(tokens)))
    return segs

# -----------------------------
# Paragraph / block segmentation (by blank lines)
# -----------------------------

_NL_MULTI_RE_CACHE: Dict[int, re.Pattern] = {}

def _compile_blankline_sep(min_blank_lines: int) -> re.Pattern:
    """
    Build a regex that matches separators made of >= min_blank_lines blank lines.
    min_blank_lines=1 -> at least one completely blank line between paragraphs (=> two consecutive newlines possibly with spaces).
    The pattern supports plain LF newlines and will also match when CR characters were present before cleaning.
    """
    m = max(1, int(min_blank_lines))
    # (?:\n[ \t]*) repeated (m+1) or more times roughly corresponds to m blank lines as separators.
    # Example: m=1 => at least 2 newlines (one blank line).
    key = m
    pat = _NL_MULTI_RE_CACHE.get(key)
    if pat is None:
        # Keep \r? for compatibility, though cleaned text no longer contains \r.
        pat = re.compile(r"(?:\r?\n[ \t]*){%d,}" % (m + 1))
        _NL_MULTI_RE_CACHE[key] = pat
    return pat

def compute_token_segments_from_blanklines(
    text: str,
    token_spans: List[Tuple[int, int]],
    min_blank_lines: int,
) -> List[Tuple[int, int]]:
    """
    Split the text in CHAR domain by runs of >= min_blank_lines blank lines,
    then map to TOKEN domain segments (start_token, end_token).

    IMPORTANT:
    To avoid CRLF vs LF domain mismatches, we compute boundaries on the SAME cleaned
    text domain used to produce token_spans, i.e., _remove_invisible_control(text).
    This keeps character coordinates aligned with token_spans.
    """
    if not token_spans:
        return []
    # Use the exact same cleaning as tokenization so char coordinates align.
    s = _remove_invisible_control(text)

    sep_re = _compile_blankline_sep(min_blank_lines)
    boundaries: List[int] = [0]
    for m in sep_re.finditer(s):
        boundaries.append(m.end())
    boundaries.append(len(s))

    segs_tok: List[Tuple[int, int]] = []
    t = 0
    n_tokens = len(token_spans)
    for i in range(len(boundaries) - 1):
        a_char = boundaries[i]
        b_char = boundaries[i + 1]
        # Advance to the first token that overlaps [a_char, b_char)
        a_tok = t
        while a_tok < n_tokens and token_spans[a_tok][1] <= a_char:
            a_tok += 1
        b_tok = a_tok
        while b_tok < n_tokens and token_spans[b_tok][0] < b_char:
            b_tok += 1
        if b_tok > a_tok:
            segs_tok.append((a_tok, b_tok))
        t = b_tok
    if not segs_tok:
        return [(0, n_tokens)]
    return segs_tok

def token_to_segment_index_map(segments: List[Tuple[int,int]], token_len: int) -> List[int]:
    idx_map = [0] * token_len
    for si, (a, b) in enumerate(segments):
        for t in range(a, b):
            idx_map[t] = si
    return idx_map

# -----------------------------
# CPU core detection & workers
# -----------------------------

def _physical_cores_psutil() -> Optional[int]:
    if not _HAS_PSUTIL:
        return None
    try:
        n = psutil.cpu_count(logical=False)
        if isinstance(n, int) and n > 0:
            return n
    except Exception:
        pass
    return None

def _physical_cores_windows() -> Optional[int]:
    try:
        cmd = [
            "powershell", "-NoProfile", "-Command",
            "(Get-CimInstance Win32_Processor | Measure-Object - Property NumberOfCores -Sum).Sum"
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        val = int(out.decode("utf-8", "ignore").strip() or "0")
        if val > 0:
            return val
    except Exception:
        pass
    return None

def _physical_cores_linux() -> Optional[int]:
    try:
        phys_core_pairs = set()
        cur_phys = None
        cur_core = None
        with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    if cur_phys is not None and cur_core is not None:
                        phys_core_pairs.add((cur_phys, cur_core))
                    cur_phys = None
                    cur_core = None
                    continue
                if line.startswith("physical id"):
                    try:
                        cur_phys = int(line.split(":", 1)[1])
                    except Exception:
                        pass
                elif line.startswith("core id"):
                    try:
                        cur_core = int(line.split(":", 1)[1])
                    except Exception:
                        pass
        if phys_core_pairs:
            return len(phys_core_pairs)
    except Exception:
        pass
    return None

def _physical_cores_darwin() -> Optional[int]:
    try:
        out = subprocess.check_output(["sysctl", "-n", "hw.physicalcpu"], stderr=subprocess.DEVNULL)
        val = int(out.decode("utf-8", "ignore").strip() or "0")
        if val > 0:
            return val
    except Exception:
        pass
    return None

def detect_physical_cpu_count() -> Optional[int]:
    n = _physical_cores_psutil()
    if n:
        return n
    system = platform.system()
    if system == "Windows":
        return _physical_cores_windows()
    if system == "Linux":
        return _physical_cores_linux()
    if system == "Darwin":
        return _physical_cores_darwin()
    return None

def choose_auto_workers(min_n: int, max_n: int, core_policy: str = "physical") -> int:
    tasks = max(1, max_n - min_n + 1)
    policy = (core_policy or "physical").lower()
    if policy not in ("physical", "logical", "auto"):
        policy = "physical"
    if policy in ("physical", "auto"):
        n_phys = detect_physical_cpu_count()
        if isinstance(n_phys, int) and n_phys > 0:
            return max(1, min(n_phys, tasks))
        if policy == "physical":
            n_log = os.cpu_count() or 1
            return max(1, min(n_log, tasks))
    n_log = os.cpu_count() or 1
    return max(1, min(n_log, tasks))

# -----------------------------
# N-gram counting
# -----------------------------

def count_ngrams_counter(
    tokens: List[str],
    segments: List[Tuple[int, int]],
    min_n: int,
    max_n: int,
    min_count: int,
) -> Dict[int, Dict[Tuple[str, ...], int]]:
    if min_n < 1 or max_n < min_n:
        raise ValueError("Invalid n-gram range.")
    out: Dict[int, Dict[Tuple[str, ...], int]] = {}
    for n in range(min_n, max_n + 1):
        ctr: Counter = Counter(
            tuple(tokens[i:i+n])
            for (a, b) in segments
            if (b - a) >= n
            for i in range(a, b - n + 1)
        )
        out[n] = {k: v for k, v in ctr.items() if v >= min_count}
    return out

def _count_for_n_worker(args) -> Dict[Tuple[str, ...], int]:
    tokens, segments, n, min_count = args
    d: Dict[Tuple[str, ...], int] = defaultdict(int)
    for a, b in segments:
        if (b - a) < n:
            continue
        for i in range(a, b - n + 1):
            d[tuple(tokens[i:i+n])] += 1
    if min_count > 1:
        d = {k: v for k, v in d.items() if v >= min_count}  # type: ignore
        return d  # type: ignore
    return d  # type: ignore

def count_ngrams_multiprocess(
    tokens: List[str],
    segments: List[Tuple[int, int]],
    min_n: int,
    max_n: int,
    min_count: int,
    workers: int,
) -> Dict[int, Dict[Tuple[str, ...], int]]:
    if min_n < 1 or max_n < min_n:
        raise ValueError("Invalid n-gram range.")
    ns = list(range(min_n, max_n + 1))
    # Security: Cap workers to prevent DoS via resource exhaustion
    effective_workers = min(max(1, workers), MAX_WORKERS)
    with get_context("spawn").Pool(processes=effective_workers) as pool:
        results = pool.map(_count_for_n_worker, [(tokens, segments, n, min_count) for n in ns])
    return {n: results[i] for i, n in enumerate(ns)}

# -----------------------------
# Punctuation policy filtering
# -----------------------------

def is_word_token(tok: str) -> bool:
    return bool(WORD_TOKEN_RE.fullmatch(tok))

def filter_counts_by_punct_policy(
    counts_by_n: Dict[int, Dict[Tuple[str, ...], int]],
    policy: str
) -> Dict[int, Dict[Tuple[str, ...], int]]:
    pol = (policy or "keep").lower()
    if pol == "keep":
        return counts_by_n
    filtered: Dict[int, Dict[Tuple[str, ...], int]] = {}
    for n, d in counts_by_n.items():
        newd: Dict[Tuple[str, ...], int] = {}
        for tup, cnt in d.items():
            if pol == "drop-edge":
                if not is_word_token(tup[0]) or not is_word_token(tup[-1]):
                    continue
            elif pol == "drop-any":
                if any(not is_word_token(t) for t in tup):
                    continue
            newd[tup] = cnt
        filtered[n] = newd
    return filtered

# -----------------------------
# Utilities
# -----------------------------

def phrase_tuple_to_text(tup: Tuple[str, ...]) -> str:
    return untokenize(list(tup))

def compute_md5_for_counts(counts_by_n: Dict[int, Dict[Tuple[str, ...], int]]) -> str:
    m = hashlib.md5()
    for n in sorted(counts_by_n.keys()):
        items = sorted(((phrase_tuple_to_text(t), c) for t, c in counts_by_n[n].items()), key=lambda x: (x[0], x[1]))
        for phrase, cnt in items:
            m.update(str(n).encode("utf-8"))
            m.update(b"|")
            m.update(phrase.encode("utf-8"))
            m.update(b"|")
            m.update(str(cnt).encode("utf-8"))
            m.update(b"\n")
    return m.hexdigest()

def percentile(sorted_vals: List[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    if p <= 0:
        return float(sorted_vals[0])
    if p >= 1:
        return float(sorted_vals[-1])
    k = (len(sorted_vals) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(sorted_vals[int(k)])
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return float(d0 + d1)

def describe_distances(vals: List[int]) -> Dict[str, float]:
    if not vals:
        return {}
    n = len(vals)
    s = float(sum(vals))
    mean = s / n
    var = float(sum((v - mean) ** 2 for v in vals)) / n
    stdev = math.sqrt(var)
    sorted_vals = sorted(vals)
    return {
        "count": float(n),
        "min": float(sorted_vals[0]),
        "q25": float(percentile(sorted_vals, 0.25)),
        "median": float(percentile(sorted_vals, 0.5)),
        "q75": float(percentile(sorted_vals, 0.75)),
        "max": float(sorted_vals[-1]),
        "mean": float(mean),
        "stdev": float(stdev),
        "cv": float(stdev / mean) if mean > 0 else 0.0,
    }

# -----------------------------
# Occurrences, clustering, windows
# -----------------------------

@dataclass
class OccurrenceData:
    starts_tokens: List[int]
    char_starts: List[int]
    char_ends: List[int]

def collect_occurrences_for_phrases(
    tokens: List[str],
    token_spans: List[Tuple[int,int]],
    segments: List[Tuple[int,int]],
    phrases_by_n: Dict[int, Set[Tuple[str, ...]]],
) -> Dict[int, Dict[Tuple[str, ...], OccurrenceData]]:
    out: Dict[int, Dict[Tuple[str, ...], OccurrenceData]] = {n: {} for n in phrases_by_n.keys()}
    for n, wanted in phrases_by_n.items():
        if not wanted:
            continue
        occ_map: Dict[Tuple[str, ...], OccurrenceData] = {p: OccurrenceData([], [], []) for p in wanted}
        for (a, b) in segments:
            if (b - a) < n:
                continue
            for i in range(a, b - n + 1):
                t = tuple(tokens[i:i+n])
                if t in occ_map:
                    occ_map[t].starts_tokens.append(i)
                    s_char = token_spans[i][0]
                    e_char = token_spans[i + n - 1][1]
                    occ_map[t].char_starts.append(s_char)
                    occ_map[t].char_ends.append(e_char)
        out[n] = occ_map
    return out

@dataclass
class Cluster:
    start_token: int
    end_token: int
    span_tokens: int
    count: int
    density_tokens: float

def cluster_by_gap_tokens(starts: List[int], n_len: int, max_gap: int) -> List[Cluster]:
    if not starts:
        return []
    clusters: List[Cluster] = []
    s0 = starts[0]
    prev = starts[0]
    cnt = 1
    for s in starts[1:]:
        if (s - prev) <= max_gap:
            prev = s
            cnt += 1
        else:
            span = (prev - s0) + n_len
            clusters.append(Cluster(s0, prev, span, cnt, (cnt / span) if span > 0 else float(cnt)))
            s0 = s
            prev = s
            cnt = 1
    span = (prev - s0) + n_len
    clusters.append(Cluster(s0, prev, span, cnt, (cnt / span) if span > 0 else float(cnt)))
    return clusters

def crop_clusters(clusters: List[Cluster], top_k: int, top_by: str) -> List[Cluster]:
    if top_k <= 0 or not clusters:
        return clusters
    key = (lambda c: (-c.count, c.span_tokens)) if top_by == "count" else (lambda c: (-c.density_tokens, c.span_tokens))
    clusters_sorted = sorted(clusters, key=key)
    return clusters_sorted[:top_k]

@dataclass
class DenseWindow:
    start_token: int
    end_token_excl: int
    count: int

def densest_windows_tokens(starts: List[int], window_size: int, top_k: int = 3) -> Tuple[int, List[DenseWindow]]:
    if window_size <= 0 or not starts:
        return 0, []
    top_k = max(0, int(top_k))  # guard against negative slicing
    best: List[DenseWindow] = []
    max_count = 0
    j = 0
    for i in range(len(starts)):
        base = starts[i]
        while j < len(starts) and starts[j] < base + window_size:
            j += 1
        cnt = j - i
        if cnt >= max_count:
            max_count = cnt
        best.append(DenseWindow(base, base + window_size, cnt))
    best.sort(key=lambda x: (-x.count, x.start_token))
    return max_count, best[:top_k]

# -----------------------------
# Config
# -----------------------------

@dataclass
class AnalysisConfig:
    # Counting
    min_n: int = 2
    max_n: int = 5
    min_count: int = 2

    # Tokenization
    tokenizer: str = "word_punct"           # "word_punct" | "alnum"
    lowercase: bool = True
    strip_accents: bool = False

    # Sentence handling
    respect_sentences: bool = False
    sentence_terminators: str = DEFAULT_SENTENCE_TERMINATORS
    sentence_mode: str = "chars"            # 'chars' | 'list'
    sentence_multis: str = ""               # CSV of multi-char terminators to add

    # Algorithm selection
    algo_mode: str = "auto"                 # auto | counter | multiprocess
    mp_threshold_tokens: int = 50000
    workers: int = 0
    core_policy: str = "physical"

    # Punctuation policy
    punct_policy: str = "drop-edge"         # keep | drop-edge | drop-any

    # Output shaping
    summary_mode: str = "counts"            # counts | ratio | both | none
    details_ratios: bool = True             # include ratio fields in details
    output_mode: str = "full"               # full | compact

    # Detail selection
    details: str = "top_count"              # none | top_count | top_ratio | all
    details_top_k: int = 50                 # note: positional metrics only surface for phrases kept in this window
    positions_preview: int = 10

    # Clustering & windows (disabled by default)
    enable_clusters: bool = False
    cluster_gap_tokens: int = 50
    clusters_top_k: int = 0                  # 0 = no cap
    clusters_top_by: str = "count"           # count | density

    enable_windows: bool = False
    window_size_tokens: int = 200
    top_windows_k: int = 3

    # Summaries
    summary_top_k: int = 50
    counts_only_limit_per_n: int = 0

    # Diagnostics
    diagnostics: str = "off"                 # off | basic | full
    diag_len_bins: str = ""                  # e.g., "3-5:5,6-10:3,11-1000:2"
    diag_max_repeat_ratio: float = 0.02
    diag_min_distance_tokens: int = 50
    diag_cluster_gap_tokens: int = 80
    diag_cluster_min_count: int = 3
    diag_cluster_max_span_tokens: int = 250
    diag_top_k: int = 100
    diag_digest_k: int = 20

    # Positional metrics (sentence/paragraph/block boundaries)
    enable_position_metrics: bool = False
    pos_bias_threshold: float = 0.60       # threshold for bias ratio (0..1)
    pos_min_count: int = 2                 # min occurrences of a phrase to evaluate bias
    pos_conf_z: float = 1.96               # z for Wilson interval (95% ~ 1.96)
    pos_report_top_k: int = 100            # max positional violations to report (0 = unlimited)
    pos_candidates_top_k: int = 0          # cap how many phrases per n we evaluate position for (0 = all)
    paragraph_break_min_blank_lines: int = 1  # 1 => \n\n separates paragraphs
    block_break_min_blank_lines: int = 2      # 2 => >=2 blank lines ~ "chapter/block"

    # Stopword filtering
    language: Optional[str] = None         # ISO language hint for stop-word filtering (e.g., 'es', 'en')
    filter_stop_words: bool = False        # Filter stop words from summaries

# -----------------------------
# Normalization & validation
# -----------------------------

def _coerce_choice(name: str, val: str, allowed: List[str], default: str, warnings: List[str]) -> str:
    v = (val or "").lower()
    if v not in allowed:
        warnings.append(f"{name}: invalid '{val}', using '{default}'.")
        return default
    return v

def _clamp_int(name: str, v: int, lo: Optional[int], hi: Optional[int], default: Optional[int], warnings: List[str]) -> int:
    orig = v
    if default is not None and v is None:
        warnings.append(f"{name}: None -> default {default}.")
        return default
    if lo is not None and v < lo:
        warnings.append(f"{name}: {orig} < {lo}, clamped to {lo}.")
        v = lo
    if hi is not None and v > hi:
        warnings.append(f"{name}: {orig} > {hi}, clamped to {hi}.")
        v = hi
    return v

def _clamp_float(name: str, v: float, lo: Optional[float], hi: Optional[float], default: Optional[float], warnings: List[str]) -> float:
    orig = v
    if default is not None and v is None:
        warnings.append(f"{name}: None -> default {default}.")
        return default
    if lo is not None and v < lo:
        warnings.append(f"{name}: {orig} < {lo}, clamped to {lo}.")
        v = lo
    if hi is not None and v > hi:
        warnings.append(f"{name}: {orig} > {hi}, clamped to {hi}.")
        v = hi
    return v

def normalize_and_validate_config(cfg: AnalysisConfig) -> Tuple[AnalysisConfig, List[str]]:
    w: List[str] = []

    # Enums
    cfg.tokenizer      = _coerce_choice("tokenizer", cfg.tokenizer, ["word_punct","alnum"], "word_punct", w)
    cfg.punct_policy   = _coerce_choice("punct_policy", cfg.punct_policy, ["keep","drop-edge","drop-any"], "drop-edge", w)
    cfg.summary_mode   = _coerce_choice("summary_mode", cfg.summary_mode, ["counts","ratio","both","none"], "counts", w)
    cfg.details        = _coerce_choice("details", cfg.details, ["none","top_count","top_ratio","all"], "top_count", w)
    cfg.clusters_top_by= _coerce_choice("clusters_top_by", cfg.clusters_top_by, ["count","density"], "count", w)
    cfg.sentence_mode  = _coerce_choice("sentence_mode", cfg.sentence_mode, ["chars","list"], "chars", w)
    cfg.algo_mode      = _coerce_choice("algo_mode", cfg.algo_mode, ["auto","counter","multiprocess"], "auto", w)
    cfg.core_policy    = _coerce_choice("core_policy", cfg.core_policy, ["physical","logical","auto"], "physical", w)
    cfg.output_mode    = _coerce_choice("output_mode", cfg.output_mode, ["full","compact"], "full", w)
    cfg.diagnostics    = _coerce_choice("diagnostics", cfg.diagnostics, ["off","basic","full"], "off", w)

    # Integers
    cfg.min_n          = _clamp_int("min_n", cfg.min_n, 1, None, None, w)
    cfg.max_n          = _clamp_int("max_n", cfg.max_n, cfg.min_n, None, None, w)
    cfg.min_count      = _clamp_int("min_count", cfg.min_count, 1, None, None, w)

    cfg.mp_threshold_tokens = _clamp_int("mp_threshold_tokens", cfg.mp_threshold_tokens, 1, None, None, w)
    cfg.workers             = _clamp_int("workers", cfg.workers, 0, None, None, w)

    cfg.cluster_gap_tokens  = _clamp_int("cluster_gap_tokens", cfg.cluster_gap_tokens, 0, None, None, w)
    cfg.clusters_top_k      = _clamp_int("clusters_top_k", cfg.clusters_top_k, 0, None, None, w)

    cfg.window_size_tokens  = _clamp_int("window_size_tokens", cfg.window_size_tokens, 0, None, None, w)
    cfg.top_windows_k       = _clamp_int("top_windows_k", cfg.top_windows_k, 0, None, None, w)

    cfg.summary_top_k       = _clamp_int("summary_top_k", cfg.summary_top_k, 0, None, None, w)
    cfg.counts_only_limit_per_n = _clamp_int("counts_only_limit_per_n", cfg.counts_only_limit_per_n, 0, None, None, w)
    cfg.details_top_k       = _clamp_int("details_top_k", cfg.details_top_k, 0, None, None, w)
    cfg.positions_preview   = _clamp_int("positions_preview", cfg.positions_preview, 0, None, None, w)

    # Diagnostics numbers
    cfg.diag_max_repeat_ratio     = _clamp_float("diag_max_repeat_ratio", float(cfg.diag_max_repeat_ratio), 0.0, 1.0, None, w)
    cfg.diag_min_distance_tokens  = _clamp_int("diag_min_distance_tokens", cfg.diag_min_distance_tokens, 0, None, None, w)
    cfg.diag_cluster_gap_tokens   = _clamp_int("diag_cluster_gap_tokens", cfg.diag_cluster_gap_tokens, 0, None, None, w)
    cfg.diag_cluster_min_count    = _clamp_int("diag_cluster_min_count", cfg.diag_cluster_min_count, 1, None, None, w)
    cfg.diag_cluster_max_span_tokens = _clamp_int("diag_cluster_max_span_tokens", cfg.diag_cluster_max_span_tokens, 1, None, None, w)
    cfg.diag_top_k                = _clamp_int("diag_top_k", cfg.diag_top_k, 0, None, None, w)
    cfg.diag_digest_k             = _clamp_int("diag_digest_k", cfg.diag_digest_k, 0, None, None, w)

    # Positional metrics clamps
    cfg.pos_bias_threshold        = _clamp_float("pos_bias_threshold", float(cfg.pos_bias_threshold), 0.0, 1.0, None, w)
    cfg.pos_min_count             = _clamp_int("pos_min_count", cfg.pos_min_count, 1, None, None, w)
    cfg.pos_conf_z                = _clamp_float("pos_conf_z", float(cfg.pos_conf_z), 0.0, None, None, w)
    cfg.pos_report_top_k          = _clamp_int("pos_report_top_k", cfg.pos_report_top_k, 0, None, None, w)
    cfg.pos_candidates_top_k      = _clamp_int("pos_candidates_top_k", cfg.pos_candidates_top_k, 0, None, None, w)
    cfg.paragraph_break_min_blank_lines = _clamp_int("paragraph_break_min_blank_lines", cfg.paragraph_break_min_blank_lines, 1, None, None, w)
    cfg.block_break_min_blank_lines     = _clamp_int("block_break_min_blank_lines", cfg.block_break_min_blank_lines, 1, None, None, w)

    # Cross-field advisories
    if cfg.respect_sentences and cfg.tokenizer == "alnum":
        w.append("respect_sentences with tokenizer=alnum: no sentence boundaries will be detected (no punctuation).")
    if cfg.tokenizer == "alnum" and cfg.punct_policy == "drop-any":
        w.append("punct_policy=drop-any with tokenizer=alnum is redundant (no punctuation tokens to drop).")
    if cfg.enable_windows and cfg.window_size_tokens <= 0:
        w.append("enable_windows ignored because window_size_tokens <= 0.")
        cfg.enable_windows = False

    return cfg, w

# -----------------------------
# Helpers: sentence segments & selection
# -----------------------------

def build_sentence_segments(tokens: List[str], cfg: AnalysisConfig) -> List[Tuple[int,int]]:
    tc = parse_sentence_terminators(cfg.sentence_terminators, mode=cfg.sentence_mode, multis_spec=cfg.sentence_multis)
    return compute_sentence_segments(tokens, tc) if cfg.respect_sentences else [(0, len(tokens))]

def token_to_sentence_index_map(segments: List[Tuple[int,int]], token_len: int) -> List[int]:
    idx_map = [0] * token_len
    for si, (a, b) in enumerate(segments):
        for t in range(a, b):
            idx_map[t] = si
    return idx_map

def select_phrases_for_details(
    counts_by_n: Dict[int, Dict[Tuple[str,...], int]],
    total_tokens: int,
    details: str,
    details_top_k: int
) -> Dict[int, Set[Tuple[str,...]]]:
    details = (details or "top_count").lower()
    out: Dict[int, Set[Tuple[str,...]]] = {}
    for n, d in counts_by_n.items():
        if details == "none":
            out[n] = set()
            continue
        items = list(d.items())
        if details == "all":
            out[n] = set(k for k, _ in items)
            continue
        if details == "top_ratio":
            ranked = sorted(items, key=lambda kv: (-(n * kv[1] / max(1, total_tokens)), phrase_tuple_to_text(kv[0])))
        else:  # top_count
            ranked = sorted(items, key=lambda kv: (-kv[1], phrase_tuple_to_text(kv[0])))
        out[n] = set(k for k, _ in ranked[:max(0, details_top_k)])
    return out

# -----------------------------
# Diagnostics (math-only) + positional bias helpers
# -----------------------------

def parse_len_bins(spec: str) -> List[Tuple[int,int,int]]:
    """
    Parse length bins like "3-5:5,6-10:3,11-1000:2" -> [(3,5,5),(6,10,3),(11,1000,2)]
    Robust to malformed entries; clamps to a>=1, b>=a, maxc>=1.
    """
    spec = (spec or "").strip()
    if not spec:
        return []
    bins: List[Tuple[int,int,int]] = []
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    for p in parts:
        if ":" not in p:
            continue
        rng, maxc = p.split(":", 1)
        try:
            max_count = int(maxc.strip())
        except Exception:
            continue
        max_count = max(1, max_count)
        rng = rng.strip()
        try:
            if "-" in rng:
                a, b = rng.split("-", 1)
                a = max(1, int(a.strip()))
                b = max(1, int(b.strip()))
                if b < a:
                    a, b = b, a
                bins.append((a, b, max_count))
            else:
                v = max(1, int(rng.strip()))
                bins.append((v, v, max_count))
        except Exception:
            continue
    bins.sort(key=lambda x: (x[0], x[1]))
    return bins

def allowed_max_for_len(n: int, bins: List[Tuple[int,int,int]]) -> Optional[int]:
    for a, b, m in bins:
        if a <= n <= b:
            return m
    return None

def wilson_lower_bound(k: int, n: int, z: float = 1.96) -> float:
    if n <= 0:
        return 0.0
    phat = k / n
    denom = 1.0 + (z * z) / n
    center = phat + (z * z) / (2.0 * n)
    adj = z * math.sqrt((phat * (1.0 - phat) + (z * z) / (4.0 * n)) / n)
    return max(0.0, (center - adj) / denom)

def compute_positional_counters(
    starts: List[int],
    n_len: int,
    sent_segments: List[Tuple[int,int]],
    sent_map: List[int],
    para_segments: List[Tuple[int,int]],
    para_map: List[int],
    block_segments: List[Tuple[int,int]],
    block_map: List[int],
) -> Dict[str, Dict[str, float]]:
    total = len(starts)
    s_start = s_end = p_start = p_end = b_start = b_end = 0
    for i in starts:
        # sentence
        si = sent_map[i]
        sa, sb = sent_segments[si]
        if i == sa:
            s_start += 1
        if i + n_len == sb:
            s_end += 1
        # paragraph
        pi = para_map[i]
        pa, pb = para_segments[pi]
        if i == pa:
            p_start += 1
        if i + n_len == pb:
            p_end += 1
        # block/chapter
        bi = block_map[i]
        ba, bb = block_segments[bi]
        if i == ba:
            b_start += 1
        if i + n_len == bb:
            b_end += 1
    def pack(c_start: int, c_end: int) -> Dict[str, float]:
        return {
            "start_count": float(c_start),
            "start_ratio": (c_start / total) if total else 0.0,
            "end_count": float(c_end),
            "end_ratio": (c_end / total) if total else 0.0,
        }
    return {
        "sentence": pack(s_start, s_end),
        "paragraph": pack(p_start, p_end),
        "block": pack(b_start, b_end),
        "_total_occurrences": float(total),
    }

# -----------------------------
# Main analysis
# -----------------------------

def analyze_text(text: str, cfg: AnalysisConfig) -> Dict[str, Any]:
    # Normalize & validate
    cfg, cfg_warnings = normalize_and_validate_config(cfg)

    # Tokenize
    tokens, token_spans = tokenize_with_spans(
        text,
        tokenizer=cfg.tokenizer,
        lowercase=cfg.lowercase,
        remove_accents_flag=cfg.strip_accents,
    )
    total_tokens = len(tokens)
    total_chars = len(text)

    # Sentences for counting (respect_sentences) and for positional analysis (always)
    segments = build_sentence_segments(tokens, cfg)
    sentence_map = token_to_sentence_index_map(segments, total_tokens) if cfg.respect_sentences else [0] * total_tokens
    total_sentences = len(segments)
    # For positional metrics we want sentence boundaries even if respect_sentences=False
    pos_tc = parse_sentence_terminators(cfg.sentence_terminators, mode=cfg.sentence_mode, multis_spec=cfg.sentence_multis)
    pos_sentence_segments = compute_sentence_segments(tokens, pos_tc)
    pos_sentence_map = token_to_segment_index_map(pos_sentence_segments, total_tokens)

    # Paragraphs and Blocks (by blank lines)
    para_segments_tok = compute_token_segments_from_blanklines(text, token_spans, cfg.paragraph_break_min_blank_lines)
    block_segments_tok = compute_token_segments_from_blanklines(text, token_spans, cfg.block_break_min_blank_lines)
    para_map = token_to_segment_index_map(para_segments_tok, total_tokens)
    block_map = token_to_segment_index_map(block_segments_tok, total_tokens)

    # Algo selection
    algo_mode_norm = (cfg.algo_mode or "auto").lower()
    if algo_mode_norm == "auto":
        algorithm_selected = "multiprocess" if total_tokens >= max(1, cfg.mp_threshold_tokens) else "counter"
    else:
        algorithm_selected = algo_mode_norm

    # Count n-grams
    if algorithm_selected == "multiprocess":
        workers_used = cfg.workers if cfg.workers > 0 else choose_auto_workers(cfg.min_n, cfg.max_n, cfg.core_policy)
        counts_by_n = count_ngrams_multiprocess(tokens, segments, cfg.min_n, cfg.max_n, cfg.min_count, workers_used)
    else:
        workers_used = 0
        counts_by_n = count_ngrams_counter(tokens, segments, cfg.min_n, cfg.max_n, cfg.min_count)

    # Punctuation filtering
    counts_by_n = filter_counts_by_punct_policy(counts_by_n, cfg.punct_policy)

    # Stopword filtering (for summaries)
    canonical_language = resolve_language_hint(cfg.language)
    stop_words = get_stopwords_for_language(cfg.language, cfg.filter_stop_words)
    stop_words_applied = bool(stop_words)

    def is_stopword_phrase(tup: Tuple[str, ...]) -> bool:
        """Return True if the phrase is entirely stopwords (for n=1) or starts/ends with stopword."""
        if not stop_words:
            return False
        if len(tup) == 1:
            return tup[0] in stop_words
        # For n>1, filter if first or last token is a stopword
        return tup[0] in stop_words or tup[-1] in stop_words

    checksum_md5 = compute_md5_for_counts(counts_by_n)

    # Build summaries (according to summary_mode)
    summary_counts: Dict[int, List[Dict[str, Any]]] = {}
    summary_ratios: Dict[int, List[Dict[str, Any]]] = {}
    include_counts = cfg.summary_mode in ("counts", "both")
    include_ratio  = cfg.summary_mode in ("ratio", "both")
    sum_k = max(0, cfg.summary_top_k)

    for n, d in counts_by_n.items():
        items = list(d.items())
        # Apply stopword filtering if enabled
        if stop_words_applied:
            items = [(k, v) for k, v in items if not is_stopword_phrase(k)]

        if include_counts:
            ranked_counts = sorted(items, key=lambda kv: (-kv[1], phrase_tuple_to_text(kv[0])))
            # apply counts_only_limit_per_n after sorting, then slice to summary_top_k
            if cfg.counts_only_limit_per_n > 0:
                ranked_counts = ranked_counts[:cfg.counts_only_limit_per_n]
            ranked_counts = ranked_counts[:sum_k]
            summary_counts[n] = [
                {"text": phrase_tuple_to_text(k), "count": v, "n": n,
                 "repeat_ratio_tokens": (n * v) / max(1, total_tokens)}
                for k, v in ranked_counts
            ]

        if include_ratio:
            ranked_ratios = sorted(items, key=lambda kv: (-(n * kv[1] / max(1, total_tokens)), phrase_tuple_to_text(kv[0])))
            if cfg.counts_only_limit_per_n > 0:
                ranked_ratios = ranked_ratios[:cfg.counts_only_limit_per_n]
            ranked_ratios = ranked_ratios[:sum_k]
            summary_ratios[n] = [
                {"text": phrase_tuple_to_text(k), "count": v, "n": n,
                 "repeat_ratio_tokens": (n * v) / max(1, total_tokens)}
                for k, v in ranked_ratios
            ]

    # Details selection
    selected_for_details = select_phrases_for_details(counts_by_n, total_tokens, cfg.details, cfg.details_top_k)

    # Diagnostics candidates (pre)
    diagnostics_mode = (cfg.diagnostics or "off").lower()
    len_bins = parse_len_bins(cfg.diag_len_bins)
    diag_candidates_by_n: Dict[int, Set[Tuple[str, ...]]] = {n: set() for n in counts_by_n.keys()}

    if diagnostics_mode in ("basic", "full"):
        for n, d in counts_by_n.items():
            for tup, cnt in d.items():
                ratio_tokens = (n * cnt) / max(1, total_tokens)
                over_ratio = (cfg.diag_max_repeat_ratio > 0 and ratio_tokens >= cfg.diag_max_repeat_ratio)
                max_allowed = allowed_max_for_len(n, len_bins) if len_bins else None
                over_count = (max_allowed is not None and cnt > max_allowed)
                if over_ratio or over_count:
                    diag_candidates_by_n[n].add(tup)

    # Positional candidates (we need occurrences to evaluate bias)
    pos_candidates_by_n: Dict[int, Set[Tuple[str, ...]]] = {n: set() for n in counts_by_n.keys()}
    if cfg.enable_position_metrics:
        for n, d in counts_by_n.items():
            items = [(k, v) for k, v in d.items() if v >= cfg.pos_min_count]
            if cfg.pos_candidates_top_k > 0 and len(items) > cfg.pos_candidates_top_k:
                # Prefer top by count to control cost; change if you prefer ratio.
                items.sort(key=lambda kv: (-kv[1], phrase_tuple_to_text(kv[0])))
                items = items[:cfg.pos_candidates_top_k]
            pos_candidates_by_n[n] = set(k for k, _ in items)

    # Collect occurrences for union (details + diag candidates)
    union_for_occ: Dict[int, Set[Tuple[str, ...]]] = {n: set() for n in counts_by_n.keys()}
    for n in counts_by_n.keys():
        union_for_occ[n] = set(selected_for_details.get(n, set()))
        union_for_occ[n].update(diag_candidates_by_n.get(n, set()))
        if cfg.enable_position_metrics:
            union_for_occ[n].update(pos_candidates_by_n.get(n, set()))
    occurrences_union = collect_occurrences_for_phrases(tokens, token_spans, segments, union_for_occ)

    # Build detailed phrases (only selected_for_details)
    phrases_detail: Dict[int, List[Dict[str, Any]]] = {}
    for n in range(cfg.min_n, cfg.max_n + 1):
        details_for_n: List[Dict[str, Any]] = []
        d_counts = counts_by_n.get(n, {})
        occ_n = occurrences_union.get(n, {})
        for tup, cnt in d_counts.items():
            if tup not in selected_for_details.get(n, set()):
                continue
            occ = occ_n.get(tup)
            if not occ:
                continue
            starts = occ.starts_tokens

            entry = {"text": phrase_tuple_to_text(tup), "n": n, "count": cnt}

            if cfg.details_ratios:
                entry["repeat_ratio_tokens"] = (n * cnt) / max(1, total_tokens)

            # Distances
            distances_tok = [starts[i+1] - starts[i] for i in range(len(starts) - 1)]
            if distances_tok:
                entry["distances_tokens"] = describe_distances(distances_tok)

            # Coverage (tokens)
            if starts:
                first_start = starts[0]
                last_start = starts[-1]
                coverage_span_tokens = (last_start - first_start) + n
                entry["coverage"] = {
                    "first_token": first_start,
                    "last_token": last_start,
                    "span_tokens": coverage_span_tokens,
                    "coverage_ratio_tokens": coverage_span_tokens / max(1, total_tokens),
                }

            # Clusters (optional)
            if cfg.enable_clusters and cfg.cluster_gap_tokens > 0 and len(starts) >= 2:
                cls = cluster_by_gap_tokens(starts, n_len=n, max_gap=cfg.cluster_gap_tokens)
                cls = crop_clusters(cls, cfg.clusters_top_k, cfg.clusters_top_by)
                entry["clusters_tokens"] = [
                    {"start_token": c.start_token, "end_token": c.end_token,
                     "span_tokens": c.span_tokens, "count": c.count, "density_tokens": c.density_tokens}
                    for c in cls
                ]

            # Windows (optional)
            if cfg.enable_windows and cfg.window_size_tokens > 0 and len(starts) >= 1:
                max_in_window_tok, win_tok = densest_windows_tokens(starts, cfg.window_size_tokens, max(0, cfg.top_windows_k))
                entry["dense_windows_tokens"] = {
                    "window_size": cfg.window_size_tokens,
                    "max_count": max_in_window_tok,
                    "top_windows": [{"start_token": w.start_token, "end_token_excl": w.end_token_excl, "count": w.count} for w in win_tok]
                }

            # Positional bias (details)
            if cfg.enable_position_metrics:
                pos = compute_positional_counters(
                    starts, n,
                    pos_sentence_segments, pos_sentence_map,
                    para_segments_tok, para_map,
                    block_segments_tok, block_map
                )
                total_occ = int(pos["_total_occurrences"])
                # Add Wilson lower bounds
                def add_wlb(d: Dict[str, float]) -> Dict[str, float]:
                    k_start = int(d.get("start_count", 0.0))
                    k_end = int(d.get("end_count", 0.0))
                    d["start_wilson_lb"] = wilson_lower_bound(k_start, total_occ, cfg.pos_conf_z)
                    d["end_wilson_lb"] = wilson_lower_bound(k_end, total_occ, cfg.pos_conf_z)
                    return d
                entry["position_bias"] = {
                    "occurrences": total_occ,
                    "sentence": add_wlb({"start_count": pos["sentence"]["start_count"], "start_ratio": pos["sentence"]["start_ratio"],
                                         "end_count": pos["sentence"]["end_count"], "end_ratio": pos["sentence"]["end_ratio"]}),
                    "paragraph": add_wlb({"start_count": pos["paragraph"]["start_count"], "start_ratio": pos["paragraph"]["start_ratio"],
                                          "end_count": pos["paragraph"]["end_count"], "end_ratio": pos["paragraph"]["end_ratio"]}),
                    "block": add_wlb({"start_count": pos["block"]["start_count"], "start_ratio": pos["block"]["start_ratio"],
                                      "end_count": pos["block"]["end_count"], "end_ratio": pos["block"]["end_ratio"]}),
                }

            # Positions preview
            preview_n = max(0, cfg.positions_preview)
            entry["occurrence_positions"] = {
                "starts_tokens_preview": starts[:preview_n] if preview_n > 0 else starts,
                "preview_count": len(starts[:preview_n]) if preview_n > 0 else len(starts),
                "total_occurrences": len(starts),
            }

            details_for_n.append(entry)

        # Reminder: downstream consumers (positional diagnostics, etc.) only see phrases retained here.
        # If details_top_k or details="top_count/top_ratio" trims lower-frequency items, any associated
        # block/paragraph start counters will also disappear from the JSON output.
        details_for_n.sort(key=lambda x: (-x["count"], x["text"]))
        phrases_detail[n] = details_for_n

    # Diagnostics report
    diagnostics_obj: Dict[str, Any] = {}
    if diagnostics_mode in ("basic", "full"):
        violations: List[Dict[str, Any]] = []
        by_rule_counts = defaultdict(int)

        for n, d in counts_by_n.items():
            for tup in diag_candidates_by_n.get(n, set()):
                cnt = d.get(tup, 0)
                if cnt <= 0:
                    continue
                ratio_tok = (n * cnt) / max(1, total_tokens)
                rules_triggered: List[str] = []

                max_allowed = allowed_max_for_len(n, len_bins) if len_bins else None
                if max_allowed is not None and cnt > max_allowed:
                    rules_triggered.append("over_count_bin")
                if cfg.diag_max_repeat_ratio > 0 and ratio_tok >= cfg.diag_max_repeat_ratio:
                    rules_triggered.append("over_ratio")

                metrics: Dict[str, Any] = {"count": cnt, "repeat_ratio_tokens": ratio_tok}

                if diagnostics_mode == "full":
                    occ = occurrences_union.get(n, {}).get(tup)
                    if occ and len(occ.starts_tokens) >= 2:
                        distances_tok = [occ.starts_tokens[i+1] - occ.starts_tokens[i] for i in range(len(occ.starts_tokens) - 1)]
                        if distances_tok:
                            min_dist = min(distances_tok)
                            metrics["min_distance_tokens"] = min_dist
                            if cfg.diag_min_distance_tokens > 0 and min_dist < cfg.diag_min_distance_tokens:
                                rules_triggered.append("too_close")

                        gap = cfg.diag_cluster_gap_tokens or cfg.cluster_gap_tokens
                        cls = cluster_by_gap_tokens(occ.starts_tokens, n_len=n, max_gap=gap)
                        dense_flag = False
                        best_cluster = None
                        for c in cls:
                            if c.count >= cfg.diag_cluster_min_count and c.span_tokens <= cfg.diag_cluster_max_span_tokens:
                                dense_flag = True
                                if best_cluster is None or c.count > best_cluster.count or (c.count == best_cluster.count and c.span_tokens < best_cluster.span_tokens):
                                    best_cluster = c
                        if dense_flag:
                            rules_triggered.append("dense_cluster")
                        if best_cluster:
                            metrics["dense_cluster"] = {
                                "count": best_cluster.count,
                                "span_tokens": best_cluster.span_tokens,
                                "start_token": best_cluster.start_token,
                                "end_token": best_cluster.end_token,
                                "density_tokens": best_cluster.density_tokens,
                            }

                if rules_triggered:
                    for r in set(rules_triggered):
                        by_rule_counts[r] += 1
                    violations.append({
                        "text": phrase_tuple_to_text(tup),
                        "n": n,
                        "rules_triggered": sorted(set(rules_triggered)),
                        "metrics": metrics
                    })

        # Positional bias diagnostics
        if cfg.enable_position_metrics:
            pos_violations: List[Dict[str, Any]] = []
            for n, d in counts_by_n.items():
                # evaluate only candidates we collected occurrences for
                for tup in pos_candidates_by_n.get(n, set()):
                    cnt = d.get(tup, 0)
                    if cnt < cfg.pos_min_count:
                        continue
                    occ = occurrences_union.get(n, {}).get(tup)
                    if not occ:
                        continue
                    pos = compute_positional_counters(
                        occ.starts_tokens, n,
                        pos_sentence_segments, pos_sentence_map,
                        para_segments_tok, para_map,
                        block_segments_tok, block_map
                    )
                    total_occ = int(pos["_total_occurrences"])
                    # compute Wilson LBs
                    def LB(start_count: float) -> float:
                        return wilson_lower_bound(int(start_count), total_occ, cfg.pos_conf_z)
                    def LB_end(end_count: float) -> float:
                        return wilson_lower_bound(int(end_count), total_occ, cfg.pos_conf_z)
                    rules: List[str] = []
                    if LB(pos["sentence"]["start_count"]) >= cfg.pos_bias_threshold: rules.append("bias_sentence_start")
                    if LB_end(pos["sentence"]["end_count"])   >= cfg.pos_bias_threshold: rules.append("bias_sentence_end")
                    if LB(pos["paragraph"]["start_count"]) >= cfg.pos_bias_threshold: rules.append("bias_paragraph_start")
                    if LB_end(pos["paragraph"]["end_count"])   >= cfg.pos_bias_threshold: rules.append("bias_paragraph_end")
                    if LB(pos["block"]["start_count"])     >= cfg.pos_bias_threshold: rules.append("bias_block_start")
                    if LB_end(pos["block"]["end_count"])       >= cfg.pos_bias_threshold: rules.append("bias_block_end")
                    if rules:
                        for r in set(rules):
                            by_rule_counts[r] += 1
                        pos_violations.append({
                            "text": phrase_tuple_to_text(tup),
                            "n": n,
                            "rules_triggered": sorted(set(rules)),
                            "metrics": {
                                "count": cnt,
                                "position": {
                                    "occurrences": total_occ,
                                    "sentence": pos["sentence"],
                                    "paragraph": pos["paragraph"],
                                    "block": pos["block"],
                                }
                            }
                        })
            # Optionally cap how many we report
            if cfg.pos_report_top_k > 0 and len(pos_violations) > cfg.pos_report_top_k:
                pos_violations.sort(key=lambda v: (-len(v["rules_triggered"]),
                                                   -v["metrics"]["position"]["sentence"]["start_ratio"],
                                                   -v["metrics"]["count"]))
                pos_violations = pos_violations[:cfg.pos_report_top_k]
            violations.extend(pos_violations)

        violations.sort(key=lambda v: (-len(v["rules_triggered"]),
                                       -v["metrics"].get("repeat_ratio_tokens", 0.0) if "repeat_ratio_tokens" in v["metrics"] else 0.0,
                                       -v["metrics"].get("count", 0)))
        if cfg.diag_top_k > 0:
            violations = violations[:cfg.diag_top_k]

        digest_k = max(0, cfg.diag_digest_k)
        digest_by_ratio = sorted(violations, key=lambda v: -v["metrics"].get("repeat_ratio_tokens", 0.0))[:digest_k]
        digest_by_count = sorted(violations, key=lambda v: -v["metrics"].get("count", 0))[:digest_k]
        diagnostics_obj = {
            "mode": diagnostics_mode,
            "policy": {
                "len_bins": len_bins,
                "max_repeat_ratio_tokens": cfg.diag_max_repeat_ratio,
                "min_distance_tokens": cfg.diag_min_distance_tokens if diagnostics_mode == "full" else None,
                "cluster_gap_tokens": cfg.diag_cluster_gap_tokens if diagnostics_mode == "full" else None,
                "cluster_min_count": cfg.diag_cluster_min_count if diagnostics_mode == "full" else None,
                "cluster_max_span_tokens": cfg.diag_cluster_max_span_tokens if diagnostics_mode == "full" else None,
            },
            "summary": {
                "total_phrases_evaluated": sum(len(s) for s in diag_candidates_by_n.values()),
                "total_violating_phrases": len(violations),
                "by_rule": dict(by_rule_counts),
            },
            "violations": violations,
            "llm_digest": {
                "top_by_ratio": [
                    {"text": v["text"], "n": v["n"],
                     "count": v["metrics"]["count"],
                     "repeat_ratio_tokens": v["metrics"].get("repeat_ratio_tokens", 0.0),
                     "rules": v["rules_triggered"]}
                    for v in digest_by_ratio
                ],
                "top_by_count": [
                    {"text": v["text"], "n": v["n"],
                     "count": v["metrics"]["count"],
                     "repeat_ratio_tokens": v["metrics"].get("repeat_ratio_tokens", 0.0),
                     "rules": v["rules_triggered"]}
                    for v in digest_by_count
                ]
            }
        }

    # Compose output
    result_full: Dict[str, Any] = {
        "version": "2.3.3",
        "meta": {
            "total_tokens": total_tokens,
            "total_chars": total_chars,
            "tokenizer": cfg.tokenizer,
            "lowercase": cfg.lowercase,
            "strip_accents": cfg.strip_accents,
            "respect_sentences": cfg.respect_sentences,
            "sentence_terminators": cfg.sentence_terminators,
            "sentence_mode": cfg.sentence_mode,
            "sentence_multis": cfg.sentence_multis,
            "algorithm_selected": algorithm_selected,
            "mp_threshold_tokens": cfg.mp_threshold_tokens,
            "workers_used": workers_used,
            "core_policy": cfg.core_policy,
            "punct_policy": cfg.punct_policy,
            "language_hint": canonical_language or cfg.language,
            "stop_words_filtered": cfg.filter_stop_words and stop_words_applied,
            "stop_words_count": len(stop_words) if stop_words_applied else 0,
            "checksum_md5": checksum_md5,
            "config_warnings": cfg_warnings,
        },
        "settings": asdict(cfg),
        "summary": {
            **({"top_by_count": summary_counts} if include_counts else {}),
            **({"top_by_ratio_tokens": summary_ratios} if include_ratio else {}),
        },
        "diagnostics": diagnostics_obj,
        "phrases": phrases_detail,
    }

    if cfg.output_mode == "compact":
        result_compact = {
            "version": result_full["version"],
            "meta": result_full["meta"],
            "settings": {
                "min_n": cfg.min_n, "max_n": cfg.max_n, "min_count": cfg.min_count,
                "tokenizer": cfg.tokenizer, "lowercase": cfg.lowercase, "strip_accents": cfg.strip_accents,
                "respect_sentences": cfg.respect_sentences, "punct_policy": cfg.punct_policy,
                "summary_mode": cfg.summary_mode, "diagnostics": cfg.diagnostics
            },
            "summary": result_full["summary"],
            "diagnostics": result_full["diagnostics"],
        }
        return result_compact

    return result_full

# -----------------------------
# CLI
# -----------------------------

def read_text_from_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def build_cli_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Repetition Analyzer -> JSON. "
            "Control output via --summary-mode, --details-*, --enable-*, --diagnostics, and --output-mode. "
            "Hardened against unexpected values; normalization warnings printed to stderr."
        )
    )
    p.add_argument("--file", type=str, default="", help="Input .txt file (utf-8). If omitted and --text not given, uses an internal demo.")
    p.add_argument("--text", type=str, default="", help="Raw text provided directly via CLI (overrides --file if both given).")

    # Counting / tokenization
    p.add_argument("--min-n", type=int, default=2, help="Minimum n-gram length (>=1).")
    p.add_argument("--max-n", type=int, default=5, help="Maximum n-gram length (>=min-n).")
    p.add_argument("--min-count", type=int, default=2, help="Report phrases with at least this frequency.")
    p.add_argument("--tokenizer", type=str, default="word_punct", choices=["word_punct","alnum"], help="Tokenizer mode.")
    p.add_argument("--lowercase", action="store_true", help="Lowercase tokens before processing.")
    p.add_argument("--no-lowercase", action="store_true", help="Do NOT lowercase tokens (overrides --lowercase).")
    p.add_argument("--strip-accents", action="store_true", help="Strip accents/diacritics via NFKD.")

    # Sentences
    p.add_argument("--respect-sentences", action="store_true", help="Do not allow n-grams to cross sentence boundaries.")
    p.add_argument("--sentence-terminators", type=str, default=DEFAULT_SENTENCE_TERMINATORS,
                   help='Terminators. With --sentence-mode=chars (default), every character here is a terminator; "..." is auto-detected.')
    p.add_argument("--sentence-mode", type=str, default="chars", choices=["chars","list"],
                   help="How to interpret --sentence-terminators (chars=list of characters, list=CSV tokens).")
    p.add_argument("--sentence-multis", type=str, default="",
                   help='Extra multi-character terminators as CSV, e.g. "...,!!,??".')

    # Algorithm selection & performance
    p.add_argument("--algo-mode", type=str, default="auto", choices=["auto","counter","multiprocess"],
                   help="Counting mode. 'auto' picks multiprocess for large files, counter otherwise.")
    p.add_argument("--mp-threshold-tokens", type=int, default=50000,
                   help="When --algo-mode=auto, switch to multiprocess at this token count (default: 50000).")
    p.add_argument("--workers", type=int, default=0,
                   help="Workers for multiprocess (0 = auto selection based on CPU; bounded by number of n-values).")
    p.add_argument("--core-policy", type=str, default="physical", choices=["physical","logical","auto"],
                   help="Auto worker selection policy when --workers=0 (default: physical cores).")

    # Punctuation policy
    p.add_argument("--punct-policy", type=str, default="drop-edge", choices=["keep","drop-edge","drop-any"],
                   help="Filter n-grams based on punctuation presence.")

    # Output shaping
    p.add_argument("--summary-mode", type=str, default="counts", choices=["counts","ratio","both","none"],
                   help="Which summary sections to include.")
    p.add_argument("--details-ratios", dest="details_ratios", action="store_true", help="Include ratio fields in details.")
    p.add_argument("--no-details-ratios", dest="details_ratios", action="store_false", help="Omit ratio fields in details.")
    p.set_defaults(details_ratios=True)

    p.add_argument("--output-mode", type=str, default="full", choices=["full","compact"],
                   help="Compact mode drops heavy 'phrases' block; keep 'summary' and 'diagnostics'.")

    # Details & summaries
    p.add_argument("--details", type=str, default="top_count", choices=["none","top_count","top_ratio","all"],
                   help="Which phrases to enrich with distances/clusters/windows.")
    p.add_argument(
        "--details-top-k",
        type=int,
        default=50,
        help=(
            "How many phrases per n to enrich (only for top_* modes). "
            "Positional metrics and boundary counters are emitted only for phrases that survive this cap; "
            "use --details all or raise this limit to retain low-frequency block/paragraph starts."
        ),
    )
    p.add_argument("--positions-preview", type=int, default=10, help="How many occurrence start indices to expose (0=all).")
    p.add_argument("--summary-top-k", type=int, default=50, help="How many phrases per n in summaries.")
    p.add_argument("--counts-only-limit-per-n", type=int, default=0, help="If >0, cap candidates per n (after sorting) before building summaries.")

    # Clustering & windows (disabled by default)
    p.add_argument("--enable-clusters", action="store_true", help="Enable cluster computation (token-gap based).")
    p.add_argument("--cluster-gap-tokens", type=int, default=50, help="Max token gap to group occurrences into a cluster.")
    p.add_argument("--clusters-top-k", type=int, default=0, help="If >0, cap clusters per phrase (top by --clusters-top-by).")
    p.add_argument("--clusters-top-by", type=str, default="count", choices=["count","density"], help="Cluster ranking key.")
    p.add_argument("--enable-windows", action="store_true", help="Enable dense window search in tokens.")
    p.add_argument("--window-size-tokens", type=int, default=200, help="Sliding window size in tokens.")
    p.add_argument("--top-windows-k", type=int, default=3, help="How many top dense windows to include per phrase.")

    # Diagnostics
    p.add_argument("--diagnostics", type=str, default="off", choices=["off","basic","full"],
                   help="Diagnostics report with mathematical rules.")
    p.add_argument("--diag-len-bins", type=str, default="", help='Length bins policy, e.g., "3-5:5,6-10:3,11-1000:2".')
    p.add_argument("--diag-max-repeat-ratio", type=float, default=0.02, help="Max allowed repeat ratio (tokens) before flagging (0..1).")
    p.add_argument("--diag-min-distance-tokens", type=int, default=50, help="Min allowed distance in tokens for 'too_close' rule (full mode).")
    p.add_argument("--diag-cluster-gap-tokens", type=int, default=80, help="Token gap for cluster detection in diagnostics (full mode).")
    p.add_argument("--diag-cluster-min-count", type=int, default=3, help="Min occurrences within a cluster to flag 'dense_cluster' (full mode).")
    p.add_argument("--diag-cluster-max-span-tokens", type=int, default=250, help="Max span in tokens for a cluster to be considered dense (full mode).")
    p.add_argument("--diag-top-k", type=int, default=100, help="Max number of violations returned.")
    p.add_argument("--diag-digest-k", type=int, default=20, help="Max number of digest entries (LLM) per list.")

    # Positional metrics (CLI)
    p.add_argument("--enable-position-metrics", action="store_true", help="Enable positional bias analysis (sentence/paragraph/block starts/ends).")
    p.add_argument("--pos-bias-threshold", type=float, default=0.60, help="Bias threshold (Wilson lower bound) for start/end ratios (0..1).")
    p.add_argument("--pos-min-count", type=int, default=2, help="Minimum occurrences of a phrase to evaluate positional bias.")
    p.add_argument("--pos-conf-z", type=float, default=1.96, help="Z-value for Wilson interval (e.g., 1.96 for ~95%).")
    p.add_argument("--pos-report-top-k", type=int, default=100, help="Cap positional bias violations reported (0 = unlimited).")
    p.add_argument("--pos-candidates-top-k", type=int, default=0, help="Cap number of phrases per n evaluated for position (0 = all).")
    p.add_argument("--paragraph-break-min-blank-lines", type=int, default=1, help="Blank lines required to split paragraphs (1 => \\n\\n).")
    p.add_argument("--block-break-min-blank-lines", type=int, default=2, help="Blank lines required to split blocks/chapters (2 => >= two blank lines).")

    p.add_argument("--json-out", type=str, default="", help="If provided, write JSON to this path; otherwise print to stdout.")
    return p

def main():
    parser = build_cli_parser()
    args = parser.parse_args()

    lowercase = True
    if args.no_lowercase:
        lowercase = False
    elif args.lowercase:
        lowercase = True

    if args.text:
        text = args.text
    elif args.file:
        text = read_text_from_file(args.file)
    else:
        text = (
            "Este es un pequeño texto de demostración. Este texto es solo un ejemplo, "
            "pero este texto tiene repeticiones: a veces; también: usa dos puntos. "
            "Este texto tiene repeticiones... ¿Ves? ¡Sí!"
        )

    cfg = AnalysisConfig(
        min_n=args.min_n,
        max_n=args.max_n,
        min_count=args.min_count,
        tokenizer=args.tokenizer,
        lowercase=lowercase,
        strip_accents=args.strip_accents,
        respect_sentences=bool(args.respect_sentences),
        sentence_terminators=args.sentence_terminators,
        sentence_mode=args.sentence_mode,
        sentence_multis=args.sentence_multis,
        algo_mode=args.algo_mode,
        mp_threshold_tokens=max(1, int(args.mp_threshold_tokens)),
        workers=args.workers,
        core_policy=args.core_policy,
        punct_policy=args.punct_policy,
        summary_mode=args.summary_mode,
        details_ratios=bool(args.details_ratios),
        output_mode=args.output_mode,
        details=args.details,
        details_top_k=args.details_top_k,
        positions_preview=args.positions_preview,
        enable_clusters=bool(args.enable_clusters),
        cluster_gap_tokens=args.cluster_gap_tokens,
        clusters_top_k=args.clusters_top_k,
        clusters_top_by=args.clusters_top_by,
        enable_windows=bool(args.enable_windows),
        window_size_tokens=args.window_size_tokens,
        top_windows_k=args.top_windows_k,
        summary_top_k=args.summary_top_k,
        counts_only_limit_per_n=args.counts_only_limit_per_n,
        diagnostics=args.diagnostics,
        diag_len_bins=args.diag_len_bins,
        diag_max_repeat_ratio=float(args.diag_max_repeat_ratio),
        diag_min_distance_tokens=args.diag_min_distance_tokens,
        diag_cluster_gap_tokens=args.diag_cluster_gap_tokens,
        diag_cluster_min_count=args.diag_cluster_min_count,
        diag_cluster_max_span_tokens=args.diag_cluster_max_span_tokens,
        diag_top_k=args.diag_top_k,
        diag_digest_k=args.diag_digest_k,
        enable_position_metrics=bool(args.enable_position_metrics),
        pos_bias_threshold=float(args.pos_bias_threshold),
        pos_min_count=args.pos_min_count,
        pos_conf_z=float(args.pos_conf_z),
        pos_report_top_k=args.pos_report_top_k,
        pos_candidates_top_k=args.pos_candidates_top_k,
        paragraph_break_min_blank_lines=args.paragraph_break_min_blank_lines,
        block_break_min_blank_lines=args.block_break_min_blank_lines,
    )

    result = analyze_text(text, cfg)

    # Print normalization warnings to stderr (if any)
    meta = result.get("meta", {})
    warns = meta.get("config_warnings", [])
    if warns:
        sys.stderr.write("Normalization warnings:\n")
        for msg in warns:
            sys.stderr.write(f"  - {msg}\n")

    out_json = json.dumps(result, ensure_ascii=False, indent=2)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            f.write(out_json)
    else:
        print(out_json)

if __name__ == "__main__":
    main()
