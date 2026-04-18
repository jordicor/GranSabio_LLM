#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lexical Diversity Analyzer (module + CLI)
Version: 0.2.0
Author: ChatGPT (GPT-5 Pro)
License: MIT

Features
--------
- Metrics:
  * MTLD (bidirectional, McCarthy & Jarvis) -> robust to length
  * HD-D (Hypergeometric Diversity, n=42 default; auto-adjust if N<n)
  * Yule's K (lower is better)
  * Herdan's C (log V / log N)
  * Brunet's W (N^(V^-alpha), alpha=0.165 default; lower is better)
  * TTR (optional) and distinct-n (optional)
- Auto metric selection by token length:
  * N < 200:    Herdan's C, Yule's K (+ optional TTR)
  * 200–399:     MTLD (bi), Herdan's C, Yule's K
  * N >= 400:    MTLD (bi), HD-D, Yule's K, Herdan's C (+ optional Brunet's W)
- Threshold-based diagnosis: GREEN / AMBER / RED (configurable)
- NEW: Top words (most frequent types) without stopword filtering; optional positions
- NEW: Window analysis
  * Mode "tokens": sliding windows with step and size (find local repetition zones)
  * Mode "paragraphs": windows per paragraph (blank-line delimited)
  * Optional per-window metrics (same policy as global)
- CLI usage with optional language-aware stop-word filtering

Notes
-----
- Tokenization aligns conceptually with repetition_analyzer.py:
  tokenizer in {"word_punct","alnum"}, lowercase / strip accents.
- Metrics are computed on "word tokens" only (no punctuation).
- "Top words" and window analysis share normalized word tokens. Optional stop-word filtering
  removes connector words from the most-repeated list when a language hint is provided.
"""

from __future__ import annotations
import argparse
import math

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json_utils as json
import os
import re
import sys
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import Counter

# --- Tokenization & normalization (compatible with your analyzer's spirit) ---
import unicodedata

from tools.stopwords_utils import (
    get_stopwords_for_language,
    resolve_language_hint,
)
from tools.string_utils import remove_invisible_control

WORD_PUNCT_RE = re.compile(r"\w+(?:['']\w+)*|[^\w\s]+", re.UNICODE)
WORD_TOKEN_RE = re.compile(r"^\w+(?:['']\w+)*$", re.UNICODE)

def strip_accents(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))

def tokenize_with_spans(
    text: str,
    tokenizer: str = "word_punct",
    lowercase: bool = True,
    remove_accents_flag: bool = False,
) -> Tuple[List[str], List[Tuple[int, int]]]:
    """
    Returns tokens and char spans (start, end) for each token.
    """
    clean = remove_invisible_control(text)
    tokens: List[str] = []
    spans: List[Tuple[int, int]] = []

    if (tokenizer or "word_punct").lower() == "alnum":
        # Alphanumeric runs only
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
    else:
        # Word + punctuation tokens
        for m in WORD_PUNCT_RE.finditer(clean):
            tok = m.group(0)
            if remove_accents_flag:
                tok = strip_accents(tok)
            if lowercase:
                tok = tok.lower()
            tokens.append(tok)
            spans.append((m.start(), m.end()))
    return tokens, spans

def word_only(tokens: List[str]) -> List[str]:
    """Keep word tokens only (drop punctuation) for lexical metrics."""
    return [t for t in tokens if WORD_TOKEN_RE.fullmatch(t)]

# --- Metrics -----------------------------------------------------------------

def calc_ttr(tokens: List[str]) -> float:
    N = len(tokens)
    if N == 0:
        return 0.0
    V = len(set(tokens))
    return V / N

def calc_herdan_c(tokens: List[str], log_base: str = "e") -> float:
    N = len(tokens)
    V = len(set(tokens))
    if N <= 1 or V <= 1:
        return 0.0
    if log_base == "10":
        return math.log10(V) / math.log10(N)
    return math.log(V) / math.log(N)

def calc_brunet_w(tokens: List[str], alpha: float = 0.165) -> float:
    N = len(tokens)
    V = len(set(tokens))
    if N == 0 or V == 0:
        return 0.0
    try:
        return pow(N, pow(V, -alpha))
    except Exception:
        return float("inf")

def calc_yules_k(tokens: List[str]) -> float:
    """
    K = 10^4 * (sum_i f_i^2 - N) / N^2, where f_i are token counts per type.
    Lower K -> more lexical variety (less repetition).
    """
    N = len(tokens)
    if N == 0:
        return 0.0
    freq = Counter(tokens)
    m2 = sum(c * c for c in freq.values())
    return (10000.0 * (m2 - N)) / (N * N)

def _log_choose(n: int, k: int) -> float:
    """Stable log(n choose k) via log-gamma; 0 <= k <= n assumed."""
    if k < 0 or k > n:
        return float("-inf")
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)

def calc_hdd(tokens: List[str], sample_size: int = 42) -> float:
    """
    HD-D (Hypergeometric Diversity):
    D = (1 / n) * sum_types [1 - C(N - f, n) / C(N, n)], with n = min(sample_size, N)
    Returns value in [0, 1].
    """
    N = len(tokens)
    if N == 0:
        return 0.0
    n = min(int(sample_size), N)
    freq = Counter(tokens)
    denom = _log_choose(N, n)
    s = 0.0
    for f in freq.values():
        if N - f >= n:
            p_zero_log = _log_choose(N - f, n) - denom
            p_zero = math.exp(p_zero_log)
        else:
            p_zero = 0.0
        p_at_least_one = 1.0 - min(max(p_zero, 0.0), 1.0)
        s += p_at_least_one
    return s / n

def _mtld_one_pass(seq: List[str], t: float = 0.72, min_factor_len: int = 10) -> float:
    """
    Compute MTLD for a single direction (forward).
    Returns #tokens / #factors (including partial factor at the end).
    """
    N = len(seq)
    if N == 0:
        return 0.0
    types: set = set()
    token_count = 0
    factors = 0.0

    for tok in seq:
        token_count += 1
        types.add(tok)
        curr_ttr = len(types) / token_count
        if token_count >= min_factor_len and curr_ttr <= t:
            factors += 1.0
            types.clear()
            token_count = 0

    # Partial factor contribution (McCarthy & Jarvis)
    if token_count > 0:
        curr_ttr = len(types) / token_count
        if curr_ttr != 1.0:  # avoid division by zero
            factors += (1.0 - curr_ttr) / (1.0 - t)
        else:
            # all tokens were unique but never dropped below threshold
            factors += 1.0

    if factors == 0.0:
        return float("inf")  # degenerate case, treat as extremely high diversity
    return N / factors

def calc_mtld_bidirectional(tokens: List[str], threshold: float = 0.72, min_factor_len: int = 10) -> float:
    if not tokens:
        return 0.0
    fwd = _mtld_one_pass(tokens, t=threshold, min_factor_len=min_factor_len)
    bwd = _mtld_one_pass(list(reversed(tokens)), t=threshold, min_factor_len=min_factor_len)
    if math.isinf(fwd) and math.isinf(bwd):
        return float("inf")
    if math.isinf(fwd):
        return bwd
    if math.isinf(bwd):
        return fwd
    return (fwd + bwd) / 2.0

def calc_distinct_n(tokens: List[str], n: int) -> float:
    """distinct-n ratio: unique n-grams / total n-grams (n>=1)."""
    if n <= 0:
        return 0.0
    N = len(tokens)
    if N < n:
        return 0.0
    seen = set(tuple(tokens[i:i+n]) for i in range(N - n + 1))
    return len(seen) / (N - n + 1)

# --- Top words ---------------------------------------------------------------

def compute_top_words(
    tokens: List[str],
    top_k: int,
    include_positions: bool = False,
    stop_words: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Returns top_k most frequent word types (optional stop-word filtering).
    If include_positions=True, include token indices where the word occurs.
    """
    if top_k <= 0 or not tokens:
        return []
    freq: Counter = Counter()
    filtered_total = 0
    positions_map: Dict[str, List[int]] = {} if include_positions else {}

    for idx, tok in enumerate(tokens):
        if stop_words and tok in stop_words:
            continue
        freq[tok] += 1
        filtered_total += 1
        if include_positions:
            positions_map.setdefault(tok, []).append(idx)

    if not freq:
        return []

    # Stable ordering: count desc, then lex asc
    items = sorted(freq.items(), key=lambda x: (-x[1], x[0]))[:top_k]

    out: List[Dict[str, Any]] = []
    for w, c in items:
        entry: Dict[str, Any] = {
            "word": w,
            "count": int(c),
            "freq": c / filtered_total if filtered_total > 0 else 0.0,
        }
        if include_positions:
            entry["positions"] = positions_map.get(w, [])
        out.append(entry)
    return out

# --- Windowing ---------------------------------------------------------------

def _split_paragraphs_with_spans(text: str) -> List[Tuple[int, int]]:
    """
    Split text into paragraphs by blank-line separators and return (start, end) char spans.
    A paragraph is any block of text separated by two or more newline groups.
    """
    spans: List[Tuple[int, int]] = []
    start = 0
    # separator: at least one blank line (two or more newline groups when allowing whitespace)
    sep = re.compile(r"(?:\r?\n\s*){2,}")
    for m in sep.finditer(text):
        end = m.start()
        if end > start:
            spans.append((start, end))
        start = m.end()
    # tail
    if start < len(text):
        spans.append((start, len(text)))
    if not spans:
        # Fallback: whole text as a single paragraph
        spans.append((0, len(text)))
    return spans

def _window_slices_over_tokens(n_tokens: int, size: int, step: int) -> List[Tuple[int, int]]:
    if n_tokens <= 0 or size <= 0 or step <= 0:
        return []
    starts = list(range(0, max(1, n_tokens - size + 1), step))
    if not starts or starts[-1] + size < n_tokens:
        # Ensure last window reaches the end
        starts.append(max(0, n_tokens - size))
    return [(s, min(s + size, n_tokens)) for s in starts]

# --- Config & thresholds -----------------------------------------------------

@dataclass
class Thresholds:
    # GREEN / AMBER / RED thresholds (seed defaults; adjust to your corpus)
    herdan_green_min: float = 0.80
    herdan_amber_min: float = 0.65

    yulek_green_max: float = 100.0
    yulek_amber_max: float = 150.0

    mtld_green_min: float = 70.0
    mtld_amber_min: float = 50.0

    hdd_green_min: float = 0.70
    hdd_amber_min: float = 0.55

    brunet_green_max: Optional[float] = None  # Optional: if set, lower is better
    brunet_amber_max: Optional[float] = None

@dataclass
class LexDivConfig:
    # Tokenization
    tokenizer: str = "word_punct"          # "word_punct" | "alnum"
    lowercase: bool = True
    strip_accents: bool = False
    language: Optional[str] = None          # ISO language hint for stop-word filtering
    filter_stop_words: bool = True          # Filter stop words in top-word summaries

    # Metric toggles
    metrics: str = "auto"                  # "auto" | "all" | CSV e.g. "mtld,hdd,yulek,c,brunet,ttr,distinct2"
    mtld_threshold: float = 0.72
    mtld_min_factor_len: int = 10
    hdd_sample_size: int = 42
    brunet_alpha: float = 0.165
    include_ttr: bool = False
    distinct_max_n: int = 0                # 0 disables

    # Decision policy
    thresholds: Thresholds = field(default_factory=Thresholds)
    decision_mode: str = "consensus"       # "consensus" | "score" (score not implemented yet)
    require_majority: int = 2              # e.g., 2-of-N for consensus

    # Output mode
    output_mode: str = "full"              # "full" | "compact"

    # --- NEW: top words (global) ---
    top_words_k: int = 0                   # 0 disables; if >0, compute top-K words globally
    include_positions: bool = False        # include global positions for top words

    # --- NEW: window analysis ---
    analyze_windows: bool = False
    window_mode: str = "tokens"            # "tokens" | "paragraphs"
    window_size: int = 200                 # tokens per window (tokens mode)
    window_step: int = 100                 # token step between windows (tokens mode)
    window_top_k: int = 10                 # top words per window
    window_preview_chars: int = 160        # preview text chars in output
    include_window_metrics: bool = False   # compute metrics per window
    window_include_positions: bool = False # include positions of top words per window (token indices relative to window)

# --- Metric selection policy -------------------------------------------------

def _select_metrics_auto(n_tokens: int) -> List[str]:
    """
    Auto policy (word-tokens count):
      - N < 200:                   ["c","yulek"] + optional ["ttr"]
      - 200 <= N < 400:            ["mtld","c","yulek"]
      - N >= 400:                  ["mtld","hdd","yulek","c","brunet"]
    """
    if n_tokens < 200:
        return ["c", "yulek"]
    if n_tokens < 400:
        return ["mtld", "c", "yulek"]
    return ["mtld", "hdd", "yulek", "c", "brunet"]

def _parse_metrics_arg(metrics: str) -> List[str]:
    m = (metrics or "auto").strip().lower()
    if m == "auto":
        return ["auto"]
    if m == "all":
        return ["mtld", "hdd", "yulek", "c", "brunet", "ttr"]
    parts = [p.strip() for p in m.split(",") if p.strip()]
    # normalize aliases
    out: List[str] = []
    for p in parts:
        if p in {"c", "herdan", "herdan_c"}:
            out.append("c")
        elif p in {"w", "brunet", "brunet_w"}:
            out.append("brunet")
        elif p in {"k", "yulek", "yules_k", "yulek"}:
            out.append("yulek")
        elif p in {"mtld", "mtld_bi"}:
            out.append("mtld")
        elif p in {"hdd", "hd-d"}:
            out.append("hdd")
        elif p in {"ttr"}:
            out.append("ttr")
        elif p.startswith("distinct"):
            out.append(p)  # e.g., distinct2, distinct3
        else:
            # ignore unknown
            pass
    return list(dict.fromkeys(out))  # de-dup preserving order

# --- Diagnosis ---------------------------------------------------------------

def _grade_higher_better(x: float, green_min: float, amber_min: float) -> str:
    if x >= green_min:
        return "GREEN"
    if x >= amber_min:
        return "AMBER"
    return "RED"

def _grade_lower_better(x: float, green_max: float, amber_max: float) -> str:
    if x <= green_max:
        return "GREEN"
    if x <= amber_max:
        return "AMBER"
    return "RED"

def diagnose(metrics: Dict[str, float], th: Thresholds) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if "c" in metrics:
        out["c"] = _grade_higher_better(metrics["c"], th.herdan_green_min, th.herdan_amber_min)
    if "yulek" in metrics:
        out["yulek"] = _grade_lower_better(metrics["yulek"], th.yulek_green_max, th.yulek_amber_max)
    if "mtld_bi" in metrics:
        out["mtld_bi"] = _grade_higher_better(metrics["mtld_bi"], th.mtld_green_min, th.mtld_amber_min)
    if "hdd" in metrics:
        out["hdd"] = _grade_higher_better(metrics["hdd"], th.hdd_green_min, th.hdd_amber_min)
    if "brunet" in metrics and th.brunet_green_max is not None and th.brunet_amber_max is not None:
        out["brunet"] = _grade_lower_better(metrics["brunet"], th.brunet_green_max, th.brunet_amber_max)
    # no default for TTR or distinct-n
    return out

def aggregate_decision(grades: Dict[str, str], require_majority: int = 2) -> Dict[str, Any]:
    """
    Majority rule: if >= require_majority metrics are RED -> RED; elif >= require_majority GREEN -> GREEN; else AMBER.
    """
    reds = sum(1 for g in grades.values() if g == "RED")
    greens = sum(1 for g in grades.values() if g == "GREEN")
    if reds >= require_majority:
        final = "RED"
        reason = f"{reds} metric(s) in RED"
    elif greens >= require_majority:
        final = "GREEN"
        reason = f"{greens} metric(s) in GREEN"
    else:
        final = "AMBER"
        reason = f"no majority; GREEN={greens}, RED={reds}"
    return {"label": final, "reason": reason, "counts": {"GREEN": greens, "RED": reds, "TOTAL": len(grades)}}

# --- Internal helpers to compute metrics from tokens -------------------------

def _metrics_for_tokens(tokens: List[str], cfg: LexDivConfig) -> Dict[str, float]:
    """Compute the selected metrics for a given token sequence (already word-only)."""
    n_tokens = len(tokens)
    parsed = _parse_metrics_arg(cfg.metrics)

    if parsed and parsed[0] == "auto":
        metrics_to_compute = _select_metrics_auto(n_tokens)
        if cfg.include_ttr:
            metrics_to_compute.append("ttr")
        if cfg.distinct_max_n > 0:
            for n in range(1, cfg.distinct_max_n + 1):
                metrics_to_compute.append(f"distinct{n}")
    else:
        metrics_to_compute = parsed

    m: Dict[str, float] = {}
    if "ttr" in metrics_to_compute:
        m["ttr"] = calc_ttr(tokens)
    if "c" in metrics_to_compute:
        m["c"] = calc_herdan_c(tokens)
    if "brunet" in metrics_to_compute:
        m["brunet"] = calc_brunet_w(tokens, alpha=cfg.brunet_alpha)
    if "yulek" in metrics_to_compute:
        m["yulek"] = calc_yules_k(tokens)
    if "mtld" in metrics_to_compute:
        m["mtld_bi"] = calc_mtld_bidirectional(tokens, threshold=cfg.mtld_threshold, min_factor_len=cfg.mtld_min_factor_len)
    if "hdd" in metrics_to_compute:
        m["hdd"] = calc_hdd(tokens, sample_size=cfg.hdd_sample_size)
    # distinct-n
    for key in list(metrics_to_compute):
        if key.startswith("distinct"):
            try:
                n = int(key.replace("distinct", "").strip())
                m[key] = calc_distinct_n(tokens, n)
            except Exception:
                pass
    return m

# --- Core analysis -----------------------------------------------------------

def analyze_lexical_diversity(text: str, cfg: LexDivConfig) -> Dict[str, Any]:
    """
    Analyze lexical diversity, optionally computing global top words and window analysis.
    """
    # Tokenize once for global analysis
    tokens_all, spans = tokenize_with_spans(
        text,
        tokenizer=cfg.tokenizer,
        lowercase=cfg.lowercase,
        remove_accents_flag=cfg.strip_accents,
    )
    word_tokens = word_only(tokens_all)
    canonical_language = resolve_language_hint(cfg.language)
    stop_words = get_stopwords_for_language(cfg.language, cfg.filter_stop_words)
    stop_words_applied = bool(stop_words)
    N_words = len(word_tokens)

    # Global metrics
    metrics = _metrics_for_tokens(word_tokens, cfg)
    grades = diagnose(metrics, cfg.thresholds)
    decision = aggregate_decision(grades, require_majority=max(1, int(cfg.require_majority)))

    # Global top words
    top_words: List[Dict[str, Any]] = []
    if cfg.top_words_k > 0:
        top_words = compute_top_words(
            word_tokens,
            top_k=int(cfg.top_words_k),
            include_positions=bool(cfg.include_positions),
            stop_words=stop_words if stop_words_applied else None,
        )

    # Window analysis
    windows: List[Dict[str, Any]] = []
    if cfg.analyze_windows:
        windows = analyze_windows(
            text=text,
            tokens_all=tokens_all,
            word_tokens=word_tokens,
            spans=spans,
            cfg=cfg,
            stop_words=stop_words if stop_words_applied else None,
        )

    # Compose output
    result: Dict[str, Any] = {
        "version": "0.2.0",
        "meta": {
            "total_tokens_all": len(tokens_all),
            "total_word_tokens": N_words,
            "tokenizer": cfg.tokenizer,
            "lowercase": cfg.lowercase,
            "strip_accents": cfg.strip_accents,
            "windows_count": len(windows),
            "language_hint": canonical_language or cfg.language,
            "stop_words_filtered": cfg.filter_stop_words and stop_words_applied,
            "stop_words_count": len(stop_words) if stop_words_applied else 0,
        },
        "settings": {
            **asdict(cfg),
            "thresholds": asdict(cfg.thresholds),
        },
        "metrics": metrics,
        "grades": grades,
        "decision": decision,
        "top_words": top_words,
        "windows": windows,
    }

    if cfg.output_mode == "compact":
        return {
            "version": result["version"],
            "meta": {"total_word_tokens": N_words, "windows_count": len(windows)},
            "metrics": metrics,
            "decision": decision,
            "top_words": top_words,
            "windows": windows,
        }
    return result

def analyze_windows(
    text: str,
    tokens_all: List[str],
    word_tokens: List[str],
    spans: List[Tuple[int, int]],
    cfg: LexDivConfig,
    stop_words: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Build windows either by token slices ("tokens" mode) or paragraphs ("paragraphs" mode).
    For each window return: char span, token span (if applicable), preview, top words, and optional metrics.
    """
    out: List[Dict[str, Any]] = []

    def _preview(s: int, e: int, limit: int) -> str:
        frag = text[s:e]
        if len(frag) <= limit:
            return frag
        return frag[:limit].rstrip() + "..."

    if cfg.window_mode == "paragraphs":
        para_spans = _split_paragraphs_with_spans(text)
        for i, (cs, ce) in enumerate(para_spans):
            # Collect word tokens whose char span lies inside paragraph
            idxs = [j for j, t in enumerate(tokens_all) if WORD_TOKEN_RE.fullmatch(t) and spans[j][0] >= cs and spans[j][1] <= ce]
            window_words = [tokens_all[j] for j in idxs]
            m: Dict[str, float] = {}
            if cfg.include_window_metrics:
                m = _metrics_for_tokens(window_words, cfg)
            topw = compute_top_words(
                window_words,
                top_k=cfg.window_top_k,
                include_positions=cfg.window_include_positions,
                stop_words=stop_words,
            )
            out.append({
                "window_id": i,
                "mode": "paragraphs",
                "char_start": cs,
                "char_end": ce,
                "token_count": len(window_words),
                "top_words": topw,
                "metrics": m,
                "text_preview": _preview(cs, ce, cfg.window_preview_chars),
            })
        return out

    # tokens mode (sliding windows)
    # Build a mapping from word token index -> char span
    word_token_indices: List[int] = [i for i, t in enumerate(tokens_all) if WORD_TOKEN_RE.fullmatch(t)]
    # Map word-index to char spans quickly
    word_spans: List[Tuple[int, int]] = [spans[i] for i in word_token_indices]
    n = len(word_tokens)
    slices = _window_slices_over_tokens(n, size=cfg.window_size, step=cfg.window_step)

    for k, (ws, we) in enumerate(slices):
        # Map to original char spans via word_spans
        if we <= 0 or we <= ws:
            continue
        cs = word_spans[ws][0]
        ce = word_spans[we - 1][1]
        window_words = word_tokens[ws:we]
        m: Dict[str, float] = {}
        if cfg.include_window_metrics:
            m = _metrics_for_tokens(window_words, cfg)
        # positions inside window: local indices 0..(we-ws-1)
        topw = compute_top_words(
            window_words,
            top_k=cfg.window_top_k,
            include_positions=cfg.window_include_positions,
            stop_words=stop_words,
        )
        out.append({
            "window_id": k,
            "mode": "tokens",
            "token_start": ws,
            "token_end": we,
            "char_start": cs,
            "char_end": ce,
            "token_count": len(window_words),
            "top_words": topw,
            "metrics": m,
            "text_preview": _preview(cs, ce, cfg.window_preview_chars),
        })
    return out

# --- File I/O ----------------------------------------------------------------

def read_text_from_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

# --- CLI ---------------------------------------------------------------------

def build_cli_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=("Lexical Diversity Analyzer -> JSON. "
                     "Auto-selects metrics by token length. Diagnoses GREEN/AMBER/RED via thresholds. "
                     "Also supports global top-words and window analysis.")
    )
    # Input
    p.add_argument("--file", type=str, default="", help="Input .txt file (utf-8). If omitted and --text not given, a small demo text is used.")
    p.add_argument("--text", type=str, default="", help="Raw text inline. If both --text and --file are provided, --text takes precedence.")

    # Tokenization
    p.add_argument("--tokenizer", type=str, default="word_punct", choices=["word_punct", "alnum"], help="Tokenizer mode.")
    p.add_argument("--lowercase", action="store_true", help="Lowercase tokens.")
    p.add_argument("--no-lowercase", action="store_true", help="Do NOT lowercase tokens (overrides --lowercase).")
    p.add_argument("--strip-accents", action="store_true", help="Strip accents (NFKD).")
    p.add_argument("--language", type=str, default="", help="Language hint for stop-word filtering (e.g., 'es', 'en').")
    p.add_argument("--keep-stop-words", action="store_true", help="Keep stop words in top-word summaries even if a language is provided.")

    # Metrics & behavior
    p.add_argument("--metrics", type=str, default="auto",
                   help='Which metrics to compute: "auto" | "all" | CSV (e.g., "mtld,hdd,yulek,c,brunet,ttr,distinct2").')
    p.add_argument("--include-ttr", action="store_true", help="Include TTR when metrics=auto.")
    p.add_argument("--distinct-max-n", type=int, default=0, help="If >0, compute distinct-n for n=1..N.")

    p.add_argument("--mtld-threshold", type=float, default=0.72, help="MTLD factor threshold (default 0.72).")
    p.add_argument("--mtld-min-factor-len", type=int, default=10, help="Minimum factor length for MTLD.")
    p.add_argument("--hdd-sample-size", type=int, default=42, help="HD-D sample size (n).")
    p.add_argument("--brunet-alpha", type=float, default=0.165, help="Brunet's W alpha.")

    # Thresholds (optional fine-tuning)
    p.add_argument("--herdan-green-min", type=float, default=0.80, help="Herdan's C GREEN minimum.")
    p.add_argument("--herdan-amber-min", type=float, default=0.65, help="Herdan's C AMBER minimum.")
    p.add_argument("--yulek-green-max", type=float, default=100.0, help="Yule's K GREEN maximum.")
    p.add_argument("--yulek-amber-max", type=float, default=150.0, help="Yule's K AMBER maximum.")
    p.add_argument("--mtld-green-min", type=float, default=70.0, help="MTLD GREEN minimum.")
    p.add_argument("--mtld-amber-min", type=float, default=50.0, help="MTLD AMBER minimum.")
    p.add_argument("--hdd-green-min", type=float, default=0.70, help="HD-D GREEN minimum.")
    p.add_argument("--hdd-amber-min", type=float, default=0.55, help="HD-D AMBER minimum.")
    p.add_argument("--brunet-green-max", type=float, default=float("nan"), help="Brunet's W GREEN maximum (lower is better). NaN = disabled.")
    p.add_argument("--brunet-amber-max", type=float, default=float("nan"), help="Brunet's W AMBER maximum. NaN = disabled.")

    p.add_argument("--decision-mode", type=str, default="consensus", choices=["consensus"], help="Decision mode.")
    p.add_argument("--require-majority", type=int, default=2, help="Majority requirement for final decision (e.g., 2 means 2-of-N).")

    # Output
    p.add_argument("--output-mode", type=str, default="full", choices=["full", "compact"], help="Output verbosity.")
    p.add_argument("--json-out", type=str, default="", help="If provided, write JSON to this path; otherwise print to stdout.")

    # --- NEW: global top words ---
    p.add_argument(
        "--top-words",
        nargs="?",
        const=25,
        type=int,
        default=0,
        help="If provided, output the top-K most frequent words globally (no stopword filtering). If given without a number, defaults to 25."
    )
    p.add_argument(
        "--include-positions",
        action="store_true",
        help="Include global token positions for top words (may increase output size)."
    )

    # --- NEW: window analysis ---
    p.add_argument("--analyze-windows", action="store_true", help="Enable window analysis to localize repetition.")
    p.add_argument("--window-mode", type=str, default="tokens", choices=["tokens", "paragraphs"], help="Windowing mode: sliding token windows or paragraph-based.")
    p.add_argument("--window-size", type=int, default=200, help="Token window size (tokens mode).")
    p.add_argument("--window-step", type=int, default=100, help="Token window step (tokens mode).")
    p.add_argument("--window-top-k", type=int, default=10, help="Top words per window.")
    p.add_argument("--window-preview-chars", type=int, default=160, help="Number of chars to include as a preview snippet per window.")
    p.add_argument("--include-window-metrics", action="store_true", help="Also compute metrics for each window (slower).")
    p.add_argument("--window-include-positions", action="store_true", help="Include local positions for window top words (indices within the window).")

    return p

def main():
    parser = build_cli_parser()
    args = parser.parse_args()

    # Lowercase logic (align with the original spirit)
    lowercase = True
    if args.no_lowercase:
        lowercase = False
    elif args.lowercase:
        lowercase = True
    language_hint = args.language.strip() or None
    filter_stop_words = not args.keep_stop_words

    # Read text
    if args.text:
        text = args.text
    elif args.file:
        text = read_text_from_file(args.file)
    else:
        text = ("This is a small demo text. "
                "It aims to exercise the lexical diversity layer and the CLI.")

    # Build thresholds
    th = Thresholds(
        herdan_green_min=float(args.herdan_green_min),
        herdan_amber_min=float(args.herdan_amber_min),
        yulek_green_max=float(args.yulek_green_max),
        yulek_amber_max=float(args.yulek_amber_max),
        mtld_green_min=float(args.mtld_green_min),
        mtld_amber_min=float(args.mtld_amber_min),
        hdd_green_min=float(args.hdd_green_min),
        hdd_amber_min=float(args.hdd_amber_min),
        brunet_green_max=(None if math.isnan(args.brunet_green_max) else float(args.brunet_green_max)),
        brunet_amber_max=(None if math.isnan(args.brunet_amber_max) else float(args.brunet_amber_max)),
    )

    # Build config (including new options)
    cfg = LexDivConfig(
        tokenizer=args.tokenizer,
        lowercase=lowercase,
        strip_accents=bool(args.strip_accents),
        language=language_hint,
        filter_stop_words=filter_stop_words,
        metrics=args.metrics,
        mtld_threshold=float(args.mtld_threshold),
        mtld_min_factor_len=int(args.mtld_min_factor_len),
        hdd_sample_size=int(args.hdd_sample_size),
        brunet_alpha=float(args.brunet_alpha),
        include_ttr=bool(args.include_ttr),
        distinct_max_n=int(args.distinct_max_n),
        thresholds=th,
        decision_mode=args.decision_mode,
        require_majority=max(1, int(args.require_majority)),
        output_mode=args.output_mode,
        # NEW: top words
        top_words_k=int(args.top_words),
        include_positions=bool(args.include_positions),
        # NEW: windows
        analyze_windows=bool(args.analyze_windows),
        window_mode=args.window_mode,
        window_size=int(args.window_size),
        window_step=int(args.window_step),
        window_top_k=int(args.window_top_k),
        window_preview_chars=int(args.window_preview_chars),
        include_window_metrics=bool(args.include_window_metrics),
        window_include_positions=bool(args.window_include_positions),
    )

    # Analyze
    result = analyze_lexical_diversity(text, cfg)

    out_json = json.dumps(result, ensure_ascii=False, indent=2)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            f.write(out_json)
    else:
        print(out_json)

if __name__ == "__main__":
    main()
