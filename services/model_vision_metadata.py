"""Read explicit image-input declarations during catalog sync, never generation."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Callable, Iterable, Mapping

import aiohttp

OPENAI_DOCS_BASE = "https://developers.openai.com/api/docs/models"
GEMINI_DOCS_BASE = "https://ai.google.dev/gemini-api/docs/models"

_DOCS_HEADERS = {"Accept": "text/markdown", "Accept-Encoding": "gzip, deflate"}
_GEMINI_INPUT_TERMS = {
    "text": "text", "image": "image", "images": "image", "video": "video",
    "videos": "video", "audio": "audio", "pdf": "pdf",
}


def api_image_support(model: Mapping[str, Any]) -> bool | None:
    """Distinguish an explicit yes/no from missing or malformed API metadata."""
    capabilities = model.get("capabilities")
    image = capabilities.get("image_input") if isinstance(capabilities, Mapping) else None
    supported = image.get("supported") if isinstance(image, Mapping) else None
    if isinstance(supported, bool):
        return supported
    modalities = model.get("input_modalities")
    if isinstance(modalities, list) and modalities and all(isinstance(m, str) for m in modalities):
        normalized = {m.strip().lower() for m in modalities}
        if normalized <= {"text", "image", "audio", "video"}:
            return "image" in normalized
    return None


def parse_openai_model_vision(markdown: str) -> tuple[set[str], bool | None]:
    """Parse published Markdown fields, not prose or model-name heuristics.

    Only the canonical ID and explicitly listed snapshots/aliases share the
    declaration. A changed/ambiguous document format supplies no new evidence.
    """
    model_ids = re.findall(r"^Model ID: `([a-zA-Z0-9._-]+)`\s*$", markdown, re.MULTILINE)
    modalities = re.findall(r"^- Input modalities: ([a-z, ]+)\s*$", markdown, re.MULTILINE)
    if len(model_ids) != 1 or len(modalities) != 1:
        return set(), None
    supported = api_image_support({"input_modalities": modalities[0].split(",")})
    if supported is None:
        return set(), None
    ids = set(model_ids)
    snapshots = re.search(r"^## Snapshots\s*\n(.*?)(?=^## |\Z)", markdown, re.MULTILINE | re.DOTALL)
    if snapshots:
        ids.update(re.findall(r"^- `([a-zA-Z0-9._-]+)`", snapshots[1], re.MULTILINE))
    return ids, supported


def _gemini_inputs_include_image(cell: str) -> bool | None:
    """Read the bold Inputs list of a data-types cell; unknown terms are no evidence."""
    match = re.fullmatch(r"\*\*Inputs?\*\*\s+(.+?)\s+\*\*Outputs?\*\*.*", cell)
    if not match:
        return None
    # Parenthetical limits such as "Audio (up to 1 hour)" qualify one input.
    listed = re.sub(r"\([^()]*\)", " ", match[1])
    terms = [term.strip().lower() for term in re.split(r",|\band\b", listed)]
    terms = [term for term in terms if term]
    if not terms or any(term not in _GEMINI_INPUT_TERMS for term in terms):
        return None
    return "image" in {_GEMINI_INPUT_TERMS[term] for term in terms}


def parse_gemini_model_vision(markdown: str) -> dict[str, bool]:
    """Parse each model property table, not prose or model-name heuristics.

    A table declares image input only through exactly one "Model code" row and
    one "Supported data types" row. Every ID in that code cell shares the
    declaration; the Versions row mixes in related models and is ignored.
    """
    declarations: dict[str, bool] = {}
    conflicts: set[str] = set()
    for table in re.findall(r"(?:^\|.*\|[ \t]*(?:\n|\Z))+", markdown, re.MULTILINE):
        rows: dict[str, list[str]] = {}
        for line in table.splitlines():
            row = re.fullmatch(r"\|\s*([^|]+?)\s*\|\s*(.*?)\s*\|\s*", line)
            if row:
                rows.setdefault(row[1], []).append(row[2])
        codes = rows.get("Model code", [])
        data_types = rows.get("Supported data types", [])
        if len(codes) != 1 or len(data_types) != 1:
            continue
        model_ids = re.findall(r"`([a-zA-Z0-9._-]+)`", codes[0])
        supported = _gemini_inputs_include_image(data_types[0])
        if not model_ids or supported is None:
            continue
        for model_id in model_ids:
            if model_id in conflicts:
                continue
            if declarations.get(model_id, supported) != supported:
                del declarations[model_id]
                conflicts.add(model_id)
                continue
            declarations[model_id] = supported
    return declarations


async def _read_markdown(session: aiohttp.ClientSession, url: str) -> str:
    try:
        async with session.get(url, headers=_DOCS_HEADERS, timeout=aiohttp.ClientTimeout(total=8)) as response:
            if response.status == 200:
                return await response.text()
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError):
        pass
    return ""


async def _collect_docs_declarations(
    session: aiohttp.ClientSession,
    urls: Iterable[str],
    parse_page: Callable[[str], Mapping[str, bool]],
    model_ids: set[str],
    logger: logging.Logger,
    provider_label: str,
) -> dict[str, tuple[bool, str]]:
    """Read pages with bounded parallelism; conflicting declarations cancel out."""
    declarations: dict[str, tuple[bool, str]] = {}
    conflicts: set[str] = set()
    semaphore = asyncio.Semaphore(6)

    async def read_page(url: str) -> None:
        async with semaphore:
            page = parse_page(await _read_markdown(session, url))
        for mid, supported in page.items():
            if mid not in model_ids or mid in conflicts:
                continue
            if mid in declarations and declarations[mid][0] != supported:
                del declarations[mid]
                conflicts.add(mid)
                continue
            declarations[mid] = (supported, url)

    try:
        await asyncio.wait_for(asyncio.gather(*(read_page(url) for url in urls)), timeout=30)
    except asyncio.TimeoutError:
        logger.warning("%s vision documentation refresh timed out; preserving unrefreshed capabilities.", provider_label)
    missing = model_ids - declarations.keys()
    if missing:
        logger.warning(
            "%s has no fresh image-input declaration for %s; preserving local values.",
            provider_label, ", ".join(sorted(missing)),
        )
    return declarations


def _openai_page_declarations(markdown: str) -> dict[str, bool]:
    ids, supported = parse_openai_model_vision(markdown)
    return {} if supported is None else dict.fromkeys(ids, supported)


async def fetch_openai_vision_metadata(
    session: aiohttp.ClientSession, model_ids: set[str], logger: logging.Logger,
) -> dict[str, tuple[bool, str]]:
    """Supplement /models with official modality fields and exact snapshot IDs.

    The model-list API currently has no modalities. Read its public docs only
    on an explicit catalog refresh, with bounded parallelism and no API key.
    Unavailable docs never turn an existing declaration into an unsupported one.
    """
    if not model_ids:
        return {}
    index = await _read_markdown(session, OPENAI_DOCS_BASE + ".md")
    # These are machine-formatted documentation links. Prefixes only select
    # candidate pages; they NEVER establish that a model has vision.
    slugs = {
        path.removesuffix(".md")
        for path in re.findall(r"\]\(/api/docs/models/([a-zA-Z0-9._-]+)\)", index)
    }
    slugs = {slug for slug in slugs if any(mid == slug or mid.startswith(slug + "-") for mid in model_ids)}
    urls = [f"{OPENAI_DOCS_BASE}/{slug}.md" for slug in sorted(slugs)]
    return await _collect_docs_declarations(
        session, urls, _openai_page_declarations, model_ids, logger, "OpenAI",
    )


async def fetch_gemini_vision_metadata(
    session: aiohttp.ClientSession, model_ids: set[str], logger: logging.Logger,
) -> dict[str, tuple[bool, str]]:
    """Supplement the Gemini Models API, which publishes no input modalities.

    Read each model's official Markdown page only on an explicit catalog
    refresh, with bounded parallelism and no API key. A page only declares the
    IDs in its own "Model code" cell, so a missing or renamed page supplies no
    evidence and never turns an existing declaration into an unsupported one.
    """
    if not model_ids:
        return {}
    # Machine IDs from the provider API select the page; they never establish vision.
    urls = [
        f"{GEMINI_DOCS_BASE}/{mid}.md.txt"
        for mid in sorted(model_ids)
        if re.fullmatch(r"[a-zA-Z0-9._-]+", mid)
    ]
    return await _collect_docs_declarations(
        session, urls, parse_gemini_model_vision, model_ids, logger, "Gemini",
    )
