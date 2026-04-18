"""
Model synchronization services for the Gran Sabio LLM admin UI.
"""

from __future__ import annotations

import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import aiohttp

import json_utils as json


MODEL_SPECS_PATH = Path(__file__).resolve().parent.parent / "model_specs.json"
MODEL_SPECS_BACKUP_DIR = Path(__file__).resolve().parent.parent / "backups"
SUPPORTED_SYNC_PROVIDERS = ("openrouter", "xai", "openai", "anthropic", "google")
FULL_SYNC_PROVIDERS = {"openrouter", "xai"}
DISCOVERY_SYNC_PROVIDERS = {"openai", "anthropic", "google"}
OPENAI_MODEL_LIST_URL = "https://api.openai.com/v1/models"
ANTHROPIC_MODEL_LIST_URL = "https://api.anthropic.com/v1/models"
XAI_LANGUAGE_MODELS_URL = "https://api.x.ai/v1/language-models"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
GOOGLE_MODELS_URL = "https://generativelanguage.googleapis.com/v1beta/models"
ANTHROPIC_API_VERSION = "2023-06-01"


class ModelSyncError(RuntimeError):
    """Raised when model sync cannot continue safely."""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _iso_from_unix_timestamp(value: Any) -> str | None:
    if value in (None, "", 0):
        return None
    try:
        timestamp = float(value)
    except (TypeError, ValueError):
        return None
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def _price_per_million_from_token_price(value: Any) -> float:
    raw_value = _safe_float(value, default=0.0)
    if raw_value <= 0:
        return 0.0
    return round(raw_value * 1_000_000, 4)


def _price_per_million_from_cents_per_100m(value: Any) -> float:
    raw_value = _safe_float(value, default=0.0)
    if raw_value <= 0:
        return 0.0
    return round(raw_value / 10_000, 4)


def _dedupe_strings(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = str(value).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def _nested_lookup(payload: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload and payload.get(key) not in (None, ""):
            return payload.get(key)
    raw_payload = payload.get("raw")
    if isinstance(raw_payload, dict):
        for key in keys:
            if key in raw_payload and raw_payload.get(key) not in (None, ""):
                return raw_payload.get(key)
    return None


def _title_case_model_name(model_id: str) -> str:
    parts = re.split(r"[-_/]+", model_id)
    pretty_parts: list[str] = []
    for part in parts:
        if not part:
            continue
        if part.lower() in {"gpt", "xai", "api", "ui"}:
            pretty_parts.append(part.upper())
            continue
        if part.lower() == "claude":
            pretty_parts.append("Claude")
            continue
        if part.lower() == "grok":
            pretty_parts.append("Grok")
            continue
        if part.isdigit():
            pretty_parts.append(part)
            continue
        pretty_parts.append(part[:1].upper() + part[1:])
    return " ".join(pretty_parts) or model_id


def _normalize_model_hint(provider: str, model_id: str) -> str:
    normalized = model_id.strip().lower()
    if provider == "openai":
        normalized = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", normalized)
        normalized = normalized.replace("-latest", "")
    elif provider == "anthropic":
        normalized = re.sub(r"-\d{8}$", "", normalized)
    return normalized


def _is_openai_text_candidate(model_id: str) -> bool:
    lowered = model_id.lower()
    if any(
        marker in lowered
        for marker in (
            "whisper",
            "tts",
            "transcribe",
            "embedding",
            "moderation",
            "image",
            "realtime",
            "search",
            "audio",
        )
    ):
        return False

    prefixes = (
        "gpt-",
        "gpt-oss-",
        "o1",
        "o3",
        "o4",
        "codex-",
    )
    return lowered.startswith(prefixes)


def _is_google_text_candidate(model: dict) -> bool:
    """Filter Google models to only include text generation candidates."""
    methods = model.get("supportedGenerationMethods", [])
    return "generateContent" in methods


def _guess_capabilities(model_id: str, *, base_capabilities: Iterable[str] | None = None) -> list[str]:
    capabilities = list(base_capabilities or ["text"])
    model_lower = model_id.lower()

    if any(marker in model_lower for marker in ("vision", "image", "4o", "gpt-5", "grok-4", "claude")):
        capabilities.append("vision")
    if any(marker in model_lower for marker in ("reason", "thinking", "gpt-5", "o1", "o3", "o4", "opus")):
        capabilities.append("reasoning")
    if any(marker in model_lower for marker in ("code", "codex", "coder")):
        capabilities.append("coding")
    if "function" in model_lower or "gpt-" in model_lower or "grok" in model_lower:
        capabilities.append("function_calling")

    return _dedupe_strings(capabilities)


def _context_length_from_remote(model: dict[str, Any], fallback: int = 0) -> int:
    for key in (
        "context_length",
        "max_context_length",
        "max_prompt_length",
        "input_token_limit",
        "input_tokens",
    ):
        value = _safe_int(model.get(key))
        if value > 0:
            return value
    return fallback


def _output_length_from_remote(model: dict[str, Any], fallback: int = 0) -> int:
    for key in (
        "max_completion_tokens",
        "output_tokens",
        "max_output_tokens",
        "max_generated_tokens",
        "completion_token_limit",
    ):
        value = _safe_int(model.get(key))
        if value > 0:
            return value
    return fallback


class ModelSyncService:
    """Coordinates provider fetch, diff, and sync workflows."""

    def __init__(self, config: Any, logger: Any, specs_path: Path | None = None):
        self.config = config
        self.logger = logger
        self.specs_path = specs_path or MODEL_SPECS_PATH
        self.backup_dir = MODEL_SPECS_BACKUP_DIR

    def get_sync_mode(self, provider: str) -> str:
        if provider in FULL_SYNC_PROVIDERS:
            return "full"
        if provider in DISCOVERY_SYNC_PROVIDERS:
            return "discovery-assisted"
        raise ModelSyncError(f"Unsupported sync provider '{provider}'.")

    def load_specs(self) -> dict[str, Any]:
        with open(self.specs_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _get_provider_specs(self, specs: dict[str, Any], provider: str) -> dict[str, dict[str, Any]]:
        return dict(specs.get("model_specifications", {}).get(provider, {}) or {})

    def get_local_catalog(self) -> dict[str, Any]:
        specs = self.load_specs()
        providers = specs.get("model_specifications", {}) or {}
        models: list[dict[str, Any]] = []
        provider_counts: dict[str, int] = {}
        needs_review = 0

        for provider, provider_models in providers.items():
            provider_counts[provider] = len(provider_models)
            for key, model_data in provider_models.items():
                sync_meta = model_data.get("sync_metadata", {}) or {}
                status = "local_only"
                if sync_meta.get("needs_review"):
                    status = "needs_review"
                    needs_review += 1
                elif sync_meta.get("managed_by_ui"):
                    status = "synced"

                models.append(
                    {
                        "provider": provider,
                        "key": key,
                        "model_id": model_data.get("model_id", key),
                        "id": model_data.get("model_id", key),
                        "name": model_data.get("name", key),
                        "description": model_data.get("description", ""),
                        "context_window": model_data.get("context_window", model_data.get("input_tokens", 0)),
                        "input_tokens": model_data.get("input_tokens", model_data.get("context_window", 0)),
                        "output_tokens": model_data.get("output_tokens", 0),
                        "capabilities": model_data.get("capabilities", []),
                        "pricing": model_data.get("pricing", {}),
                        "verified_at": model_data.get("verified_at"),
                        "source": model_data.get("source"),
                        "source_status": status,
                        "status": status,
                        "sync_mode": sync_meta.get("sync_mode"),
                        "remote_created_at": sync_meta.get("remote_created_at"),
                        "created_at": sync_meta.get("remote_created_at"),
                        "enabled": model_data.get("enabled", True),
                    }
                )

        models.sort(key=lambda item: (item["provider"], item["name"].lower(), item["key"].lower()))
        return {
            "providers": sorted(provider_counts.keys()),
            "models": models,
            "stats": {
                "total_models": len(models),
                "needs_review": needs_review,
                "provider_counts": provider_counts,
            },
            "default_models": self.config.model_specs.get("default_models", {}),
        }

    async def fetch_remote_models(self, provider: str) -> dict[str, Any]:
        provider = provider.strip().lower()
        if provider not in SUPPORTED_SYNC_PROVIDERS:
            raise ModelSyncError(f"Unsupported provider '{provider}'.")

        fetchers = {
            "openrouter": self._fetch_openrouter_remote,
            "xai": self._fetch_xai_remote,
            "openai": self._fetch_openai_remote,
            "anthropic": self._fetch_anthropic_remote,
            "google": self._fetch_google_remote,
        }
        remote_models = await fetchers[provider]()
        merged = self._merge_remote_with_local(provider, remote_models)
        return {
            "provider": provider,
            "sync_mode": self.get_sync_mode(provider),
            "models": merged,
            "enabled_models": [item["key"] for item in merged if item.get("exists_locally")],
            "stats": {
                **self._summarize_remote_models(merged),
                "fetched_at": _utc_now_iso(),
            },
        }

    async def _fetch_openrouter_remote(self) -> list[dict[str, Any]]:
        if not self.config.OPENROUTER_API_KEY:
            raise ModelSyncError("OpenRouter API key not configured (OPENROUTER_API_KEY).")

        headers = {"Authorization": f"Bearer {self.config.OPENROUTER_API_KEY}"}
        async with aiohttp.ClientSession() as session:
            async with session.get(
                OPENROUTER_MODELS_URL,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                if response.status != 200:
                    raise ModelSyncError(f"OpenRouter API error: {await response.text()}")
                payload = await response.json()

        remote_models: list[dict[str, Any]] = []
        for model in payload.get("data", []) or []:
            model_id = str(model.get("id", "")).strip()
            if not model_id:
                continue

            pricing = model.get("pricing", {}) or {}
            architecture = model.get("architecture", {}) or {}
            input_modalities = architecture.get("input_modalities", []) or []
            capabilities = ["text"]
            if "image" in input_modalities:
                capabilities.append("vision")
            if "audio" in input_modalities:
                capabilities.append("audio")
            if any(marker in model_id.lower() for marker in ("code", "coder", "instruct")):
                capabilities.append("coding")

            context_length = _context_length_from_remote(model)
            top_provider = model.get("top_provider", {}) or {}
            output_tokens = _output_length_from_remote(
                top_provider,
                fallback=max(context_length // 4, 4096) if context_length else 4096,
            )

            remote_models.append(
                {
                    "key": model_id,
                    "id": model_id,
                    "model_id": model_id,
                    "name": model.get("name", model_id),
                    "description": model.get("description", ""),
                    "provider": "openrouter",
                    "vendor": model_id.split("/", 1)[0] if "/" in model_id else "openrouter",
                    "context_window": context_length,
                    "input_tokens": context_length,
                    "output_tokens": output_tokens,
                    "pricing": {
                        "input_per_million": _price_per_million_from_token_price(pricing.get("prompt")),
                        "output_per_million": _price_per_million_from_token_price(pricing.get("completion")),
                    },
                    "capabilities": _dedupe_strings(capabilities),
                    "supported": True,
                    "needs_review": False,
                    "sync_mode": "full",
                    "source": f"https://openrouter.ai/models/{model_id}",
                    "remote_created_at": None,
                    "local_only": False,
                }
            )

        remote_models.sort(key=lambda item: (item.get("vendor", ""), item["name"].lower()))
        return remote_models

    async def _fetch_xai_remote(self) -> list[dict[str, Any]]:
        if not self.config.XAI_API_KEY:
            raise ModelSyncError("xAI API key not configured (XAI_API_KEY).")

        headers = {"Authorization": f"Bearer {self.config.XAI_API_KEY}"}
        async with aiohttp.ClientSession() as session:
            async with session.get(
                XAI_LANGUAGE_MODELS_URL,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                if response.status != 200:
                    raise ModelSyncError(f"xAI API error: {await response.text()}")
                payload = await response.json()

        current_xai_specs = self._get_provider_specs(self.load_specs(), "xai")
        remote_models: list[dict[str, Any]] = []
        for model in payload.get("models", []) or []:
            model_id = str(model.get("id", "")).strip()
            if not model_id:
                continue

            local_template = self._find_best_local_template("xai", model_id, current_xai_specs)
            context_length = _context_length_from_remote(model, fallback=_safe_int(local_template.get("context_window")))
            output_tokens = _output_length_from_remote(model, fallback=_safe_int(local_template.get("output_tokens")))
            if context_length <= 0:
                context_length = _safe_int(local_template.get("context_window"), 128000)
            if output_tokens <= 0:
                output_tokens = _safe_int(local_template.get("output_tokens"), max(context_length // 4, 16384))

            input_modalities = model.get("input_modalities", []) or []
            capabilities = _guess_capabilities(model_id, base_capabilities=input_modalities or local_template.get("capabilities", ["text"]))
            if "text" not in capabilities:
                capabilities.insert(0, "text")

            remote_models.append(
                {
                    "key": model_id,
                    "id": model_id,
                    "model_id": model_id,
                    "name": _title_case_model_name(model_id),
                    "description": local_template.get("description", f"{_title_case_model_name(model_id)} discovered from xAI language models API."),
                    "provider": "xai",
                    "vendor": "xai",
                    "context_window": context_length,
                    "input_tokens": context_length,
                    "output_tokens": output_tokens,
                    "pricing": {
                        "input_per_million": _price_per_million_from_cents_per_100m(model.get("prompt_text_token_price")),
                        "output_per_million": _price_per_million_from_cents_per_100m(model.get("completion_text_token_price")),
                    },
                    "capabilities": capabilities,
                    "supported": True,
                    "needs_review": False,
                    "sync_mode": "full",
                    "source": f"https://docs.x.ai/developers/models/{model_id}",
                    "remote_created_at": _iso_from_unix_timestamp(model.get("created")),
                    "aliases": model.get("aliases", []),
                    "local_only": False,
                }
            )

        remote_models.sort(key=lambda item: (item.get("remote_created_at") or "", item["name"].lower()), reverse=True)
        return remote_models

    async def _fetch_openai_remote(self) -> list[dict[str, Any]]:
        if not self.config.OPENAI_API_KEY:
            raise ModelSyncError("OpenAI API key not configured (OPENAI_KEY / OPENAI_API_KEY).")

        headers = {"Authorization": f"Bearer {self.config.OPENAI_API_KEY}"}
        async with aiohttp.ClientSession() as session:
            async with session.get(
                OPENAI_MODEL_LIST_URL,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                if response.status != 200:
                    raise ModelSyncError(f"OpenAI API error: {await response.text()}")
                payload = await response.json()

        current_specs = self._get_provider_specs(self.load_specs(), "openai")
        remote_models: list[dict[str, Any]] = []
        for model in payload.get("data", []) or []:
            model_id = str(model.get("id", "")).strip()
            if not model_id or not _is_openai_text_candidate(model_id):
                continue
            remote_models.append(
                self._build_discovery_entry(
                    provider="openai",
                    model_id=model_id,
                    display_name=model_id,
                    remote_created_at=_iso_from_unix_timestamp(model.get("created")),
                    current_provider_specs=current_specs,
                )
            )

        remote_models.sort(key=lambda item: (item.get("remote_created_at") or "", item["model_id"]), reverse=True)
        return remote_models

    async def _fetch_anthropic_remote(self) -> list[dict[str, Any]]:
        if not self.config.ANTHROPIC_API_KEY:
            raise ModelSyncError("Anthropic API key not configured (ANTHROPIC_API_KEY).")

        headers = {
            "x-api-key": self.config.ANTHROPIC_API_KEY,
            "anthropic-version": ANTHROPIC_API_VERSION,
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(
                ANTHROPIC_MODEL_LIST_URL,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                if response.status != 200:
                    raise ModelSyncError(f"Anthropic API error: {await response.text()}")
                payload = await response.json()

        current_specs = self._get_provider_specs(self.load_specs(), "anthropic")
        remote_models: list[dict[str, Any]] = []
        for model in payload.get("data", []) or []:
            model_id = str(model.get("id", "")).strip()
            if not model_id or not model_id.startswith("claude-"):
                continue
            remote_models.append(
                self._build_discovery_entry(
                    provider="anthropic",
                    model_id=model_id,
                    display_name=model.get("display_name", model_id),
                    remote_created_at=model.get("created_at"),
                    current_provider_specs=current_specs,
                )
            )

        remote_models.sort(key=lambda item: (item.get("remote_created_at") or "", item["model_id"]), reverse=True)
        return remote_models

    async def _fetch_google_remote(self) -> list[dict[str, Any]]:
        """Fetch available models from Google Generative Language API."""
        api_key = self.config.GOOGLE_API_KEY
        if not api_key:
            raise ModelSyncError("GOOGLE_API_KEY / GEMINI_KEY not configured.")

        current_specs = self._get_provider_specs(self.load_specs(), "google")
        headers = {"x-goog-api-key": api_key}
        all_models: list[dict[str, Any]] = []

        async with aiohttp.ClientSession() as session:
            page_token = None
            while True:
                params = {"pageSize": "100"}
                if page_token:
                    params["pageToken"] = page_token
                async with session.get(GOOGLE_MODELS_URL, headers=headers, params=params) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        raise ModelSyncError(f"Google API returned {resp.status}: {text[:300]}")
                    data = await resp.json()

                for model in data.get("models", []):
                    if not _is_google_text_candidate(model):
                        continue

                    # Strip "models/" prefix from name
                    raw_name = model.get("name", "")
                    model_id = raw_name.replace("models/", "", 1) if raw_name.startswith("models/") else raw_name
                    if not model_id:
                        continue

                    display_name = model.get("displayName", _title_case_model_name(model_id))
                    description = model.get("description", "")
                    input_tokens = _safe_int(model.get("inputTokenLimit"))
                    output_tokens = _safe_int(model.get("outputTokenLimit"))

                    entry = self._build_discovery_entry(
                        provider="google",
                        model_id=model_id,
                        display_name=display_name,
                        remote_created_at=None,  # Google API doesn't return creation date
                        current_provider_specs=current_specs,
                        remote_description=description,
                    )
                    # Override with actual API values
                    entry["input_tokens"] = input_tokens
                    entry["output_tokens"] = output_tokens
                    entry["context_window"] = input_tokens
                    entry["source"] = GOOGLE_MODELS_URL

                    all_models.append(entry)

                page_token = data.get("nextPageToken")
                if not page_token:
                    break

        self.logger.info("Fetched %d text-generation models from Google API", len(all_models))
        return self._merge_remote_with_local("google", all_models)

    def _build_discovery_entry(
        self,
        *,
        provider: str,
        model_id: str,
        display_name: str,
        remote_created_at: str | None,
        current_provider_specs: dict[str, dict[str, Any]],
        remote_description: str = "",
    ) -> dict[str, Any]:
        existing = current_provider_specs.get(model_id)
        template = existing or self._find_best_local_template(provider, model_id, current_provider_specs)
        existing_sync_meta = dict(existing.get("sync_metadata", {}) or {}) if existing else {}
        capabilities = list(template.get("capabilities", [])) if template else []
        if not capabilities:
            capabilities = _guess_capabilities(model_id)
        suggested = not bool(existing)
        needs_review = bool(existing_sync_meta.get("needs_review")) if existing else True

        context_window = _safe_int(template.get("context_window"), 0) if template else 0
        output_tokens = _safe_int(template.get("output_tokens"), 0) if template else 0
        input_tokens = _safe_int(template.get("input_tokens"), context_window) if template else context_window

        if provider == "openai" and context_window <= 0:
            context_window = 128000
        elif provider == "anthropic" and context_window <= 0:
            context_window = 200000
        if input_tokens <= 0:
            input_tokens = context_window
        if output_tokens <= 0:
            output_tokens = max(context_window // 4, 8192) if context_window else 8192

        # Description priority: remote API > existing local entry > auto-generated disclaimer.
        # Always prefer remote_description when available (avoids stale local descriptions
        # causing perpetual "updated" status in the sync UI).
        description = remote_description or (existing.get("description", "") if existing else "")
        pricing = {"input_per_million": 0.0, "output_per_million": 0.0}
        if existing:
            pricing = dict(existing.get("pricing", {}) or pricing)
        elif template:
            pricing = dict(template.get("pricing", {}) or pricing)
        if not description and not existing:
            # Only add disclaimer for brand-new discoveries, not for models
            # already in the catalog (avoids perpetual fingerprint mismatches)
            description = (
                f"{display_name} auto-discovered from the {provider.title()} models API. "
                "Review metadata before relying on it in production."
            )

        source_urls = {
            "openai": "https://developers.openai.com/api/reference/resources/models/methods/list",
            "anthropic": "https://docs.anthropic.com/en/api/models-list",
            "google": GOOGLE_MODELS_URL,
        }
        source_url = source_urls.get(provider, "")

        return {
            "key": model_id,
            "id": model_id,
            "model_id": model_id,
            "name": existing.get("name", _title_case_model_name(display_name)) if existing else _title_case_model_name(display_name),
            "description": description,
            "provider": provider,
            "vendor": provider,
            "context_window": context_window,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "pricing": {
                "input_per_million": round(_safe_float(pricing.get("input_per_million")), 4),
                "output_per_million": round(_safe_float(pricing.get("output_per_million")), 4),
            },
            "capabilities": _dedupe_strings(capabilities or ["text"]),
            "supported": bool(template) or provider in DISCOVERY_SYNC_PROVIDERS,
            "needs_review": needs_review,
            "sync_mode": "discovery-assisted",
            "source": source_url,
            "remote_created_at": remote_created_at,
            "local_only": False,
            "suggested_from": template.get("model_id", template.get("name")) if template else None,
            "suggested": suggested,
        }

    def _find_best_local_template(
        self,
        provider: str,
        model_id: str,
        current_provider_specs: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        normalized_target = _normalize_model_hint(provider, model_id)
        best_score = -1
        best_template: dict[str, Any] | None = None

        for key, spec in current_provider_specs.items():
            candidates = [key]
            model_identifier = spec.get("model_id")
            if model_identifier:
                candidates.append(str(model_identifier))

            for candidate in candidates:
                normalized_candidate = _normalize_model_hint(provider, candidate)
                if normalized_candidate == normalized_target:
                    return spec
                common_prefix = os.path.commonprefix([normalized_candidate, normalized_target])
                score = len(common_prefix)
                if score > best_score and score >= 4:
                    best_score = score
                    best_template = spec

        return dict(best_template or {})

    def _merge_remote_with_local(self, provider: str, remote_models: list[dict[str, Any]]) -> list[dict[str, Any]]:
        specs = self.load_specs()
        current_provider_specs = self._get_provider_specs(specs, provider)
        merged: list[dict[str, Any]] = []
        remote_keys: set[str] = set()

        for model in remote_models:
            key = model["key"]
            remote_keys.add(key)
            local_spec = current_provider_specs.get(key)
            entry = dict(model)
            entry["exists_locally"] = local_spec is not None
            entry["status"] = self._determine_remote_status(provider, entry, local_spec)
            merged.append(entry)

        if provider in FULL_SYNC_PROVIDERS:
            for key, local_spec in current_provider_specs.items():
                if key in remote_keys:
                    continue
                merged.append(
                    {
                        "key": key,
                        "id": local_spec.get("model_id", key),
                        "model_id": local_spec.get("model_id", key),
                        "name": local_spec.get("name", key),
                        "description": local_spec.get("description", ""),
                        "provider": provider,
                        "vendor": provider,
                        "context_window": local_spec.get("context_window", local_spec.get("input_tokens", 0)),
                        "input_tokens": local_spec.get("input_tokens", local_spec.get("context_window", 0)),
                        "output_tokens": local_spec.get("output_tokens", 0),
                        "pricing": local_spec.get("pricing", {}),
                        "capabilities": local_spec.get("capabilities", []),
                        "supported": True,
                        "needs_review": False,
                        "sync_mode": self.get_sync_mode(provider),
                        "source": local_spec.get("source"),
                        "remote_created_at": (local_spec.get("sync_metadata", {}) or {}).get("remote_created_at"),
                        "local_only": True,
                        "exists_locally": True,
                        "status": "stale",
                    }
                )

        sort_key = (
            (lambda item: (item.get("vendor", ""), item["name"].lower()))
            if provider == "openrouter"
            else (lambda item: ((item.get("remote_created_at") or ""), item["name"].lower()))
        )
        merged.sort(key=sort_key, reverse=provider in {"openai", "anthropic", "xai"})
        return merged

    def _determine_remote_status(
        self,
        provider: str,
        remote_model: dict[str, Any],
        local_spec: dict[str, Any] | None,
    ) -> str:
        if local_spec is None:
            return "review" if provider in DISCOVERY_SYNC_PROVIDERS else "new"
        if provider in DISCOVERY_SYNC_PROVIDERS:
            sync_meta = local_spec.get("sync_metadata", {}) or {}
            return "review" if sync_meta.get("needs_review") else "enabled"

        generated_spec = self._build_provider_spec(provider, remote_model, local_spec)
        relevant_fields = (
            "model_id",
            "name",
            "description",
            "input_tokens",
            "output_tokens",
            "context_window",
            "pricing",
            "capabilities",
            "source",
        )
        has_changes = any(local_spec.get(field) != generated_spec.get(field) for field in relevant_fields)
        return "update" if has_changes else "enabled"

    def _summarize_remote_models(self, models: list[dict[str, Any]]) -> dict[str, int]:
        summary = {
            "total": len(models),
            "enabled": 0,
            "new": 0,
            "update": 0,
            "review": 0,
            "stale": 0,
        }
        for model in models:
            status = model.get("status")
            if status in summary:
                summary[status] += 1
        return summary

    def sync_provider(self, provider: str, selected_models: list[dict[str, Any]]) -> dict[str, Any]:
        provider = provider.strip().lower()
        if provider not in SUPPORTED_SYNC_PROVIDERS:
            raise ModelSyncError(f"Unsupported provider '{provider}'.")
        if not isinstance(selected_models, list):
            raise ModelSyncError("Invalid sync payload: 'models' must be a list.")

        specs = self.load_specs()
        current_provider_specs = self._get_provider_specs(specs, provider)
        current_keys = set(current_provider_specs.keys())
        new_provider_specs: dict[str, dict[str, Any]]

        if provider in FULL_SYNC_PROVIDERS:
            new_provider_specs = {}
            for model in selected_models:
                key = str(
                    _nested_lookup(model, "key", "model_id", "id", "model")
                    or ""
                ).strip()
                if not key:
                    continue
                current_spec = current_provider_specs.get(key)
                if bool(_nested_lookup(model, "local_only")) and current_spec:
                    new_provider_specs[key] = dict(current_spec)
                    continue
                new_provider_specs[key] = self._build_provider_spec(provider, model, current_spec)
        else:
            new_provider_specs = dict(current_provider_specs)
            for model in selected_models:
                key = str(
                    _nested_lookup(model, "key", "model_id", "id", "model")
                    or ""
                ).strip()
                if not key:
                    continue
                current_spec = new_provider_specs.get(key)
                new_provider_specs[key] = self._build_provider_spec(provider, model, current_spec)

        if "model_specifications" not in specs:
            specs["model_specifications"] = {}
        specs["model_specifications"][provider] = new_provider_specs
        self._validate_model_references(specs)

        added_keys = set(new_provider_specs.keys()) - current_keys
        removed_keys = current_keys - set(new_provider_specs.keys())
        updated = 0
        for key in set(new_provider_specs.keys()) & current_keys:
            if current_provider_specs.get(key) != new_provider_specs.get(key):
                updated += 1

        backup_path = self._create_backup()
        self._write_specs_atomic(specs)
        self.config.reload_model_specifications()

        self.logger.info(
            "Model sync complete for provider=%s added=%d updated=%d removed=%d backup=%s",
            provider,
            len(added_keys),
            updated,
            len(removed_keys),
            backup_path,
        )

        return {
            "success": True,
            "provider": provider,
            "sync_mode": self.get_sync_mode(provider),
            "added": len(added_keys),
            "updated": updated,
            "removed": len(removed_keys),
            "backup_path": str(backup_path),
            "reloaded": True,
            "message": (
                f"{provider} sync complete. "
                f"Added {len(added_keys)}, updated {updated}, removed {len(removed_keys)}."
            ),
        }

    def delete_model(self, provider: str, model_id: str) -> dict[str, Any]:
        """Remove a single model from the catalog."""
        model_id = model_id.strip()
        if not model_id or len(model_id) > 256:
            raise ModelSyncError("Invalid model_id.")
        provider = provider.strip().lower()
        if provider not in SUPPORTED_SYNC_PROVIDERS:
            raise ModelSyncError(f"Unsupported provider '{provider}'.")

        specs = self.load_specs()
        provider_specs = self._get_provider_specs(specs, provider)
        if model_id not in provider_specs:
            raise ModelSyncError(f"Model '{model_id}' not found in {provider}.")

        del provider_specs[model_id]
        specs["model_specifications"][provider] = provider_specs
        self._validate_model_references(specs)

        backup_path = self._create_backup()
        self._write_specs_atomic(specs)
        self.config.reload_model_specifications()

        self.logger.info("Deleted model %s from provider %s, backup=%s", model_id, provider, backup_path)
        return {"success": True, "deleted": model_id, "provider": provider}

    def toggle_model(self, provider: str, model_id: str, enabled: bool) -> dict[str, Any]:
        """Enable or disable a model in the catalog."""
        model_id = model_id.strip()
        if not model_id or len(model_id) > 256:
            raise ModelSyncError("Invalid model_id.")
        provider = provider.strip().lower()
        if provider not in SUPPORTED_SYNC_PROVIDERS:
            raise ModelSyncError(f"Unsupported provider '{provider}'.")

        specs = self.load_specs()
        provider_specs = self._get_provider_specs(specs, provider)
        if model_id not in provider_specs:
            raise ModelSyncError(f"Model '{model_id}' not found in {provider}.")

        if not enabled:
            # Prevent disabling models referenced by defaults or aliases.
            # Temporarily remove and validate (same pattern as delete_model).
            test_specs = json.loads(json.dumps(specs))
            del test_specs["model_specifications"][provider][model_id]
            self._validate_model_references(test_specs)

        provider_specs[model_id]["enabled"] = enabled
        self._create_backup()
        self._write_specs_atomic(specs)
        self.config.reload_model_specifications()

        self.logger.info("Toggled model %s in %s: enabled=%s", model_id, provider, enabled)
        return {"success": True, "model_id": model_id, "provider": provider, "enabled": enabled}

    def sync_providers_bulk(self, providers: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        """Sync multiple providers in a single atomic operation."""
        specs = self.load_specs()
        per_provider_stats: dict[str, dict[str, Any]] = {}

        for provider, selected_models in providers.items():
            provider = provider.strip().lower()
            if provider not in SUPPORTED_SYNC_PROVIDERS:
                raise ModelSyncError(f"Unsupported provider '{provider}'.")
            if not isinstance(selected_models, list):
                raise ModelSyncError(f"Invalid payload for '{provider}': must be a list.")

            current_provider_specs = self._get_provider_specs(specs, provider)
            current_keys = set(current_provider_specs.keys())
            new_provider_specs: dict[str, dict[str, Any]]

            if provider in FULL_SYNC_PROVIDERS:
                new_provider_specs = {}
                for model in selected_models:
                    key = str(
                        _nested_lookup(model, "key", "model_id", "id", "model") or ""
                    ).strip()
                    if not key:
                        continue
                    current_spec = current_provider_specs.get(key)
                    if bool(_nested_lookup(model, "local_only")) and current_spec:
                        new_provider_specs[key] = dict(current_spec)
                        continue
                    new_provider_specs[key] = self._build_provider_spec(provider, model, current_spec)
            else:
                new_provider_specs = dict(current_provider_specs)
                for model in selected_models:
                    key = str(
                        _nested_lookup(model, "key", "model_id", "id", "model") or ""
                    ).strip()
                    if not key:
                        continue
                    current_spec = new_provider_specs.get(key)
                    new_provider_specs[key] = self._build_provider_spec(provider, model, current_spec)

            if "model_specifications" not in specs:
                specs["model_specifications"] = {}
            specs["model_specifications"][provider] = new_provider_specs

            added_keys = set(new_provider_specs.keys()) - current_keys
            removed_keys = current_keys - set(new_provider_specs.keys())
            updated = sum(
                1 for key in set(new_provider_specs.keys()) & current_keys
                if current_provider_specs.get(key) != new_provider_specs.get(key)
            )
            per_provider_stats[provider] = {
                "added": len(added_keys),
                "updated": updated,
                "removed": len(removed_keys),
            }

        self._validate_model_references(specs)
        backup_path = self._create_backup()
        self._write_specs_atomic(specs)
        self.config.reload_model_specifications()

        self.logger.info(
            "Bulk sync complete: %s, backup=%s",
            ", ".join(f"{p}: +{s['added']}~{s['updated']}-{s['removed']}" for p, s in per_provider_stats.items()),
            backup_path,
        )

        return {
            "success": True,
            "providers": per_provider_stats,
            "backup_path": str(backup_path),
            "message": "Bulk sync complete.",
        }

    def _build_provider_spec(
        self,
        provider: str,
        model: dict[str, Any],
        current_spec: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        current_spec = dict(current_spec or {})
        key = str(_nested_lookup(model, "key", "model_id", "id", "model") or "").strip()
        model_id = str(_nested_lookup(model, "model_id", "id", "key", "model") or key).strip()
        name = str(model.get("name") or current_spec.get("name") or _title_case_model_name(model_id))
        description = str(model.get("description") or current_spec.get("description") or "")
        context_window = _safe_int(model.get("context_window"), _safe_int(current_spec.get("context_window")))
        input_tokens = _safe_int(model.get("input_tokens"), _safe_int(current_spec.get("input_tokens"), context_window))
        output_tokens = _safe_int(model.get("output_tokens"), _safe_int(current_spec.get("output_tokens")))
        if input_tokens <= 0:
            input_tokens = context_window
        if context_window <= 0:
            context_window = input_tokens
        if output_tokens <= 0:
            output_tokens = max(context_window // 4, 4096) if context_window else 4096

        pricing = model.get("pricing") or {}
        capabilities = model.get("capabilities") or current_spec.get("capabilities") or ["text"]
        sync_mode = self.get_sync_mode(provider)
        sync_meta = dict(current_spec.get("sync_metadata", {}) or {})
        sync_meta.update(
            {
                "managed_by_ui": True,
                "provider_sync": provider,
                "sync_mode": sync_mode,
                "last_synced_at": _utc_now_iso(),
                "remote_created_at": _nested_lookup(model, "remote_created_at", "created_at"),
                "needs_review": bool(
                    _nested_lookup(model, "needs_review")
                    if _nested_lookup(model, "needs_review") is not None
                    else provider in DISCOVERY_SYNC_PROVIDERS
                ),
            }
        )

        new_spec = dict(current_spec)
        new_spec.update(
            {
                "model_id": model_id,
                "name": name,
                "description": description,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "context_window": context_window,
                "pricing": {
                    "input_per_million": round(_safe_float(pricing.get("input_per_million")), 4),
                    "output_per_million": round(_safe_float(pricing.get("output_per_million")), 4),
                },
                "capabilities": _dedupe_strings(capabilities or ["text"]),
                "verified_at": _utc_now_iso(),
                "source": _nested_lookup(model, "source", "url") or current_spec.get("source"),
                "sync_metadata": sync_meta,
            }
        )

        if provider in DISCOVERY_SYNC_PROVIDERS and sync_meta.get("needs_review"):
            if "Review metadata before relying on it in production." not in new_spec["description"]:
                new_spec["description"] = (
                    f"{new_spec['description']} Review metadata before relying on it in production."
                ).strip()

        return new_spec

    def _collect_model_identifiers(self, specs: dict[str, Any]) -> set[str]:
        identifiers: set[str] = set()
        for provider_models in (specs.get("model_specifications", {}) or {}).values():
            for key, model in (provider_models or {}).items():
                identifiers.add(str(key))
                model_id = model.get("model_id")
                if model_id:
                    identifiers.add(str(model_id))
        return identifiers

    def _validate_model_references(self, specs: dict[str, Any]) -> None:
        identifiers = self._collect_model_identifiers(specs)
        missing_references: list[str] = []

        defaults = specs.get("default_models", {}) or {}
        for key, value in defaults.items():
            if isinstance(value, str) and value and value not in identifiers:
                missing_references.append(f"default_models.{key} -> {value}")
            elif isinstance(value, list):
                for index, model_name in enumerate(value):
                    if model_name and model_name not in identifiers:
                        missing_references.append(f"default_models.{key}[{index}] -> {model_name}")

        for alias, target in (specs.get("aliases", {}) or {}).items():
            if target and target not in identifiers:
                missing_references.append(f"aliases.{alias} -> {target}")

        if missing_references:
            raise ModelSyncError(
                "Sync blocked because it would leave invalid model references: "
                + "; ".join(missing_references)
            )

    def _create_backup(self) -> Path:
        self.backup_dir.mkdir(exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"model_specs_{timestamp}.json"
        shutil.copy2(self.specs_path, backup_path)
        return backup_path

    def _write_specs_atomic(self, specs: dict[str, Any]) -> None:
        temp_path = self.specs_path.with_suffix(".json.tmp")
        payload = json.dumps(specs, indent=2, ensure_ascii=False)
        with open(temp_path, "w", encoding="utf-8") as handle:
            handle.write(payload)
            if not payload.endswith("\n"):
                handle.write("\n")
        temp_path.replace(self.specs_path)
