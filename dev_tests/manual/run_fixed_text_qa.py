"""Prepare or execute exactly one fixed-text QA cell through the engine's QAEngine.

Usage:
    python dev_tests/manual/run_fixed_text_qa.py manifest.json output --cell ID
    python dev_tests/manual/run_fixed_text_qa.py manifest.json output --cell ID \
        --max-estimated-cost-usd AMOUNT --execute

The manifest has a ``cells`` list. Each cell declares ``id``, ``request`` (JSON
path), ``candidate`` (UTF-8 text path), ``layer`` (exact request layer name),
``qa_model`` (QAModelConfig), ``estimated_max_cost_usd``, and ``max_attempts: 1``.
Paths are relative to the manifest. The caller supplies the estimate from its
verified model prices and token allowances; it is not an engine spending cap.
Only the selected cell runs. Its output directory must not already exist, even
after preparation; use a separate output root for a preview and a live run.

No service is started, no candidate is generated, and no edits are applied.
Live calls require --execute and an explicit local estimate ceiling. The helper
blocks a second evaluator dispatch, including the normal schema fallback, and
disables supported provider SDK retries in its own isolated Python process.
It does not enforce a cumulative budget across separately launched processes.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json_utils as json  # noqa: E402
from ai_service import AIService  # noqa: E402
from config import config  # noqa: E402
from models import ContentRequest, QAModelConfig  # noqa: E402
from qa_engine import QAEngine  # noqa: E402
from qa_evaluation_service import QAEvaluationService  # noqa: E402
from usage_tracking import UsageTracker  # noqa: E402


class Cell(BaseModel):
    """One explicitly budgeted candidate/layer/evaluator combination."""

    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,95}$")
    request: Path
    candidate: Path
    layer: str = Field(min_length=1)
    qa_model: QAModelConfig
    estimated_max_cost_usd: float = Field(gt=0, allow_inf_nan=False)
    max_attempts: Literal[1]

    @model_validator(mode="after")
    def require_explicit_limits(self) -> Cell:
        """Reject unlimited output or time before preparing a paid call."""
        if self.qa_model.max_tokens is None or self.qa_model.timeout_seconds is None:
            raise ValueError("qa_model must declare max_tokens and timeout_seconds")
        return self


class Manifest(BaseModel):
    """A selectable inventory, never an automatically executable batch."""

    model_config = ConfigDict(extra="forbid")
    cells: list[Cell] = Field(min_length=1)

    @model_validator(mode="after")
    def reject_duplicate_ids(self) -> Manifest:
        """Keep output identity unambiguous."""
        if len({cell.id for cell in self.cells}) != len(self.cells):
            raise ValueError("Cell IDs must be unique")
        return self


def save_json(path: Path, value: Any) -> None:
    """Write one artifact without replacing existing evidence."""
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


class SingleAttemptService:
    """Constrain normal AIService dispatch without replacing QA behavior."""

    def __init__(self, service: AIService, output: Path):
        self.service = service
        self.output = output
        self.attempts = 0

    async def generate_content(self, **kwargs: Any) -> str:
        """Reject retries before they can make another paid provider call."""
        if self.attempts:
            raise RuntimeError("The selected QA cell permits one evaluator dispatch only")
        self.attempts += 1
        kwargs["strict_single_generation_attempt"] = True
        response = await self.service.generate_content(**kwargs)
        (self.output / "raw_response.txt").write_text(response, encoding="utf-8")
        return response


async def disable_sdk_retries(service: AIService, provider: str) -> None:
    """Configure only this process's selected provider for a single attempt."""
    client_names = {
        "openai": ("openai_client",),
        "claude": ("anthropic_client",),
        "openrouter": ("openrouter_client",),
        "xai": ("xai_client",),
    }
    if provider in client_names:
        for name in client_names[provider]:
            client = getattr(service, name)
            if client is None:
                raise RuntimeError(f"The selected provider client is unavailable: {name}")
            # SDK clones retain their existing HTTP transport; service.close()
            # closes that transport when this isolated run finishes.
            setattr(service, name, client.with_options(max_retries=0))
        return
    if provider == "gemini":
        from google import genai
        from google.genai import types

        previous = service.google_new_client
        if previous is None:
            raise RuntimeError("The selected Google client is unavailable")
        await previous.aio.aclose()
        previous.close()
        service.google_new_client = genai.Client(
            api_key=config.GOOGLE_API_KEY,
            http_options=types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=1)),
        )
        return
    raise ValueError("This runner supports direct OpenAI, Anthropic, Google, xAI, and OpenRouter only")


async def execute_cell(
    cell: Cell, request: ContentRequest, candidate: str, output: Path, provider: str
) -> bool:
    """Evaluate one unchanged candidate and retain verdict, usage, and timing."""
    tracker = UsageTracker(detail_level=2)
    raw_usage: list[dict[str, Any]] = []
    tracked_callback = tracker.create_callback(
        phase="qa", role="evaluation", iteration=1, layer=cell.layer,
        metadata={"cell_id": cell.id, "qa_model": cell.qa_model.model},
    )

    def capture_usage(payload: dict[str, Any]) -> None:
        raw_usage.append(dict(payload))
        tracked_callback(payload)

    started = time.monotonic()
    service = AIService()
    guarded = SingleAttemptService(service, output)
    succeeded = False
    try:
        await disable_sdk_retries(service, provider)
        engine = QAEngine(ai_service=guarded)
        layer = next(layer for layer in request.qa_layers if layer.name == cell.layer)
        evaluation = await asyncio.wait_for(
            engine.evaluate_content(
                content=candidate,
                layer=layer,
                model=cell.qa_model,
                original_request=request,
                request_edit_info=False,
                usage_callback=capture_usage,
                request_timeout=cell.qa_model.timeout_seconds,
            ),
            timeout=cell.qa_model.timeout_seconds,
        )
        save_json(output / "evaluation.json", evaluation.model_dump(mode="json"))
        succeeded = True
    except Exception as exc:
        save_json(output / "failure.json", {"type": type(exc).__name__, "message": str(exc)})
    finally:
        save_json(output / "usage.json", {
            "raw_events": raw_usage,
            "costs": tracker.build_summary(2),
            "query_stats": tracker.build_query_stats(3),
        })
        save_json(output / "execution.json", {
            "cell_id": cell.id,
            "succeeded": succeeded,
            "evaluator_dispatches": guarded.attempts,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "estimated_max_cost_usd": cell.estimated_max_cost_usd,
            "budget_note": "Caller estimate gate, not an enforced provider charge ceiling.",
        })
        await service.close()
    return succeeded


def main() -> None:
    """Prepare one cell; require explicit execution and estimate approval."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--max-estimated-cost-usd", type=float)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    # Provider exceptions are retained locally; console output is an explicit
    # status summary rather than SDK logs or request/configuration objects.
    logging.disable(logging.CRITICAL)

    manifest = Manifest.model_validate(json.loads(args.manifest.read_text(encoding="utf-8")))
    selected = [cell for cell in manifest.cells if cell.id == args.cell]
    if len(selected) != 1:
        raise ValueError("The selected cell ID does not exist")
    cell = selected[0]
    if args.execute:
        ceiling = args.max_estimated_cost_usd
        if ceiling is None or not math.isfinite(ceiling) or ceiling <= 0:
            raise ValueError("Execution requires a positive --max-estimated-cost-usd")
        if cell.estimated_max_cost_usd > ceiling:
            raise ValueError("The selected cell estimate exceeds the explicit local ceiling")

    source = args.manifest.resolve().parent
    original_payload = json.loads((source / cell.request).read_text(encoding="utf-8"))
    request = ContentRequest.model_validate(original_payload)
    if request.context_documents or request.images:
        raise ValueError("Fixed-text QA requires resolved inline evidence; attachments are unsupported")
    matching_layers = [layer for layer in request.qa_layers if layer.name == cell.layer]
    if len(matching_layers) != 1:
        raise ValueError("The selected layer must occur exactly once in request.qa_layers")
    request.qa_tools_mode = "never"
    request.smart_editing_mode = "never"
    candidate = (source / cell.candidate).read_text(encoding="utf-8")
    if not candidate.strip():
        raise ValueError("The fixed candidate must not be empty")
    provider = config.get_model_info(cell.qa_model.model, require_api_key=False)["provider"]
    if provider not in {"openai", "claude", "gemini", "xai", "openrouter"}:
        raise ValueError("The selected provider is not supported by this bounded runner")
    if provider == "openrouter":
        options = cell.qa_model.provider_options or {}
        if options.get("allow_fallbacks") is not False:
            raise ValueError("OpenRouter QA cells must explicitly set provider_options.allow_fallbacks=false")

    rendered = QAEvaluationService.render_qa_evaluation_prompt(
        content=candidate,
        criteria=matching_layers[0].criteria,
        layer_name=cell.layer,
        min_score=matching_layers[0].min_score,
        deal_breaker_criteria=matching_layers[0].deal_breaker_criteria,
        concise_on_pass=matching_layers[0].concise_on_pass,
        original_request=request,
        request_edit_info=False,
    )
    output = args.output.resolve() / cell.id
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "cell.json", cell.model_dump(mode="json"))
    save_json(output / "request.json", original_payload)
    save_json(output / "prompt.json", {
        "system_prompt": rendered.system_prompt,
        "evaluation_prompt": rendered.evaluation_prompt,
        "response_schema": rendered.response_schema,
        "request_edit_info": False,
        "qa_tools_mode": "never",
        "candidate_sha256": hashlib.sha256(candidate.encode("utf-8")).hexdigest(),
        "provider": provider,
        "authorized_estimate_ceiling_usd": args.max_estimated_cost_usd,
    })
    (output / "candidate.txt").write_text(candidate, encoding="utf-8")
    if not args.execute:
        print(json.dumps({"cell_id": cell.id, "status": "prepared", "output": str(output)}))
        return
    succeeded = asyncio.run(execute_cell(cell, request, candidate, output, provider))
    print(json.dumps({"cell_id": cell.id, "status": "evaluated" if succeeded else "failed", "output": str(output)}))
    if not succeeded:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
