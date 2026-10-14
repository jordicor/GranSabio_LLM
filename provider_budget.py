"""Optional durable spend ceiling for explicitly bounded provider models.

This is dispatch accounting, not a token estimator. Each HTTP attempt reserves
a conservative rendered-input/output-limit quote before it can leave the
process. The output allowance includes reasoning; unused allowance is released. Uncertain attempts retain their reservation across restarts.
"""

from __future__ import annotations

import contextvars
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, replace
from decimal import ROUND_CEILING, Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterator, Mapping

import json_utils as json
from ai_runtime.openrouter_provider_options import (
    OpenRouterProviderOptionsError,
    validate_openrouter_provider_options,
)
from context_budget import ContextAdmissionError, RenderedCallBudget

MICRO_USD = Decimal(1_000_000)
SUPPORTED_CALLS = frozenset({"ai_service.generate_content", "ai_service.generate_content_stream"})
SYSTEM_ONE_CALLS = frozenset({"systemone.evaluate", "qa.evaluate_layer", "preflight.validate"})


class ProviderBudgetError(ContextAdmissionError):
    """Terminal local budget rejection; existing AI retries must propagate it."""

    error_code = "provider_budget_blocked"


def _decimal(value: Any) -> Decimal:
    if isinstance(value, bool) or value is None:
        raise ValueError("A finite non-negative money amount is required.")
    try:
        amount = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError("A finite non-negative money amount is required.") from exc
    if not amount.is_finite() or amount < 0:
        raise ValueError("A finite non-negative money amount is required.")
    return amount


def _micro(value: Decimal) -> int:
    return int((value * MICRO_USD).to_integral_value(rounding=ROUND_CEILING))


def _field(value: Any, name: str) -> Any:
    return value.get(name) if isinstance(value, Mapping) else getattr(value, name, None)


@dataclass(frozen=True)
class BudgetPolicy:
    path: Path
    limit_usd: Decimal
    model: str
    input_usd_per_million: Decimal
    output_usd_per_million: Decimal
    max_input_tokens: int
    max_output_tokens: int
    typesafe_model: str = ""
    typesafe_input_usd_per_million: Decimal = Decimal("0")
    typesafe_max_input_tokens: int = 0

    def __post_init__(self) -> None:
        if not self.path.is_absolute():
            raise ValueError("PROVIDER_BUDGET_LEDGER_PATH must be absolute.")
        if (not self.model or len(self.model) > 256 or any(character.isspace() for character in self.model)
                or any(ord(character) < 32 for character in self.model)):
            raise ValueError("PROVIDER_BUDGET_OPENROUTER_MODEL must identify one model.")
        for value in (self.limit_usd, self.input_usd_per_million, self.output_usd_per_million):
            if _decimal(value) <= 0:
                raise ValueError("Provider budget amounts must be positive.")
        if _micro(self.limit_usd) > 10**15:
            raise ValueError("Provider budget limit is outside the supported accounting range.")
        for value in (self.max_input_tokens, self.max_output_tokens):
            if type(value) is not int or not 0 < value <= 100_000_000:
                raise ValueError("Provider budget requires explicit positive token ceilings.")
        if self.typesafe_model:
            if (len(self.typesafe_model) > 256
                    or any(character.isspace() or ord(character) < 32 for character in self.typesafe_model)
                    or _decimal(self.typesafe_input_usd_per_million) <= 0
                    or type(self.typesafe_max_input_tokens) is not int
                    or not 0 < self.typesafe_max_input_tokens <= 100_000_000):
                raise ValueError("TypeSafe budget requires one model, a positive input tariff and token ceiling.")
        elif self.typesafe_input_usd_per_million != 0 or self.typesafe_max_input_tokens != 0:
            raise ValueError("TypeSafe budget settings require an explicit model.")

    @classmethod
    def from_config(cls, settings: Any) -> BudgetPolicy | None:
        path = getattr(settings, "PROVIDER_BUDGET_LEDGER_PATH", "")
        if not path:
            return None
        return cls(path=Path(path), limit_usd=_decimal(settings.PROVIDER_BUDGET_LIMIT_USD),
            model=settings.PROVIDER_BUDGET_OPENROUTER_MODEL,
            input_usd_per_million=_decimal(settings.PROVIDER_BUDGET_INPUT_USD_PER_MILLION),
            output_usd_per_million=_decimal(settings.PROVIDER_BUDGET_OUTPUT_USD_PER_MILLION),
            max_input_tokens=settings.PROVIDER_BUDGET_MAX_INPUT_TOKENS,
            max_output_tokens=settings.PROVIDER_BUDGET_MAX_OUTPUT_TOKENS,
            typesafe_model=getattr(settings, "PROVIDER_BUDGET_TYPESAFE_MODEL", ""),
            typesafe_input_usd_per_million=_decimal(
                getattr(settings, "PROVIDER_BUDGET_TYPESAFE_INPUT_USD_PER_MILLION", "") or "0"),
            typesafe_max_input_tokens=getattr(settings, "PROVIDER_BUDGET_TYPESAFE_MAX_INPUT_TOKENS", 0))

    def signature(self) -> str:
        signature = {"version": 1, "limit_micro_usd": _micro(self.limit_usd), "model": self.model,
            "input_usd_per_million": str(self.input_usd_per_million),
            "output_usd_per_million": str(self.output_usd_per_million),
            "max_input_tokens": self.max_input_tokens, "max_output_tokens": self.max_output_tokens,
            "reservation_multiplier": 2, "max_request_usd": 0}
        # Existing OpenRouter-only ledgers retain their exact signature. Enabling
        # TypeSafe requires the same explicit, offline transition as any tariff change.
        if self.typesafe_model:
            signature["typesafe"] = {
                "model": self.typesafe_model,
                "input_usd_per_million": str(self.typesafe_input_usd_per_million),
                "max_input_tokens": self.typesafe_max_input_tokens,
                "output_usd_per_million": "0",
            }
        return json.dumps(signature, sort_keys=True)


@dataclass(frozen=True)
class Reservation:
    id: str
    output_tokens_limit: int
    upper_micro_usd: int
    provider: str = "openrouter"


class ProviderBudget:
    def __init__(self, policy: BudgetPolicy):
        self.policy = policy
        self._admitted: contextvars.ContextVar[tuple[str, str, str, int, int] | None] = contextvars.ContextVar(
            "provider_budget_admission", default=None)
        policy.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            descriptor = os.open(policy.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            pass
        else:
            os.close(descriptor)
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                db.execute("CREATE TABLE IF NOT EXISTS budget_policy (id INTEGER PRIMARY KEY CHECK(id=1),"
                           "signature TEXT NOT NULL, halted INTEGER NOT NULL DEFAULT 0)")
                db.execute("CREATE TABLE IF NOT EXISTS provider_attempts (id TEXT PRIMARY KEY, model TEXT NOT NULL,"
                    "created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL, state TEXT NOT NULL "
                    "CHECK(state IN ('reserved','settled','uncertain')), output_tokens_limit INTEGER NOT NULL,"
                    "reserved_micro_usd INTEGER NOT NULL, accounted_micro_usd INTEGER NOT NULL,"
                    "account_cost_micro_usd INTEGER, upstream_cost_micro_usd INTEGER,"
                    "input_tokens INTEGER, output_tokens INTEGER, total_tokens INTEGER,"
                    "cost_basis TEXT CHECK(cost_basis IN ('actual','conservative')), is_byok INTEGER)")
                row = db.execute("SELECT signature FROM budget_policy WHERE id=1").fetchone()
                if row is None:
                    db.execute("INSERT INTO budget_policy(id,signature) VALUES(1,?)", (policy.signature(),))
                elif row[0] != policy.signature():
                    raise ValueError("The durable budget policy differs from the runtime configuration.")
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.policy.path.as_uri() + "?mode=rw", uri=True, timeout=10, isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=FULL")
            yield db
        finally:
            db.close()

    def transition_policy(self, replacement: BudgetPolicy, *, expected_signature: str,
                          reason: str) -> dict[str, Any]:
        """Explicit offline maintenance for a new model/tariff under the same cap.

        The operator must first close the engine and all provider dispatches.
        No reserved rows is necessary but does not prove that streams are closed.
        Prior attempts, including uncertain amounts, remain untouched. History
        appends full policy signatures and the last existing attempt's rowid/ID.
        This instance retains its old policy; reopen with the replacement before
        reserving again. Normal construction never transitions a policy.
        """
        if replacement.path != self.policy.path or replacement.limit_usd != self.policy.limit_usd:
            raise ValueError("A budget transition must preserve the ledger path and spend ceiling.")
        return self._transition_policy(replacement, expected_signature=expected_signature, reason=reason)

    def transition_limit(self, limit_usd: Decimal, *, expected_signature: str,
                         reason: str) -> dict[str, Any]:
        """Change only the spend ceiling during explicit offline maintenance.

        Close the engine and all dispatches first, as for transition_policy.
        The new ceiling must cover every accounted amount, including uncertain
        attempts. This instance retains its old policy; reopen with the new
        ceiling before dispatch. No attempt or other policy field is changed.
        """
        replacement = replace(self.policy, limit_usd=_decimal(limit_usd))
        return self._transition_policy(replacement, expected_signature=expected_signature, reason=reason)

    def _transition_policy(self, replacement: BudgetPolicy, *, expected_signature: str,
                           reason: str) -> dict[str, Any]:
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 2000:
            raise ValueError("A non-empty maintenance reason of at most 2000 characters is required.")
        replacement_signature = replacement.signature()
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                current = db.execute("SELECT signature,halted FROM budget_policy WHERE id=1").fetchone()
                if (current is None or current["signature"] != expected_signature
                        or current["signature"] != self.policy.signature()):
                    raise ProviderBudgetError("The durable budget policy differs from the expected policy.")
                if current["halted"]:
                    raise ProviderBudgetError("The provider budget is halted pending accounting verification.")
                if db.execute("SELECT 1 FROM provider_attempts WHERE state='reserved' LIMIT 1").fetchone():
                    raise ProviderBudgetError("Provider reservations must be closed before a policy transition.")
                if replacement_signature == current["signature"]:
                    raise ValueError("The replacement budget policy must differ from the current policy.")
                last_attempt = db.execute(
                    "SELECT rowid,id FROM provider_attempts ORDER BY rowid DESC LIMIT 1").fetchone()
                totals = db.execute("SELECT COUNT(*),COALESCE(SUM(accounted_micro_usd),0) "
                                    "FROM provider_attempts").fetchone()
                if (replacement.limit_usd != self.policy.limit_usd
                        and replacement.limit_usd * MICRO_USD < totals[1]):
                    raise ProviderBudgetError("The new provider spend ceiling is below the accounted spend.")
                record = {"created_at": int(time.time()), "old_signature": current["signature"],
                    "new_signature": replacement_signature,
                    "through_attempt_rowid": last_attempt[0] if last_attempt else 0,
                    "through_attempt_id": last_attempt[1] if last_attempt else None,
                    "attempt_count": totals[0], "accounted_micro_usd": totals[1], "reason": reason}
                db.execute("CREATE TABLE IF NOT EXISTS budget_policy_transitions (id INTEGER PRIMARY KEY,"
                    "created_at INTEGER NOT NULL, old_signature TEXT NOT NULL, new_signature TEXT NOT NULL,"
                    "through_attempt_rowid INTEGER NOT NULL, through_attempt_id TEXT,"
                    "attempt_count INTEGER NOT NULL, accounted_micro_usd INTEGER NOT NULL, reason TEXT NOT NULL)")
                inserted = db.execute("INSERT INTO budget_policy_transitions (created_at,old_signature,new_signature,"
                    "through_attempt_rowid,through_attempt_id,attempt_count,accounted_micro_usd,reason) VALUES "
                    "(:created_at,:old_signature,:new_signature,:through_attempt_rowid,:through_attempt_id,"
                    ":attempt_count,:accounted_micro_usd,:reason)", record)
                record["id"] = inserted.lastrowid
                db.execute("UPDATE budget_policy SET signature=? WHERE id=1", (replacement_signature,))
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise
        return record

    def validate(self, budget: RenderedCallBudget) -> None:
        self._admitted.set(None)
        if budget.provider == "typesafe":
            if (not self.policy.typesafe_model or budget.model != self.policy.typesafe_model
                    or budget.call_id not in SYSTEM_ONE_CALLS
                    or budget.reserved_output_tokens != 0 or budget.thinking_budget_tokens
                    or any(component.kind != "message" for component in budget.components)
                    or not budget.components
                    or sum(component.tokens_estimate for component in budget.components)
                    > self.policy.typesafe_max_input_tokens):
                raise ProviderBudgetError("This System One capability has no configured spend bound.")
            return
        if (budget.provider != "openrouter" or budget.model != self.policy.model
                or budget.call_id not in SUPPORTED_CALLS or budget.framing != "chat"
                or any(component.kind not in {"system", "message", "schema", "framing"}
                       for component in budget.components)
                or budget.thinking_budget_tokens
                or not budget.components
                or any(type(c.tokens_estimate) is not int or c.tokens_estimate < 0
                       or type(c.chars) is not int or c.chars < 0 for c in budget.components)
                or sum(c.tokens_estimate for c in budget.components) > self.policy.max_input_tokens
                or type(budget.reserved_output_tokens) is not int
                or not 0 < budget.reserved_output_tokens <= self.policy.max_output_tokens):
            raise ProviderBudgetError("This provider capability has no configured spend bound.")

    def admit(self, budget: RenderedCallBudget) -> None:
        self.validate(budget)
        # Count the rendered messages/schema, with byte-level and framing margin.
        # This is deliberately wider than the normal token estimate, but a
        # small prompt no longer reserves an entire million-token window.
        input_bound = min(self.policy.max_input_tokens, 512 + sum(
            max(2 * c.tokens_estimate,
                len(c.text.encode("utf-8")) if c.text is not None else 4 * c.chars)
            for c in budget.components
        ))
        self._admitted.set((budget.provider, budget.model, budget.call_id,
                            budget.reserved_output_tokens, input_bound))

    def require_operation(self, provider: str, model: str, operation: str) -> None:
        if (provider == "typesafe" and self.policy.typesafe_model
                and model == self.policy.typesafe_model and operation == "systemone.evaluate"):
            return
        if (provider != "openrouter" or model != self.policy.model
                or ("ai_service." + operation not in SUPPORTED_CALLS and operation != "tool_loop_generation")):
            raise ProviderBudgetError("This provider operation has no configured spend bound.")

    @contextmanager
    def tool_dispatch(self, *, provider: str, model: str, params: Mapping[str, Any],
                      request_kwargs: Mapping[str, Any]) -> Iterator[tuple[dict[str, Any], Reservation]]:
        """Bound one rendered text/validate_draft HTTP request, including retries.

        Tool loops already perform their normal context admission. Here the
        complete wire payload, rather than a reusable loop-level admission,
        establishes the text-only capability and actual output ceiling.
        """
        self.require_operation(provider, model, "tool_loop_generation")
        allowed = {"model", "messages", "max_tokens", "temperature", "response_format", "tools",
                   "tool_choice", "parallel_tool_calls", "stream", "stream_options", "extra_body",
                   "reasoning_effort"}
        extra = params.get("extra_body") or {}
        messages = params.get("messages")
        tools = params.get("tools") or []
        if (set(params) - allowed or set(request_kwargs) - {"timeout"}
                or params.get("model") != model or params.get("stream") is not True
                or not isinstance(extra, Mapping) or set(extra) - {"provider", "transforms", "plugins", "reasoning"}
                or extra.get("transforms", []) != []
                or extra.get("plugins", []) not in ([], [{"id": "context-compression", "enabled": False}])
                or params.get("stream_options") != {"include_usage": True}
                or not isinstance(messages, list) or not messages or not isinstance(tools, list)):
            raise ProviderBudgetError("This tool request has no configured spend bound.")
        if "reasoning" in extra:
            reasoning = extra["reasoning"]
            # Effort shares the existing max_tokens ceiling; separate token budgets
            # and model-changing reasoning modes are outside this bounded scope.
            if (not isinstance(reasoning, Mapping) or set(reasoning) != {"effort"}
                    or not isinstance(reasoning["effort"], str)
                    or reasoning["effort"] not in {"none", "minimal", "low", "medium", "high", "xhigh", "max"}):
                raise ProviderBudgetError("This reasoning configuration has no configured spend bound.")
        for tool in tools:
            if (not isinstance(tool, Mapping) or tool.get("type") != "function"
                    or not isinstance(tool.get("function"), Mapping)
                    or tool["function"].get("name") != "validate_draft"):
                raise ProviderBudgetError("Only the local validate_draft tool has a configured spend bound.")
        for message in messages:
            if (not isinstance(message, Mapping)
                    or set(message) - {"role", "content", "tool_calls", "tool_call_id", "name"}
                    or message.get("role") not in {"system", "user", "assistant", "tool"}
                    or (message.get("content") is not None and not isinstance(message["content"], str))):
                raise ProviderBudgetError("Only text tool-loop messages have a configured spend bound.")
            calls = message.get("tool_calls") or []
            if not isinstance(calls, list) or any(
                    not isinstance(call, Mapping) or call.get("type") != "function"
                    or not isinstance(call.get("function"), Mapping)
                    or call["function"].get("name") != "validate_draft" for call in calls):
                raise ProviderBudgetError("Only local validation calls have a configured spend bound.")
        bounded = dict(params)
        try:
            routing = validate_openrouter_provider_options(extra.get("provider"))
        except OpenRouterProviderOptionsError as exc:
            raise ProviderBudgetError("The tool request has unsupported provider routing.") from exc
        bounded["extra_body"] = self.constrain_routing({**extra, "provider": routing})
        rendered = {key: bounded[key] for key in ("messages", "tools", "response_format") if key in bounded}
        input_bound = min(self.policy.max_input_tokens, 512 + len(json.dumps(rendered).encode("utf-8")))
        with self._reservation_scope(bounded.get("max_tokens"), input_tokens=input_bound) as reservation:
            yield bounded, reservation

    def constrain_routing(self, extra_body: Mapping[str, Any] | None) -> dict[str, Any]:
        extra = dict(extra_body or {})
        provider = dict(extra.get("provider") or {})
        requested_prices = dict(provider.get("max_price") or {})
        ceilings = {"prompt": float(self.policy.input_usd_per_million),
                    "completion": float(self.policy.output_usd_per_million), "request": 0.0}
        provider["max_price"] = {key: min(value, requested_prices.get(key, value)) for key, value in ceilings.items()}
        provider["require_parameters"] = True
        provider["allow_fallbacks"] = False
        extra["provider"] = provider
        return extra

    def reserve(self, output_tokens_limit: int, *, input_tokens: int | None = None) -> Reservation:
        if type(output_tokens_limit) is not int or not 0 < output_tokens_limit <= self.policy.max_output_tokens:
            raise ProviderBudgetError("The output limit has no configured spend bound.")
        if input_tokens is None:
            input_tokens = self.policy.max_input_tokens
        if type(input_tokens) is not int or not 0 < input_tokens <= self.policy.max_input_tokens:
            raise ProviderBudgetError("The input limit has no configured spend bound.")
        # Keep the full requested output, including billed reasoning.
        upper = _micro(Decimal(2) * (
            Decimal(input_tokens) * self.policy.input_usd_per_million
            + Decimal(output_tokens_limit) * self.policy.output_usd_per_million) / MICRO_USD)
        return self._reserve(self.policy.model, output_tokens_limit, upper)

    def _reserve(self, model: str, output_tokens_limit: int, upper: int,
                 *, provider: str = "openrouter") -> Reservation:
        reservation = Reservation(str(uuid.uuid4()), output_tokens_limit, upper, provider)
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                current = db.execute("SELECT signature,halted FROM budget_policy WHERE id=1").fetchone()
                if current is None or current["signature"] != self.policy.signature():
                    raise ProviderBudgetError("The durable budget policy differs from this client; reopen the ledger.")
                if current["halted"]:
                    raise ProviderBudgetError("The provider budget is halted pending accounting verification.")
                committed = db.execute("SELECT COALESCE(SUM(accounted_micro_usd),0) FROM provider_attempts").fetchone()[0]
                if committed + upper > _micro(self.policy.limit_usd):
                    raise ProviderBudgetError("The accumulated provider spend ceiling cannot admit another attempt.")
                now = int(time.time())
                db.execute("INSERT INTO provider_attempts(id,model,created_at,updated_at,state,output_tokens_limit,"
                           "reserved_micro_usd,accounted_micro_usd) VALUES(?,?,?,?,'reserved',?,?,?)",
                           (reservation.id, model, now, now, output_tokens_limit, upper, upper))
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise
        return reservation

    @contextmanager
    def dispatch(self, *, model: str, max_tokens: int, call_id: str, images: Any = None) -> Iterator[Reservation]:
        admitted = self._admitted.get()
        self._admitted.set(None)
        if (admitted is None or admitted[:3] != ("openrouter", model, call_id)
                or model != self.policy.model or images
                or type(max_tokens) is not int or not 0 < max_tokens <= admitted[3]):
            raise ProviderBudgetError("The provider dispatch lacks a matching bounded admission.")
        with self._reservation_scope(max_tokens, input_tokens=admitted[4]) as reservation:
            yield reservation

    @contextmanager
    def _reservation_scope(self, max_tokens: int, *, input_tokens: int | None = None) -> Iterator[Reservation]:
        reservation = self.reserve(max_tokens, input_tokens=input_tokens)
        with self._attempt_scope(reservation):
            yield reservation

    @contextmanager
    def system_one_dispatch(self, *, model: str, call_id: str) -> Iterator[Reservation]:
        """Charge one typed request to the existing shared ceiling, without text output."""
        admitted = self._admitted.get()
        self._admitted.set(None)
        self.require_operation("typesafe", model, "systemone.evaluate")
        if admitted is None or admitted[:4] != ("typesafe", model, call_id, 0):
            raise ProviderBudgetError("The provider dispatch lacks a matching bounded admission.")
        upper = _micro(Decimal(2) * Decimal(self.policy.typesafe_max_input_tokens)
                       * self.policy.typesafe_input_usd_per_million / MICRO_USD)
        reservation = self._reserve(model, 0, upper, provider="typesafe")
        with self._attempt_scope(reservation):
            yield reservation

    @contextmanager
    def _attempt_scope(self, reservation: Reservation) -> Iterator[Reservation]:
        try:
            yield reservation
        finally:
            with self._connection() as db:
                db.execute("UPDATE provider_attempts SET state='uncertain',updated_at=? WHERE id=? AND state='reserved'",
                           (int(time.time()), reservation.id))

    def settle(self, reservation: Reservation, usage: Any, *, response: Any = None) -> None:
        """Use complete raw provider metadata; missing fields never mean zero."""
        byok = _field(usage, "is_byok")
        if type(byok) is not bool:
            byok = _field(response, "is_byok")
        if type(byok) is not bool:
            byok = None
        account_cost, upstream_cost = None, None
        for key in ("cost", "upstream_inference_cost"):
            value = _field(usage, key) if key == "cost" else _field(_field(usage, "cost_details"), key)
            try:
                amount = _micro(_decimal(value))
            except ValueError:
                continue
            if key == "cost":
                account_cost = amount
            else:
                upstream_cost = amount
        counts = [_field(usage, name) for name in ("prompt_tokens", "completion_tokens", "total_tokens")]
        complete_counts = all(type(value) is int and value >= 0 for value in counts)
        complete_counts = complete_counts and counts[0] + counts[1] == counts[2]
        complete_cost = account_cost is not None and (byok is False or upstream_cost is not None)
        complete = complete_counts and complete_cost
        observed = (account_cost + (upstream_cost or 0) if byok is not False else account_cost) if complete else None
        bounds_broken = complete_counts and (counts[0] > self.policy.max_input_tokens
                                              or counts[1] > reservation.output_tokens_limit)
        basis = ("actual" if byok is not None else "conservative") if complete else None
        if reservation.provider == "typesafe":
            # Native input usage plus the explicitly verified tariff is a
            # conservative catalog estimate, never a claimed invoice amount.
            input_tokens = _field(usage, "input_tokens")
            output_tokens = _field(usage, "output_tokens")
            complete_counts = all(type(value) is int and value >= 0
                                  for value in (input_tokens, output_tokens))
            counts = ([input_tokens, output_tokens, input_tokens + output_tokens]
                      if complete_counts else [None, None, None])
            complete = complete_counts
            observed = (_micro(Decimal(input_tokens) * self.policy.typesafe_input_usd_per_million / MICRO_USD)
                        if complete else None)
            bounds_broken = complete_counts and input_tokens > self.policy.typesafe_max_input_tokens
            account_cost = upstream_cost = byok = None
            basis = "conservative" if complete else None
        bounds_broken = bounds_broken or (observed is not None and observed > reservation.upper_micro_usd)
        state = "settled" if complete and not bounds_broken else "uncertain"
        accounted = observed if state == "settled" else max(reservation.upper_micro_usd, observed or 0)
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                current = db.execute("SELECT state FROM provider_attempts WHERE id=?", (reservation.id,)).fetchone()
                if current is None:
                    raise ProviderBudgetError("The provider budget reservation is missing.")
                if current[0] == "settled":
                    db.execute("COMMIT")
                    return
                db.execute("UPDATE provider_attempts SET state=?,accounted_micro_usd=?,account_cost_micro_usd=?,"
                    "upstream_cost_micro_usd=?,input_tokens=?,output_tokens=?,total_tokens=?,cost_basis=?,is_byok=?,"
                    "updated_at=? WHERE id=?", (state, accounted, account_cost, upstream_cost,
                    *(counts if complete_counts else [None, None, None]),
                    basis,
                    int(byok) if byok is not None else None, int(time.time()), reservation.id))
                if bounds_broken:
                    db.execute("UPDATE budget_policy SET halted=1 WHERE id=1")
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise
        if bounds_broken:
            raise ProviderBudgetError("Provider usage exceeded the configured accounting bounds; dispatch is halted.")

    def summary(self) -> dict[str, Any]:
        """Content-free totals, distinguishing observed cost and conservative risk."""
        with self._connection() as db:
            rows = list(db.execute("SELECT state,cost_basis,COUNT(*) AS attempts,"
                "COALESCE(SUM(accounted_micro_usd),0) AS accounted_micro_usd FROM provider_attempts GROUP BY state,cost_basis"))
            halted = bool(db.execute("SELECT halted FROM budget_policy WHERE id=1").fetchone()[0])
        accounted = sum(row["accounted_micro_usd"] for row in rows)
        return {"limit_micro_usd": _micro(self.policy.limit_usd), "accounted_micro_usd": accounted,
                "remaining_micro_usd": max(0, _micro(self.policy.limit_usd) - accounted),
                "halted": halted, "groups": [dict(row) for row in rows]}
