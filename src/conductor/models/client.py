"""Provider-agnostic model client: resolve a route, invoke, fall back, account for cost.

Every model call in the system goes through `ModelClient.invoke`. Callers name a **role**
("planner", "implementer"); they never name a provider, a model id, or a parameter. That
is what makes swapping Gemini in for Opus on the Reviewer a `models.yaml` edit.

Three behaviours worth knowing about, because each is a decision rather than a default:

**Fallback is per-attempt and audited.** LangChain's `Runnable.with_fallbacks` would do
the retry, but it swallows which candidate was tried and why the first one failed. Since
FR-A1 wants every attempt traceable, the loop lives here and writes an ERROR event per
failed candidate before moving on.

**A refusal is not retried.** A provider declining a request is a judgement about the
request, not a transport failure. Rerouting it to a different vendor until one complies
is a safety-bypass pattern, so refusals raise immediately and the fallback chain is
abandoned. Transport errors, rate limits, and outages do fall through.

**Structured output keeps its usage.** `with_structured_output(Schema)` returns the parsed
object and drops the raw message — and with it the token counts FR-A3 needs. Passing
`include_raw=True` returns both, so cost accounting survives structured calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from conductor.audit import AuditStore, EventType
from conductor.models.providers import Capability, Effort, build_model_kwargs
from conductor.models.registry import ModelRegistry, ModelSpec

# Message inputs accepted by LangChain chat models: (role, content) tuples or messages.
MessageInput = list[BaseMessage] | list[tuple[str, str]] | str


class ModelRefusal(RuntimeError):
    """The provider declined the request. Deliberately not retried on another provider."""

    def __init__(self, spec: ModelSpec, reason: str, partial_text: str = "") -> None:
        super().__init__(f"{spec.alias} ({spec.provider}) refused the request: {reason}")
        self.spec = spec
        self.reason = reason
        self.partial_text = partial_text


class AllCandidatesFailed(RuntimeError):
    """Every model on the route errored. Carries the per-candidate failures."""

    def __init__(self, role: str, failures: list[tuple[str, Exception]]) -> None:
        detail = "; ".join(f"{alias}: {exc!r}" for alias, exc in failures)
        super().__init__(f"all candidates for role {role!r} failed — {detail}")
        self.role = role
        self.failures = failures


@dataclass
class ModelResult:
    """One completed call, with everything the audit log and the caller both need."""

    text: str
    spec: ModelSpec
    usage: dict[str, int] = field(default_factory=dict)
    cost_usd: float | None = None
    parsed: Any = None
    attempts: list[str] = field(default_factory=list)
    #: The raw assistant message. Needed verbatim when continuing a tool-calling loop —
    #: providers reject a tool result whose call does not appear in the history.
    message: Any = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)

    @property
    def fell_back(self) -> bool:
        return len(self.attempts) > 1


# Stop/finish reasons that mean "declined", normalized across providers.
_REFUSAL_REASONS = {"refusal", "content_filter", "safety", "blocklist", "prohibited_content"}


def _refusal_reason(message: BaseMessage) -> str | None:
    meta = getattr(message, "response_metadata", None) or {}
    for key in ("stop_reason", "finish_reason"):
        value = meta.get(key)
        if isinstance(value, str) and value.lower() in _REFUSAL_REASONS:
            return value
    return None


def _normalize_usage(message: BaseMessage) -> dict[str, int]:
    """Flatten LangChain's normalized `usage_metadata` into the audit payload shape.

    LangChain already reconciles the providers' differing field names; this only lifts
    the nested detail dicts up so `cost_breakdown` can sum flat integers.
    """
    um = getattr(message, "usage_metadata", None) or {}
    inp = um.get("input_token_details") or {}
    out = um.get("output_token_details") or {}
    return {
        "input_tokens": int(um.get("input_tokens") or 0),
        "output_tokens": int(um.get("output_tokens") or 0),
        "total_tokens": int(um.get("total_tokens") or 0),
        "cache_read": int(inp.get("cache_read") or 0),
        "cache_creation": int(inp.get("cache_creation") or 0),
        "reasoning_tokens": int(out.get("reasoning") or 0),
    }


def _text_of(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    # Anthropic and Gemini return content blocks; keep only the text ones.
    parts = [
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    return "".join(parts)


class ModelClient:
    """Builds and caches chat models per (route candidate, effort, token cap)."""

    def __init__(
        self,
        registry: ModelRegistry,
        *,
        audit: AuditStore | None = None,
        run_id: str | None = None,
    ) -> None:
        self._registry = registry
        self._audit = audit
        self._run_id = run_id
        self._cache: dict[tuple[str, str, int | None], BaseChatModel] = {}

    @property
    def registry(self) -> ModelRegistry:
        return self._registry

    def build(self, spec: ModelSpec, *, effort: Effort, max_tokens: int | None) -> BaseChatModel:
        key = (spec.alias, str(effort), max_tokens)
        if key not in self._cache:
            kwargs = build_model_kwargs(
                spec.provider,
                effort=effort,
                max_tokens=max_tokens,
                supports_thinking=spec.can(Capability.THINKING),
                extra=spec.params,
            )
            # `model_provider=` rather than the "provider:id" prefix form: Ollama ids
            # carry their own colon (`qwen3:8b`), which the prefix parser would split.
            self._cache[key] = init_chat_model(
                model=spec.id, model_provider=str(spec.provider), **kwargs
            )
        return self._cache[key]

    def invoke(
        self,
        role: str,
        messages: MessageInput,
        *,
        actor: str | None = None,
        effort: Effort = Effort.HIGH,
        max_tokens: int | None = None,
        schema: type[BaseModel] | None = None,
        tools: list[dict[str, Any]] | None = None,
        requires: set[Capability] | None = None,
    ) -> ModelResult:
        """Run `messages` on the first candidate for `role` that works.

        `schema` requests a typed result and implies STRUCTURED_OUTPUT; `tools` implies
        TOOLS. Candidates lacking a required capability are skipped rather than attempted
        and failed — a local model with no tool support should never be handed a task that
        needs one.
        """
        required = set(requires or set())
        if schema is not None:
            required.add(Capability.STRUCTURED_OUTPUT)
        if tools:
            required.add(Capability.TOOLS)

        candidates = self._registry.chain(role, requires=required)
        attempts: list[str] = []
        failures: list[tuple[str, Exception]] = []

        for spec in candidates:
            attempts.append(spec.alias)
            try:
                return self._invoke_one(
                    spec,
                    messages,
                    role=role,
                    actor=actor or role,
                    effort=effort,
                    max_tokens=max_tokens,
                    schema=schema,
                    tools=tools,
                    attempts=list(attempts),
                )
            except ModelRefusal:
                raise  # a judgement about the request — do not shop it around
            except Exception as exc:
                failures.append((spec.alias, exc))
                self._log(
                    actor or role,
                    EventType.ERROR,
                    {
                        "role": role,
                        "model": spec.alias,
                        "provider": str(spec.provider),
                        "error": repr(exc),
                        "falling_back_to": (
                            candidates[len(attempts)].alias if len(attempts) < len(candidates)
                            else None
                        ),
                    },
                )

        raise AllCandidatesFailed(role, failures)

    def _invoke_one(
        self,
        spec: ModelSpec,
        messages: MessageInput,
        *,
        role: str,
        actor: str,
        effort: Effort,
        max_tokens: int | None,
        schema: type[BaseModel] | None,
        tools: list[dict[str, Any]] | None,
        attempts: list[str],
    ) -> ModelResult:
        base = self.build(spec, effort=effort, max_tokens=max_tokens)

        if schema is None:
            model = base.bind_tools(tools) if tools else base
            message = model.invoke(messages)
            parsed = None
        else:
            if tools:
                # Structured output is itself implemented as a forced tool call on most
                # providers, so stacking user tools on top produces provider-specific
                # misbehaviour rather than a clean error. Ask for one or the other.
                raise ValueError(
                    "cannot combine `schema` and `tools` in one call — run the tool loop "
                    "first, then make a separate structured call for the final result"
                )
            # include_raw=True widens the return type to a dict of raw/parsed/parsing_error.
            raw = base.with_structured_output(schema, include_raw=True).invoke(messages)
            if not isinstance(raw, dict):  # pragma: no cover — contract of include_raw
                raise TypeError(f"{spec.alias}: expected include_raw dict, got {type(raw)}")
            message = raw["raw"]
            if raw.get("parsing_error"):
                raise ValueError(
                    f"{spec.alias} returned unparseable output: {raw['parsing_error']}"
                )
            parsed = raw["parsed"]

        text = _text_of(message)

        reason = _refusal_reason(message)
        if reason is not None:
            self._log(
                actor,
                EventType.ERROR,
                {"role": role, "model": spec.alias, "refusal": reason},
            )
            raise ModelRefusal(spec, reason, partial_text=text)

        usage = _normalize_usage(message)
        cost = (
            spec.price.cost_usd(
                input_tokens=usage["input_tokens"],
                output_tokens=usage["output_tokens"],
                cache_read=usage["cache_read"],
            )
            if spec.price
            else None
        )

        if self._audit and self._run_id:
            self._audit.record_usage(
                self._run_id,
                actor=actor,
                model=spec.qualified,
                usage={
                    **usage,
                    "model_alias": spec.alias,
                    "provider": str(spec.provider),
                    "role": role,
                    "cost_usd": cost,
                    "effort_requested": str(effort),
                },
            )

        return ModelResult(
            text=text,
            spec=spec,
            usage=usage,
            cost_usd=cost,
            parsed=parsed,
            attempts=attempts,
            message=message,
            tool_calls=list(getattr(message, "tool_calls", None) or []),
        )

    def _log(self, actor: str, event_type: EventType, payload: dict[str, Any]) -> None:
        if self._audit and self._run_id:
            self._audit.append(self._run_id, actor=actor, event_type=event_type, payload=payload)
