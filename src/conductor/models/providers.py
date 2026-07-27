"""Provider vocabulary and the translation from declared intent to provider params.

The premise of this layer: a subagent declares *how hard to think* (`effort: xhigh`),
never *which knob to turn*. Providers disagree about the knob, its name, and its range,
and that disagreement is not something a manifest author should have to know.

What was actually verified against the installed integration packages:

| provider      | effort param       | accepted values     | output cap        |
|---------------|--------------------|---------------------|-------------------|
| anthropic     | `reasoning_effort` | low..max (all five) | `max_tokens`      |
| openai        | `reasoning_effort` | minimal..high       | `max_tokens`      |
| google_genai  | `reasoning_effort` | minimal..high       | `max_output_tokens` |
| ollama        | `reasoning` (bool) | n/a                 | `num_predict`     |

So `effort: xhigh` is honoured natively on Anthropic, clamps to `high` on OpenAI and
Gemini, and becomes `reasoning=True` on Ollama. Clamping is always *downward* — we never
silently buy more reasoning than was asked for, because effort is a cost lever as much as
a quality one.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class Provider(StrEnum):
    """Values double as `init_chat_model` provider prefixes — do not rename casually.

    `google_genai` is the Gemini Developer API. LangChain infers a bare `gemini-...`
    string as `google_vertexai` instead, and warns that the default changes in the next
    major version, which is why every model in the registry names its provider
    explicitly rather than relying on prefix inference.
    """

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google_genai"
    OLLAMA = "ollama"


class Capability(StrEnum):
    """What a model can do, declared per model in `models.yaml`.

    Used to skip a fallback candidate that cannot serve a request at all — asking a
    model without `STRUCTURED_OUTPUT` for a typed task graph fails slowly and confusingly,
    so the router declines it up front and moves to the next candidate.
    """

    TOOLS = "tools"
    STRUCTURED_OUTPUT = "structured_output"
    THINKING = "thinking"
    VISION = "vision"


class Effort(StrEnum):
    """Canonical, provider-independent reasoning intent. Ascending."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    MAX = "max"


_CANONICAL_ORDER: tuple[Effort, ...] = (
    Effort.LOW,
    Effort.MEDIUM,
    Effort.HIGH,
    Effort.XHIGH,
    Effort.MAX,
)

# Highest canonical effort each provider can express natively. Anything above clamps here.
_EFFORT_CEILING: dict[Provider, Effort | None] = {
    Provider.ANTHROPIC: Effort.MAX,
    Provider.OPENAI: Effort.HIGH,
    Provider.GOOGLE: Effort.HIGH,
    Provider.OLLAMA: None,  # no effort scale; see _ollama_kwargs
}

_MAX_TOKENS_PARAM: dict[Provider, str] = {
    Provider.ANTHROPIC: "max_tokens",
    Provider.OPENAI: "max_tokens",
    Provider.GOOGLE: "max_output_tokens",
    Provider.OLLAMA: "num_predict",
}


def clamp_effort(provider: Provider, effort: Effort) -> Effort | None:
    """Reduce `effort` to the highest value `provider` actually accepts.

    Returns None when the provider has no effort scale at all.
    """
    ceiling = _EFFORT_CEILING[provider]
    if ceiling is None:
        return None
    if _CANONICAL_ORDER.index(effort) <= _CANONICAL_ORDER.index(ceiling):
        return effort
    return ceiling


def max_tokens_param(provider: Provider) -> str:
    return _MAX_TOKENS_PARAM[provider]


def build_model_kwargs(
    provider: Provider,
    *,
    effort: Effort,
    max_tokens: int | None,
    supports_thinking: bool,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Translate declared intent into constructor kwargs for this provider's chat model.

    `extra` is the per-model escape hatch from `models.yaml` and is applied last, so a
    model entry can always override the translation when a provider does something this
    table does not anticipate.
    """
    kwargs: dict[str, Any] = {}

    if max_tokens is not None:
        kwargs[max_tokens_param(provider)] = max_tokens

    if provider is Provider.OLLAMA:
        kwargs.update(_ollama_kwargs(effort, supports_thinking))
    else:
        clamped = clamp_effort(provider, effort)
        if clamped is not None and supports_thinking:
            kwargs["reasoning_effort"] = str(clamped)

    return kwargs | (extra or {})


def _ollama_kwargs(effort: Effort, supports_thinking: bool) -> dict[str, Any]:
    """Ollama exposes reasoning as an on/off switch, not a scale.

    Treat HIGH and above as "on". Local models are the cheap tier by definition, so the
    cost argument for a finer gradient does not apply.
    """
    if not supports_thinking:
        return {}
    on = _CANONICAL_ORDER.index(effort) >= _CANONICAL_ORDER.index(Effort.HIGH)
    return {"reasoning": on}
