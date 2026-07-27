"""The model routing registry — one file to answer "what runs which role, on whose API".

`models.yaml` has two halves:

* ``models:`` — what is available, what it can do, what it costs.
* ``routes:`` — role -> ordered fallback chain. First entry preferred; later entries are
  tried only when an earlier one errors out or lacks a capability the call requires.

Subagent manifests name a **route**, never a model id. That indirection is the whole
point: swapping the Reviewer from Opus to Gemini is a one-line edit here, with no manifest
touched and no Python changed, and `conductor models` prints the current picture.

Pricing is USD per million tokens and is deliberately optional. A model with no price
still runs — its cost reports as unpriced rather than as zero, because a silent $0 in a
cost dashboard is worse than an honest gap.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator

from conductor.models.providers import Capability, Effort, Provider

DEFAULT_REGISTRY_PATH = Path("models.yaml")


class Pricing(BaseModel):
    """USD per million tokens."""

    input: float = Field(ge=0)
    output: float = Field(ge=0)
    cache_read: float | None = Field(default=None, ge=0)

    def cost_usd(self, *, input_tokens: int, output_tokens: int, cache_read: int = 0) -> float:
        # Cached reads are billed at their own rate when the provider publishes one, and
        # are otherwise already counted in input_tokens by the normalizer.
        billed_input = input_tokens - cache_read if self.cache_read is not None else input_tokens
        total = billed_input * self.input + output_tokens * self.output
        if self.cache_read is not None:
            total += cache_read * self.cache_read
        return total / 1_000_000


class ModelSpec(BaseModel):
    """One concrete model on one provider."""

    alias: str = Field(default="", description="Registry key; filled in by the loader")
    provider: Provider
    id: str = Field(description="Provider-native model id, exactly as the provider spells it")
    supports: set[Capability] = Field(default_factory=set)
    price: Pricing | None = None
    context_window: int | None = Field(default=None, gt=0)
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Provider-specific constructor overrides, applied after translation",
    )
    notes: str = ""

    @property
    def qualified(self) -> str:
        """The `provider:id` string `init_chat_model` takes."""
        return f"{self.provider}:{self.id}"

    def can(self, capability: Capability) -> bool:
        return capability in self.supports

    def missing(self, required: set[Capability]) -> set[Capability]:
        return required - self.supports


class ModelRegistry(BaseModel):
    """Loaded `models.yaml`. Validates that every route points at a real model."""

    models: dict[str, ModelSpec]
    routes: dict[str, list[str]]
    default_route: str | None = None

    @model_validator(mode="after")
    def _check(self) -> ModelRegistry:
        for alias, spec in self.models.items():
            spec.alias = alias

        for role, chain in self.routes.items():
            if not chain:
                raise ValueError(f"route {role!r} has an empty fallback chain")
            unknown = [a for a in chain if a not in self.models]
            if unknown:
                known = ", ".join(sorted(self.models))
                raise ValueError(
                    f"route {role!r} references unknown model(s) {unknown}; declared: {known}"
                )
        if self.default_route is not None and self.default_route not in self.routes:
            raise ValueError(f"default_route {self.default_route!r} is not a declared route")
        return self

    @classmethod
    def from_file(cls, path: Path | str = DEFAULT_REGISTRY_PATH) -> ModelRegistry:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"model registry not found: {path}")
        raw = yaml.safe_load(path.read_text()) or {}
        return cls.model_validate(raw)

    # ---- resolution -------------------------------------------------------

    def has_route(self, role: str) -> bool:
        return role in self.routes

    def chain(self, role: str, *, requires: set[Capability] | None = None) -> list[ModelSpec]:
        """Ordered candidates for `role`, filtered to those that can serve the call.

        Falling back to `default_route` keeps a newly added subagent runnable before
        anyone has decided where to route it — a missing route should not be a crash.
        """
        if role not in self.routes:
            if self.default_route is None:
                known = ", ".join(sorted(self.routes)) or "<none>"
                raise KeyError(f"no route for role {role!r} and no default_route; routes: {known}")
            role = self.default_route

        candidates = [self.models[a] for a in self.routes[role]]
        if not requires:
            return candidates

        eligible = [m for m in candidates if not m.missing(requires)]
        if not eligible:
            needed = ", ".join(sorted(requires))
            tried = ", ".join(m.alias for m in candidates)
            raise ValueError(
                f"no model on route {role!r} supports [{needed}]; candidates were: {tried}"
            )
        return eligible

    def primary(self, role: str) -> ModelSpec:
        return self.chain(role)[0]

    # ---- inspection -------------------------------------------------------

    def describe(self) -> str:
        """What `conductor models` prints — the "which model for which job" view."""
        width = max((len(r) for r in self.routes), default=0)
        lines = ["ROUTES"]
        for role in sorted(self.routes):
            parts = []
            for i, alias in enumerate(self.routes[role]):
                spec = self.models[alias]
                label = f"{alias} ({spec.provider})"
                parts.append(label if i == 0 else f"-> {label}")
            marker = "  *" if role == self.default_route else "   "
            lines.append(f"{marker}{role.ljust(width)}  {'  '.join(parts)}")

        lines.append("")
        lines.append("MODELS")
        for alias in sorted(self.models):
            spec = self.models[alias]
            price = (
                f"${spec.price.input:g}/${spec.price.output:g} per MTok"
                if spec.price
                else "unpriced"
            )
            caps = ", ".join(sorted(spec.supports)) or "none declared"
            lines.append(f"   {alias}: {spec.qualified}  [{price}]  ({caps})")
        return "\n".join(lines)

    def unpriced(self) -> list[str]:
        """Aliases with no pricing — cost reporting will show tokens but no dollars."""
        return sorted(a for a, s in self.models.items() if s.price is None)


def effort_for(spec: ModelSpec, requested: Effort) -> Effort:
    """Convenience re-export point so callers need not import providers directly."""
    from conductor.models.providers import clamp_effort

    return clamp_effort(spec.provider, requested) or requested
