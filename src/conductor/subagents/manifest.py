"""Subagent manifests and the registry that loads them.

FR-S4: "New subagent types register via a manifest (role, tools, prompt, risk profile)
without changing the orchestrator core."

That requirement is why this module exists in Phase 0 with only two subagents. The
orchestrator builds its graph nodes by iterating the registry — it has no import of,
or branch on, any specific subagent. Adding the Phase 1 Tester and Doc-writer should
be two YAML files and no Python change; that is the acceptance test for this design.

A manifest also carries the tool allow-list, which is the enforcement input for
least-privilege connector scoping (FR-T2). A subagent cannot call a tool it did not
declare, regardless of what its prompt talks it into.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from pydantic import BaseModel, Field, field_validator

from conductor.models.providers import Effort

if TYPE_CHECKING:
    from conductor.models.registry import ModelRegistry

DEFAULT_MANIFEST_DIR = Path("subagents")


class RiskProfile(StrEnum):
    """The worst thing this subagent is allowed to do.

    Consumed by the policy engine to decide gating. `READ_ONLY` subagents can never
    trip an approval gate, which is what makes them cheap to run unattended.
    """

    READ_ONLY = "read_only"
    WRITE = "write"
    IRREVERSIBLE = "irreversible"


class SubagentManifest(BaseModel):
    """One specialist worker: scoped prompt, allow-listed tools, isolated context."""

    name: str = Field(description="Unique identifier, also the actor name in the audit log")
    role: str = Field(description="One-line description the planner reads when routing")
    route: str = Field(
        description="Route name in models.yaml — never a model id, so provider choice "
        "stays in one file",
    )
    effort: Effort = Effort.HIGH
    max_tokens: int = Field(default=32000, gt=0)
    risk_profile: RiskProfile = RiskProfile.READ_ONLY
    tools: list[str] = Field(default_factory=list, description="Allow-list of tool names")
    prompt: str = Field(description="System prompt — scoped to this role only")
    description: str = ""

    @field_validator("name")
    @classmethod
    def _name_is_identifier(cls, v: str) -> str:
        if not v.replace("_", "").isalnum():
            raise ValueError(f"subagent name must be alphanumeric/underscore, got {v!r}")
        return v

    @field_validator("route")
    @classmethod
    def _route_is_not_a_model_id(cls, v: str) -> str:
        """Catch the most likely migration mistake: pasting a model id into `route`.

        A model id here would resolve to no route and fall through to `default_route`,
        which runs successfully on the wrong model — the worst kind of failure. Provider
        prefixes and vendor-shaped names are rejected outright.

        Note there is deliberately no date-suffix check any more. That guard was correct
        while every model was Anthropic, where dated ids 404; it is actively wrong for
        OpenAI and Gemini, where pinned ids like `gpt-4o-2024-08-06` are the recommended
        form. Model id validity is now the provider's business, not the manifest's.
        """
        if ":" in v or v.startswith(("claude-", "gpt-", "gemini-", "o1", "o3")):
            raise ValueError(
                f"route {v!r} looks like a model id. Manifests name a route from "
                "models.yaml (e.g. 'implementer'); model ids live only in that file"
            )
        return v

    def allows(self, tool_name: str) -> bool:
        return tool_name in self.tools


class SubagentRegistry:
    """Loads manifests at boot and answers 'who can do what'.

    The planner is handed `capability_summary()` so it routes against declared roles
    rather than a hardcoded list — FR-O3, "the user never picks a subagent", requires
    the orchestrator not to have one either.
    """

    def __init__(self, manifests: dict[str, SubagentManifest] | None = None) -> None:
        self._manifests: dict[str, SubagentManifest] = manifests or {}

    @classmethod
    def from_dir(cls, directory: Path | str = DEFAULT_MANIFEST_DIR) -> SubagentRegistry:
        directory = Path(directory)
        if not directory.is_dir():
            raise FileNotFoundError(f"manifest directory not found: {directory}")

        manifests: dict[str, SubagentManifest] = {}
        for path in sorted(directory.glob("*.yaml")) + sorted(directory.glob("*.yml")):
            raw = yaml.safe_load(path.read_text())
            if raw is None:
                continue
            manifest = SubagentManifest.model_validate(raw)
            if manifest.name in manifests:
                raise ValueError(
                    f"duplicate subagent name {manifest.name!r} "
                    f"(second definition in {path})"
                )
            manifests[manifest.name] = manifest

        if not manifests:
            raise ValueError(f"no subagent manifests found in {directory}")
        return cls(manifests)

    def __len__(self) -> int:
        return len(self._manifests)

    def __contains__(self, name: object) -> bool:
        return name in self._manifests

    def get(self, name: str) -> SubagentManifest:
        try:
            return self._manifests[name]
        except KeyError:
            known = ", ".join(sorted(self._manifests)) or "<none>"
            raise KeyError(f"unknown subagent {name!r}; registered: {known}") from None

    def names(self) -> list[str]:
        return sorted(self._manifests)

    def all(self) -> list[SubagentManifest]:
        return [self._manifests[n] for n in self.names()]

    def validate_routes(self, models: ModelRegistry) -> list[tuple[str, str]]:
        """Return (subagent, route) pairs whose route no model registry entry defines.

        Called at boot for the same reason `ConnectorRegistry.validate_manifest` is: a
        renamed route should fail immediately and by name, not three subagents into a
        run. Routes covered by `default_route` are not reported — falling back is the
        documented behaviour, not an error.
        """
        return [
            (m.name, m.route)
            for m in self.all()
            if not models.has_route(m.route) and models.default_route is None
        ]

    def capability_summary(self) -> str:
        """Rendered into the planner prompt so routing is driven by the registry."""
        lines = []
        for m in self.all():
            tools = ", ".join(m.tools) if m.tools else "no tools"
            lines.append(f"- {m.name}: {m.role} (risk: {m.risk_profile}; tools: {tools})")
        return "\n".join(lines)
