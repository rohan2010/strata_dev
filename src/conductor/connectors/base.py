"""Connector interface, tool specs, and the scoping layer that enforces least privilege.

Design note — why this is not MCP yet
-------------------------------------
FR-T1 commits to MCP as the integration standard, and Phase 1 keeps that commitment.
For the Phase 0 spike the GitHub connector talks to the REST API directly, *behind this
interface*, so the MCP-backed implementation is a swap of `Connector` subclass rather
than a change to any caller. What matters architecturally — a registry the planner
queries at plan time (FR-T4), and per-subagent scoped permissions (FR-T2) — is real
here and does not change when the transport does.

The scoping model
-----------------
A subagent's manifest declares a tool allow-list. `ScopedToolbox` is the only thing
agents are handed, and it refuses anything outside that list. This means "Reviewer is
read-only" is enforced by a `PermissionError` in code, not by a sentence in a prompt
that a sufficiently confused model can talk itself past.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from conductor.audit import AuditStore
from conductor.subagents import SubagentManifest


class ToolSpec(BaseModel):
    """Describes one callable tool.

    `mutates` and `reversible` are the inputs the Phase 1 policy engine classifies on
    (FR-H1). They live on the tool definition rather than in the policy engine's own
    table so that adding a connector cannot silently introduce an unclassified action.
    """

    name: str
    description: str
    parameters: dict[str, str] = Field(default_factory=dict)
    optional: list[str] = Field(
        default_factory=list, description="Parameter names the caller may omit"
    )
    mutates: bool = False
    reversible: bool = True

    @property
    def qualified_risk(self) -> str:
        if not self.mutates:
            return "read"
        return "reversible_write" if self.reversible else "irreversible"

    @property
    def wire_name(self) -> str:
        """The name given to the model. Not the same as `name`.

        Tool names are namespaced `<connector>.<tool>`, but every provider validates tool
        names against `^[a-zA-Z0-9_-]{1,64}$` — a dot is rejected by Anthropic and OpenAI
        alike. The dot is swapped for a double underscore on the wire and mapped back
        before dispatch, so the audit log and the allow-list keep the readable name.
        """
        return self.name.replace(".", "__")

    def json_schema(self) -> dict[str, Any]:
        """Parameter schema for tool binding.

        Everything is typed `string` deliberately: the connectors coerce (`int(number)`),
        and models are markedly more reliable at emitting strings than at matching a
        numeric type across three providers' differing tool-call encodings.
        """
        return {
            "type": "object",
            "properties": {
                name: {"type": "string", "description": desc}
                for name, desc in self.parameters.items()
            },
            "required": [p for p in self.parameters if p not in self.optional],
        }

    def as_tool_definition(self) -> dict[str, Any]:
        """OpenAI-function shape, which `bind_tools` accepts for every provider."""
        return {
            "name": self.wire_name,
            "description": self.description,
            "parameters": self.json_schema(),
        }


class ConnectorError(RuntimeError):
    """A connector failed to execute a tool. Contained, not fatal to the run (FR-O7)."""


class Connector(ABC):
    """One external system. Connect once, reuse across agents (FR-T1)."""

    name: str

    @abstractmethod
    def tools(self) -> list[ToolSpec]:
        """Tool specs, names already namespaced as `<connector>.<tool>`."""

    @abstractmethod
    def invoke(self, tool: str, arguments: dict[str, Any]) -> Any:
        """Execute `tool`. Raise ConnectorError on failure."""

    def health(self) -> tuple[bool, str]:
        """Cheap reachability/credential check, surfaced by `conductor doctor`."""
        return True, "ok"


class ConnectorRegistry:
    """Resolves a tool name to the connector that owns it.

    The planner queries `capability_summary()` at plan time so it only ever proposes
    work the system can actually do — FR-T4.
    """

    def __init__(self) -> None:
        self._connectors: dict[str, Connector] = {}
        self._tools: dict[str, tuple[Connector, ToolSpec]] = {}

    def register(self, connector: Connector) -> None:
        if connector.name in self._connectors:
            raise ValueError(f"connector {connector.name!r} already registered")
        self._connectors[connector.name] = connector
        for spec in connector.tools():
            if spec.name in self._tools:
                raise ValueError(f"duplicate tool name {spec.name!r}")
            self._tools[spec.name] = (connector, spec)

    def connectors(self) -> list[Connector]:
        return [self._connectors[n] for n in sorted(self._connectors)]

    def spec(self, tool: str) -> ToolSpec:
        try:
            return self._tools[tool][1]
        except KeyError:
            raise KeyError(f"unknown tool {tool!r}") from None

    def has(self, tool: str) -> bool:
        return tool in self._tools

    def tool_names(self) -> list[str]:
        return sorted(self._tools)

    def invoke(self, tool: str, arguments: dict[str, Any]) -> Any:
        try:
            connector, _ = self._tools[tool]
        except KeyError:
            raise KeyError(f"unknown tool {tool!r}") from None
        return connector.invoke(tool, arguments)

    def capability_summary(self) -> str:
        lines = []
        for name in sorted(self._tools):
            spec = self._tools[name][1]
            lines.append(f"- {name}: {spec.description} [{spec.qualified_risk}]")
        return "\n".join(lines)

    def validate_manifest(self, manifest: SubagentManifest) -> list[str]:
        """Return tools the manifest declares that no connector provides.

        Called at boot so a typo in a manifest fails immediately rather than as a
        mid-run KeyError three subagents deep.
        """
        return [t for t in manifest.tools if not self.has(t)]


class ScopedToolbox:
    """What a single subagent is handed. Enforces the manifest allow-list.

    Every call is written to the audit log write-ahead (FR-A1), so a tool that crashes
    or hangs still leaves evidence of what was attempted.
    """

    def __init__(
        self,
        registry: ConnectorRegistry,
        manifest: SubagentManifest,
        audit: AuditStore,
        run_id: str,
    ) -> None:
        self._registry = registry
        self._manifest = manifest
        self._audit = audit
        self._run_id = run_id

    @property
    def manifest(self) -> SubagentManifest:
        return self._manifest

    def available(self) -> list[ToolSpec]:
        return [self._registry.spec(t) for t in self._manifest.tools if self._registry.has(t)]

    def describe(self) -> str:
        return "\n".join(f"- {s.name}: {s.description}" for s in self.available())

    def tool_definitions(self) -> list[dict[str, Any]]:
        """What gets bound to the model — only this subagent's allow-listed tools.

        Least privilege starts here: a tool the manifest does not declare is never even
        described to the model, so refusing it later is a backstop rather than the plan.
        """
        return [spec.as_tool_definition() for spec in self.available()]

    def resolve(self, wire_name: str) -> str:
        """Map a model-emitted tool name back to its namespaced form."""
        for spec in self.available():
            if spec.wire_name == wire_name:
                return spec.name
        # Fall through to the raw name so `invoke` raises the normal permission error
        # rather than a confusing lookup failure.
        return wire_name

    def invoke(self, tool: str, arguments: dict[str, Any]) -> Any:
        # Allow-list check comes first — before existence, before audit. A subagent
        # probing for tools it has no right to should not learn whether they exist.
        if not self._manifest.allows(tool):
            raise PermissionError(
                f"subagent {self._manifest.name!r} is not permitted to call {tool!r}; "
                f"allowed: {', '.join(self._manifest.tools) or '<none>'}"
            )
        if not self._registry.has(tool):
            raise KeyError(f"unknown tool {tool!r}")

        with self._audit.tool_call(
            self._run_id, actor=self._manifest.name, tool=tool, arguments=arguments
        ) as outcome:
            outcome["result"] = self._registry.invoke(tool, arguments)
            return outcome["result"]
