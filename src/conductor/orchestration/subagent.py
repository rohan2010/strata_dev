"""Running one subagent: isolated context, allow-listed tools, structured result out.

The tool loop lives here rather than in the graph because the graph must not see it. A
subagent's messages — including every tool call and every blob of file content it read —
are local to this function. The orchestrator gets a `SubagentResult`.

Tool calls go through `ScopedToolbox`, never to a connector directly, which is what makes
the manifest allow-list an enforcement point instead of documentation (FR-T2) and what puts
every call in the audit log write-ahead (FR-A1). When the policy engine arrives in Phase 1
it slots in at that same boundary, and nothing here changes.
"""

from __future__ import annotations

import json
from typing import Any

from conductor.audit import AuditStore, EventType
from conductor.connectors import ScopedToolbox
from conductor.models import AllCandidatesFailed, ModelClient, ModelRefusal
from conductor.orchestration.state import SubagentResult
from conductor.planning import Task, TaskStatus
from conductor.subagents import SubagentManifest

#: Ceiling on tool-call round trips within one subagent invocation. A subagent that has not
#: finished in this many steps is looping, not working — the run should surface that rather
#: than spend a budget discovering it.
MAX_TOOL_ITERATIONS = 12

TASK_TEMPLATE = """\
Task: {objective}

{inputs}{acceptance}{context}
Work the task and report your result."""


def _render_task(task: Task, upstream: dict[str, SubagentResult]) -> str:
    inputs = ""
    if task.inputs:
        rendered = "\n".join(f"  {k}: {v}" for k, v in task.inputs.items())
        inputs = f"Inputs:\n{rendered}\n\n"

    acceptance = f"Done when: {task.acceptance}\n\n" if task.acceptance else ""

    context = ""
    if upstream:
        # Only summaries and artifacts cross the boundary — never the upstream subagent's
        # transcript. This is the isolation rule applied between siblings, not just upward.
        parts = []
        for dep_id in task.depends_on:
            result = upstream.get(dep_id)
            if result is None:
                continue
            block = f"  {dep_id} ({result.subagent}): {result.summary}"
            if result.artifacts:
                block += f"\n    artifacts: {json.dumps(result.artifacts, default=str)}"
            parts.append(block)
        if parts:
            context = "Results from work this task depends on:\n" + "\n".join(parts) + "\n\n"

    return TASK_TEMPLATE.format(
        objective=task.objective, inputs=inputs, acceptance=acceptance, context=context
    )


def run_subagent(
    task: Task,
    manifest: SubagentManifest,
    *,
    client: ModelClient,
    toolbox: ScopedToolbox,
    audit: AuditStore | None = None,
    run_id: str | None = None,
    upstream: dict[str, SubagentResult] | None = None,
    max_iterations: int = MAX_TOOL_ITERATIONS,
) -> SubagentResult:
    """Execute one task. Never raises — a failure is a result, not an exception.

    Containing failures here rather than letting them propagate is FR-O7: one subagent
    dying must degrade the run to partial, not abort it.
    """
    if audit and run_id:
        audit.append(
            run_id,
            actor="orchestrator",
            event_type=EventType.DELEGATION,
            payload={
                "task_id": task.id,
                "subagent": manifest.name,
                "route": manifest.route,
                "objective": task.objective,
            },
        )

    # This list is the isolation boundary. It does not leave this function.
    messages: list[Any] = [
        ("system", manifest.prompt),
        ("human", _render_task(task, upstream or {})),
    ]

    tools = toolbox.tool_definitions()
    tool_calls = 0
    last_model = ""
    cost = 0.0
    priced = False

    try:
        for _ in range(max_iterations):
            result = client.invoke(
                manifest.route,
                messages,
                actor=manifest.name,
                effort=manifest.effort,
                max_tokens=manifest.max_tokens,
                tools=tools or None,
            )
            last_model = result.spec.alias
            if result.cost_usd is not None:
                cost += result.cost_usd
                priced = True

            requested = result.tool_calls
            if not requested:
                return SubagentResult(
                    task_id=task.id,
                    subagent=manifest.name,
                    status=TaskStatus.COMPLETED,
                    summary=result.text.strip(),
                    tool_calls=tool_calls,
                    model=last_model,
                    cost_usd=cost if priced else None,
                )

            messages.append(result.message)
            for call in requested:
                tool_calls += 1
                messages.append(_execute(toolbox, call))

        return _failed(
            task,
            manifest,
            f"exceeded {max_iterations} tool iterations without producing a result",
            tool_calls,
            last_model,
            cost if priced else None,
        )

    except ModelRefusal as exc:
        return _failed(task, manifest, f"model refused: {exc.reason}", tool_calls, last_model, None)
    except AllCandidatesFailed as exc:
        return _failed(task, manifest, str(exc), tool_calls, last_model, None)
    except Exception as exc:  # noqa: BLE001 — containment is the point
        return _failed(task, manifest, repr(exc), tool_calls, last_model, None)


def _execute(toolbox: ScopedToolbox, call: dict[str, Any]) -> Any:
    """Run one tool call and return the message to feed back to the model.

    A blocked or failing tool returns its error *to the model* rather than raising: a
    subagent that asked for something it may not have should be told so and given the
    chance to proceed differently, which is more useful than killing the task.
    """
    from langchain_core.messages import ToolMessage

    name = toolbox.resolve(call.get("name", ""))
    try:
        output = toolbox.invoke(name, call.get("args") or {})
        content = json.dumps(output, default=str)
    except PermissionError as exc:
        content = f"ERROR: {exc}"
    except Exception as exc:  # noqa: BLE001 — surfaced to the model, and already audited
        content = f"ERROR: {exc!r}"
    return ToolMessage(content=content, tool_call_id=call.get("id") or name)


def _failed(
    task: Task,
    manifest: SubagentManifest,
    error: str,
    tool_calls: int,
    model: str,
    cost: float | None,
) -> SubagentResult:
    return SubagentResult(
        task_id=task.id,
        subagent=manifest.name,
        status=TaskStatus.FAILED,
        summary="",
        error=error,
        tool_calls=tool_calls,
        model=model,
        cost_usd=cost,
    )
