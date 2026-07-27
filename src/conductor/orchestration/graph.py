"""The LangGraph skeleton: plan -> dispatch -> aggregate, with a resumable checkpoint.

Shape:

    START -> plan -> dispatch --(Send per ready task)--> run_task -+
                        ^                                          |
                        +------------------------------------------+
                        |
                        +--(nothing ready)--> aggregate -> END

`dispatch` is a loop hub rather than a one-shot fan-out. A single `Send` fan-out would run
every task in one superstep, which only works for a flat plan; looping back through dispatch
runs the graph one dependency *layer* at a time, so independent tasks still go in parallel
(FR-S3) while dependent ones wait. It also means resumption is free: the checkpointer knows
exactly which layer was in flight.

Why the graph is built from the registry and never names a subagent: FR-S4. Adding a Tester
is a YAML file. There is no branch here to extend, because there is one `run_task` node that
looks up its manifest at dispatch time.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from conductor.audit import AuditStore, EventType, RunStatus
from conductor.connectors import ConnectorRegistry, ScopedToolbox
from conductor.models import ModelClient
from conductor.orchestration.state import (
    RunState,
    SubagentResult,
    completed_ids,
    failed_ids,
)
from conductor.orchestration.subagent import run_subagent
from conductor.planning import Planner, PlanningError, Task, TaskStatus
from conductor.subagents import SubagentRegistry

DEFAULT_CHECKPOINT_PATH = Path(".conductor/checkpoints.db")


class TaskEnvelope(TypedDict):
    """What one `run_task` invocation receives. Deliberately not the whole RunState."""

    run_id: str
    task: Task
    upstream: dict[str, SubagentResult]


class Orchestrator:
    """Owns the compiled graph and the registries the nodes close over."""

    def __init__(
        self,
        *,
        client: ModelClient,
        subagents: SubagentRegistry,
        connectors: ConnectorRegistry,
        audit: AuditStore,
        checkpointer: Any | None = None,
        max_tool_iterations: int | None = None,
    ) -> None:
        self._client = client
        self._subagents = subagents
        self._connectors = connectors
        self._audit = audit
        self._max_tool_iterations = max_tool_iterations
        self._graph = self._build().compile(checkpointer=checkpointer)

    @property
    def graph(self) -> Any:
        return self._graph

    # ---- nodes ------------------------------------------------------------

    def _plan(self, state: RunState) -> dict[str, Any]:
        planner = Planner(
            self._client,
            self._subagents,
            connectors=self._connectors,
            audit=self._audit,
            run_id=state["run_id"],
        )
        try:
            plan = planner.plan(state["goal"])
        except PlanningError as exc:
            # Planning is the one failure that is not partial-able: with no task graph
            # there is nothing to degrade to.
            return {"plan": None, "status": RunStatus.FAILED, "summary": str(exc)}
        return {"plan": plan, "status": RunStatus.RUNNING}

    def _dispatch(self, state: RunState) -> dict[str, Any]:
        """Mark the next layer as in-flight. The router turns that into Sends."""
        plan = state.get("plan")
        if plan is None:
            return {}
        ready = plan.ready(completed_ids(state), set(state.get("dispatched", [])))
        return {"dispatched": [t.id for t in ready]}

    def _route(self, state: RunState) -> list[Send] | str:
        plan = state.get("plan")
        if plan is None:
            return "aggregate"

        results = state.get("results", {})
        pending = [
            t
            for t in plan.tasks
            if t.id in set(state.get("dispatched", [])) and t.id not in results
        ]
        if not pending:
            return "aggregate"

        return [
            Send(
                "run_task",
                TaskEnvelope(
                    run_id=state["run_id"],
                    task=task,
                    upstream={
                        dep: results[dep] for dep in task.depends_on if dep in results
                    },
                ),
            )
            for task in pending
        ]

    def _run_task(self, state: TaskEnvelope) -> dict[str, Any]:
        # LangGraph inspects the parameter name; it must be `state`, even though what
        # arrives here is one Send envelope rather than the whole RunState.
        task = state["task"]
        run_id = state["run_id"]

        try:
            manifest = self._subagents.get(task.subagent)
        except KeyError as exc:
            # The planner validates its roster, so reaching here means the plan was
            # resumed against a registry that has since changed.
            return {
                "results": {
                    task.id: SubagentResult(
                        task_id=task.id,
                        subagent=task.subagent,
                        status=TaskStatus.FAILED,
                        error=str(exc),
                    )
                }
            }

        toolbox = ScopedToolbox(self._connectors, manifest, self._audit, run_id)
        kwargs: dict[str, Any] = {}
        if self._max_tool_iterations is not None:
            kwargs["max_iterations"] = self._max_tool_iterations

        result = run_subagent(
            task,
            manifest,
            client=self._client,
            toolbox=toolbox,
            audit=self._audit,
            run_id=run_id,
            upstream=state.get("upstream") or {},
            **kwargs,
        )

        self._audit.append(
            run_id,
            actor=manifest.name,
            event_type=EventType.SUBAGENT_RESULT,
            payload={
                "task_id": task.id,
                "status": str(result.status),
                "summary": result.summary,
                "artifacts": result.artifacts,
                "error": result.error,
                "tool_calls": result.tool_calls,
                "model": result.model,
            },
        )
        return {"results": {task.id: result}}

    def _aggregate(self, state: RunState) -> dict[str, Any]:
        """Deterministic roll-up. No model call — there is nothing here to reason about.

        Tasks that never ran because an upstream task failed are recorded as SKIPPED
        rather than left absent, so the run view can show why the graph stopped short
        (FR-O7 partial-result aggregation).
        """
        plan = state.get("plan")
        if plan is None:
            status = RunStatus.FAILED
            summary = state.get("summary") or "planning failed"
            self._finish(state, status, summary)
            return {"status": status, "summary": summary}

        results = dict(state.get("results", {}))
        failed = failed_ids(state)
        completed = completed_ids(state)

        skipped: dict[str, SubagentResult] = {}
        for task_id in plan.blocked_by(failed):
            if task_id in results:
                continue
            task = plan.get(task_id)
            blocking = sorted(set(task.depends_on) & (failed | set(plan.blocked_by(failed))))
            skipped[task_id] = SubagentResult(
                task_id=task_id,
                subagent=task.subagent,
                status=TaskStatus.SKIPPED,
                error=f"upstream task(s) did not complete: {', '.join(blocking)}",
            )

        if failed:
            status = RunStatus.PARTIAL if completed else RunStatus.FAILED
        else:
            status = RunStatus.COMPLETED

        parts = [f"{len(completed)}/{len(plan.tasks)} tasks completed"]
        if failed:
            parts.append(f"failed: {', '.join(sorted(failed))}")
        if skipped:
            parts.append(f"skipped: {', '.join(sorted(skipped))}")
        summary = "; ".join(parts)

        self._finish(state, status, summary)
        return {"status": status, "summary": summary, "results": skipped}

    def _finish(self, state: RunState, status: RunStatus, summary: str) -> None:
        self._audit.finish_run(state["run_id"], status, summary)

    # ---- assembly ---------------------------------------------------------

    def _build(self) -> StateGraph:
        graph: StateGraph = StateGraph(RunState)
        graph.add_node("plan", self._plan)
        graph.add_node("dispatch", self._dispatch)
        graph.add_node("run_task", self._run_task, input_schema=TaskEnvelope)
        graph.add_node("aggregate", self._aggregate)

        graph.add_edge(START, "plan")
        graph.add_edge("plan", "dispatch")
        graph.add_conditional_edges("dispatch", self._route, ["run_task", "aggregate"])
        graph.add_edge("run_task", "dispatch")
        graph.add_edge("aggregate", END)
        return graph

    # ---- execution --------------------------------------------------------

    def run(self, goal: str, *, run_id: str | None = None) -> RunState:
        """Start a run and drive it to completion. Returns the final state."""
        run_id = run_id or self._audit.start_run(goal)
        initial: RunState = {
            "run_id": run_id,
            "goal": goal,
            "plan": None,
            "results": {},
            "dispatched": [],
            "status": str(RunStatus.RUNNING),
        }
        config = {"configurable": {"thread_id": run_id}}
        return self._graph.invoke(initial, config=config)  # type: ignore[arg-type]

    def resume(self, run_id: str) -> RunState:
        """Continue a checkpointed run from wherever it stopped.

        Requires a checkpointer — without one the graph keeps no state between calls.
        """
        config = {"configurable": {"thread_id": run_id}}
        return self._graph.invoke(None, config=config)  # type: ignore[arg-type]


def sqlite_checkpointer(path: Path | str = DEFAULT_CHECKPOINT_PATH) -> SqliteSaver:
    """Checkpointer for the Phase 0 spike.

    Postgres replaces this in Phase 1 (`langgraph-checkpoint-postgres`); the swap is this
    function, because nothing else names the backend. `check_same_thread=False` because
    parallel task branches execute on worker threads.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    return SqliteSaver(conn)
