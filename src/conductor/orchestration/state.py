"""Graph state, and the isolation boundary that keeps it small.

The architectural commitment from EXECUTION-PLAN.md section 1: *each subagent node owns its
own `messages` list; only a structured summary is written back to graph state.* This module
is where that is enforced — `RunState` has nowhere to put a transcript. A subagent's
conversation lives inside its node invocation and is discarded when the node returns; what
survives is a `SubagentResult`.

That is not tidiness. Orchestrator context that grows with every subagent's transcript is
the failure mode this architecture exists to avoid: it caps how many subagents a run can
use, and it re-pays for the whole accumulated history on every subsequent orchestrator turn.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from pydantic import BaseModel, Field

from conductor.planning import TaskGraph, TaskStatus


class SubagentResult(BaseModel):
    """The only thing a subagent hands back to the orchestrator."""

    task_id: str
    subagent: str
    status: TaskStatus
    summary: str = Field(default="", description="Prose result, kept short by the prompt")
    artifacts: dict[str, Any] = Field(
        default_factory=dict, description="Structured outputs: PR urls, branch names, findings"
    )
    error: str | None = None
    tool_calls: int = 0
    model: str = ""
    cost_usd: float | None = None

    @property
    def ok(self) -> bool:
        return self.status is TaskStatus.COMPLETED


def merge_results(
    left: dict[str, SubagentResult], right: dict[str, SubagentResult]
) -> dict[str, SubagentResult]:
    """Reducer for parallel branches.

    Independent tasks in one layer complete concurrently and each write one key, so the
    merge is a plain dict update — but it must exist, or LangGraph rejects concurrent
    writes to the same channel.
    """
    return {**left, **right}


class RunState(TypedDict, total=False):
    """What flows through the graph. Note what is absent: any subagent's messages."""

    run_id: str
    goal: str
    plan: TaskGraph | None
    results: Annotated[dict[str, SubagentResult], merge_results]
    dispatched: Annotated[list[str], lambda a, b: sorted(set(a) | set(b))]
    status: str
    summary: str


def completed_ids(state: RunState) -> set[str]:
    return {tid for tid, r in state.get("results", {}).items() if r.ok}


def failed_ids(state: RunState) -> set[str]:
    return {tid for tid, r in state.get("results", {}).items() if not r.ok}
