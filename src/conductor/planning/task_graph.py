"""The task graph — FR-O2's "decompose a goal into a task graph", as a real object.

The point of making this a validated Pydantic model rather than prose in a prompt: the
orchestrator can *reason about* it. Dependencies give layered parallel dispatch (FR-S3),
cycle detection catches an unrunnable plan at plan time instead of as a hang, and a task
naming an unregistered subagent fails before any work starts.

Dependencies are declared, never inferred. A planner that lists every task as depending on
the previous one produces a correct but fully serial run, so the prompt asks explicitly for
*real* dependencies only — that single instruction is most of what parallelism costs here.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class TaskStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class Task(BaseModel):
    """One unit of delegated work. Maps 1:1 onto a subagent invocation."""

    id: str = Field(description="Short stable identifier, e.g. 'implement' or 'review'")
    subagent: str = Field(description="Name of a registered subagent — the planner picks")
    objective: str = Field(
        description="What this subagent must accomplish, in one or two sentences. "
        "Scoped to this task only."
    )
    depends_on: list[str] = Field(
        default_factory=list,
        description="Task ids that must complete first. Declare only REAL data or ordering "
        "dependencies — anything listed here cannot run in parallel.",
    )
    inputs: dict[str, str] = Field(
        default_factory=dict,
        description="Concrete parameters the subagent needs, e.g. {'issue': '42'}",
    )
    acceptance: str = Field(
        default="", description="How the orchestrator will know this task succeeded"
    )


class TaskGraph(BaseModel):
    """A validated, acyclic plan. Structural validity is guaranteed by construction."""

    goal: str
    tasks: list[Task] = Field(min_length=1)
    rationale: str = Field(
        default="", description="One or two sentences on why this decomposition"
    )

    @model_validator(mode="after")
    def _structurally_sound(self) -> TaskGraph:
        ids = [t.id for t in self.tasks]
        if len(ids) != len(set(ids)):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"duplicate task ids: {dupes}")

        known = set(ids)
        for task in self.tasks:
            unknown = [d for d in task.depends_on if d not in known]
            if unknown:
                raise ValueError(f"task {task.id!r} depends on unknown task(s) {unknown}")
            if task.id in task.depends_on:
                raise ValueError(f"task {task.id!r} depends on itself")

        self._layers()  # raises on a cycle
        return self

    # ---- topology ---------------------------------------------------------

    def _layers(self) -> list[list[Task]]:
        """Kahn's algorithm. Each layer is a set of tasks that may run in parallel."""
        by_id = {t.id: t for t in self.tasks}
        remaining = {t.id: set(t.depends_on) for t in self.tasks}
        layers: list[list[Task]] = []

        while remaining:
            ready = sorted(tid for tid, deps in remaining.items() if not deps)
            if not ready:
                raise ValueError(
                    f"task graph has a dependency cycle among: {sorted(remaining)}"
                )
            layers.append([by_id[t] for t in ready])
            for tid in ready:
                del remaining[tid]
            for deps in remaining.values():
                deps.difference_update(ready)
        return layers

    def layers(self) -> list[list[Task]]:
        return self._layers()

    @property
    def width(self) -> int:
        """Widest parallel layer — how much concurrency this plan actually allows."""
        return max((len(layer) for layer in self.layers()), default=0)

    def get(self, task_id: str) -> Task:
        for task in self.tasks:
            if task.id == task_id:
                return task
        raise KeyError(f"no task {task_id!r} in plan")

    # ---- dispatch ---------------------------------------------------------

    def ready(self, completed: set[str], started: set[str] | None = None) -> list[Task]:
        """Tasks whose dependencies are satisfied and which have not been dispatched.

        `started` is separate from `completed` so the dispatch loop does not re-send a
        task that is still in flight.
        """
        started = started or set()
        return [
            t
            for t in self.tasks
            if t.id not in started
            and t.id not in completed
            and set(t.depends_on) <= completed
        ]

    def blocked_by(self, failed: set[str]) -> set[str]:
        """Tasks that can never run because something upstream failed.

        Used for partial-result aggregation (FR-O7): a failed implementation should mark
        the review as skipped rather than leaving the run hanging on it.
        """
        blocked: set[str] = set()
        changed = True
        while changed:
            changed = False
            for task in self.tasks:
                if task.id in blocked or task.id in failed:
                    continue
                if set(task.depends_on) & (failed | blocked):
                    blocked.add(task.id)
                    changed = True
        return blocked

    def unknown_subagents(self, registered: set[str]) -> dict[str, str]:
        """{task_id: subagent} for tasks routed to a subagent that does not exist."""
        return {t.id: t.subagent for t in self.tasks if t.subagent not in registered}

    def render(self) -> str:
        """Compact human/audit rendering, one line per task."""
        lines = [f"goal: {self.goal}"]
        for i, layer in enumerate(self.layers(), start=1):
            for task in layer:
                deps = f" <- {', '.join(task.depends_on)}" if task.depends_on else ""
                lines.append(f"  [{i}] {task.id} ({task.subagent}){deps}: {task.objective}")
        return "\n".join(lines)
