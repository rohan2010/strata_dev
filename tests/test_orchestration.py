import pytest
from langchain_core.messages import AIMessage

from conductor.audit import AuditStore, EventType, RunStatus
from conductor.connectors import Connector, ConnectorRegistry, ToolSpec
from conductor.models import ModelResult, ModelSpec
from conductor.orchestration import Orchestrator, sqlite_checkpointer
from conductor.planning import TaskGraph
from conductor.subagents import SubagentRegistry

SPEC = ModelSpec(alias="fake", provider="anthropic", id="x")


class FakeConnector(Connector):
    name = "github"

    def __init__(self):
        self.calls = []

    def tools(self):
        return [
            ToolSpec(name="github.get_issue", description="read an issue",
                     parameters={"number": "issue number"}),
            ToolSpec(name="github.list_files", description="list files",
                     parameters={"path": "dir"}, optional=["path"]),
            ToolSpec(name="github.read_file", description="read a file",
                     parameters={"path": "p", "ref": "r"}, optional=["ref"]),
            ToolSpec(name="github.get_pr_diff", description="pr diff",
                     parameters={"number": "n"}),
            ToolSpec(name="github.create_branch", description="branch",
                     parameters={"name": "n", "from_ref": "r"}, optional=["from_ref"],
                     mutates=True),
            ToolSpec(name="github.write_file", description="write",
                     parameters={"path": "p", "content": "c", "branch": "b", "message": "m"},
                     mutates=True),
            ToolSpec(name="github.open_pr", description="open pr",
                     parameters={"head": "h", "base": "b", "title": "t", "body": "d"},
                     optional=["body"], mutates=True),
        ]

    def invoke(self, tool, arguments):
        self.calls.append((tool, arguments))
        return {"ok": tool}


class ScriptedClient:
    """Returns queued responses per actor. Anything unqueued gets a plain completion."""

    def __init__(self, plan: TaskGraph, scripts: dict | None = None):
        self.plan = plan
        self.scripts = {k: list(v) for k, v in (scripts or {}).items()}
        self.calls = []

    def invoke(self, role, messages, **kw):
        actor = kw.get("actor", role)
        self.calls.append((actor, messages, kw))

        if kw.get("schema") is not None:
            return ModelResult(text="", spec=SPEC, parsed=self.plan)

        queued = self.scripts.get(actor)
        if queued:
            item = queued.pop(0)
            if isinstance(item, Exception):
                raise item
            if isinstance(item, AIMessage):
                return ModelResult(
                    text=item.text,
                    spec=SPEC,
                    message=item,
                    tool_calls=list(item.tool_calls or []),
                    cost_usd=0.01,
                )
            return ModelResult(text=item, spec=SPEC, message=AIMessage(item), cost_usd=0.01)

        return ModelResult(text=f"{actor} done", spec=SPEC, message=AIMessage("done"),
                           cost_usd=0.01)


def plan_of(*tasks, goal="fix login") -> TaskGraph:
    return TaskGraph.model_validate({"goal": goal, "tasks": list(tasks)})


SERIAL = plan_of(
    {"id": "impl", "subagent": "implementer", "objective": "fix it"},
    {"id": "rev", "subagent": "reviewer", "objective": "review", "depends_on": ["impl"]},
)

PARALLEL = plan_of(
    {"id": "impl", "subagent": "implementer", "objective": "fix it"},
    {"id": "rev", "subagent": "reviewer", "objective": "read the issue"},
)


@pytest.fixture
def audit(tmp_path):
    s = AuditStore(tmp_path / "t.db")
    yield s
    s.close()


@pytest.fixture
def connectors():
    r = ConnectorRegistry()
    r.register(FakeConnector())
    return r


def build(audit, connectors, plan, scripts=None, **kw):
    client = ScriptedClient(plan, scripts)
    orch = Orchestrator(
        client=client,
        subagents=SubagentRegistry.from_dir("subagents"),
        connectors=connectors,
        audit=audit,
        **kw,
    )
    return orch, client


# ---- happy path ----------------------------------------------------------


def test_serial_plan_runs_to_completion(audit, connectors):
    orch, _ = build(audit, connectors, SERIAL)
    final = orch.run("fix login")

    assert final["status"] == RunStatus.COMPLETED
    assert set(final["results"]) == {"impl", "rev"}
    assert all(r.ok for r in final["results"].values())
    assert "2/2 tasks completed" in final["summary"]


def test_run_is_fully_traced(audit, connectors):
    orch, _ = build(audit, connectors, SERIAL)
    final = orch.run("fix login")

    types = [e.event_type for e in audit.events(final["run_id"])]
    assert types[0] is EventType.RUN_STARTED
    assert types[-1] is EventType.RUN_FINISHED
    assert types.count(EventType.DELEGATION) == 2
    assert types.count(EventType.SUBAGENT_RESULT) == 2
    assert EventType.PLAN_CREATED in types
    assert audit.get_run(final["run_id"])["status"] == RunStatus.COMPLETED


def test_independent_tasks_share_a_dispatch_layer(audit, connectors):
    """FR-S3: the plan's width is what makes parallel dispatch possible."""
    assert PARALLEL.width == 2
    orch, _ = build(audit, connectors, PARALLEL)
    final = orch.run("fix login")
    assert set(final["results"]) == {"impl", "rev"}


def test_dependent_task_sees_upstream_summary_but_not_its_transcript(audit, connectors):
    orch, client = build(
        audit, connectors, SERIAL, scripts={"implementer": ["opened PR #7 on branch fix/login"]}
    )
    orch.run("fix login")

    reviewer_msgs = [m for actor, m, _ in client.calls if actor == "reviewer"][0]
    human_turn = reviewer_msgs[1][1]
    assert "opened PR #7" in human_turn
    # The implementer's own conversation must not appear anywhere in the reviewer's context.
    assert "system" not in human_turn.lower() or "You are the Implementer" not in human_turn


# ---- isolation -----------------------------------------------------------


def test_orchestrator_state_never_accumulates_transcripts(audit, connectors):
    """The load-bearing architectural claim: state holds summaries, not messages."""
    orch, _ = build(audit, connectors, SERIAL)
    final = orch.run("fix login")

    assert "messages" not in final
    for result in final["results"].values():
        assert not hasattr(result, "messages")
    assert set(final) <= {"run_id", "goal", "plan", "results", "dispatched", "status", "summary"}


# ---- tools ---------------------------------------------------------------


def test_subagent_tool_call_is_dispatched_and_audited(audit, connectors):
    call = AIMessage(
        content="",
        tool_calls=[{"name": "github__get_issue", "args": {"number": "42"}, "id": "c1"}],
    )
    orch, _ = build(audit, connectors, SERIAL, scripts={"implementer": [call, "fixed it"]})
    final = orch.run("fix login")

    assert connectors.connectors()[0].calls == [("github.get_issue", {"number": "42"})]
    assert final["results"]["impl"].tool_calls == 1

    events = audit.events(final["run_id"])
    assert any(
        e.event_type is EventType.TOOL_CALL and e.payload["tool"] == "github.get_issue"
        for e in events
    )


def test_wire_names_round_trip_through_the_dot(audit, connectors):
    """Providers reject dots in tool names; the toolbox must map back before dispatch."""
    from conductor.connectors import ScopedToolbox

    manifest = SubagentRegistry.from_dir("subagents").get("implementer")
    box = ScopedToolbox(connectors, manifest, audit, audit.start_run("g"))

    names = [d["name"] for d in box.tool_definitions()]
    assert "github__get_issue" in names
    assert not any("." in n for n in names)
    assert box.resolve("github__get_issue") == "github.get_issue"


def test_only_allow_listed_tools_are_offered_to_the_model(audit, connectors):
    from conductor.connectors import ScopedToolbox

    reviewer = SubagentRegistry.from_dir("subagents").get("reviewer")
    box = ScopedToolbox(connectors, reviewer, audit, audit.start_run("g"))
    offered = {d["name"] for d in box.tool_definitions()}

    assert "github__get_pr_diff" in offered
    assert "github__write_file" not in offered
    assert "github__open_pr" not in offered


def test_blocked_tool_is_reported_to_the_model_not_fatal(audit, connectors):
    """A reviewer reaching for a write tool should be told no, then carry on."""
    call = AIMessage(
        content="",
        tool_calls=[{"name": "github__write_file", "args": {}, "id": "c1"}],
    )
    orch, _ = build(
        audit, connectors, PARALLEL, scripts={"reviewer": [call, "reviewed, no write needed"]}
    )
    final = orch.run("fix login")

    assert final["results"]["rev"].ok
    assert connectors.connectors()[0].calls == []


def test_runaway_tool_loop_is_capped(audit, connectors):
    call = AIMessage(
        content="", tool_calls=[{"name": "github__get_issue", "args": {"number": "1"}, "id": "c"}]
    )
    orch, _ = build(
        audit, connectors, PARALLEL,
        scripts={"implementer": [call] * 10},
        max_tool_iterations=3,
    )
    final = orch.run("fix login")

    assert not final["results"]["impl"].ok
    assert "exceeded 3 tool iterations" in final["results"]["impl"].error


# ---- failure containment (FR-O7) -----------------------------------------


def test_failed_task_skips_its_dependents_and_run_is_partial(audit, connectors):
    orch, _ = build(
        audit, connectors, SERIAL, scripts={"implementer": [RuntimeError("provider exploded")]}
    )
    final = orch.run("fix login")

    # Nothing completed, so the run is FAILED rather than PARTIAL — but `rev` must still
    # appear, marked skipped with its cause, rather than silently missing from the trace.
    assert final["status"] == RunStatus.FAILED
    assert not final["results"]["impl"].ok
    assert final["results"]["rev"].status == "skipped"
    assert "impl" in final["results"]["rev"].error


def test_all_tasks_failing_is_a_failed_run_not_a_partial(audit, connectors):
    orch, _ = build(
        audit,
        connectors,
        PARALLEL,
        scripts={
            "implementer": [RuntimeError("boom")],
            "reviewer": [RuntimeError("boom")],
        },
    )
    final = orch.run("fix login")
    assert final["status"] == RunStatus.FAILED


def test_one_failure_among_independent_tasks_is_partial(audit, connectors):
    orch, _ = build(
        audit, connectors, PARALLEL, scripts={"implementer": [RuntimeError("boom")]}
    )
    final = orch.run("fix login")

    assert final["status"] == RunStatus.PARTIAL
    assert final["results"]["rev"].ok
    assert audit.get_run(final["run_id"])["status"] == RunStatus.PARTIAL


# ---- checkpointing -------------------------------------------------------


def test_run_is_checkpointed_and_resumable(audit, connectors, tmp_path):
    checkpointer = sqlite_checkpointer(tmp_path / "cp.db")
    orch, _ = build(audit, connectors, SERIAL, checkpointer=checkpointer)
    final = orch.run("fix login")

    state = orch.graph.get_state({"configurable": {"thread_id": final["run_id"]}})
    assert state.values["status"] == RunStatus.COMPLETED
    assert set(state.values["results"]) == {"impl", "rev"}
